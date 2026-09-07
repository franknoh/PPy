"""AST -> canonical IR for the natively lowerable subset.

One function at a time: parameters become entry-block arguments held in
stack slots (a later pass promotes them), Python control flow becomes an
explicit graph of blocks with arguments, and every place the old lowering
branched to its fallback is a `core.guard`. Integer arithmetic carries
`overflow = "python"` -- or `wrap` when the safeguards are off -- so that a
backend emits the guard the source semantics require and never guesses.

What the subset excludes is refused with `Unsupported` and the reason, and
the caller records the function as running on CPython.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from ppy_runtime.abi import NativeSignature

from ..analysis import types as T
from ..analysis.checker import FunctionAnalysis, ModuleAnalysis
from ..analysis.symbols import FunctionInfo
from ..backend.llvm.lowering import (
    _ALLOCATIONS,
    _MATH_INTRINSICS,
    _MAX_TUPLE_WIDTH,
    _NARROW,
    ClassLayouts,
    LoweredFunction,
    Unsupported,
    _declared_bounds,
    _read_as,
    _return_atoms,
    _signature,
    eligible,
    should_lower_native,
)
from ..backend.llvm.obligations import BinOp, Const, Obligation, Relation, Term, Var, variables
from ..backend.llvm.prover import Prover
from ..ir import (
    BOOL,
    F64,
    I8,
    I64,
    U8,
    Block,
    BlockArgument,
    BufferType,
    Builder,
    IRFunction,
    IRModule,
    IRType,
    Operation,
    PtrType,
    SourceLocation,
    StructType,
    Successor,
    TupleType,
    Value,
)
from ..ir.dialects import core
from ..ir.dialects import math as math_dialect

__all__ = ["Frontend", "Lowered", "lower_function", "lower_module_to_ir"]

_SCALAR_TYPES: dict[str, IRType] = {"int": I64, "float": F64, "bool": BOOL, "i8": I8, "u8": U8}
_KINDS: dict[IRType, str] = {I64: "int", F64: "float", BOOL: "bool", I8: "i8", U8: "u8"}
_COMPARISONS = {
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
}
_ARITHMETIC = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}
_BITWISE = {ast.BitAnd: "and", ast.BitOr: "or", ast.BitXor: "xor"}


def _scalar_type(kind: str) -> IRType:
    return _SCALAR_TYPES[kind]


def _kind(t: IRType) -> str:
    found = _KINDS.get(t)
    if found is None:
        if isinstance(t, PtrType):
            raise Unsupported("a pointer is moved with `ppy.native.offset`, not arithmetic")
        raise Unsupported(f"{t} has no scalar kind")
    return found


def _struct_type(class_name: str, fields: tuple[tuple[str, str], ...]) -> StructType:
    return StructType(class_name.replace(".", "_"), tuple((f, _scalar_type(k)) for f, k in fields))


@dataclass(slots=True)
class Lowered:
    """One module's IR with what the driver records per function."""

    module: IRModule
    functions: dict[str, LoweredFunction] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)
    #: Per function, the arithmetic whose overflow guard a proof left out.
    proved: dict[str, tuple[str, ...]] = field(default_factory=dict)


def lower_module_to_ir(
    module: ModuleAnalysis,
    functions: dict[str, tuple[FunctionInfo, FunctionAnalysis, ast.FunctionDef]],
    layouts: ClassLayouts | None = None,
    *,
    safeguards: str = "hoisted",
    standalone: bool = False,
    prover: Prover | None = None,
    root: Path | None = None,
) -> Lowered:
    """The IR of every eligible function in one module."""
    frontend = Frontend(
        module, layouts, safeguards=safeguards, standalone=standalone, prover=prover, root=root
    )
    return frontend.build(functions)


def lower_function(
    module: ModuleAnalysis,
    info: FunctionInfo,
    node: ast.FunctionDef,
    *,
    constants: dict[str, object] | None = None,
    symbol: str | None = None,
    layouts: ClassLayouts | None = None,
    safeguards: str = "hoisted",
    prover: Prover | None = None,
) -> IRModule:
    """One function, optionally with parameters pinned to constants.

    A specialization keeps the generic ABI -- the argument is still passed --
    and the body uses the constant, which is what lets the optimizer fold
    around it.
    """
    frontend = Frontend(module, layouts, safeguards=safeguards, prover=prover)
    analysis = module.functions.get(info.qualname)
    signature = _signature(info, layouts, analysis)
    if symbol is not None:
        signature = NativeSignature(
            qualname=signature.qualname,
            symbol=symbol,
            parameters=signature.parameters,
            returns=signature.returns,
            releases_gil=signature.releases_gil,
        )
    frontend.declare(info, signature)
    frontend.define(info, node, constants or {})
    return frontend.module


class Frontend:
    """Builds one IR module for one analyzed Python module."""

    def __init__(
        self,
        analysis: ModuleAnalysis,
        layouts: ClassLayouts | None = None,
        *,
        safeguards: str = "hoisted",
        standalone: bool = False,
        prover: Prover | None = None,
        root: Path | None = None,
    ) -> None:
        self.analysis = analysis
        self.layouts: ClassLayouts = dict(layouts or {})
        self.safeguards = safeguards
        self.standalone = standalone
        self.prover = prover
        #: Source locations are spelled relative to this, so the IR text is
        #: the same wherever the project sits.
        self.root = root
        self.module = IRModule(analysis.name)
        #: qualname -> (IR function, its native signature), for calls.
        self.declared: dict[str, tuple[IRFunction, NativeSignature]] = {}
        #: Generic functions by qualname, lowered per instantiation.
        self.generics: dict[str, tuple[FunctionInfo, FunctionAnalysis, ast.FunctionDef]] = {}
        #: Instantiations made so far: (qualname, type arguments) -> declaration.
        self.instances: dict[tuple[str, tuple[str, ...]], tuple[IRFunction, NativeSignature]] = {}
        #: C bindings by qualname: the stub and its directive's options.
        self.externs: dict[str, tuple[FunctionInfo, dict[str, object]]] = {}
        self._instantiating: list[tuple[str, tuple[str, ...]]] = []

    def build(
        self, functions: dict[str, tuple[FunctionInfo, FunctionAnalysis, ast.FunctionDef]]
    ) -> Lowered:
        lowered = Lowered(self.module)
        candidates: dict[str, tuple[FunctionInfo, FunctionAnalysis, ast.FunctionDef]] = {}
        self.generics: dict[str, tuple[FunctionInfo, FunctionAnalysis, ast.FunctionDef]] = {}
        for qualname, (info, analysis, node) in functions.items():
            extern = info.directive("native.extern")
            if extern is not None:
                # A C binding: called, never lowered.
                self.externs[qualname] = (info, dict(extern.options))
                lowered.rejected[qualname] = "a C binding has no body of its own"
                continue
            if info.type_params:
                # Lowered per instantiation, when a native caller names one.
                self.generics[qualname] = (info, analysis, node)
                lowered.rejected[qualname] = (
                    "a generic function is specialized where it is called; "
                    "it has no single native entry point"
                )
                continue
            ok, reason = eligible(info, analysis, self.layouts, allow_io=self.standalone)
            if ok:
                candidates[qualname] = (info, analysis, node)
            else:
                lowered.rejected[qualname] = reason
        for info, analysis, _node in candidates.values():
            self.declare(info, _signature(info, self.layouts, analysis))
        for qualname, (info, analysis, node) in candidates.items():
            try:
                proved = self.define(info, node, {})
            except Unsupported as error:
                lowered.rejected[qualname] = str(error)
                self._drop(qualname)
                continue
            if proved:
                lowered.proved[qualname] = tuple(proved)
            exposed, why = should_lower_native(info, analysis)
            lowered.functions[qualname] = LoweredFunction(
                info, self.declared[qualname][1], exposed=exposed, exposure_reason=why
            )
        self._reject_callers_of_rejected(lowered)
        return lowered

    def _effects_of(self, info: FunctionInfo) -> tuple[str, ...]:
        """What the IR says the function may do: the analysis's effects, with
        a write through a buffer spelled as the native memory write it is."""
        analysis = self.analysis.functions.get(info.qualname)
        effects = info.effects if analysis is None else analysis.effects
        spelled = set(effects.spelled())
        if analysis is not None and (analysis.mutated_params or analysis.delegated_writes):
            spelled.add("write_memory")
        if any(p.is_buffer for p in _signature(info, self.layouts).parameters):
            spelled.add("read_memory")
        return tuple(sorted(spelled))

    def declare(self, info: FunctionInfo, signature: NativeSignature) -> IRFunction:
        params: list[tuple[str, IRType]] = []
        attributes: dict[str, object] = {}
        kinds: list[dict[str, object]] = []
        facts_by_name = {p.name: p.facts for p in info.params}
        for parameter in signature.parameters:
            params.append((parameter.name, _param_type(parameter)))
            described: dict[str, object] = {}
            if parameter.is_buffer:
                described["ppy.kind"] = parameter.kind
            elif parameter.is_object:
                described["ppy.class"] = parameter.class_name
            facts = facts_by_name.get(parameter.name)
            ownership = facts.ownership if facts is not None else None
            if ownership is None and parameter.is_buffer:
                # A buffer is borrowed for the call unless the program says
                # otherwise; a `Sequence` or list is copied in, so it is owned.
                ownership = "borrowed" if parameter.is_borrowed else "owned"
            if ownership is not None:
                described["ownership"] = ownership
            if facts is not None and facts.no_alias:
                described["noalias"] = True
            kinds.append(described)
        results = _result_types(info)
        function = self.module.add_function(
            info.qualname.replace(".", "_"),
            params,
            results,
            attributes={
                "ppy.symbol": signature.symbol,
                "ppy.qualname": info.qualname,
                "ppy.abi": "ppy",
                "ppy.releases_gil": signature.releases_gil,
                "effects": self._effects_of(info),
                **({"fastmath": True} if info.directive("fastmath") is not None else {}),
                **attributes,
            },
            location=SourceLocation(self.spell(info.path), info.node.lineno, info.node.col_offset),
        )
        function.param_attributes = kinds
        export = info.directive("native.export")
        if export is not None:
            if len(results) != 1 or isinstance(results[0], TupleType):
                raise Unsupported("a C export returns one scalar")
            function.attributes["ppy.export"] = str(export.options.get("name") or info.name)
        self.declared[info.qualname] = (function, signature)
        return function

    def spell(self, path: Path) -> str:
        """A source path as the IR writes it: relative to the project root."""
        if self.root is not None:
            try:
                return str(Path(path).resolve().relative_to(Path(self.root).resolve()))
            except ValueError:
                pass
        return Path(path).name

    def define(
        self, info: FunctionInfo, node: ast.FunctionDef, constants: dict[str, object]
    ) -> list[str]:
        """Lower the body; returns the chains a proof freed of their guard."""
        function, signature = self.declared[info.qualname]
        lowering = _FunctionLowering(self, function, signature, info, constants)
        lowering.run(node)
        return lowering.proved

    def _drop(self, qualname: str) -> None:
        self.declared[qualname][0].body.blocks.clear()

    def instantiate(
        self, qualname: str, arguments: tuple[T.Type, ...]
    ) -> tuple[IRFunction, NativeSignature] | None:
        """The native function `qualname[arguments]`, made now if it is new.

        Monomorphization: the generic's body is lowered with its type
        parameters replaced by the arguments, under a name that spells them,
        so two callers with the same arguments share one function and two
        with different ones get two. A body the arguments do not lower --
        `a + b` on a type with no native `+` -- refuses with the reason.
        """
        entry = self.generics.get(qualname)
        if entry is None:
            return None
        info, analysis, node = entry
        key = (qualname, tuple(str(a) for a in arguments))
        found = self.instances.get(key)
        if found is not None:
            return found
        if key in self._instantiating:
            raise Unsupported(f"`{qualname}` instantiates itself with the same arguments")
        bindings = dict(zip(info.type_params, arguments, strict=True))
        specialized = _specialized_info(info, bindings, key[1])
        ok, reason = eligible(specialized, analysis, self.layouts, allow_io=self.standalone)
        if not ok:
            raise Unsupported(f"`{qualname}[{', '.join(key[1])}]` has no native lowering: {reason}")
        signature = _signature(specialized, self.layouts, analysis)
        function = self.declare(specialized, signature)
        function.attributes["ppy.generic"] = qualname
        function.attributes["ppy.type_arguments"] = key[1]
        self.instances[key] = (function, signature)
        self._instantiating.append(key)
        try:
            _FunctionLowering(self, function, signature, specialized, {}).run(node)
        except Unsupported:
            del self.instances[key]
            del self.declared[specialized.qualname]
            del self.module.functions[function.name]
            raise
        finally:
            self._instantiating.pop()
        return self.instances[key]

    def _reject_callers_of_rejected(self, lowered: Lowered) -> None:
        """A caller of a function that did not lower runs on CPython too."""
        while True:
            blocked: dict[str, str] = {}
            for qualname in lowered.functions:
                function = self.declared[qualname][0]
                for op in function.operations():
                    if op.name != "core.call":
                        continue
                    callee = op.attributes["callee"].name  # type: ignore[union-attr]
                    target = self.module.functions.get(callee)
                    if target is None or target.is_declaration:
                        blocked[qualname] = callee
                        break
            if not blocked:
                return
            for qualname, callee in blocked.items():
                source = next(
                    (q for q, (f, _s) in self.declared.items() if f.name == callee), callee
                )
                lowered.rejected[qualname] = (
                    f"`{source}` has no native lowering, so this call cannot be made"
                )
                del lowered.functions[qualname]
                self._drop(qualname)


def _param_type(parameter) -> IRType:  # type: ignore[no-untyped-def]
    if parameter.is_buffer:
        return BufferType(_scalar_type(parameter.element))
    if parameter.is_pointer:
        return PtrType(_scalar_type(parameter.element), mutable=parameter.kind == "ptr")
    if parameter.is_tuple:
        return TupleType(tuple(_scalar_type(e) for e in parameter.elements))
    if parameter.is_object:
        return _struct_type(parameter.class_name, parameter.fields)
    return _scalar_type(parameter.kind)


def _result_types(info: FunctionInfo) -> tuple[IRType, ...]:
    atoms = _return_atoms(info.ret)
    if atoms is None:
        return ()
    if len(atoms) == 1:
        return (_scalar_type(atoms[0]),)
    return (TupleType(tuple(_scalar_type(a) for a in atoms)),)


class _GuardSite:
    """An open block ahead of one loop, collecting that loop's hoisted guards.

    The block sits between the loop's bound computation and its setup, so
    every check in it runs once per loop *entry* instead of once per
    iteration -- and a body freed of its side exits is one the optimizer can
    strength-reduce and vectorize. A failed check takes the road a failed
    inline guard takes: the function's fallback, and CPython.
    """

    def __init__(self, block: Block) -> None:
        self.block = block
        self.b = Builder(block)

    def bail_if(self, overflowed: Value, kind: str) -> None:
        ok = core.bitwise(self.b, "xor", overflowed, core.const(self.b, True, BOOL))
        core.guard(self.b, ok, kind, "hoisted guard")

    def finish(self, setup: Block) -> None:
        core.br(self.b, Successor(setup))


class _FunctionLowering:
    """Lowers one function body."""

    def __init__(
        self,
        frontend: Frontend,
        function: IRFunction,
        signature: NativeSignature,
        info: FunctionInfo,
        constants: dict[str, object],
    ) -> None:
        self.frontend = frontend
        self.function = function
        self.signature = signature
        self.info = info
        self.constants = constants
        self.overflow = "wrap" if frontend.safeguards == "off" else "python"
        self.hoist = frontend.safeguards != "inline"
        #: Proves an arithmetic chain fits the word, so its guard is left out.
        self.prover = frontend.prover if frontend.safeguards != "off" else None
        self.entry: Block | None = None
        self.b = Builder()
        #: Scalar locals: name -> the stack slot holding it.
        self.slots: dict[str, Value] = {}
        #: Buffer parameters, and standalone allocations: name -> buffer value.
        self.buffers: dict[str, Value] = {}
        #: Tuple locals: name -> the stack slot holding the whole tuple.
        self.tuples: dict[str, Value] = {}
        #: Value-class parameters: name -> the struct value.
        self.objects: dict[str, Value] = {}
        self._loops: list[tuple[Block, Block]] = []
        self._labels = 0
        #: What each integer value is, as a statement about integers: the
        #: term the prover reasons over. A value with no term keeps its guard.
        self._terms: dict[Value, Term] = {}
        #: The relations a loaded value carries: an induction variable's
        #: bounds, from the `range()` that drives it.
        self._relations: dict[Var, tuple[Relation, ...]] = {}
        self._induction_terms: dict[str, tuple[Term | None, Term | None]] = {}
        self._loads = 0
        #: Every chain a proof freed of its guard, spelled.
        self.proved: list[str] = []
        #: Open guard blocks, one per active hoistable loop, outermost first.
        self._guard_sites: list[_GuardSite] = []
        #: Induction variable name -> (lo, hi, depth): values that dominate
        #: the loop's guard block.
        self._induction: dict[str, tuple[Value, Value, int]] = {}
        #: Value -> (lo, hi, depth): a proven range for a body value.
        self._ranges: dict[Value, tuple[Value, Value, int]] = {}
        #: Results whose guards were hoisted; their consumers hoist too.
        self._hoisted: set[Value] = set()
        #: Integer parameters the body never rebinds, and their entry loads.
        self._stable: set[str] = set()
        self._entry_loads: dict[str, Value] = {}

    # -- setup ------------------------------------------------------------

    def run(self, node: ast.FunctionDef) -> None:
        self.entry = self.function.add_entry_block()
        self.b.at_end(self.entry)
        self._location(node)
        for argument, parameter in zip(
            self.entry.arguments, self.signature.parameters, strict=True
        ):
            if parameter.is_buffer:
                self.buffers[parameter.name] = argument
                continue
            if parameter.is_pointer:
                slot = self._alloca(argument.type, parameter.name)
                core.store(self.b, argument, slot)
                self.slots[parameter.name] = slot
                continue
            if parameter.is_object:
                self.objects[parameter.name] = argument
                continue
            if parameter.is_tuple:
                slot = self._alloca(argument.type, parameter.name)
                core.store(self.b, argument, slot)
                self.tuples[parameter.name] = slot
                continue
            slot = self._alloca(argument.type, parameter.name)
            pinned = self.constants.get(parameter.name)
            initial = argument
            if pinned is not None:
                initial = self._literal(pinned, parameter.kind)
            core.store(self.b, initial, slot)
            self.slots[parameter.name] = slot
        if self.prover is not None:
            self._guard_declared_ranges()
        stored = {
            name.id
            for statement in node.body
            for name in ast.walk(statement)
            if isinstance(name, ast.Name) and isinstance(name.ctx, (ast.Store, ast.Del))
        }
        # Integer parameters the body never rebinds: their slot holds one
        # value forever, so a load in the entry block speaks for every read.
        self._stable = {
            name for name, slot in self.slots.items() if slot.type == PtrType(I64, "stack")
        } - stored
        self._body(node.body)
        if self._open():
            self._return_default()

    def _alloca(self, t: IRType, name: str) -> Value:
        """A stack slot in the entry block, where it is made once."""
        return core.alloca(self._entry_builder(), t, name=f"{name}.addr")

    def _entry_builder(self) -> Builder:
        assert self.entry is not None
        if self.b.block is self.entry and self.b.anchor is None:
            return self.b
        if self.entry.terminator is not None:
            return Builder().before(self.entry.operations[-1])
        return Builder(self.entry)

    def _entry_load(self, name: str) -> Value:
        """One load of a never-rebound parameter, placed to dominate everything."""
        existing = self._entry_loads.get(name)
        if existing is not None:
            return existing
        value = core.load(self._entry_builder(), self.slots[name], name=f"{name}.entry")
        self._entry_loads[name] = value
        return value

    def _block(self, label: str) -> Block:
        self._labels += 1
        return self.function.body.add_block(f"{label}{self._labels}")

    def _open(self) -> bool:
        block = self.b.block
        return block is not None and block.terminator is None

    def _location(self, node: ast.AST) -> None:
        line = getattr(node, "lineno", None)
        if line is not None:
            self.b.location = SourceLocation(
                self.frontend.spell(self.info.path), line, getattr(node, "col_offset", 0)
            )

    def _literal(self, value: object, kind: str) -> Value:
        if kind == "float":
            return core.const(self.b, float(value), F64)  # type: ignore[arg-type]
        if kind == "bool":
            return core.const(self.b, bool(value), BOOL)
        return self._int_constant(int(value), _scalar_type(kind))  # type: ignore[call-overload]

    def _int_constant(self, value: int, t: IRType = I64) -> Value:
        """An integer literal, made in the entry block so it dominates every use.

        A literal is its own range and its own term; a hoisted corner check
        in a loop's guard block may read it, and a value made in the body
        would not reach there.
        """
        constant = core.const(self._entry_builder(), value, t)
        if t == I64:
            self._ranges[constant] = (constant, constant, 0)
            self._terms[constant] = Const(value)
        return constant

    # -- statements -------------------------------------------------------

    def _body(self, body: list[ast.stmt]) -> None:
        for statement in body:
            if not self._open():
                return
            self._location(statement)
            self._statement(statement)

    def _statement(self, node: ast.stmt) -> None:
        match node:
            case ast.Return():
                self._return(node)
            case ast.Assign():
                self._assign(node)
            case ast.AnnAssign():
                if node.value is not None:
                    if (
                        isinstance(node.target, ast.Name)
                        and self.frontend.standalone
                        and self._standalone_buffer(node.target.id, node.value)
                    ):
                        return
                    self._store(node.target, self._expr(node.value))
            case ast.AugAssign():
                self._augassign(node)
            case ast.If():
                self._if(node)
            case ast.Break():
                if not self._loops:
                    raise Unsupported("`break` outside a loop")
                core.br(self.b, Successor(self._loops[-1][1]))
            case ast.Continue():
                if not self._loops:
                    raise Unsupported("`continue` outside a loop")
                core.br(self.b, Successor(self._loops[-1][0]))
            case ast.While():
                self._while(node)
            case ast.For():
                self._for(node)
            case ast.Pass():
                return
            case ast.Expr(value=ast.Constant()):
                return
            case ast.Expr(value=ast.Call()):
                self._expr(node.value)
            case _:
                raise Unsupported(f"`{type(node).__name__}` has no native lowering")

    def _return(self, node: ast.Return) -> None:
        if node.value is None:
            raise Unsupported("a native function must return a value")
        results = self.function.results
        if not results:
            raise Unsupported("a native function must return a value")
        expected = results[0]
        if isinstance(expected, TupleType):
            values = self._tuple_expr(node.value)
            if values is None or len(values) != len(expected.items):
                raise Unsupported("the returned tuple does not match the declared shape")
            items = [
                self._coerce(item, _kind(t)) for item, t in zip(values, expected.items, strict=True)
            ]
            core.ret(self.b, core.tuple_make(self.b, *items))
            return
        core.ret(self.b, self._coerce(self._expr(node.value), _kind(expected)))

    def _return_default(self) -> None:
        if self.info.ret == T.NONE and not self.function.results:
            core.ret(self.b)
            return
        raise Unsupported("control flow can fall off the end without returning a value")

    def _assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise Unsupported("chained assignment has no native lowering")
        target = node.targets[0]
        if (
            self.frontend.standalone
            and isinstance(target, ast.Name)
            and self._standalone_buffer(target.id, node.value)
        ):
            return
        values = self._tuple_expr(node.value)
        if values is not None:
            self._store_tuple(target, values)
            return
        self._store(target, self._expr(node.value))

    def _standalone_buffer(self, name: str, value: ast.expr) -> bool:
        """`xs = ppy.buffer[int](n)`: a zeroed allocation from the C support."""
        if not isinstance(value, ast.Call) or len(value.args) != 1:
            return False
        described = _ALLOCATIONS.get(ast.unparse(value.func))
        if described is None:
            return False
        element, reads = described
        count = self._coerce(self._expr(value.args[0]), "int")
        element_type = _scalar_type(element)
        width = core.const(self.b, 1 if element in _NARROW else 8, I64)
        raw = core.call_extern(
            self.b, "ppy_rt_alloc", (count, width), (PtrType(I8),), names=(f"{name}.raw",)
        ).results[0]
        data = core.cast(self.b, raw, PtrType(element_type)) if element_type != I8 else raw
        if reads:
            core.call_extern(self.b, "ppy_rt_read_ints", (data, count), (I64,))
        buffer = self.b.create(
            "core.call_intrinsic",
            (data, count),
            (BufferType(element_type),),
            {"intrinsic": "ppy.buffer_from_parts"},
            result_names=(name,),
        ).result
        self.buffers[name] = buffer
        self.slots.pop(name, None)
        return True

    def _store_tuple(self, target: ast.expr, values: list[Value]) -> None:
        if isinstance(target, ast.Name):
            packed = core.tuple_make(self.b, *values)
            slot = self.tuples.get(target.id)
            if slot is None or slot.type != PtrType(packed.type, "stack"):
                slot = self._alloca(packed.type, target.id)
                self.tuples[target.id] = slot
                self.slots.pop(target.id, None)
            core.store(self.b, packed, slot)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if len(target.elts) != len(values):
                raise Unsupported("unpacking width does not match the tuple")
            for element_target, item in zip(target.elts, values, strict=True):
                self._store(element_target, item)
            return
        raise Unsupported("this assignment target has no native lowering")

    def _tuple_expr(self, node: ast.expr) -> list[Value] | None:
        if isinstance(node, ast.Tuple):
            if not node.elts or len(node.elts) > _MAX_TUPLE_WIDTH:
                return None
            if any(isinstance(e, ast.Starred) for e in node.elts):
                raise Unsupported("a starred element has no fixed native width")
            return [self._expr(element) for element in node.elts]
        if isinstance(node, ast.Name) and node.id in self.tuples:
            packed = core.load(self.b, self.tuples[node.id])
            assert isinstance(packed.type, TupleType)
            return [core.tuple_extract(self.b, packed, i) for i in range(len(packed.type.items))]
        return None

    def _augassign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Subscript):
            if not (
                isinstance(node.target.value, ast.Name) and node.target.value.id in self.buffers
            ):
                raise Unsupported("this augmented assignment has no native lowering")
            current = self._buffer_element(node.target.value.id, self._expr(node.target.slice))
            value = self._expr(node.value)
            self._store(node.target, self._binary(current, value, type(node.op)))
            return
        if not isinstance(node.target, ast.Name):
            raise Unsupported("augmented assignment to a non-local has no native lowering")
        current = self._load(node.target.id)
        if self.prover is not None and current.type == I64:
            self._term_for_load(current, node.target)
        value = self._expr(node.value)
        self._store(node.target, self._binary(current, value, type(node.op)))

    def _store(self, target: ast.expr, value: Value) -> None:
        if isinstance(target, ast.Attribute):
            raise Unsupported("a flattened value class cannot be written back")
        if isinstance(target, ast.Subscript):
            if isinstance(target.value, ast.Name) and target.value.id in self.buffers:
                self._store_element(target.value.id, target.slice, value)
                return
            raise Unsupported("this subscript assignment has no native lowering")
        if not isinstance(target, ast.Name):
            raise Unsupported("assignment to a non-local has no native lowering")
        slot = self.slots.get(target.id)
        if slot is None:
            slot = self._alloca(value.type, target.id)
            self.slots[target.id] = slot
            self.tuples.pop(target.id, None)
            core.store(self.b, value, slot)
            return
        assert isinstance(slot.type, PtrType)
        if isinstance(slot.type.pointee, PtrType):
            if value.type != slot.type.pointee:
                raise Unsupported("a pointer local keeps one pointer type")
            core.store(self.b, value, slot)
            return
        core.store(self.b, self._coerce(value, _kind(slot.type.pointee)), slot)

    def _if(self, node: ast.If) -> None:
        condition = self._truth(self._expr(node.test))
        then_block = self._block("then")
        else_block = self._block("else")
        merge = self._block("endif")
        core.cond_br(self.b, condition, Successor(then_block), Successor(else_block))
        self.b.at_end(then_block)
        self._body(node.body)
        if self._open():
            core.br(self.b, Successor(merge))
        self.b.at_end(else_block)
        self._body(node.orelse)
        if self._open():
            core.br(self.b, Successor(merge))
        self.b.at_end(merge)

    def _while(self, node: ast.While) -> None:
        if node.orelse:
            raise Unsupported("`while ... else` has no native lowering")
        header = self._block("while.head")
        body = self._block("while.body")
        done = self._block("while.end")
        core.br(self.b, Successor(header))
        self.b.at_end(header)
        core.cond_br(self.b, self._truth(self._expr(node.test)), Successor(body), Successor(done))
        self.b.at_end(body)
        self._loops.append((header, done))
        self._body(node.body)
        self._loops.pop()
        if self._open():
            core.br(self.b, Successor(header))
        self.b.at_end(done)

    def _for(self, node: ast.For) -> None:
        if node.orelse or not isinstance(node.target, ast.Name):
            raise Unsupported("only `for NAME in range(...)` or over a list parameter is lowered")
        if isinstance(node.iter, ast.Name) and node.iter.id in self.buffers:
            self._for_buffer(node, node.iter.id)
            return
        if not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            raise Unsupported("only `for NAME in range(...)` or over a list parameter is lowered")
        bounds = [self._coerce(self._expr(a), "int") for a in node.iter.args]
        step_value = 1
        if len(bounds) == 1:
            start, stop = self._int_constant(0), bounds[0]
        elif len(bounds) == 2:
            start, stop = bounds
        elif len(bounds) == 3:
            start, stop, step = bounds
            constant = _constant_of(step)
            if constant is None:
                raise Unsupported("a non-constant `range` step has no native lowering")
            if constant == 0:
                raise Unsupported("`range` with a zero step has no native lowering")
            step_value = int(constant)  # type: ignore[call-overload]
        else:
            raise Unsupported("`range` takes at most three arguments")
        step = self._int_constant(step_value)
        name = node.target.id

        site: _GuardSite | None = None
        saved_induction = self._induction.get(name)
        if self.hoist and not _rebinds(node.body, name):
            guards = self._block("for.guards")
            setup = self._block("for.setup")
            core.br(self.b, Successor(guards))
            site = _GuardSite(guards)
            self.b.at_end(setup)
            self._guard_sites.append(site)
            one = core.const(site.b, 1, I64)
            if step_value > 0:
                # Body loads sit in [start, stop-1]; a wrapping stop-1 can
                # only happen when the loop is empty and the body never runs.
                lo, hi = start, core.sub(site.b, stop, one, overflow="wrap")
            else:
                lo, hi = core.add(site.b, stop, one, overflow="wrap"), start
            self._induction[name] = (lo, hi, len(self._guard_sites))
            if self.prover is not None:
                self._induction_terms[name] = self._induction_bounds(start, stop, step_value > 0)

        slot = self.slots.get(name)
        if slot is None or slot.type != PtrType(I64, "stack"):
            slot = self._alloca(I64, name)
            self.slots[name] = slot
            self.tuples.pop(name, None)
        core.store(self.b, start, slot)

        header = self._block("for.head")
        body = self._block("for.body")
        latch = self._block("for.latch")
        done = self._block("for.end")
        loop_setup = self.b.block
        assert loop_setup is not None
        core.br(self.b, Successor(header))
        self.b.at_end(header)
        current = core.load(self.b, slot)
        condition = core.cmp(self.b, "gt" if step_value < 0 else "lt", current, stop)
        core.cond_br(self.b, condition, Successor(body), Successor(done))
        self.b.at_end(body)
        self._loops.append((latch, done))
        self._body(node.body)
        self._loops.pop()
        if self._open():
            core.br(self.b, Successor(latch))
        self.b.at_end(latch)
        value = core.load(self.b, slot)
        if site is not None:
            self._ranges[value] = self._induction[name]
        if self.prover is not None:
            self._term_for_load(value, node.target)
        core.store(self.b, self._checked_binary(value, step, "add"), slot)
        core.br(self.b, Successor(header))
        self.b.at_end(done)
        if site is not None:
            site.finish(loop_setup)
            self._guard_sites.pop()
            if saved_induction is None:
                self._induction.pop(name, None)
                self._induction_terms.pop(name, None)
            else:
                self._induction[name] = saved_induction

    def _for_buffer(self, node: ast.For, name: str) -> None:
        buffer = self.buffers[name]
        assert isinstance(buffer.type, BufferType)
        element = buffer.type.element
        index = self._alloca(I64, f"{name}.i")
        core.store(self.b, core.const(self.b, 0, I64), index)
        target = node.target
        assert isinstance(target, ast.Name)
        carried = _scalar_type(_read_as(_kind(element)))
        slot = self.slots.get(target.id)
        if slot is None or slot.type != PtrType(carried, "stack"):
            slot = self._alloca(carried, target.id)
            self.slots[target.id] = slot
        length = core.cast(self.b, core.buffer_len(self.b, buffer), I64)

        header = self._block("each.head")
        body = self._block("each.body")
        latch = self._block("each.latch")
        done = self._block("each.end")
        core.br(self.b, Successor(header))
        self.b.at_end(header)
        current = core.load(self.b, index)
        core.cond_br(
            self.b, core.cmp(self.b, "lt", current, length), Successor(body), Successor(done)
        )
        self.b.at_end(body)
        position = core.load(self.b, index)
        loaded = core.buffer_load(self.b, buffer, position)
        core.store(self.b, self._coerce(loaded, _kind(carried)), slot)
        self._loops.append((latch, done))
        self._body(node.body)
        self._loops.pop()
        if self._open():
            core.br(self.b, Successor(latch))
        self.b.at_end(latch)
        one = core.const(self.b, 1, I64)
        core.store(self.b, core.add(self.b, core.load(self.b, index), one, overflow="wrap"), index)
        core.br(self.b, Successor(header))
        self.b.at_end(done)

    # -- buffers ----------------------------------------------------------

    def _store_element(self, name: str, index: ast.expr, value: Value) -> None:
        buffer = self.buffers[name]
        kind = next((p for p in self.signature.parameters if p.name == name and p.is_buffer), None)
        if kind is not None and not kind.is_borrowed:
            raise Unsupported(f"`{name}` is copied in, so writing to it has no effect")
        if isinstance(index, ast.Slice):
            raise Unsupported("slice assignment has no native lowering")
        assert isinstance(buffer.type, BufferType)
        position = self._coerce(self._expr(index), "int")
        self._guard_index(position, buffer)
        core.buffer_store(self.b, self._coerce(value, _kind(buffer.type.element)), buffer, position)

    def _guard_index(self, position: Value, buffer: Value) -> None:
        """A negative or out-of-range index is left to CPython.

        An index with a proven range checks its extremes once in the loop's
        guard block instead -- removing the side exit is what lets the loop
        vectorize -- provided the buffer is a parameter, whose length the
        guard block can read.
        """
        if self.hoist:
            entry = self._ranges.get(position)
            if entry is not None:
                lo, hi, depth = entry
                dominates = isinstance(buffer, BlockArgument) and buffer.block is self.entry
                if 1 <= depth <= len(self._guard_sites) and dominates:
                    site = self._guard_sites[depth - 1]
                    zero = core.const(site.b, 0, I64)
                    length = core.cast(site.b, core.buffer_len(site.b, buffer), I64)
                    inside = core.bitwise(
                        site.b,
                        "and",
                        core.cmp(site.b, "ge", lo, zero),
                        core.cmp(site.b, "lt", hi, length),
                    )
                    core.guard(site.b, inside, "bounds", "hoisted bounds check")
                    return
        zero = core.const(self.b, 0, I64)
        length = core.cast(self.b, core.buffer_len(self.b, buffer), I64)
        in_range = core.bitwise(
            self.b,
            "and",
            core.cmp(self.b, "ge", position, zero),
            core.cmp(self.b, "lt", position, length),
        )
        core.guard(self.b, in_range, "bounds", "index out of range")

    def _buffer_element(self, name: str, index: Value) -> Value:
        buffer = self.buffers[name]
        assert isinstance(buffer.type, BufferType)
        position = self._coerce(index, "int")
        self._guard_index(position, buffer)
        loaded = core.buffer_load(self.b, buffer, position)
        return self._coerce(loaded, _read_as(_kind(buffer.type.element)))

    def _buffer_reduction(self, name: str, operation: str) -> Value:
        buffer = self.buffers[name]
        assert isinstance(buffer.type, BufferType)
        length = core.cast(self.b, core.buffer_len(self.b, buffer), I64)
        zero = core.const(self.b, 0, I64)
        carried_as = _read_as(_kind(buffer.type.element))
        carried_type = _scalar_type(carried_as)
        if operation in {"min", "max"}:
            core.guard(
                self.b, core.cmp(self.b, "ne", length, zero), "contract", f"{operation}() of empty"
            )
        accumulator = self._alloca(carried_type, f"{operation}.acc")
        index = self._alloca(I64, f"{operation}.i")
        if operation == "sum":
            core.store(self.b, self._literal(0, carried_as), accumulator)
            core.store(self.b, zero, index)
        else:
            first = self._coerce(core.buffer_load(self.b, buffer, zero), carried_as)
            core.store(self.b, first, accumulator)
            core.store(self.b, core.const(self.b, 1, I64), index)
        header = self._block("reduce.head")
        body = self._block("reduce.body")
        done = self._block("reduce.end")
        core.br(self.b, Successor(header))
        self.b.at_end(header)
        current = core.load(self.b, index)
        core.cond_br(
            self.b, core.cmp(self.b, "lt", current, length), Successor(body), Successor(done)
        )
        self.b.at_end(body)
        position = core.load(self.b, index)
        value = self._coerce(core.buffer_load(self.b, buffer, position), carried_as)
        carried = core.load(self.b, accumulator)
        if operation == "sum":
            updated = self._binary(carried, value, ast.Add)
        else:
            keep = core.cmp(self.b, "lt" if operation == "min" else "gt", value, carried)
            updated = core.select(self.b, keep, value, carried)
        core.store(self.b, updated, accumulator)
        one = core.const(self.b, 1, I64)
        core.store(self.b, core.add(self.b, position, one, overflow="wrap"), index)
        core.br(self.b, Successor(header))
        self.b.at_end(done)
        return core.load(self.b, accumulator)

    # -- expressions ------------------------------------------------------

    def _expr(self, node: ast.expr) -> Value:
        match node:
            case ast.Constant(value=bool() as value):
                return core.const(self.b, value, BOOL)
            case ast.Constant(value=int() as value):
                if not -(1 << 63) <= value < (1 << 63):
                    raise Unsupported("an integer literal exceeds the native machine range")
                return self._int_constant(value)
            case ast.Constant(value=float() as value):
                return core.const(self.b, value, F64)
            case ast.Name():
                loaded = self._load(node.id)
                if loaded.type == I64:
                    interval = self._induction.get(node.id)
                    if interval is not None:
                        self._ranges[loaded] = interval
                    elif self.hoist and node.id in self._stable:
                        anchor = self._entry_load(node.id)
                        self._ranges[loaded] = (anchor, anchor, 0)
                    if self.prover is not None:
                        self._term_for_load(loaded, node)
                return loaded
            case ast.BinOp():
                return self._binary(self._expr(node.left), self._expr(node.right), type(node.op))
            case ast.UnaryOp():
                return self._unary(node)
            case ast.BoolOp():
                return self._boolop(node)
            case ast.Compare():
                return self._compare(node)
            case ast.IfExp():
                return self._ifexp(node)
            case ast.Call():
                return self._call(node)
            case ast.Attribute():
                if isinstance(node.value, ast.Name) and node.value.id in self.objects:
                    return self._field(node.value.id, node.attr)
                raise Unsupported("attribute access on this value has no native lowering")
            case ast.Subscript():
                if isinstance(node.value, ast.Name) and node.value.id in self.tuples:
                    return self._tuple_element(node.value.id, node.slice)
                if isinstance(node.value, ast.Name) and node.value.id in self.buffers:
                    if isinstance(node.slice, ast.Slice):
                        raise Unsupported("slicing a buffer allocates, so it stays boxed")
                    return self._buffer_element(node.value.id, self._expr(node.slice))
                raise Unsupported("subscripting this value has no native lowering")
        raise Unsupported(f"`{type(node).__name__}` has no native lowering")

    def _field(self, name: str, attr: str) -> Value:
        struct = self.objects[name]
        assert isinstance(struct.type, StructType)
        if struct.type.field_type(attr) is None:
            raise Unsupported(f"`{name}.{attr}` is not a native field")
        return core.struct_extract(self.b, struct, attr)

    def _tuple_element(self, name: str, index: ast.expr) -> Value:
        packed = core.load(self.b, self.tuples[name])
        assert isinstance(packed.type, TupleType)
        if not isinstance(index, ast.Constant) or not isinstance(index.value, int):
            raise Unsupported("a fixed tuple must be indexed by a constant")
        position = index.value
        if position < 0:
            position += len(packed.type.items)
        if not 0 <= position < len(packed.type.items):
            raise Unsupported(f"index {index.value} is out of range for `{name}`")
        return core.tuple_extract(self.b, packed, position)

    def _module_constant(self, name: str) -> Value | None:
        symbols = getattr(self.frontend.analysis, "symbols", None)
        if symbols is None:
            return None
        value = symbols.constant_globals.get(name)
        if isinstance(value, bool):
            return core.const(self.b, value, BOOL)
        if isinstance(value, int) and -(1 << 63) <= value < (1 << 63):
            return self._int_constant(value)
        if isinstance(value, float):
            return core.const(self.b, value, F64)
        return None

    def _load(self, name: str) -> Value:
        if name in self.objects:
            raise Unsupported(f"`{name}` is a value class, which has no single scalar value")
        if name in self.tuples:
            raise Unsupported(f"`{name}` is a tuple, which has no single scalar value")
        if name in self.buffers:
            raise Unsupported(f"`{name}` is a buffer, which has no scalar value")
        slot = self.slots.get(name)
        if slot is None:
            constant = self._module_constant(name)
            if constant is not None:
                return constant
            raise Unsupported(f"`{name}` is not a native local")
        return core.load(self.b, slot)

    def _unary(self, node: ast.UnaryOp) -> Value:
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            literal = node.operand.value
            if isinstance(literal, bool):
                pass
            elif isinstance(literal, int) and -(1 << 63) < -literal < (1 << 63):
                return self._int_constant(-literal)
            elif isinstance(literal, float):
                return core.const(self.b, -literal, F64)
        operand = self._expr(node.operand)
        match node.op:
            case ast.USub():
                if operand.type == F64:
                    return core.neg(self.b, operand)
                promoted = self._coerce(operand, "int")
                return self._checked_binary(self._int_constant(0), promoted, "sub")
            case ast.UAdd():
                return operand
            case ast.Invert():
                promoted = self._coerce(operand, "int")
                minus_one = core.const(self.b, -1, I64)
                return core.bitwise(self.b, "xor", promoted, minus_one)
            case ast.Not():
                truth = self._truth(operand)
                return core.bitwise(self.b, "xor", truth, core.const(self.b, True, BOOL))
        raise Unsupported("unary operator has no native lowering")

    def _boolop(self, node: ast.BoolOp) -> Value:
        """`and`/`or` with short-circuit evaluation, joined by a block argument."""
        done = self._block("boolop.end")
        result = done.add_argument(BOOL, "boolop")
        is_and = isinstance(node.op, ast.And)
        for index, value_node in enumerate(node.values):
            truth = self._truth(self._expr(value_node))
            if index == len(node.values) - 1:
                core.br(self.b, Successor(done, [truth]))
                break
            following = self._block("boolop")
            if is_and:
                core.cond_br(self.b, truth, Successor(following), Successor(done, [truth]))
            else:
                core.cond_br(self.b, truth, Successor(done, [truth]), Successor(following))
            self.b.at_end(following)
        self.b.at_end(done)
        return result

    def _compare(self, node: ast.Compare) -> Value:
        if len(node.ops) != 1:
            raise Unsupported("chained comparison has no native lowering")
        predicate = _COMPARISONS.get(type(node.ops[0]))
        if predicate is None:
            raise Unsupported("comparison operator has no native lowering")
        left = self._expr(node.left)
        right = self._expr(node.comparators[0])
        kind = self._unify(_kind(left.type), _kind(right.type))
        return core.cmp(self.b, predicate, self._coerce(left, kind), self._coerce(right, kind))

    def _ifexp(self, node: ast.IfExp) -> Value:
        condition = self._truth(self._expr(node.test))
        then_value = self._expr(node.body)
        else_value = self._expr(node.orelse)
        kind = self._unify(_kind(then_value.type), _kind(else_value.type))
        return core.select(
            self.b, condition, self._coerce(then_value, kind), self._coerce(else_value, kind)
        )

    def _call(self, node: ast.Call) -> Value:
        if node.keywords:
            raise Unsupported("keyword arguments have no native ABI")
        target = ast.unparse(node.func)
        if self.frontend.standalone and target == "print":
            return self._standalone_print(node)
        if self.frontend.standalone and target == "ppy.input[int]" and not node.args:
            return core.call_extern(self.b, "ppy_rt_read_int", (), (I64,)).results[0]
        if target.startswith("math."):
            return self._math_call(target.removeprefix("math."), node)
        if target.startswith(("ppy.native.", "native.")):
            return self._native_op(target.rpartition("native.")[2], node)
        for qualname, (info, options) in self.frontend.externs.items():
            if qualname.rpartition(".")[2] == target:
                return self._extern_call(info, options, node)
        if target == "len" and len(node.args) == 1:
            argument = node.args[0]
            if isinstance(argument, ast.Name) and argument.id in self.tuples:
                width = len(self.tuples[argument.id].type.pointee.items)  # type: ignore[attr-defined]
                return self._int_constant(width)
        if target in {"len", "sum", "min", "max"} and len(node.args) == 1:
            argument = node.args[0]
            if isinstance(argument, ast.Name) and argument.id in self.buffers:
                if target == "len":
                    length = core.buffer_len(self.b, self.buffers[argument.id])
                    return core.cast(self.b, length, I64)
                return self._buffer_reduction(argument.id, target)
        if target in {"min", "max"} and len(node.args) >= 2:
            return self._extremum(target, node)
        if target in {"abs", "float", "int", "bool"}:
            return self._builtin_call(target, node)
        for qualname, (function, signature) in self.frontend.declared.items():
            if qualname.rpartition(".")[2] == target and "ppy.generic" not in function.attributes:
                return self._native_call(function, signature, qualname, node)
        for qualname, (info, _analysis, _node) in self.frontend.generics.items():
            if qualname.rpartition(".")[2] == target:
                return self._generic_call(qualname, info, node)
        raise Unsupported(f"`{target}` has no native lowering")

    def _generic_call(self, qualname: str, info: FunctionInfo, node: ast.Call) -> Value:
        """Instantiate a generic on the argument types this body has in hand."""
        if len(node.args) != len(info.params):
            raise Unsupported(f"`{qualname}` called with the wrong number of arguments")
        values = [self._expr(argument) for argument in node.args]
        bindings: dict[T.TypeVar_, T.Type] = {}
        for value, param in zip(values, info.params, strict=True):
            if not T.infer(param.type, _analysis_type(value.type), bindings):
                raise Unsupported(f"`{qualname}` cannot take a `{value.type}` for `{param.name}`")
        arguments = tuple(bindings.get(v, v.bound or T.ANY) for v in info.type_params)
        instance = self.frontend.instantiate(qualname, arguments)
        if instance is None:
            raise Unsupported(f"`{qualname}` has no native lowering")
        function, signature = instance
        if not function.results:
            raise Unsupported(f"`{qualname}` returns nothing a caller can use")
        converted = [
            self._coerce(value, parameter.kind)
            for value, parameter in zip(values, signature.parameters, strict=True)
        ]
        return core.call(self.b, function.name, tuple(converted), function.results).results[0]

    def _native_op(self, operation: str, node: ast.Call) -> Value:
        """`ppy.native.load` and the rest, as the pointer operations they are."""
        subscript = node.func.slice if isinstance(node.func, ast.Subscript) else None
        if operation.startswith("load"):
            pointer = self._pointer(node.args[0])
            loaded = core.load(self.b, pointer)
            return self._coerce(loaded, _read_as(_kind(loaded.type)))
        if operation.startswith("store"):
            pointer = self._pointer(node.args[0])
            assert isinstance(pointer.type, PtrType)
            if not pointer.type.mutable:
                raise Unsupported("a store through a const pointer")
            value = self._coerce(self._expr(node.args[1]), _kind(pointer.type.pointee))
            core.store(self.b, value, pointer)
            return core.const(self.b, 0, I64)
        if operation.startswith("offset"):
            pointer = self._pointer(node.args[0])
            count = self._coerce(self._expr(node.args[1]), "int")
            return core.ptr_offset(self.b, pointer, count)
        if subscript is None:
            raise Unsupported(f"`ppy.native.{operation}` has no native lowering")
        element = _element_type(subscript)
        if operation.startswith(("sizeof", "alignof")):
            width = 8 if element is None else {I64: 8, F64: 8, BOOL: 1, I8: 1, U8: 1}[element]
            return self._int_constant(width)
        if element is None:
            raise Unsupported(f"`{ast.unparse(subscript)}` is not an element native memory holds")
        if operation.startswith("stack_alloc"):
            count = _constant_of(self._expr(node.args[0]))
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise Unsupported("a stack allocation needs a constant, positive size")
            return core.alloca(self.b, element, count=int(count))
        if operation.startswith("cast"):
            pointer = self._pointer(node.args[0])
            assert isinstance(pointer.type, PtrType)
            target = PtrType(element, pointer.type.address_space, pointer.type.mutable)
            return core.cast(self.b, pointer, target)
        raise Unsupported(f"`ppy.native.{operation}` has no native lowering")

    def _pointer(self, node: ast.expr) -> Value:
        value = self._expr(node)
        if not isinstance(value.type, PtrType):
            raise Unsupported("a `ppy.native` operation needs a pointer")
        return value

    def _extern_call(self, info: FunctionInfo, options: dict[str, object], node: ast.Call) -> Value:
        """A C binding: `core.call_extern` with the stub's signature."""
        if len(node.args) != len(info.params):
            raise Unsupported(f"`{info.qualname}` called with the wrong number of arguments")
        signature = _signature(info, self.frontend.layouts)
        operands: list[Value] = []
        for argument, parameter in zip(node.args, signature.parameters, strict=True):
            if parameter.is_pointer:
                operands.append(self._pointer(argument))
            elif parameter.is_buffer:
                if not isinstance(argument, ast.Name) or argument.id not in self.buffers:
                    raise Unsupported("a buffer argument must be a buffer this function holds")
                operands.append(core.buffer_data(self.b, self.buffers[argument.id]))
            else:
                operands.append(self._coerce(self._expr(argument), parameter.kind))
        results: tuple[IRType, ...] = ()
        if info.ret != T.NONE:
            atoms = _return_atoms(info.ret)
            if atoms is None or len(atoms) != 1:
                raise Unsupported(f"`{info.qualname}` returns something C cannot")
            results = (_scalar_type(atoms[0]),)
        library = options.get("library")
        if isinstance(library, str):
            known = self.frontend.module.attributes.get("ppy.libraries", ())
            assert isinstance(known, tuple)
            if library not in known:
                self.frontend.module.attributes["ppy.libraries"] = (*known, library)
        call = core.call_extern(
            self.b,
            str(options.get("symbol") or info.name),
            tuple(operands),
            results,
            abi=str(options.get("convention", "c")),
        )
        return call.results[0] if results else core.const(self.b, 0, I64)

    def _standalone_print(self, node: ast.Call) -> Value:
        for index, argument in enumerate(node.args):
            if index:
                core.call_extern(self.b, "ppy_rt_print_sep", (), ())
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                text = self.frontend.module.add_global(
                    f"ppy.str.{len(self.frontend.module.globals)}",
                    BufferType(U8),
                    argument.value,
                    visibility="private",
                )
                pointer = self.b.create(
                    "core.call_intrinsic",
                    (),
                    (PtrType(U8),),
                    {"intrinsic": "ppy.string_data", "symbol": text.symbol.name},
                ).result
                length = core.const(self.b, len(argument.value.encode("utf-8")), I64)
                core.call_extern(self.b, "ppy_rt_print_str", (pointer, length), ())
                continue
            value = self._expr(argument)
            if value.type == I64:
                core.call_extern(self.b, "ppy_rt_print_i64", (value,), ())
            elif value.type == BOOL:
                core.call_extern(self.b, "ppy_rt_print_bool", (value,), ())
            else:
                raise Unsupported("only integers, booleans, and string literals print natively")
        core.call_extern(self.b, "ppy_rt_print_nl", (), ())
        return core.const(self.b, 0, I64)

    def _extremum(self, target: str, node: ast.Call) -> Value:
        values = [self._expr(argument) for argument in node.args]
        kinds = {_kind(value.type) for value in values}
        if len(kinds) != 1:
            raise Unsupported(f"`{target}` over mixed {sorted(kinds)} would change the result type")
        kind = values[0].type
        if kind not in {I64, F64}:
            raise Unsupported(f"`{target}` over `{_kind(kind)}` has no native lowering")
        best = values[0]
        for candidate in values[1:]:
            wins = core.cmp(self.b, "lt" if target == "min" else "gt", candidate, best)
            best = core.select(self.b, wins, candidate, best)
        return best

    def _native_call(
        self, function: IRFunction, signature: NativeSignature, qualname: str, node: ast.Call
    ) -> Value:
        if len(node.args) != len(signature.parameters):
            raise Unsupported(f"`{qualname}` called with the wrong number of arguments")
        arguments: list[Value] = []
        for argument, parameter in zip(node.args, signature.parameters, strict=True):
            if parameter.is_buffer:
                if not isinstance(argument, ast.Name) or argument.id not in self.buffers:
                    raise Unsupported("a buffer argument must be a buffer this function holds")
                buffer = self.buffers[argument.id]
                assert isinstance(buffer.type, BufferType)
                if _kind(buffer.type.element) != parameter.element:
                    raise Unsupported(
                        f"`{qualname}` expects a `{parameter.element}` buffer, "
                        f"and `{argument.id}` holds `{_kind(buffer.type.element)}`"
                    )
                arguments.append(buffer)
                continue
            if parameter.is_object:
                if not isinstance(argument, ast.Name) or argument.id not in self.objects:
                    raise Unsupported("a value class argument must be a flattened local")
                struct = self.objects[argument.id]
                assert isinstance(struct.type, StructType)
                for attr, _scalar in parameter.fields:
                    if struct.type.field_type(attr) is None:
                        raise Unsupported(f"`{argument.id}` has no field `{attr}`")
                arguments.append(struct)
                continue
            if parameter.is_tuple:
                values = self._tuple_expr(argument)
                if values is None or len(values) != len(parameter.elements):
                    raise Unsupported("a tuple argument does not match the callee's shape")
                items = [
                    self._coerce(item, element)
                    for item, element in zip(values, parameter.elements, strict=True)
                ]
                arguments.append(core.tuple_make(self.b, *items))
                continue
            arguments.append(self._coerce(self._expr(argument), parameter.kind))
        if signature.returns_tuple:
            raise Unsupported("a tuple result cannot be forwarded between native calls yet")
        if not function.results:
            raise Unsupported(f"`{qualname}` returns nothing a caller can use")
        return core.call(self.b, function.name, tuple(arguments), function.results).results[0]

    def _math_call(self, name: str, node: ast.Call) -> Value:
        if name not in _MATH_INTRINSICS:
            raise Unsupported(f"`math.{name}` has no native lowering")
        arity = 2 if name == "pow" else 1
        if len(node.args) != arity:
            raise Unsupported(f"`math.{name}` takes {arity} argument(s)")
        arguments = tuple(self._coerce(self._expr(a), "float") for a in node.args)
        self.frontend.module.require("math", 1)
        return math_dialect.call(self.b, "abs" if name == "fabs" else name, *arguments)

    def _builtin_call(self, name: str, node: ast.Call) -> Value:
        if len(node.args) != 1:
            raise Unsupported(f"`{name}` with this arity has no native lowering")
        value = self._expr(node.args[0])
        if name == "float":
            return self._coerce(value, "float")
        if name == "bool":
            return self._truth(value)
        if name == "int":
            if value.type == F64:
                return core.cast(self.b, value, I64)  # truncates toward zero, like `int()`
            return self._coerce(value, "int")
        if name == "abs":
            if value.type == F64:
                self.frontend.module.require("math", 1)
                return math_dialect.call(self.b, "abs", value)
            promoted = self._coerce(value, "int")
            zero = self._int_constant(0)
            negative = core.cmp(self.b, "lt", promoted, zero)
            negated = self._checked_binary(zero, promoted, "sub")
            return core.select(self.b, negative, negated, promoted)
        raise Unsupported(f"`{name}` has no native lowering")

    def _binary(self, left: Value, right: Value, op: type[ast.operator]) -> Value:
        if op is ast.Div:
            left, right = self._coerce(left, "float"), self._coerce(right, "float")
            self._guard_nonzero(right)
            return core.div(self.b, left, right)
        dispatched = self._struct_operator(left, right, op)
        if dispatched is not None:
            return dispatched
        kind = self._unify(_kind(left.type), _kind(right.type))
        left, right = self._coerce(left, kind), self._coerce(right, kind)
        if kind == "float":
            if op in _ARITHMETIC:
                return getattr(core, _ARITHMETIC[op])(self.b, left, right)
            if op is ast.Pow:
                self.frontend.module.require("math", 1)
                return math_dialect.call(self.b, "pow", left, right)
            raise Unsupported("floating-point operator has no native lowering")
        if op in _ARITHMETIC:
            return self._checked_binary(left, right, _ARITHMETIC[op])
        if op in {ast.FloorDiv, ast.Mod}:
            self._guard_nonzero(right)
            if op is ast.FloorDiv:
                return core.div(self.b, left, right, overflow=self.overflow, rounding="floor")
            return core.mod(self.b, left, right, overflow=self.overflow, rounding="floor")
        if op in _BITWISE:
            return core.bitwise(self.b, _BITWISE[op], left, right)
        if op in {ast.LShift, ast.RShift}:
            return self._shift(left, right, op)
        raise Unsupported("integer operator has no native lowering")

    # -- overflow: proofs, hoisting, and the guard --------------------------

    def _checked_binary(self, left: Value, right: Value, op: str) -> Value:
        """Machine arithmetic with an overflow guard back to CPython.

        A chain the prover shows fits the word, or whose operands carry
        proven ranges whose corners the loop's guard block checks, runs the
        plain instruction; anything else carries the semantics the source
        has and the backend guards it.
        """
        proven = self._proven_binary(left, right, op)
        if proven is not None:
            return proven
        hoisted = self._hoisted_binary(left, right, op)
        if hoisted is not None:
            return hoisted
        return getattr(core, op)(self.b, left, right, overflow=self.overflow)

    def _proven_binary(self, left: Value, right: Value, op: str) -> Value | None:
        if self.prover is None:
            return None
        left_term = self._terms.get(left)
        right_term = self._terms.get(right)
        if left_term is None or right_term is None:
            return None
        term = BinOp(_SPELLING[op], left_term, right_term)
        obligation = Obligation(term, self._hypotheses(term))
        if not self.prover.proves(obligation):
            return None
        result = getattr(core, op)(self.b, left, right, overflow="proven")
        self._terms[result] = term
        self.proved.append(str(obligation))
        return result

    def _hypotheses(self, term: Term) -> tuple[Relation, ...]:
        found: list[Relation] = []
        seen: set[Relation] = set()
        pending = list(variables(term))
        while pending:
            variable = pending.pop()
            for relation in self._relations.get(variable, ()):
                if relation in seen:
                    continue
                seen.add(relation)
                found.append(relation)
                pending.extend(variables(relation.left))
                pending.extend(variables(relation.right))
        return tuple(found)

    def _hoisted_binary(self, left: Value, right: Value, op: str) -> Value | None:
        if not self.hoist:
            return None
        left_range = self._ranges.get(left)
        right_range = self._ranges.get(right)
        if left_range is None or right_range is None:
            return None
        # Only chains that contain a multiplication hoist. A guarded add of
        # an induction variable is something LLVM proves and folds by itself;
        # a guarded multiply is the one thing it cannot see through, and the
        # additions that consume its result must follow it out or the chain
        # stays opaque.
        if op != "mul" and left not in self._hoisted and right not in self._hoisted:
            return None
        depth = max(left_range[2], right_range[2])
        if depth < 1 or depth > len(self._guard_sites):
            return None
        site = self._guard_sites[depth - 1]
        bounds = self._corner_check(site, left_range, right_range, op)
        result = getattr(core, op)(self.b, left, right, overflow="proven")
        self._ranges[result] = (*bounds, depth)
        self._hoisted.add(result)
        return result

    def _corner_check(
        self,
        site: _GuardSite,
        left_range: tuple[Value, Value, int],
        right_range: tuple[Value, Value, int],
        op: str,
    ) -> tuple[Value, Value]:
        """Prove the operation safe for its extreme operands, once.

        Every value the body can hold lies between its operands' corners, so
        if no corner overflows, no iteration can. Any corner overflowing --
        or an already-empty loop with absurd bounds -- bails to CPython.
        """
        left_lo, left_hi, _ = left_range
        right_lo, right_hi, _ = right_range
        if op == "add":
            corners = [(left_lo, right_lo), (left_hi, right_hi)]
        elif op == "sub":
            corners = [(left_lo, right_hi), (left_hi, right_lo)]
        else:
            corners = [
                (left_lo, right_lo),
                (left_lo, right_hi),
                (left_hi, right_lo),
                (left_hi, right_hi),
            ]
        b = site.b
        values: list[Value] = []
        overflowed: Value | None = None
        for first, second in corners:
            value, flag = core.checked(b, op, first, second)
            values.append(value)
            overflowed = flag if overflowed is None else core.bitwise(b, "or", overflowed, flag)
        assert overflowed is not None
        site.bail_if(overflowed, "overflow")
        if len(values) == 2:
            return values[0], values[1]
        lo, hi = values[0], values[0]
        for value in values[1:]:
            lo = core.select(b, core.cmp(b, "lt", value, lo), value, lo)
            hi = core.select(b, core.cmp(b, "gt", value, hi), value, hi)
        return lo, hi

    def _guard_declared_ranges(self) -> None:
        """Check each parameter against the range it declares, at entry.

        `Range(lo, hi)` is a refinement the checker propagates, not a check
        the runtime makes; a proof that rests on it is a proof about the
        calls inside it. So a function whose guards a proof may leave out
        checks its declared ranges once, on entry, and a call outside them
        takes the fallback.
        """
        for param in self.info.params:
            slot = self.slots.get(param.name)
            if slot is None or slot.type != PtrType(I64, "stack"):
                continue
            low, high = _declared_bounds(param.facts.int_range)
            if low is None and high is None:
                continue
            value = core.load(self.b, slot)
            inside: Value | None = None
            if low is not None:
                inside = core.cmp(self.b, "ge", value, core.const(self.b, low, I64))
            if high is not None:
                below = core.cmp(self.b, "le", value, core.const(self.b, high, I64))
                inside = below if inside is None else core.bitwise(self.b, "and", inside, below)
            assert inside is not None
            core.guard(self.b, inside, "range", "declared range", label=f"{param.name}.declared")

    def _term_for_load(self, value: Value, node: ast.expr) -> None:
        """The loaded value as a variable, with its range and its relations.

        Every load is its own variable: between two loads of one name the
        name may have been assigned, and a proof that took them for the same
        value would be a proof about a program that was not written.
        """
        if not isinstance(node, ast.Name):
            return
        facts = self.frontend.analysis.facts_of(node) if isinstance(node.ctx, ast.Load) else None
        low, high = _declared_bounds(None if facts is None else facts.int_range)
        self._loads += 1
        variable = Var(f"{node.id}#{self._loads}", low, high)
        self._terms[value] = variable
        induction = self._induction_terms.get(node.id)
        if induction is not None:
            low_term, high_term = induction
            relations: list[Relation] = []
            if low_term is not None:
                relations.append(Relation(low_term, "<=", variable))
            if high_term is not None:
                relations.append(Relation(variable, "<=", high_term))
            if relations:
                self._relations[variable] = tuple(relations)

    def _induction_bounds(
        self, start: Value, stop: Value, ascending: bool
    ) -> tuple[Term | None, Term | None]:
        start_term = self._terms.get(start)
        stop_term = self._terms.get(stop)
        if ascending:
            high = None if stop_term is None else BinOp("-", stop_term, Const(1))
            return start_term, high
        low = None if stop_term is None else BinOp("+", stop_term, Const(1))
        return low, start_term

    # -- the rest of the arithmetic ---------------------------------------

    def _struct_operator(self, left: Value, right: Value, op: type[ast.operator]) -> Value | None:
        """`a + b` on a value class: the class's own `__add__`, called directly.

        Static dispatch: the receiver's type is known, so the method is the
        one the class defines, lowered like any other native function. A
        class without a native `__add__` refuses rather than falling back
        to Python's dynamic dispatch inside native code.
        """
        if not isinstance(left.type, StructType):
            return None
        dunder = _OPERATOR_DUNDERS.get(op)
        if dunder is None:
            raise Unsupported(f"no operator method for `{op.__name__}` on `{left.type.name}`")
        for qualname, (function, signature) in self.frontend.declared.items():
            if (
                qualname.endswith(f".{dunder}")
                and function.params
                and (function.params[0][1] == left.type)
            ):
                if not function.results:
                    raise Unsupported(f"`{qualname}` returns nothing a caller can use")
                other = (
                    self._coerce(right, signature.parameters[1].kind)
                    if not isinstance(right.type, StructType)
                    else right
                )
                return core.call(self.b, function.name, (left, other), function.results).results[0]
        raise Unsupported(
            f"`{left.type.name}` has no native `{dunder}`; native code never dispatches dynamically"
        )

    def _shift(self, left: Value, right: Value, op: type[ast.operator]) -> Value:
        zero = core.const(self.b, 0, I64)
        limit = core.const(self.b, 63, I64)
        in_range = core.bitwise(
            self.b,
            "and",
            core.cmp(self.b, "ge", right, zero),
            core.cmp(self.b, "le", right, limit),
        )
        core.guard(self.b, in_range, "range", "shift count outside the machine word")
        if op is ast.LShift:
            return self.b.create(
                "core.shl", (left, right), (I64,), {"overflow": self.overflow}
            ).result
        return core.shift(self.b, "shr", left, right)

    def _guard_nonzero(self, value: Value) -> None:
        zero = self._literal(0, _kind(value.type))
        core.guard(self.b, core.cmp(self.b, "ne", value, zero), "zero_division", "division by zero")

    def _unify(self, left: str, right: str) -> str:
        if left == "float" or right == "float":
            return "float"
        if "int" in (left, right) or left in _NARROW or right in _NARROW:
            return "int"
        return "bool"

    def _coerce(self, value: Value, kind: str) -> Value:
        current = _kind(value.type)
        if current == kind:
            return value
        if kind == "const_ptr" and isinstance(value.type, PtrType) and value.type.mutable:
            # Memory one may write is memory one may read.
            pointer = value.type
            return core.cast(self.b, value, PtrType(pointer.pointee, pointer.address_space, False))
        if current in _NARROW:
            widened = core.cast(self.b, value, I64)
            return self._coerce(widened, kind)
        if kind in _NARROW:
            narrowed = self._coerce(value, "int")
            low, high = (-128, 127) if _NARROW[kind] else (0, 255)
            fits = core.bitwise(
                self.b,
                "and",
                core.cmp(self.b, "ge", narrowed, core.const(self.b, low, I64)),
                core.cmp(self.b, "le", narrowed, core.const(self.b, high, I64)),
            )
            core.guard(self.b, fits, "range", "value does not fit a byte")
            return core.cast(self.b, narrowed, _scalar_type(kind))
        if kind == "float":
            widened = self._coerce(value, "int") if current == "bool" else value
            return core.cast(self.b, widened, F64)
        if kind == "int":
            if current == "bool":
                return core.cast(self.b, value, I64)
            raise Unsupported("an implicit float-to-int conversion is not a Python semantic")
        if kind == "bool":
            return self._truth(value)
        raise Unsupported(f"cannot convert `{current}` to `{kind}`")

    def _truth(self, value: Value) -> Value:
        if value.type == BOOL:
            return value
        return core.cmp(self.b, "ne", value, self._literal(0, _kind(value.type)))


_SPELLING = {"add": "+", "sub": "-", "mul": "*"}


def _rebinds(body: list[ast.stmt], name: str) -> bool:
    """Does the loop body bind `name` itself, invalidating its range?"""
    for statement in body:
        for child in ast.walk(statement):
            if (
                isinstance(child, ast.Name)
                and child.id == name
                and isinstance(child.ctx, (ast.Store, ast.Del))
            ):
                return True
    return False


def _constant_of(value: Value) -> object | None:
    owner = value.owner
    if isinstance(owner, Operation) and owner.name == "core.const":
        return owner.attributes.get("value")
    return None


_OPERATOR_DUNDERS: dict[type[ast.operator], str] = {
    ast.Add: "__add__",
    ast.Sub: "__sub__",
    ast.Mult: "__mul__",
    ast.Div: "__truediv__",
    ast.FloorDiv: "__floordiv__",
    ast.Mod: "__mod__",
    ast.Pow: "__pow__",
    ast.MatMult: "__matmul__",
    ast.BitAnd: "__and__",
    ast.BitOr: "__or__",
    ast.BitXor: "__xor__",
}


def _specialized_info(
    info: FunctionInfo, bindings: dict[T.TypeVar_, T.Type], arguments: tuple[str, ...]
) -> FunctionInfo:
    """`info` with its type parameters replaced: what one instantiation is."""
    from dataclasses import replace

    spelled = "_".join(a.replace(".", "_").replace("[", "_").replace("]", "") for a in arguments)
    params = [replace(p, type=T.substitute(p.type, bindings)) for p in info.params]
    return replace(
        info,
        qualname=f"{info.qualname}__{spelled}",
        params=params,
        ret=T.substitute(info.ret, bindings),
        type_params=(),
    )


def _analysis_type(t: IRType) -> T.Type:
    """The analysis type an IR value's type corresponds to."""
    if t == I64:
        return T.INT
    if t == F64:
        return T.FLOAT
    if t == BOOL:
        return T.BOOL
    if isinstance(t, BufferType):
        return T.instance("Buffer", _analysis_type(t.element))
    if isinstance(t, TupleType):
        return T.Tuple_(tuple(_analysis_type(i) for i in t.items))
    if isinstance(t, StructType):
        return T.Instance(t.name.replace("_", "."), (), ())
    return T.ANY


_ELEMENTS: dict[str, IRType] = {
    "int": I64,
    "float": F64,
    "bool": BOOL,
    "ppy.i8": I8,
    "ppy.u8": U8,
    "i8": I8,
    "u8": U8,
    "ppy.i64": I64,
    "ppy.f64": F64,
    "i64": I64,
    "f64": F64,
}


def _element_type(annotation: ast.expr) -> IRType | None:
    """The element an `[T]` subscript on the native namespace names."""
    return _ELEMENTS.get(ast.unparse(annotation))
