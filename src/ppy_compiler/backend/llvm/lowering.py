"""LLVM IR generation for the natively lowerable subset (spec 16).

Only effect-free functions are lowered. That restriction is what makes the
guard-and-fall-back model sound: when a native fast path bails out, the
original Python implementation can be re-executed with identical observable
behavior (spec 16.8).

Floating-point ordering is strict by default: no reassociation, no
contraction, no reciprocal substitution. `@ppy.fastmath` is what permits those,
and nothing else does -- optimization level alone never will (spec 3.4, 12.5).

A function may take scalars, fixed-size tuples, value classes whose fields
are all scalars, and homogeneous `list[int]` / `list[float]` arguments. A fixed tuple flattens into scalar SSA
values rather than an allocated object, and a list is unboxed into a borrowed
native buffer at the Python boundary -- transparent because a lowered function
may not mutate it (spec 13.2, 13.3, 13.5).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ...analysis import types as T
from ...analysis.checker import FunctionAnalysis, ModuleAnalysis
from ...analysis.effects import Effect
from ...analysis.symbols import FunctionInfo

__all__ = [
    "Unsupported",
    "lower_specialization",
    "NativeParam",
    "NativeSignature",
    "LoweredFunction",
    "eligible",
    "lower_module",
    "STATUS_OK",
    "STATUS_FALLBACK",
]

#: The native entry point returns a status; a non-zero status means the caller
#: must re-run the Python implementation (spec 16.9).
STATUS_OK = 0
STATUS_FALLBACK = 1

_SCALARS = {"int", "float", "bool"}

#: Element types a native buffer parameter may carry.
_BUFFER_ELEMENTS = {"int", "float"}

#: A tuple wider than this stays boxed rather than expanding the ABI.
_MAX_TUPLE_WIDTH = 8

#: The transformations `@ppy.fastmath` permits, and nothing permits otherwise.
FASTMATH_FLAGS = ("fast",)

#: A class with more fields than this stays boxed rather than expanding the ABI.
_MAX_CLASS_WIDTH = 8

_MATH_INTRINSICS = {
    "sqrt": "llvm.sqrt.f64",
    "sin": "llvm.sin.f64",
    "cos": "llvm.cos.f64",
    "exp": "llvm.exp.f64",
    "log": "llvm.log.f64",
    "log2": "llvm.log2.f64",
    "log10": "llvm.log10.f64",
    "fabs": "llvm.fabs.f64",
    "floor": "llvm.floor.f64",
    "ceil": "llvm.ceil.f64",
    "pow": "llvm.pow.f64",
    "trunc": "llvm.trunc.f64",
}

_OVERFLOW_INTRINSICS = {
    ast.Add: "llvm.sadd.with.overflow.i64",
    ast.Sub: "llvm.ssub.with.overflow.i64",
    ast.Mult: "llvm.smul.with.overflow.i64",
}


class Unsupported(Exception):
    """Raised when a construct has no native lowering."""


@dataclass(frozen=True, slots=True)
class NativeParam:
    """One source-level parameter and the ABI atoms it expands to."""

    name: str
    kind: str
    element: str = ""
    elements: tuple[str, ...] = ()
    fields: tuple[tuple[str, str], ...] = ()
    class_name: str = ""

    @property
    def is_buffer(self) -> bool:
        return self.kind in {"list", "view"}

    @property
    def is_object(self) -> bool:
        return self.kind == "object"

    @property
    def is_borrowed(self) -> bool:
        """A `ppy.Buffer[T]` is borrowed in place; a list is copied out.

        A Python list holds boxed elements, so there is no contiguous array to
        point at. A buffer-protocol object already has one (spec 6.4, 13.8).
        """
        return self.kind == "view"

    @property
    def is_tuple(self) -> bool:
        return self.kind == "tuple"

    @property
    def abi(self) -> tuple[str, ...]:
        if self.is_buffer:
            return (f"{_abi_name(self.element)}*", "i64")
        if self.is_tuple:
            return tuple(_abi_name(element) for element in self.elements)
        if self.is_object:
            return tuple(_abi_name(scalar) for _field, scalar in self.fields)
        return (_abi_name(self.kind),)

    def __str__(self) -> str:
        if self.is_buffer:
            borrow = " borrowed" if self.is_borrowed else ""
            return f"{_abi_name(self.element)}*{borrow} {self.name}, i64 {self.name}_len"
        if self.is_tuple:
            return ", ".join(
                f"{_abi_name(element)} {self.name}{index}"
                for index, element in enumerate(self.elements)
            )
        if self.is_object:
            return ", ".join(
                f"{_abi_name(scalar)} {self.name}_{field}" for field, scalar in self.fields
            )
        return f"{_abi_name(self.kind)} {self.name}"


@dataclass(frozen=True, slots=True)
class NativeSignature:
    """The PPY native ABI for one function (spec 16.4)."""

    qualname: str
    symbol: str
    parameters: tuple[NativeParam, ...]
    returns: tuple[str, ...]
    #: The body touches no Python object once its arguments are unpacked, so
    #: the boundary may drop the GIL around the call (spec 16.6).
    releases_gil: bool = False

    @property
    def ret(self) -> str:
        return self.returns[0] if len(self.returns) == 1 else "{" + ", ".join(self.returns) + "}"

    @property
    def returns_tuple(self) -> bool:
        return len(self.returns) > 1

    @property
    def params(self) -> tuple[str, ...]:
        return tuple(atom for parameter in self.parameters for atom in parameter.abi)

    def __str__(self) -> str:
        rendered = ", ".join(str(p) for p in self.parameters)
        return f"{self.ret} {self.symbol}({rendered})"


@dataclass(slots=True)
class LoweredFunction:
    info: FunctionInfo
    signature: NativeSignature
    reason: str = ""


@dataclass(slots=True)
class LoweringResult:
    ir: str
    functions: dict[str, LoweredFunction] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)


def _scalar_name(t: T.Type) -> str | None:
    base = T.strip_literal(t)
    if isinstance(base, T.Instance) and base.name in _SCALARS:
        return base.name
    return None


def _buffer_element(t: T.Type) -> tuple[str, str] | None:
    """The kind and element scalar of a buffer parameter, if it is one."""
    base = T.strip_literal(t)
    if not isinstance(base, T.Instance) or len(base.args) != 1:
        return None
    if base.name not in {"list", "Buffer", "memoryview", "array"}:
        return None
    element = _scalar_name(base.args[0])
    if element not in _BUFFER_ELEMENTS:
        return None
    return ("list" if base.name == "list" else "view"), element


def _tuple_elements(t: T.Type) -> tuple[str, ...] | None:
    """The scalar element kinds of a fixed-size tuple, if it has any."""
    base = T.strip_literal(t)
    if not isinstance(base, T.Tuple_) or base.homogeneous:
        return None
    if not base.items or len(base.items) > _MAX_TUPLE_WIDTH:
        return None
    kinds = [_scalar_name(item) for item in base.items]
    if any(kind is None for kind in kinds):
        return None
    return tuple(kind for kind in kinds if kind is not None)


#: Layouts of the value classes this module may flatten, by qualified name.
ClassLayouts = dict


def _class_fields(t: T.Type, layouts: ClassLayouts) -> tuple[str, tuple[tuple[str, str], ...]] | None:
    """The flattened field layout of a value-class parameter, if it has one."""
    base = T.strip_literal(t)
    if not isinstance(base, T.Instance):
        return None
    fields = layouts.get(base.name)
    if not fields or len(fields) > _MAX_CLASS_WIDTH:
        return None
    return base.name, tuple(fields)


def _native_param(name: str, t: T.Type, layouts: ClassLayouts | None = None) -> NativeParam | None:
    scalar = _scalar_name(t)
    if scalar is not None:
        return NativeParam(name, scalar)
    buffer = _buffer_element(t)
    if buffer is not None:
        kind, element = buffer
        return NativeParam(name, kind, element)
    elements = _tuple_elements(t)
    if elements is not None:
        return NativeParam(name, "tuple", elements=elements)
    described = _class_fields(t, layouts or {})
    if described is not None:
        class_name, fields = described
        return NativeParam(name, "object", fields=fields, class_name=class_name)
    return None


def _return_atoms(t: T.Type) -> tuple[str, ...] | None:
    scalar = _scalar_name(t)
    if scalar is not None:
        return (scalar,)
    return _tuple_elements(t)


def eligible(
    info: FunctionInfo, analysis: FunctionAnalysis, layouts: ClassLayouts | None = None
) -> tuple[bool, str]:
    """Can this function be lowered to a native scalar entry point?"""
    if info.is_async or info.is_generator:
        return False, "coroutines and generators use the boxed runtime"
    for name in sorted(analysis.mutated_params):
        declared = next((p.type for p in info.params if p.name == name), None)
        described = _buffer_element(declared) if declared is not None else None
        # Writing through a borrowed buffer is visible to the caller, which is
        # what borrowing means. Anything else would lose the write.
        if described is None or described[0] != "view":
            return False, f"mutates `{name}`, which is not a borrowed buffer"
    if analysis.foreign_writes:
        return False, "writes through a target the compiler cannot identify"

    violations = set(analysis.effects.violations())
    if analysis.mutated_params:
        # Those writes land in memory the caller lent us, and nowhere else.
        violations.discard(Effect.WRITE_OBJECT)
    if violations:
        listed = ", ".join(sorted(str(e) for e in violations))
        return False, f"has effects that must run on CPython: {listed}"
    for param in info.params:
        if param.kind in {"var_positional", "var_keyword"}:
            return False, "variadic parameters have no native ABI"
        if _native_param(param.name, param.type, layouts) is None:
            return False, f"parameter `{param.name}` is `{param.type}`, which has no native ABI"
    if _return_atoms(info.ret) is None:
        return False, f"returns `{info.ret}`, which has no native ABI"
    return True, ""


def lower_specialization(
    module: ModuleAnalysis,
    info: FunctionInfo,
    node: ast.FunctionDef,
    constants: dict[str, object],
    symbol: str,
    layouts: ClassLayouts | None = None,
) -> str:
    """Emit an LLVM module holding one specialized copy of a function.

    The specialization keeps the generic ABI, so the caller may pass the same
    arguments; the guards that select it are what make the pinned values safe
    (spec 16.9).
    """
    from llvmlite import ir

    llvm_module = ir.Module(name=f"{module.name}.{symbol}")
    llvm_module.triple = _default_triple()

    # A specialization is the same body with values pinned, so it inherits the
    # generic function's decision about the GIL.
    signature = _signature(info, layouts, module.functions.get(info.qualname))
    function = ir.Function(llvm_module, _function_type(ir, info, layouts), name=symbol)
    function.linkage = "external"
    declarations = {info.qualname: (function, signature)}
    _FunctionLowering(
        ir, llvm_module, function, info, module, declarations, constants, layouts
    ).run(node)
    return str(llvm_module)


def lower_module(
    module: ModuleAnalysis,
    functions: dict[str, tuple[FunctionInfo, FunctionAnalysis, ast.FunctionDef]],
    layouts: ClassLayouts | None = None,
) -> LoweringResult:
    """Emit an LLVM module for every eligible function."""
    from llvmlite import ir

    llvm_module = ir.Module(name=module.name)
    llvm_module.triple = _default_triple()

    candidates: dict[str, tuple[FunctionInfo, FunctionAnalysis, ast.FunctionDef]] = {}
    rejected: dict[str, str] = {}
    for qualname, (info, analysis, node) in functions.items():
        ok, reason = eligible(info, analysis, layouts)
        if ok:
            candidates[qualname] = (info, analysis, node)
        else:
            rejected[qualname] = reason

    declarations: dict[str, tuple[object, NativeSignature]] = {}
    for qualname, (info, analysis, _node) in candidates.items():
        signature = _signature(info, layouts, analysis)
        function_type = _function_type(ir, info, layouts)
        function = ir.Function(llvm_module, function_type, name=signature.symbol)
        function.linkage = "external"
        declarations[qualname] = (function, signature)

    lowered: dict[str, LoweredFunction] = {}
    for qualname, (info, analysis, node) in candidates.items():
        function, signature = declarations[qualname]
        try:
            _FunctionLowering(
                ir, llvm_module, function, info, module, declarations, layouts=layouts
            ).run(node)
        except Unsupported as exc:
            rejected[qualname] = str(exc)
            function.blocks.clear()
            continue
        lowered[qualname] = LoweredFunction(info, signature)

    for qualname in list(rejected):
        entry = declarations.get(qualname)
        if entry is not None and not entry[0].blocks:
            entry[0].linkage = "external"

    return LoweringResult(ir=str(llvm_module), functions=lowered, rejected=rejected)


def _default_triple() -> str:
    try:
        import llvmlite.binding as llvm

        return llvm.get_process_triple()
    except Exception:  # noqa: BLE001 - triple is informational for IR dumps
        return ""


def _signature(
    info: FunctionInfo,
    layouts: ClassLayouts | None = None,
    analysis: FunctionAnalysis | None = None,
) -> NativeSignature:
    parameters = tuple(
        _native_param(p.name, p.type, layouts) or NativeParam(p.name, "int")
        for p in info.params
    )
    atoms = _return_atoms(info.ret) or ("int",)
    return NativeSignature(
        qualname=info.qualname,
        symbol="ppy_" + info.qualname.replace(".", "_"),
        parameters=parameters,
        returns=tuple(_abi_name(atom) for atom in atoms),
        releases_gil=_releases_gil(analysis) if analysis is not None else False,
    )


#: Effects that mean the body can reach the interpreter while it runs, so the
#: GIL has to be held for the whole call.
_NEEDS_GIL = (Effect.PYTHON_CALLBACK, Effect.EXTERNAL_UNKNOWN, Effect.IO)


def _releases_gil(analysis: FunctionAnalysis) -> bool:
    """May the boundary drop the GIL around this call? (spec 16.6)

    Arguments are unpacked into machine values before the call and the result
    is built after it, so the only question is whether the body itself can
    touch a Python object. A borrowed buffer does not count: the caller holds
    the reference and the boundary pins the memory for the whole call, which is
    the same guarantee NumPy relies on.
    """
    return not any(effect in analysis.effects for effect in _NEEDS_GIL)


def _abi_name(scalar: str) -> str:
    return {"int": "i64", "float": "double", "bool": "i8"}[scalar]


def _llvm_type(ir, scalar: str):  # type: ignore[no-untyped-def]
    return {"int": ir.IntType(64), "float": ir.DoubleType(), "bool": ir.IntType(8)}[scalar]


def _constant_value(ir, value, scalar: str):  # type: ignore[no-untyped-def]
    """An LLVM constant for a pinned parameter value."""
    if scalar == "float":
        return ir.Constant(ir.DoubleType(), float(value))
    if scalar == "bool":
        return ir.Constant(ir.IntType(8), int(bool(value)))
    return ir.Constant(ir.IntType(64), int(value))


def _function_type(ir, info: FunctionInfo, layouts: ClassLayouts | None = None):  # type: ignore[no-untyped-def]
    atoms = []
    for parameter in _signature(info, layouts).parameters:
        if parameter.is_buffer:
            atoms.append(_llvm_type(ir, parameter.element).as_pointer())
            atoms.append(ir.IntType(64))
        elif parameter.is_tuple:
            atoms.extend(_llvm_type(ir, element) for element in parameter.elements)
        elif parameter.is_object:
            atoms.extend(_llvm_type(ir, scalar) for _field, scalar in parameter.fields)
        else:
            atoms.append(_llvm_type(ir, parameter.kind))
    outs = [
        _llvm_type(ir, atom).as_pointer() for atom in (_return_atoms(info.ret) or ("int",))
    ]
    return ir.FunctionType(ir.IntType(32), [*atoms, *outs])


@dataclass(slots=True)
class _Value:
    value: object
    scalar: str


@dataclass(slots=True)
class _Tuple:
    """A fixed tuple held as scalar SSA values, never as an object (spec 13.2)."""

    items: list[_Value]

    @property
    def width(self) -> int:
        return len(self.items)


class _FunctionLowering:
    """Lowers one function body to LLVM IR."""

    def __init__(  # type: ignore[no-untyped-def]
        self, ir, module, function, info, analysis, declarations, constants=None, layouts=None
    ) -> None:
        # Strict Python floating-point ordering unless the function opted out.
        self.fp_flags = (
            list(FASTMATH_FLAGS) if info.directive("fastmath") is not None else []
        )
        self.ir = ir
        self.module = module
        self.function = function
        self.info = info
        self.analysis = analysis
        self.declarations = declarations
        # Parameter values a specialization has pinned. The ABI is unchanged:
        # the argument is still passed, but the body uses the constant, which
        # is what lets LLVM fold, unroll, and vectorize around it (spec 16.9).
        self.constants: dict[str, object] = dict(constants or {})
        self.builder = None
        self.locals: dict[str, tuple[object, str]] = {}
        self.buffers: dict[str, tuple[object, object, str, bool]] = {}
        self.tuples: dict[str, list[tuple[object, str]]] = {}
        #: Value-class parameters, flattened to one slot per field.
        self.objects: dict[str, dict[str, tuple[object, str]]] = {}
        self.layouts = dict(layouts or {})
        self.returns = _return_atoms(info.ret) or ("int",)
        self.outs = list(function.args[-len(self.returns):])
        self.out = function.args[-1]
        self.fallback_block = None

    def run(self, node: ast.FunctionDef) -> None:
        ir = self.ir
        entry = self.function.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)
        self.fallback_block = self.function.append_basic_block("fallback")

        position = 0
        for parameter in _signature(self.info, self.layouts).parameters:
            if parameter.is_buffer:
                data = self.function.args[position]
                length = self.function.args[position + 1]
                pinned = self.constants.get(f"len({parameter.name})")
                if isinstance(pinned, int):
                    length = ir.Constant(ir.IntType(64), pinned)
                self.buffers[parameter.name] = (
                    data, length, parameter.element, parameter.is_borrowed
                )
                position += 2
                continue
            if parameter.is_object:
                fields: dict[str, tuple[object, str]] = {}
                for index, (field, scalar) in enumerate(parameter.fields):
                    slot = self.builder.alloca(
                        _llvm_type(ir, scalar), name=f"{parameter.name}_{field}"
                    )
                    self.builder.store(self.function.args[position + index], slot)
                    fields[field] = (slot, scalar)
                self.objects[parameter.name] = fields
                position += len(parameter.fields)
                continue
            if parameter.is_tuple:
                slots = []
                for index, element in enumerate(parameter.elements):
                    slot = self.builder.alloca(
                        _llvm_type(ir, element), name=f"{parameter.name}{index}"
                    )
                    self.builder.store(self.function.args[position + index], slot)
                    slots.append((slot, element))
                self.tuples[parameter.name] = slots
                position += len(parameter.elements)
                continue
            slot = self.builder.alloca(_llvm_type(ir, parameter.kind), name=parameter.name)
            pinned = self.constants.get(parameter.name)
            initial = (
                _constant_value(ir, pinned, parameter.kind)
                if pinned is not None
                else self.function.args[position]
            )
            self.builder.store(initial, slot)
            self.locals[parameter.name] = (slot, parameter.kind)
            position += 1

        self._body(node.body)
        if not self.builder.block.is_terminated:
            self._return_default()

        with self.builder.goto_block(self.fallback_block):
            self.builder.ret(ir.Constant(ir.IntType(32), STATUS_FALLBACK))

    def _body(self, body: list[ast.stmt]) -> None:
        for statement in body:
            if self.builder.block.is_terminated:
                return
            self._statement(statement)

    def _statement(self, node: ast.stmt) -> None:
        match node:
            case ast.Return():
                self._return(node)
            case ast.Assign():
                self._assign(node)
            case ast.AnnAssign():
                if node.value is not None:
                    self._store(node.target, self._expr(node.value))
            case ast.AugAssign():
                self._augassign(node)
            case ast.If():
                self._if(node)
            case ast.While():
                self._while(node)
            case ast.For():
                self._for(node)
            case ast.Pass():
                return
            case ast.Expr(value=ast.Constant()):
                return
            case _:
                raise Unsupported(f"`{type(node).__name__}` has no native lowering")

    def _return(self, node: ast.Return) -> None:
        ir = self.ir
        if node.value is None:
            raise Unsupported("a native function must return a value")
        if len(self.returns) > 1:
            values = self._tuple_expr(node.value)
            if values is None or values.width != len(self.returns):
                raise Unsupported("the returned tuple does not match the declared shape")
            for slot, item, target in zip(self.outs, values.items, self.returns):
                self.builder.store(self._coerce(item, target).value, slot)
        else:
            value = self._coerce(self._expr(node.value), self.returns[0])
            self.builder.store(value.value, self.outs[0])
        self.builder.ret(ir.Constant(ir.IntType(32), STATUS_OK))

    def _return_default(self) -> None:
        raise Unsupported("control flow can fall off the end without returning a value")

    def _assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise Unsupported("chained assignment has no native lowering")
        target = node.targets[0]
        values = self._tuple_expr(node.value)
        if values is not None:
            self._store_tuple(target, values)
            return
        self._store(target, self._expr(node.value))

    def _store_tuple(self, target: ast.expr, values: "_Tuple") -> None:
        """Bind a tuple to a name, or unpack it into several names."""
        if isinstance(target, ast.Name):
            slots = []
            for index, item in enumerate(values.items):
                slot = self.builder.alloca(
                    _llvm_type(self.ir, item.scalar), name=f"{target.id}{index}"
                )
                self.builder.store(item.value, slot)
                slots.append((slot, item.scalar))
            self.tuples[target.id] = slots
            self.locals.pop(target.id, None)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if len(target.elts) != values.width:
                raise Unsupported("unpacking width does not match the tuple")
            for element_target, item in zip(target.elts, values.items):
                self._store(element_target, item)
            return
        raise Unsupported("this assignment target has no native lowering")

    def _tuple_expr(self, node: ast.expr) -> "_Tuple | None":
        """Evaluate an expression that produces a fixed tuple, if it does."""
        if isinstance(node, ast.Tuple):
            if not node.elts or len(node.elts) > _MAX_TUPLE_WIDTH:
                return None
            if any(isinstance(e, ast.Starred) for e in node.elts):
                raise Unsupported("a starred element has no fixed native width")
            return _Tuple([self._expr(element) for element in node.elts])
        if isinstance(node, ast.Name) and node.id in self.tuples:
            return _Tuple(
                [_Value(self.builder.load(slot), scalar) for slot, scalar in self.tuples[node.id]]
            )
        return None

    def _augassign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Subscript):
            if not (
                isinstance(node.target.value, ast.Name)
                and node.target.value.id in self.buffers
            ):
                raise Unsupported("this augmented assignment has no native lowering")
            current = self._buffer_element(
                node.target.value.id, self._expr(node.target.slice)
            )
            value = self._expr(node.value)
            self._store(node.target, self._binary(current, value, type(node.op)))
            return
        if not isinstance(node.target, ast.Name):
            raise Unsupported("augmented assignment to a non-local has no native lowering")
        current = self._load(node.target.id)
        value = self._expr(node.value)
        self._store(node.target, self._binary(current, value, type(node.op)))

    def _store(self, target: ast.expr, value: _Value) -> None:
        if isinstance(target, ast.Attribute):
            raise Unsupported("a flattened value class cannot be written back")
        if isinstance(target, ast.Subscript):
            if isinstance(target.value, ast.Name) and target.value.id in self.buffers:
                self._store_element(target.value.id, target.slice, value)
                return
            raise Unsupported("this subscript assignment has no native lowering")
        if not isinstance(target, ast.Name):
            raise Unsupported("assignment to a non-local has no native lowering")
        existing = self.locals.get(target.id)
        if existing is None:
            slot = self.builder.alloca(_llvm_type(self.ir, value.scalar), name=target.id)
            self.locals[target.id] = (slot, value.scalar)
            self.builder.store(value.value, slot)
            return
        slot, scalar = existing
        coerced = self._coerce(value, scalar)
        self.builder.store(coerced.value, slot)

    def _if(self, node: ast.If) -> None:
        condition = self._truth(self._expr(node.test))
        then_block = self.function.append_basic_block("then")
        else_block = self.function.append_basic_block("else")
        merge_block = self.function.append_basic_block("endif")
        self.builder.cbranch(condition, then_block, else_block)

        self.builder.position_at_end(then_block)
        self._body(node.body)
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_block)

        self.builder.position_at_end(else_block)
        self._body(node.orelse)
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_block)

        self.builder.position_at_end(merge_block)
        if not then_block.is_terminated and not else_block.is_terminated:
            return

    def _while(self, node: ast.While) -> None:
        if node.orelse:
            raise Unsupported("`while ... else` has no native lowering")
        header = self.function.append_basic_block("while.head")
        body = self.function.append_basic_block("while.body")
        exit_block = self.function.append_basic_block("while.end")
        self.builder.branch(header)

        self.builder.position_at_end(header)
        condition = self._truth(self._expr(node.test))
        self.builder.cbranch(condition, body, exit_block)

        self.builder.position_at_end(body)
        self._body(node.body)
        if not self.builder.block.is_terminated:
            self.builder.branch(header)

        self.builder.position_at_end(exit_block)

    def _for(self, node: ast.For) -> None:
        ir = self.ir
        if node.orelse or not isinstance(node.target, ast.Name):
            raise Unsupported("only `for NAME in range(...)` or over a list parameter is lowered")
        if isinstance(node.iter, ast.Name) and node.iter.id in self.buffers:
            self._for_buffer(node, node.iter.id)
            return
        if not (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"):
            raise Unsupported("only `for NAME in range(...)` or over a list parameter is lowered")

        bounds = [self._coerce(self._expr(a), "int") for a in node.iter.args]
        i64 = ir.IntType(64)
        if len(bounds) == 1:
            start, stop, step = ir.Constant(i64, 0), bounds[0].value, ir.Constant(i64, 1)
        elif len(bounds) == 2:
            start, stop, step = bounds[0].value, bounds[1].value, ir.Constant(i64, 1)
        elif len(bounds) == 3:
            start, stop, step = bounds[0].value, bounds[1].value, bounds[2].value
            if not isinstance(step, ir.Constant):
                raise Unsupported("a non-constant `range` step has no native lowering")
        else:
            raise Unsupported("`range` takes at most three arguments")

        slot = self.builder.alloca(i64, name=node.target.id)
        self.builder.store(start, slot)
        self.locals[node.target.id] = (slot, "int")

        header = self.function.append_basic_block("for.head")
        body = self.function.append_basic_block("for.body")
        exit_block = self.function.append_basic_block("for.end")
        self.builder.branch(header)

        self.builder.position_at_end(header)
        current = self.builder.load(slot)
        descending = isinstance(step, ir.Constant) and step.constant < 0
        condition = self.builder.icmp_signed(">" if descending else "<", current, stop)
        self.builder.cbranch(condition, body, exit_block)

        self.builder.position_at_end(body)
        self._body(node.body)
        if not self.builder.block.is_terminated:
            value = self.builder.load(slot)
            self.builder.store(self._checked_add(value, step), slot)
            self.builder.branch(header)

        self.builder.position_at_end(exit_block)

    def _for_buffer(self, node: ast.For, name: str) -> None:
        """`for x in xs:` over a borrowed native buffer."""
        ir = self.ir
        data, length, element, _borrowed = self.buffers[name]
        i64 = ir.IntType(64)

        index = self.builder.alloca(i64, name=f"{name}.i")
        self.builder.store(ir.Constant(i64, 0), index)
        slot = self.builder.alloca(_llvm_type(ir, element), name=node.target.id)
        self.locals[node.target.id] = (slot, element)

        header = self.function.append_basic_block("each.head")
        body = self.function.append_basic_block("each.body")
        exit_block = self.function.append_basic_block("each.end")
        self.builder.branch(header)

        self.builder.position_at_end(header)
        current = self.builder.load(index)
        self.builder.cbranch(self.builder.icmp_signed("<", current, length), body, exit_block)

        self.builder.position_at_end(body)
        position = self.builder.load(index)
        self.builder.store(self.builder.load(self.builder.gep(data, [position])), slot)
        self._body(node.body)
        if not self.builder.block.is_terminated:
            self.builder.store(self.builder.add(self.builder.load(index), ir.Constant(i64, 1)), index)
            self.builder.branch(header)

        self.builder.position_at_end(exit_block)

    def _store_element(self, name: str, index: ast.expr, value: _Value) -> None:
        """`xs[i] = v` through a borrowed buffer: the caller sees the write."""
        data, length, element, borrowed = self.buffers[name]
        if not borrowed:
            # A list is copied in, so a write to it would be lost.
            raise Unsupported(f"`{name}` is copied in, so writing to it has no effect")
        if isinstance(index, ast.Slice):
            raise Unsupported("slice assignment has no native lowering")
        position = self._coerce(self._expr(index), "int")
        self._guard_index(position, length)
        self.builder.store(
            self._coerce(value, element).value, self.builder.gep(data, [position.value])
        )

    def _guard_index(self, position: _Value, length) -> None:  # type: ignore[no-untyped-def]
        """A negative or out-of-range index is left to CPython."""
        ir = self.ir
        zero = ir.Constant(ir.IntType(64), 0)
        negative = self.builder.icmp_signed("<", position.value, zero)
        beyond = self.builder.icmp_signed(">=", position.value, length)
        ok = self.function.append_basic_block("index.ok")
        self.builder.cbranch(self.builder.or_(negative, beyond), self.fallback_block, ok)
        self.builder.position_at_end(ok)

    def _buffer_element(self, name: str, index: _Value) -> _Value:
        """`xs[i]` with the bounds and negative-index checks Python requires."""
        data, length, element, _borrowed = self.buffers[name]
        position = self._coerce(index, "int")
        self._guard_index(position, length)
        return _Value(self.builder.load(self.builder.gep(data, [position.value])), element)

    def _buffer_reduction(self, name: str, operation: str) -> _Value:
        """`sum`, `min`, or `max` over a buffer, in strict source order."""
        ir = self.ir
        data, length, element, _borrowed = self.buffers[name]
        i64 = ir.IntType(64)
        zero = ir.Constant(i64, 0)

        if operation in {"min", "max"}:
            # An empty sequence must raise ValueError, which CPython does.
            self.builder.cbranch(
                self.builder.icmp_signed("==", length, zero),
                self.fallback_block,
                (nonempty := self.function.append_basic_block("reduce.nonempty")),
            )
            self.builder.position_at_end(nonempty)

        accumulator = self.builder.alloca(_llvm_type(ir, element), name=f"{operation}.acc")
        index = self.builder.alloca(i64, name=f"{operation}.i")
        if operation == "sum":
            start = ir.Constant(_llvm_type(ir, element), 0)
            self.builder.store(start, accumulator)
            self.builder.store(zero, index)
        else:
            self.builder.store(self.builder.load(self.builder.gep(data, [zero])), accumulator)
            self.builder.store(ir.Constant(i64, 1), index)

        header = self.function.append_basic_block("reduce.head")
        body = self.function.append_basic_block("reduce.body")
        exit_block = self.function.append_basic_block("reduce.end")
        self.builder.branch(header)

        self.builder.position_at_end(header)
        current = self.builder.load(index)
        self.builder.cbranch(self.builder.icmp_signed("<", current, length), body, exit_block)

        self.builder.position_at_end(body)
        position = self.builder.load(index)
        value = _Value(self.builder.load(self.builder.gep(data, [position])), element)
        carried = _Value(self.builder.load(accumulator), element)
        if operation == "sum":
            # Sequential accumulation keeps strict Python ordering; no
            # reassociation happens without an explicit directive (spec 17.2).
            updated = self._binary(carried, value, ast.Add)
        else:
            symbol = "<" if operation == "min" else ">"
            if element == "float":
                keep = self.builder.fcmp_ordered(symbol, value.value, carried.value)
            else:
                keep = self.builder.icmp_signed(symbol, value.value, carried.value)
            updated = _Value(self.builder.select(keep, value.value, carried.value), element)
        self.builder.store(updated.value, accumulator)
        self.builder.store(self.builder.add(self.builder.load(index), ir.Constant(i64, 1)), index)
        self.builder.branch(header)

        self.builder.position_at_end(exit_block)
        return _Value(self.builder.load(accumulator), element)

    def _expr(self, node: ast.expr) -> _Value:
        ir = self.ir
        match node:
            case ast.Constant(value=bool(value)):
                return _Value(ir.Constant(ir.IntType(8), int(value)), "bool")
            case ast.Constant(value=int(value)):
                if not -(1 << 63) <= value < (1 << 63):
                    raise Unsupported("an integer literal exceeds the native machine range")
                return _Value(ir.Constant(ir.IntType(64), value), "int")
            case ast.Constant(value=float(value)):
                return _Value(ir.Constant(ir.DoubleType(), value), "float")
            case ast.Name():
                return self._load(node.id)
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

    def _field(self, name: str, field: str) -> _Value:
        """`p.x` for a flattened value class: one of its SSA values."""
        slots = self.objects[name]
        found = slots.get(field)
        if found is None:
            raise Unsupported(f"`{name}.{field}` is not a native field")
        slot, scalar = found
        return _Value(self.builder.load(slot), scalar)

    def _tuple_element(self, name: str, index: ast.expr) -> _Value:
        """`t[i]` for a constant `i`, which is just one of the SSA values."""
        slots = self.tuples[name]
        if not isinstance(index, ast.Constant) or not isinstance(index.value, int):
            raise Unsupported("a fixed tuple must be indexed by a constant")
        position = index.value
        if position < 0:
            position += len(slots)
        if not 0 <= position < len(slots):
            raise Unsupported(f"index {index.value} is out of range for `{name}`")
        slot, scalar = slots[position]
        return _Value(self.builder.load(slot), scalar)

    def _load(self, name: str) -> _Value:
        if name in self.objects:
            raise Unsupported(f"`{name}` is a value class, which has no single scalar value")
        if name in self.tuples:
            raise Unsupported(f"`{name}` is a tuple, which has no single scalar value")
        if name in self.buffers:
            raise Unsupported(f"`{name}` is a buffer, which has no scalar value")
        found = self.locals.get(name)
        if found is None:
            raise Unsupported(f"`{name}` is not a native local")
        slot, scalar = found
        return _Value(self.builder.load(slot), scalar)

    def _unary(self, node: ast.UnaryOp) -> _Value:
        operand = self._expr(node.operand)
        ir = self.ir
        match node.op:
            case ast.USub():
                if operand.scalar == "float":
                    return _Value(self.builder.fneg(operand.value), "float")
                promoted = self._coerce(operand, "int")
                zero = ir.Constant(ir.IntType(64), 0)
                return _Value(self._checked_binary(zero, promoted.value, ast.Sub), "int")
            case ast.UAdd():
                return operand
            case ast.Not():
                truth = self._truth(operand)
                return _Value(self.builder.zext(self.builder.not_(truth), ir.IntType(8)), "bool")
        raise Unsupported("unary operator has no native lowering")

    def _boolop(self, node: ast.BoolOp) -> _Value:
        """`and`/`or` with native short-circuit evaluation."""
        ir = self.ir
        result = self.builder.alloca(ir.IntType(8), name="boolop")
        done = self.function.append_basic_block("boolop.end")
        is_and = isinstance(node.op, ast.And)

        for index, value_node in enumerate(node.values):
            value = self._expr(value_node)
            truth = self._truth(value)
            self.builder.store(self.builder.zext(truth, ir.IntType(8)), result)
            if index == len(node.values) - 1:
                self.builder.branch(done)
                break
            following = self.function.append_basic_block(f"boolop.{index}")
            if is_and:
                self.builder.cbranch(truth, following, done)
            else:
                self.builder.cbranch(truth, done, following)
            self.builder.position_at_end(following)

        self.builder.position_at_end(done)
        return _Value(self.builder.load(result), "bool")

    def _compare(self, node: ast.Compare) -> _Value:
        ir = self.ir
        if len(node.ops) != 1:
            raise Unsupported("chained comparison has no native lowering")
        left = self._expr(node.left)
        right = self._expr(node.comparators[0])
        symbol = {
            ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<",
            ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
        }.get(type(node.ops[0]))
        if symbol is None:
            raise Unsupported("comparison operator has no native lowering")
        scalar = self._unify(left.scalar, right.scalar)
        left, right = self._coerce(left, scalar), self._coerce(right, scalar)
        if scalar == "float":
            result = self.builder.fcmp_ordered(symbol, left.value, right.value)
        else:
            result = self.builder.icmp_signed(symbol, left.value, right.value)
        return _Value(self.builder.zext(result, ir.IntType(8)), "bool")

    def _ifexp(self, node: ast.IfExp) -> _Value:
        condition = self._truth(self._expr(node.test))
        then_value = self._expr(node.body)
        else_value = self._expr(node.orelse)
        scalar = self._unify(then_value.scalar, else_value.scalar)
        then_value = self._coerce(then_value, scalar)
        else_value = self._coerce(else_value, scalar)
        return _Value(self.builder.select(condition, then_value.value, else_value.value), scalar)

    def _call(self, node: ast.Call) -> _Value:
        if node.keywords:
            raise Unsupported("keyword arguments have no native ABI")
        target = ast.unparse(node.func)
        if target.startswith("math."):
            return self._math_call(target.removeprefix("math."), node)
        if target == "len" and len(node.args) == 1:
            argument = node.args[0]
            if isinstance(argument, ast.Name) and argument.id in self.tuples:
                return _Value(
                    self.ir.Constant(self.ir.IntType(64), len(self.tuples[argument.id])), "int"
                )
        if target in {"len", "sum", "min", "max"} and len(node.args) == 1:
            argument = node.args[0]
            if isinstance(argument, ast.Name) and argument.id in self.buffers:
                if target == "len":
                    return _Value(self.buffers[argument.id][1], "int")
                return self._buffer_reduction(argument.id, target)
        if target in {"min", "max"} and len(node.args) >= 2:
            return self._extremum(target, node)
        if target in {"abs", "float", "int", "bool"}:
            return self._builtin_call(target, node)
        for qualname, (function, signature) in self.declarations.items():
            if qualname.rpartition(".")[2] == target:
                return self._native_call(function, signature, qualname, node)
        raise Unsupported(f"`{target}` has no native lowering")

    def _extremum(self, target: str, node: ast.Call) -> _Value:
        """`min(a, b, ...)` and `max(a, b, ...)` over scalars.

        Written as a fold of pairwise selects so that the comparison keeps the
        argument order Python guarantees: the first of equal values wins.
        """
        values = [self._expr(argument) for argument in node.args]
        kinds = {value.scalar for value in values}
        if len(kinds) != 1:
            # CPython returns the winning object, so `max(3, 2.5)` is the int
            # `3` and not `3.0`. One scalar kind cannot express that.
            raise Unsupported(f"`{target}` over mixed {sorted(kinds)} would change the result type")
        kind = values[0].scalar
        if kind not in {"int", "float"}:
            raise Unsupported(f"`{target}` over `{kind}` has no native lowering")
        best = self._coerce(values[0], kind)
        for candidate in values[1:]:
            other = self._coerce(candidate, kind)
            if kind == "float":
                # NaN propagates the way `min`/`max` do: a comparison against it
                # is false, so the incumbent is kept.
                wins = self.builder.fcmp_ordered(
                    "<" if target == "min" else ">", other.value, best.value
                )
            else:
                wins = self.builder.icmp_signed(
                    "<" if target == "min" else ">", other.value, best.value
                )
            best = _Value(self.builder.select(wins, other.value, best.value), kind)
        return best

    def _native_call(self, function, signature: NativeSignature, qualname: str, node: ast.Call) -> _Value:
        ir = self.ir
        info = self.declarations[qualname][1]
        if len(node.args) != len(signature.parameters):
            raise Unsupported(f"`{qualname}` called with the wrong number of arguments")
        scalars = {"i64": "int", "double": "float", "i8": "bool"}
        arguments: list[object] = []
        for argument, parameter in zip(node.args, signature.parameters):
            if parameter.is_buffer:
                raise Unsupported("a buffer cannot be forwarded between native calls yet")
            if parameter.is_object:
                if not isinstance(argument, ast.Name) or argument.id not in self.objects:
                    raise Unsupported("a value class argument must be a flattened local")
                slots = self.objects[argument.id]
                for field, scalar in parameter.fields:
                    entry = slots.get(field)
                    if entry is None:
                        raise Unsupported(f"`{argument.id}` has no field `{field}`")
                    arguments.append(self._coerce(self._field(argument.id, field), scalar).value)
                continue
            if parameter.is_tuple:
                values = self._tuple_expr(argument)
                if values is None or values.width != len(parameter.elements):
                    raise Unsupported("a tuple argument does not match the callee's shape")
                arguments.extend(
                    self._coerce(item, element).value
                    for item, element in zip(values.items, parameter.elements)
                )
                continue
            arguments.append(self._coerce(self._expr(argument), parameter.kind).value)
        if signature.returns_tuple:
            raise Unsupported("a tuple result cannot be forwarded between native calls yet")
        result_scalar = scalars[signature.returns[0]]
        slot = self.builder.alloca(_llvm_type(ir, result_scalar), name="callresult")
        status = self.builder.call(function, [*arguments, slot])
        ok = self.builder.icmp_signed("==", status, ir.Constant(ir.IntType(32), STATUS_OK))
        continue_block = self.function.append_basic_block("call.ok")
        self.builder.cbranch(ok, continue_block, self.fallback_block)
        self.builder.position_at_end(continue_block)
        return _Value(self.builder.load(slot), result_scalar)

    def _math_call(self, name: str, node: ast.Call) -> _Value:
        intrinsic = _MATH_INTRINSICS.get(name)
        if intrinsic is None:
            raise Unsupported(f"`math.{name}` has no native lowering")
        arity = 2 if name == "pow" else 1
        if len(node.args) != arity:
            raise Unsupported(f"`math.{name}` takes {arity} argument(s)")
        arguments = [self._coerce(self._expr(a), "float").value for a in node.args]
        function = self._intrinsic(intrinsic, arity)
        return _Value(self.builder.call(function, arguments), "float")

    def _builtin_call(self, name: str, node: ast.Call) -> _Value:
        if len(node.args) != 1:
            raise Unsupported(f"`{name}` with this arity has no native lowering")
        value = self._expr(node.args[0])
        if name == "float":
            return self._coerce(value, "float")
        if name == "bool":
            return _Value(self.builder.zext(self._truth(value), self.ir.IntType(8)), "bool")
        if name == "int":
            if value.scalar == "float":
                # Python `int()` truncates toward zero, which matches fptosi.
                return _Value(self.builder.fptosi(value.value, self.ir.IntType(64)), "int")
            return self._coerce(value, "int")
        if name == "abs":
            if value.scalar == "float":
                return _Value(self.builder.call(self._intrinsic("llvm.fabs.f64", 1), [value.value]), "float")
            promoted = self._coerce(value, "int")
            zero = self.ir.Constant(self.ir.IntType(64), 0)
            negative = self.builder.icmp_signed("<", promoted.value, zero)
            negated = self._checked_binary(zero, promoted.value, ast.Sub)
            return _Value(self.builder.select(negative, negated, promoted.value), "int")
        raise Unsupported(f"`{name}` has no native lowering")

    def _intrinsic(self, name: str, arity: int):  # type: ignore[no-untyped-def]
        ir = self.ir
        existing = self.module.globals.get(name)
        if existing is not None:
            return existing
        double = ir.DoubleType()
        return ir.Function(self.module, ir.FunctionType(double, [double] * arity), name=name)

    def _binary(self, left: _Value, right: _Value, op: type[ast.operator]) -> _Value:
        ir = self.ir
        if op is ast.Div:
            left, right = self._coerce(left, "float"), self._coerce(right, "float")
            self._guard_nonzero(right)
            return _Value(self._fp(self.builder.fdiv, left.value, right.value), "float")

        scalar = self._unify(left.scalar, right.scalar)
        left, right = self._coerce(left, scalar), self._coerce(right, scalar)

        if scalar == "float":
            operations = {
                ast.Add: self.builder.fadd,
                ast.Sub: self.builder.fsub,
                ast.Mult: self.builder.fmul,
            }
            operation = operations.get(op)
            if operation is None:
                if op is ast.Pow:
                    return _Value(
                        self.builder.call(self._intrinsic("llvm.pow.f64", 2), [left.value, right.value]),
                        "float",
                    )
                raise Unsupported("floating-point operator has no native lowering")
            return _Value(self._fp(operation, left.value, right.value), "float")

        if op in _OVERFLOW_INTRINSICS:
            return _Value(self._checked_binary(left.value, right.value, op), "int")
        if op in {ast.FloorDiv, ast.Mod}:
            self._guard_nonzero(right)
            return _Value(self._python_floordiv(left.value, right.value, op), "int")
        if op in {ast.LShift, ast.RShift, ast.BitOr, ast.BitAnd, ast.BitXor}:
            return _Value(self._bitwise(left.value, right.value, op), "int")
        raise Unsupported("integer operator has no native lowering")

    def _fp(self, operation, left, right):  # type: ignore[no-untyped-def]
        """Emit a floating-point operation with the function's FP contract."""
        if self.fp_flags:
            return operation(left, right, flags=self.fp_flags)
        return operation(left, right)

    def _checked_add(self, value, step):  # type: ignore[no-untyped-def]
        return self._checked_binary(value, step, ast.Add)

    def _checked_binary(self, left, right, op: type[ast.operator]):  # type: ignore[no-untyped-def]
        """Machine arithmetic with an overflow guard back to CPython (spec 12.2)."""
        ir = self.ir
        name = _OVERFLOW_INTRINSICS[op]
        i64 = ir.IntType(64)
        signature = ir.FunctionType(ir.LiteralStructType([i64, ir.IntType(1)]), [i64, i64])
        function = self.module.globals.get(name) or ir.Function(self.module, signature, name=name)
        packed = self.builder.call(function, [left, right])
        result = self.builder.extract_value(packed, 0)
        overflowed = self.builder.extract_value(packed, 1)
        continue_block = self.function.append_basic_block("arith.ok")
        self.builder.cbranch(overflowed, self.fallback_block, continue_block)
        self.builder.position_at_end(continue_block)
        return result

    def _python_floordiv(self, left, right, op: type[ast.operator]):  # type: ignore[no-untyped-def]
        """`//` and `%` with Python's floor semantics, not C truncation."""
        ir = self.ir
        i64 = ir.IntType(64)
        zero = ir.Constant(i64, 0)
        quotient = self.builder.sdiv(left, right)
        remainder = self.builder.srem(left, right)
        nonzero = self.builder.icmp_signed("!=", remainder, zero)
        left_negative = self.builder.icmp_signed("<", left, zero)
        right_negative = self.builder.icmp_signed("<", right, zero)
        signs_differ = self.builder.xor(left_negative, right_negative)
        adjust = self.builder.and_(nonzero, signs_differ)
        if op is ast.FloorDiv:
            adjusted = self.builder.sub(quotient, ir.Constant(i64, 1))
            return self.builder.select(adjust, adjusted, quotient)
        adjusted = self.builder.add(remainder, right)
        return self.builder.select(adjust, adjusted, remainder)

    def _bitwise(self, left, right, op: type[ast.operator]):  # type: ignore[no-untyped-def]
        ir = self.ir
        i64 = ir.IntType(64)
        if op is ast.BitOr:
            return self.builder.or_(left, right)
        if op is ast.BitAnd:
            return self.builder.and_(left, right)
        if op is ast.BitXor:
            return self.builder.xor(left, right)
        # Python shifts have no width limit, so guard the shift count.
        limit = ir.Constant(i64, 63)
        too_wide = self.builder.icmp_signed(">", right, limit)
        negative = self.builder.icmp_signed("<", right, ir.Constant(i64, 0))
        bail = self.builder.or_(too_wide, negative)
        continue_block = self.function.append_basic_block("shift.ok")
        self.builder.cbranch(bail, self.fallback_block, continue_block)
        self.builder.position_at_end(continue_block)
        if op is ast.LShift:
            shifted = self.builder.shl(left, right)
            # A left shift that loses bits must fall back to arbitrary precision.
            restored = self.builder.ashr(shifted, right)
            lost = self.builder.icmp_signed("!=", restored, left)
            safe_block = self.function.append_basic_block("shl.ok")
            self.builder.cbranch(lost, self.fallback_block, safe_block)
            self.builder.position_at_end(safe_block)
            return shifted
        return self.builder.ashr(left, right)

    def _guard_nonzero(self, value: _Value) -> None:
        """Division by zero must raise the Python exception, so bail out."""
        ir = self.ir
        if value.scalar == "float":
            zero = ir.Constant(ir.DoubleType(), 0.0)
            is_zero = self.builder.fcmp_ordered("==", value.value, zero)
        else:
            zero = ir.Constant(ir.IntType(64), 0)
            is_zero = self.builder.icmp_signed("==", value.value, zero)
        continue_block = self.function.append_basic_block("div.ok")
        self.builder.cbranch(is_zero, self.fallback_block, continue_block)
        self.builder.position_at_end(continue_block)

    def _unify(self, left: str, right: str) -> str:
        if left == "float" or right == "float":
            return "float"
        if left == "int" or right == "int":
            return "int"
        return "bool"

    def _coerce(self, value: _Value, scalar: str) -> _Value:
        ir = self.ir
        if value.scalar == scalar:
            return value
        if scalar == "float":
            widened = self._coerce(value, "int") if value.scalar == "bool" else value
            return _Value(self.builder.sitofp(widened.value, ir.DoubleType()), "float")
        if scalar == "int":
            if value.scalar == "bool":
                return _Value(self.builder.zext(value.value, ir.IntType(64)), "int")
            raise Unsupported("an implicit float-to-int conversion is not a Python semantic")
        if scalar == "bool":
            return _Value(self.builder.zext(self._truth(value), ir.IntType(8)), "bool")
        raise Unsupported(f"cannot convert `{value.scalar}` to `{scalar}`")

    def _truth(self, value: _Value):  # type: ignore[no-untyped-def]
        ir = self.ir
        if value.scalar == "float":
            return self.builder.fcmp_ordered("!=", value.value, ir.Constant(ir.DoubleType(), 0.0))
        if value.scalar == "bool":
            return self.builder.icmp_signed("!=", value.value, ir.Constant(ir.IntType(8), 0))
        return self.builder.icmp_signed("!=", value.value, ir.Constant(ir.IntType(64), 0))
