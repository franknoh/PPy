"""Canonical IR -> LLVM IR.

The backend reads the IR and nothing else. Every IR function becomes an
LLVM function with the native ABI the runtime binds -- machine atoms in,
result slots out, an `i32` status back -- and every `core.guard`, every
Python-semantics overflow, every failing call takes one road: the
function's fallback block, which returns the status that sends the caller
to CPython.

Types map by structure: `bool` is `i1` in registers and `i8` at the
boundary, a buffer is a pointer and a length, a tuple or struct is an LLVM
aggregate that the boundary flattens. Block arguments become phis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ppy_runtime.abi import STATUS_FALLBACK, STATUS_OK

from ...ir import (
    Block,
    BoolType,
    BufferType,
    FloatType,
    IndexType,
    IntType,
    IRFunction,
    IRModule,
    IRType,
    Operation,
    PtrType,
    StructType,
    TupleType,
    Value,
    VectorType,
    VoidType,
)
from ...ir.dialects.gpu import kind_of
from ...target import TargetInfo, host_target
from .dialect_lowerings import EmitError as _DialectEmitError
from .dialect_lowerings import (
    lower_atomic,
    lower_concurrency,
    lower_cpu,
    lower_simd,
    lower_special,
)
from .lowering import FASTMATH_FLAGS, _default_triple

__all__ = ["EmitError", "emit_module"]

#: Backend-neutral intrinsic names the frontend emits, and LLVM's spelling.
_MATH = {
    "math.sqrt": "llvm.sqrt.f64",
    "math.sin": "llvm.sin.f64",
    "math.cos": "llvm.cos.f64",
    "math.exp": "llvm.exp.f64",
    "math.log": "llvm.log.f64",
    "math.log2": "llvm.log2.f64",
    "math.log10": "llvm.log10.f64",
    "math.fabs": "llvm.fabs.f64",
    "math.floor": "llvm.floor.f64",
    "math.ceil": "llvm.ceil.f64",
    "math.pow": "llvm.pow.f64",
    "math.trunc": "llvm.trunc.f64",
}


class EmitError(Exception):
    """IR the backend cannot lower; the verifier should have caught it."""


def emit_module(module: IRModule, target: TargetInfo | None = None) -> str:
    """The LLVM IR text of every defined function in `module`, for `target`.

    The text is target-neutral but for what a dialect makes of the machine
    -- a spin-wait hint is one instruction on x86 and another on arm --
    so a cross build lowers for its target, and the lowering cache keys
    on it.
    """
    from llvmlite import ir

    try:
        return str(_ModuleEmitter(ir, module, target or host_target()).run())
    except _DialectEmitError as error:
        raise EmitError(str(error)) from error


@dataclass(slots=True)
class _Buffer:
    data: object
    length: object


@dataclass(slots=True)
class _Phi:
    """A block argument's phi, filled in as its predecessors are emitted."""

    node: object
    incoming: list[tuple[object, object]] = field(default_factory=list)


class _ModuleEmitter:
    def __init__(self, ir, module: IRModule, target: TargetInfo) -> None:  # type: ignore[no-untyped-def]
        self.ir = ir
        self.module = module
        self.target = target
        self.llvm = ir.Module(name=module.name)
        self.llvm.triple = target.triple if not target.is_host else _default_triple()
        self._data_layout = None
        self.functions: dict[str, object] = {}
        self.strings: dict[str, object] = {}

    def run(self):  # type: ignore[no-untyped-def]
        ir = self.ir
        for name, item in self.module.globals.items():
            if isinstance(item.type, BufferType) and isinstance(item.value, str):
                data = bytearray(item.value.encode("utf-8"))
                array_type = ir.ArrayType(ir.IntType(8), len(data))
                variable = ir.GlobalVariable(self.llvm, array_type, name=name)
                variable.global_constant = True
                variable.linkage = "private"
                variable.initializer = ir.Constant(array_type, data)
                self.strings[name] = variable
        for function in self.module.functions.values():
            if function.is_declaration or kind_of(function) != "host":
                # Device code is the GPU backends'; the CPU never sees it.
                continue
            symbol = str(function.attributes.get("ppy.symbol", function.name))
            declared = ir.Function(self.llvm, self.function_type(function), name=symbol)
            declared.linkage = "external"
            features = function.attributes.get("cpu.features")
            if features:
                # llvmlite knows only the enum attributes; a string attribute
                # goes into the set behind its check, and prints as LLVM wants.
                spelled = ",".join(f"+{f}" for f in features)  # type: ignore[union-attr]
                set.add(declared.attributes, f'"target-features"="{spelled}"')
            self.functions[function.name] = declared
        for function in self.module.functions.values():
            if not function.is_declaration and kind_of(function) == "host":
                _FunctionEmitter(self, function).run()
        for function in self.module.functions.values():
            if not function.is_declaration and "ppy.export" in function.attributes:
                self._export(function)
        return self.llvm

    def _export(self, function: IRFunction) -> None:
        """The public C symbol: the internal function behind a C signature.

        A C caller has no Python to fall back to, so a failed guard is a
        trap, never a wrong answer.
        """
        ir = self.ir
        name = str(function.attributes["ppy.export"])
        atoms = [atom for _n, t in function.params for atom in self.boundary_atoms(t)]
        result = function.results[0] if function.results else None
        result_type = ir.VoidType() if result is None else self.boundary_atoms(result)[0]
        wrapper = ir.Function(self.llvm, ir.FunctionType(result_type, atoms), name=name)
        wrapper.linkage = "external"
        block = wrapper.append_basic_block("entry")
        builder = ir.IRBuilder(block)
        slot = builder.alloca(
            self.boundary_atoms(result)[0] if result is not None else ir.IntType(64)
        )
        status = builder.call(self.functions[function.name], [*wrapper.args, slot])
        ok = builder.icmp_signed("==", status, ir.Constant(ir.IntType(32), STATUS_OK))
        good = wrapper.append_basic_block("ok")
        bad = wrapper.append_basic_block("fail")
        builder.cbranch(ok, good, bad)
        builder.position_at_end(bad)
        trap = self.llvm.globals.get("llvm.trap") or ir.Function(
            self.llvm, ir.FunctionType(ir.VoidType(), []), name="llvm.trap"
        )
        builder.call(trap, [])
        builder.unreachable()
        builder.position_at_end(good)
        if result is None:
            builder.ret_void()
        else:
            builder.ret(builder.load(slot))

    @property
    def data_layout(self):  # type: ignore[no-untyped-def]
        """The target's data layout, for the sizes of what the IR allocates."""
        if self._data_layout is None:
            from llvmlite import binding

            self._data_layout = binding.create_target_data(self.target.data_layout)
        return self._data_layout

    # -- types -----------------------------------------------------------

    def llvm_type(self, t: IRType):  # type: ignore[no-untyped-def]
        ir = self.ir
        if isinstance(t, VectorType):
            return ir.VectorType(self.llvm_type(t.element), t.count)
        if isinstance(t, BoolType):
            return ir.IntType(1)
        if isinstance(t, IntType):
            return ir.IntType(t.width)
        if isinstance(t, IndexType):
            return ir.IntType(64)
        if isinstance(t, FloatType):
            return {16: ir.HalfType(), 32: ir.FloatType(), 64: ir.DoubleType()}[t.width]
        if isinstance(t, PtrType):
            return self.llvm_type(t.pointee).as_pointer()
        if isinstance(t, TupleType):
            return ir.LiteralStructType([self.llvm_type(item) for item in t.items])
        if isinstance(t, StructType):
            return ir.LiteralStructType([self.llvm_type(item) for _name, item in t.fields])
        if isinstance(t, VoidType):
            return ir.VoidType()
        raise EmitError(f"{t} has no LLVM representation")

    def boundary_atoms(self, t: IRType) -> list:  # type: ignore[type-arg]
        """The machine atoms a value of type `t` crosses the boundary as."""
        ir = self.ir
        if isinstance(t, BoolType):
            return [ir.IntType(8)]
        if isinstance(t, BufferType):
            return [self.llvm_type(t.element).as_pointer(), ir.IntType(64)]
        if isinstance(t, TupleType):
            return [atom for item in t.items for atom in self.boundary_atoms(item)]
        if isinstance(t, StructType):
            return [atom for _name, item in t.fields for atom in self.boundary_atoms(item)]
        return [self.llvm_type(t)]

    def function_type(self, function: IRFunction):  # type: ignore[no-untyped-def]
        ir = self.ir
        atoms = [atom for _name, t in function.params for atom in self.boundary_atoms(t)]
        outs = [atom.as_pointer() for t in function.results for atom in self.boundary_atoms(t)]
        if not outs:
            outs = [ir.IntType(64).as_pointer()]
        return ir.FunctionType(ir.IntType(32), [*atoms, *outs])


class _FunctionEmitter:
    def __init__(self, owner: _ModuleEmitter, function: IRFunction) -> None:
        self.owner = owner
        self.ir = owner.ir
        self.function = function
        self.llvm = owner.functions[function.name]
        self.values: dict[int, object] = {}
        self.blocks: dict[int, object] = {}
        self.phis: dict[int, list[_Phi]] = {}
        self.fastmath = list(FASTMATH_FLAGS) if function.attributes.get("fastmath") else []
        self.builder = None
        self.entry = None
        self.fallback = None
        self.outs: list = []
        self._slots = 0

    # -- helpers ----------------------------------------------------------

    def value(self, v: Value):  # type: ignore[no-untyped-def]
        found = self.values.get(id(v))
        if found is None:
            raise EmitError(f"value %{v.name or '?'} was never emitted")
        return found

    def set(self, v: Value, llvm_value) -> None:  # type: ignore[no-untyped-def]
        self.values[id(v)] = llvm_value

    def entry_alloca(self, llvm_type, name: str = ""):  # type: ignore[no-untyped-def]
        """A slot in the entry block, made once however deep the loop."""
        if self.builder.block is self.entry:
            return self.builder.alloca(llvm_type, name=name)
        builder = self.ir.IRBuilder(self.entry)
        if self.entry.is_terminated:
            builder.position_before(self.entry.instructions[-1])
        return builder.alloca(llvm_type, name=name)

    def fail_if(self, condition, label: str) -> None:  # type: ignore[no-untyped-def]
        keep = self.llvm.append_basic_block(label)
        self.builder.cbranch(condition, self.fallback, keep)
        self.builder.position_at_end(keep)

    def continue_if(self, condition, label: str) -> None:  # type: ignore[no-untyped-def]
        keep = self.llvm.append_basic_block(label)
        self.builder.cbranch(condition, keep, self.fallback)
        self.builder.position_at_end(keep)

    def intrinsic(self, name: str, result, arguments: list):  # type: ignore[no-untyped-def]
        existing = self.owner.llvm.globals.get(name)
        if existing is not None:
            return existing
        return self.ir.Function(self.owner.llvm, self.ir.FunctionType(result, arguments), name=name)

    def extern(self, name: str, result, arguments: list):  # type: ignore[no-untyped-def]
        return self.intrinsic(name, result, arguments)

    # -- the function -----------------------------------------------------

    def run(self) -> None:
        ir = self.ir
        body = self.function.body
        for block in body.blocks:
            self.blocks[id(block)] = self.llvm.append_basic_block(block.name)
        self.entry = self.blocks[id(body.blocks[0])]
        self.fallback = self.llvm.append_basic_block("fallback")
        self.builder = ir.IRBuilder(self.entry)
        self._bind_parameters()
        for block in body.blocks[1:]:
            self._make_phis(block)
        for block in body.blocks:
            self.builder.position_at_end(self.blocks[id(block)])
            for op in block.operations:
                self._op(op)
        with self.builder.goto_block(self.fallback):
            self.builder.ret(ir.Constant(ir.IntType(32), STATUS_FALLBACK))
        for phis in self.phis.values():
            for phi in phis:
                for value, block in phi.incoming:
                    phi.node.add_incoming(value, block)

    def _bind_parameters(self) -> None:
        entry = self.function.entry
        assert entry is not None
        position = 0
        args = self.llvm.args
        for argument in entry.arguments:
            value, position = self._from_atoms(argument.type, args, position)
            self.set(argument, value)
        self.outs = list(args[position:])

    def _from_atoms(self, t: IRType, args, position: int):  # type: ignore[no-untyped-def]
        """Rebuild a value of type `t` from the boundary atoms at `position`."""
        ir = self.ir
        if isinstance(t, BoolType):
            return self.builder.trunc(args[position], ir.IntType(1)), position + 1
        if isinstance(t, BufferType):
            return _Buffer(args[position], args[position + 1]), position + 2
        if isinstance(t, (TupleType, StructType)):
            items = t.items if isinstance(t, TupleType) else tuple(i for _n, i in t.fields)
            aggregate = ir.Constant(self.owner.llvm_type(t), ir.Undefined)
            for index, item in enumerate(items):
                value, position = self._from_atoms(item, args, position)
                aggregate = self.builder.insert_value(aggregate, value, index)
            return aggregate, position
        return args[position], position + 1

    def _to_atoms(self, t: IRType, value) -> list:  # type: ignore[no-untyped-def]
        """Flatten a value of type `t` into its boundary atoms."""
        ir = self.ir
        if isinstance(t, BoolType):
            return [self.builder.zext(value, ir.IntType(8))]
        if isinstance(t, BufferType):
            assert isinstance(value, _Buffer)
            return [value.data, value.length]
        if isinstance(t, (TupleType, StructType)):
            items = t.items if isinstance(t, TupleType) else tuple(i for _n, i in t.fields)
            atoms: list = []
            for index, item in enumerate(items):
                atoms.extend(self._to_atoms(item, self.builder.extract_value(value, index)))
            return atoms
        return [value]

    def _make_phis(self, block: Block) -> None:
        if not block.arguments:
            return
        builder = self.ir.IRBuilder(self.blocks[id(block)])
        phis: list[_Phi] = []
        for argument in block.arguments:
            node = builder.phi(self.owner.llvm_type(argument.type), name=argument.name or "")
            phis.append(_Phi(node))
            self.set(argument, node)
        self.phis[id(block)] = phis

    def _branch_arguments(self, successor) -> None:  # type: ignore[no-untyped-def]
        phis = self.phis.get(id(successor.block), [])
        for phi, value in zip(phis, successor.arguments, strict=True):
            phi.incoming.append((self.value(value), self.builder.block))

    # -- operations -------------------------------------------------------

    def _op(self, op: Operation) -> None:
        ir = self.ir
        b = self.builder
        name = op.local_name
        if op.dialect == "math":
            self._math(op)
            return
        lowering = _DIALECTS.get(op.dialect)
        if lowering is not None:
            lowering(self, op)
            return
        if op.dialect == "parallel":
            raise EmitError(
                f"{op.name} reached the LLVM backend: the `openmp` parallel backend is for "
                "`ppy emit c`; select `threads`, `serial`, or `simd` for a native build"
            )
        if op.dialect != "core":
            raise EmitError(
                f"{op.name}: the LLVM backend has no lowering for the {op.dialect} dialect"
            )
        match name:
            case "const":
                self.set(op.result, self._constant(op.result.type, op.attributes["value"]))
            case "add" | "sub" | "mul":
                self._arith(op, name)
            case "div" | "mod":
                self._divmod(op, name)
            case "neg":
                self._neg(op)
            case "and" | "or" | "xor":
                left, right = (self.value(v) for v in op.operands)
                emit = {"and": b.and_, "or": b.or_, "xor": b.xor}[name]
                self.set(op.result, emit(left, right))
            case "shl" | "shr":
                self._shift(op, name)
            case "cmp":
                self._cmp(op)
            case "select":
                condition, a, c = (self.value(v) for v in op.operands)
                self.set(op.result, b.select(condition, a, c))
            case "cast":
                self._cast(op)
            case "br":
                self._branch_arguments(op.successors[0])
                b.branch(self.blocks[id(op.successors[0].block)])
            case "cond_br":
                for successor in op.successors:
                    self._branch_arguments(successor)
                b.cbranch(
                    self.value(op.operands[0]),
                    self.blocks[id(op.successors[0].block)],
                    self.blocks[id(op.successors[1].block)],
                )
            case "ret":
                self._ret(op)
            case "unreachable":
                b.unreachable()
            case "alloca":
                pointer = op.result.type
                assert isinstance(pointer, PtrType)
                count = op.attributes.get("count", 1)
                llvm_type = self.owner.llvm_type(pointer.pointee)
                if count == 1:
                    self.set(op.result, self.entry_alloca(llvm_type, op.result.name or ""))
                else:
                    slot = self.entry_alloca(ir.ArrayType(llvm_type, int(count)))
                    zero = ir.Constant(ir.IntType(32), 0)
                    self.set(op.result, b.gep(slot, [zero, zero]))
            case "load":
                self.set(op.result, b.load(self.value(op.operands[0])))
            case "store":
                b.store(self.value(op.operands[0]), self.value(op.operands[1]))
            case "ptr_offset":
                pointer, offset = (self.value(v) for v in op.operands)
                self.set(op.result, b.gep(pointer, [offset]))
            case "buffer_data":
                self.set(op.result, self.value(op.operands[0]).data)
            case "buffer_len":
                self.set(op.result, self.value(op.operands[0]).length)
            case "buffer_load":
                buffer, index = self.value(op.operands[0]), self.value(op.operands[1])
                self.set(op.result, b.load(b.gep(buffer.data, [index])))
            case "buffer_store":
                value, buffer, index = (self.value(v) for v in op.operands)
                b.store(value, b.gep(buffer.data, [index]))
            case "tuple_make" | "struct_make":
                aggregate = ir.Constant(self.owner.llvm_type(op.result.type), ir.Undefined)
                for index, operand in enumerate(op.operands):
                    aggregate = b.insert_value(aggregate, self.value(operand), index)
                self.set(op.result, aggregate)
            case "tuple_extract":
                index = int(op.attributes["index"])
                self.set(op.result, b.extract_value(self.value(op.operands[0]), index))
            case "struct_extract":
                struct = op.operands[0].type
                assert isinstance(struct, StructType)
                index = struct.field_index(str(op.attributes["field"]))
                assert index is not None
                self.set(op.result, b.extract_value(self.value(op.operands[0]), index))
            case "call":
                self._call(op)
            case "call_extern":
                self._call_extern(op)
            case "call_intrinsic":
                self._call_intrinsic(op)
            case "guard":
                label = str(op.attributes.get("label") or f"{op.attributes['kind']}.ok")
                self.continue_if(self.value(op.operands[0]), label)
            case _:
                raise EmitError(f"{op.name} has no LLVM lowering")

    def _math(self, op: Operation) -> None:
        """A math operation is the LLVM intrinsic of that name over its type."""
        b = self.builder
        t = op.results[0].type
        llvm_type = self.owner.llvm_type(t)
        width = t.element.width if isinstance(t, VectorType) else t.width  # type: ignore[union-attr]
        suffix = {16: "f16", 32: "f32", 64: "f64"}[width]
        if isinstance(t, VectorType):
            suffix = f"v{t.count}{suffix}"
        local = "fabs" if op.local_name == "abs" else op.local_name
        arguments = [self.value(v) for v in op.operands]
        function = self.intrinsic(f"llvm.{local}.{suffix}", llvm_type, [llvm_type] * len(arguments))
        self.set(op.results[0], b.call(function, arguments))

    def _constant(self, t: IRType, value):  # type: ignore[no-untyped-def]
        ir = self.ir
        llvm_type = self.owner.llvm_type(t)
        if isinstance(t, BoolType):
            return ir.Constant(llvm_type, int(bool(value)))
        if isinstance(t, FloatType):
            return ir.Constant(llvm_type, float(value))
        return ir.Constant(llvm_type, int(value))

    def _fp(self, emit, left, right):  # type: ignore[no-untyped-def]
        return emit(left, right, flags=self.fastmath) if self.fastmath else emit(left, right)

    def _arith(self, op: Operation, name: str) -> None:
        b = self.builder
        left, right = (self.value(v) for v in op.operands)
        t = op.result.type
        if isinstance(t.element if isinstance(t, VectorType) else t, FloatType):
            emit = {"add": b.fadd, "sub": b.fsub, "mul": b.fmul}[name]
            self.set(op.result, self._fp(emit, left, right))
            return
        overflow = op.attributes.get("overflow", "python")
        emit = {"add": b.add, "sub": b.sub, "mul": b.mul}[name]
        if overflow == "wrap":
            self.set(op.result, emit(left, right))
            return
        if isinstance(t, VectorType) and overflow != "proven":
            raise EmitError(f"vector {name} carries `wrap` or `proven`, not `{overflow}`")
        if overflow == "proven":
            # The proof says no signed wrap; saying so keeps SCEV able to
            # fold the inductions this value feeds.
            scalar = t.element if isinstance(t, VectorType) else t
            signed = scalar.signed if isinstance(scalar, IntType) else True
            self.set(op.result, emit(left, right, flags=("nsw",) if signed else ("nuw",)))
            return
        self.set(op.result, self._checked(name, left, right, t))

    def _checked(self, name: str, left, right, t: IRType):  # type: ignore[no-untyped-def]
        """Machine arithmetic that takes the fallback on overflow."""
        ir = self.ir
        width = t.width if isinstance(t, IntType) else 64
        signed = t.signed if isinstance(t, IntType) else True
        letter = "s" if signed else "u"
        intrinsic = f"llvm.{letter}{name}.with.overflow.i{width}"
        word = ir.IntType(width)
        function = self.intrinsic(
            intrinsic, ir.LiteralStructType([word, ir.IntType(1)]), [word, word]
        )
        packed = self.builder.call(function, [left, right])
        result = self.builder.extract_value(packed, 0)
        self.fail_if(self.builder.extract_value(packed, 1), "arith.ok")
        return result

    def _neg(self, op: Operation) -> None:
        b = self.builder
        operand = self.value(op.operands[0])
        t = op.result.type
        if isinstance(t.element if isinstance(t, VectorType) else t, FloatType):
            self.set(op.result, b.fneg(operand))
            return
        zero = self.ir.Constant(self.owner.llvm_type(t), None if isinstance(t, VectorType) else 0)
        if op.attributes.get("overflow", "python") == "wrap":
            self.set(op.result, b.sub(zero, operand))
            return
        if isinstance(t, VectorType):
            raise EmitError("vector neg carries `wrap`")
        self.set(op.result, self._checked("sub", zero, operand, t))

    def _divmod(self, op: Operation, name: str) -> None:
        ir = self.ir
        b = self.builder
        left, right = (self.value(v) for v in op.operands)
        t = op.result.type
        if isinstance(t, VectorType):
            if isinstance(t.element, FloatType) and name == "div":
                self.set(op.result, self._fp(b.fdiv, left, right))
                return
            raise EmitError(f"vector {name} over {t.element} has no LLVM lowering")
        if isinstance(t, FloatType):
            if name == "mod":
                raise EmitError("float remainder has no LLVM lowering with Python semantics")
            self.set(op.result, self._fp(b.fdiv, left, right))
            return
        signed = t.signed if isinstance(t, IntType) else True
        word = self.owner.llvm_type(t)
        rounding = op.attributes.get("rounding", "floor")
        if not signed:
            self.set(op.result, b.udiv(left, right) if name == "div" else b.urem(left, right))
            return
        if op.attributes.get("overflow", "python") != "wrap":
            minimum = ir.Constant(word, -(1 << (word.width - 1)))
            minus_one = ir.Constant(word, -1)
            overflows = b.and_(
                b.icmp_signed("==", left, minimum), b.icmp_signed("==", right, minus_one)
            )
            self.fail_if(overflows, "div.ok")
        if rounding == "trunc":
            self.set(op.result, b.sdiv(left, right) if name == "div" else b.srem(left, right))
            return
        power = _power_of_two(right)
        if power is not None:
            if name == "div":
                self.set(op.result, b.ashr(left, ir.Constant(word, power)))
            else:
                self.set(op.result, b.and_(left, ir.Constant(word, (1 << power) - 1)))
            return
        zero = ir.Constant(word, 0)
        quotient = b.sdiv(left, right)
        remainder = b.srem(left, right)
        nonzero = b.icmp_signed("!=", remainder, zero)
        signs_differ = b.xor(b.icmp_signed("<", left, zero), b.icmp_signed("<", right, zero))
        adjust = b.and_(nonzero, signs_differ)
        if name == "div":
            self.set(op.result, b.select(adjust, b.sub(quotient, ir.Constant(word, 1)), quotient))
        else:
            self.set(op.result, b.select(adjust, b.add(remainder, right), remainder))

    def _shift(self, op: Operation, name: str) -> None:
        b = self.builder
        left, right = (self.value(v) for v in op.operands)
        t = op.result.type
        signed = t.signed if isinstance(t, IntType) else True
        if name == "shr":
            self.set(op.result, b.ashr(left, right) if signed else b.lshr(left, right))
            return
        shifted = b.shl(left, right)
        if op.attributes.get("overflow", "wrap") != "wrap":
            restored = b.ashr(shifted, right) if signed else b.lshr(shifted, right)
            self.fail_if(b.icmp_signed("!=", restored, left), "shl.ok")
        self.set(op.result, shifted)

    def _cmp(self, op: Operation) -> None:
        b = self.builder
        left, right = (self.value(v) for v in op.operands)
        t = op.operands[0].type
        if isinstance(t, VectorType):
            t = t.element
        predicate = str(op.attributes["predicate"])
        symbol = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}[predicate]
        if isinstance(t, FloatType):
            self.set(op.result, b.fcmp_ordered(symbol, left, right))
        elif isinstance(t, IntType) and not t.signed:
            self.set(op.result, b.icmp_unsigned(symbol, left, right))
        else:
            self.set(op.result, b.icmp_signed(symbol, left, right))

    def _cast(self, op: Operation) -> None:
        ir = self.ir
        b = self.builder
        value = self.value(op.operands[0])
        source = op.operands[0].type
        target = op.result.type
        llvm_target = self.owner.llvm_type(target)
        if isinstance(source, VectorType) and isinstance(target, VectorType):
            # The lanes convert as the scalars would; the builder takes vectors.
            source, target = source.element, target.element
        if isinstance(source, PtrType) and isinstance(target, PtrType):
            self.set(op.result, b.bitcast(value, llvm_target))
            return
        if isinstance(source, BoolType):
            if isinstance(target, FloatType):
                self.set(op.result, b.uitofp(value, llvm_target))
            else:
                self.set(op.result, b.zext(value, llvm_target))
            return
        if isinstance(target, BoolType):
            zero = None if isinstance(value.type, ir.VectorType) else 0
            if isinstance(source, FloatType):
                self.set(op.result, b.fcmp_ordered("!=", value, ir.Constant(value.type, zero)))
            else:
                self.set(op.result, b.icmp_signed("!=", value, ir.Constant(value.type, zero)))
            return
        source_int = isinstance(source, (IntType, IndexType))
        target_int = isinstance(target, (IntType, IndexType))
        if source_int and target_int:
            from_width = source.width if isinstance(source, IntType) else 64
            to_width = target.width if isinstance(target, IntType) else 64
            signed = source.signed if isinstance(source, IntType) else True
            if from_width == to_width:
                self.set(op.result, value)
            elif from_width < to_width:
                self.set(
                    op.result, b.sext(value, llvm_target) if signed else b.zext(value, llvm_target)
                )
            else:
                self.set(op.result, b.trunc(value, llvm_target))
            return
        if source_int and isinstance(target, FloatType):
            signed = source.signed if isinstance(source, IntType) else True
            self.set(
                op.result, b.sitofp(value, llvm_target) if signed else b.uitofp(value, llvm_target)
            )
            return
        if isinstance(source, FloatType) and target_int:
            signed = target.signed if isinstance(target, IntType) else True
            self.set(
                op.result, b.fptosi(value, llvm_target) if signed else b.fptoui(value, llvm_target)
            )
            return
        if isinstance(source, FloatType) and isinstance(target, FloatType):
            if source.width < target.width:
                self.set(op.result, b.fpext(value, llvm_target))
            elif source.width > target.width:
                self.set(op.result, b.fptrunc(value, llvm_target))
            else:
                self.set(op.result, value)
            return
        raise EmitError(f"no LLVM cast from {source} to {target}")

    def _ret(self, op: Operation) -> None:
        ir = self.ir
        b = self.builder
        if not op.operands:
            b.store(ir.Constant(ir.IntType(64), 0), self.outs[0])
        else:
            position = 0
            for value, t in zip(op.operands, self.function.results, strict=True):
                for atom in self._to_atoms(t, self.value(value)):
                    b.store(atom, self.outs[position])
                    position += 1
        b.ret(ir.Constant(ir.IntType(32), STATUS_OK))

    def _call(self, op: Operation) -> None:
        ir = self.ir
        b = self.builder
        callee_name = op.attributes["callee"].name  # type: ignore[union-attr]
        callee = self.owner.functions.get(callee_name)
        target = self.owner.module.functions.get(callee_name)
        if callee is None or target is None:
            raise EmitError(f"call to @{callee_name}, which was not emitted")
        arguments: list = []
        for operand, (_name, t) in zip(op.operands, target.params, strict=True):
            arguments.extend(self._to_atoms(t, self.value(operand)))
        slots: list[tuple[IRType, list]] = []
        for t in target.results:
            atoms = [self.entry_alloca(atom, "callresult") for atom in self.owner.boundary_atoms(t)]
            slots.append((t, atoms))
            arguments.extend(atoms)
        if not target.results:
            arguments.append(self.entry_alloca(ir.IntType(64), "callvoid"))
        status = b.call(callee, arguments)
        results = list(op.results)
        if op.attributes.get("capture_status"):
            self.set(results.pop(), b.zext(status, ir.IntType(64)))
        else:
            ok = b.icmp_signed("==", status, ir.Constant(ir.IntType(32), STATUS_OK))
            self.continue_if(ok, "call.ok")
        for result, (t, atoms) in zip(results, slots, strict=True):
            loaded = [b.load(slot) for slot in atoms]
            self.set(result, self._from_atoms(t, loaded, 0)[0])

    def _call_extern(self, op: Operation) -> None:
        ir = self.ir
        b = self.builder
        symbol = str(op.attributes["callee"])
        arguments = [self._extern_atom(v) for v in op.operands]
        result_type = ir.VoidType() if not op.results else self.owner.llvm_type(op.results[0].type)
        if op.results and isinstance(op.results[0].type, BoolType):
            result_type = ir.IntType(8)
        function = self.extern(symbol, result_type, [a.type for a in arguments])
        result = b.call(function, arguments)
        if op.results:
            if isinstance(op.results[0].type, BoolType):
                result = b.trunc(result, ir.IntType(1))
            self.set(op.results[0], result)

    def _extern_atom(self, value: Value):  # type: ignore[no-untyped-def]
        """How a value crosses into C: a bool is a byte, a buffer its pointer."""
        emitted = self.value(value)
        if isinstance(value.type, BoolType):
            return self.builder.zext(emitted, self.ir.IntType(8))
        if isinstance(emitted, _Buffer):
            return emitted.data
        return emitted

    def _call_intrinsic(self, op: Operation) -> None:
        ir = self.ir
        b = self.builder
        name = str(op.attributes["intrinsic"])
        if name in _MATH:
            double = ir.DoubleType()
            arguments = [self.value(v) for v in op.operands]
            function = self.intrinsic(_MATH[name], double, [double] * len(arguments))
            self.set(op.results[0], b.call(function, arguments))
            return
        if name == "ppy.buffer_from_parts":
            data, length = (self.value(v) for v in op.operands)
            self.set(op.results[0], _Buffer(data, length))
            return
        if name == "ppy.string_data":
            variable = self.owner.strings[str(op.attributes["symbol"])]
            self.set(op.results[0], b.bitcast(variable, ir.IntType(8).as_pointer()))
            return
        if name in {"llvm.smin.i64", "llvm.smax.i64"}:
            word = ir.IntType(64)
            function = self.intrinsic(name, word, [word, word])
            self.set(op.results[0], b.call(function, [self.value(v) for v in op.operands]))
            return
        if name.startswith("ppy.checked_"):
            operation = name.removeprefix("ppy.checked_")
            t = op.operands[0].type
            width = t.width if isinstance(t, IntType) else 64
            signed = t.signed if isinstance(t, IntType) else True
            word = ir.IntType(width)
            function = self.intrinsic(
                f"llvm.{'s' if signed else 'u'}{operation}.with.overflow.i{width}",
                ir.LiteralStructType([word, ir.IntType(1)]),
                [word, word],
            )
            packed = b.call(function, [self.value(v) for v in op.operands])
            self.set(op.results[0], b.extract_value(packed, 0))
            self.set(op.results[1], b.extract_value(packed, 1))
            return
        raise EmitError(f"intrinsic {name!r} has no LLVM lowering")


#: The dialects beyond core and math, each lowered by its own module.
_DIALECTS = {
    "simd": lower_simd,
    "cpu": lower_cpu,
    "atomic": lower_atomic,
    "concurrency": lower_concurrency,
    "special": lower_special,
}


def _power_of_two(value) -> int | None:  # type: ignore[no-untyped-def]
    constant = getattr(value, "constant", None)
    if not isinstance(constant, int) or constant <= 0 or constant & (constant - 1):
        return None
    return constant.bit_length() - 1
