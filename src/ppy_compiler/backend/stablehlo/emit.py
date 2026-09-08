"""The StableHLO backend: the canonical and tensor IR as an MLIR module XLA compiles.

A function of scalars and tensors -- one block, no memory, static shapes --
becomes a `func.func` over `tensor<...>` values: a scalar is a rank-0
tensor, `core` arithmetic and `math` are the StableHLO elementwise
operations, `select` and `compare` are theirs, and the tensor dialect maps
one to one: `fill` and `broadcast` are `broadcast_in_dim`, `reshape`,
`transpose`, `reduce`, `dot_general` for `matmul` and `linalg.dot`,
`convert`, and a `fused` region is its body over the broadcast shape --
XLA fuses again on its own terms. What XLA cannot take -- a buffer, a
guard, a branch, a symbolic shape, an operation without a rule -- is
refused with the reason, and the function stays where it was (spec 62, 63).
"""

from __future__ import annotations

import math
import struct

from ...ir import shape as shapes
from ...ir.dialects import tensor as tensors
from ...ir.model import Block, IRFunction, IRModule, Operation, Value
from ...ir.types import BoolType, FloatType, IntType, IRType

__all__ = ["StableHloError", "emit_module", "prepare", "supports"]

_LN2 = math.log(2.0)
_LN10 = math.log(10.0)

_BINARY = {
    "add": "add",
    "sub": "subtract",
    "mul": "multiply",
    "div": "divide",
    "pow": "power",
    "min": "minimum",
    "max": "maximum",
}
_UNARY = {
    "sin": "sine",
    "cos": "cosine",
    "tan": "tan",
    "exp": "exponential",
    "log": "log",
    "sqrt": "sqrt",
    "abs": "abs",
    "floor": "floor",
    "ceil": "ceil",
    "neg": "negate",
}
_COMPARE = {"eq": "EQ", "ne": "NE", "lt": "LT", "le": "LE", "gt": "GT", "ge": "GE"}


class StableHloError(ValueError):
    """IR the StableHLO backend cannot express, with the reason."""


def supports(function: IRFunction) -> str | None:
    """Why `function` cannot become StableHLO, or None when it can."""
    try:
        _Emitter(function).run()
    except StableHloError as error:
        return str(error)
    return None


def emit_module(
    module: IRModule, functions: tuple[str, ...] | None = None, *, entry: str | None = None
) -> str:
    """The MLIR text of `module`'s functions -- those named, or all defined ones.

    The function named `entry` is written as `@main`, the one PJRT runs.
    """
    chosen = [
        f
        for name, f in module.functions.items()
        if not f.is_declaration and (functions is None or name in functions)
    ]
    body = "\n".join(_Emitter(f, "main" if f.name == entry else None).run() for f in chosen)
    return f"module @{_symbol(module.name)} {{\n{body}\n}}\n"


def prepare(module: IRModule) -> None:
    """The passes before emission: slots to values, folding, fusion -- no lowering to memory.

    Verified first, as the frontend's IR is before every backend.
    """
    from ...ir import PassContext, PassManager, verify_or_raise
    from ...ir.transforms import (
        Canonicalize,
        DeadCodeElimination,
        FuseTensor,
        PromoteSlots,
        SimplifyCFG,
        TensorCanonicalize,
    )

    verify_or_raise(module)
    manager = PassManager(PassContext())
    for step in (
        PromoteSlots(),
        Canonicalize(),
        SimplifyCFG(),
        DeadCodeElimination(),
        Canonicalize(),
        TensorCanonicalize(),
        FuseTensor(),
        DeadCodeElimination(),
    ):
        manager.add(step)
    manager.run(module)


def _symbol(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


class _Emitter:
    def __init__(self, function: IRFunction, symbol: str | None = None) -> None:
        self.function = function
        self.symbol = symbol or function.name
        self.names: dict[int, str] = {}
        self.shapes: dict[int, tuple[shapes.Shape, IRType]] = {}
        self.lines: list[str] = []
        self.counter = 0

    # -- types ----------------------------------------------------------------------------

    def element(self, t: IRType) -> str:
        if isinstance(t, FloatType):
            return f"f{t.width}"
        if isinstance(t, BoolType):
            return "i1"
        if isinstance(t, IntType):
            return f"i{t.width}"
        raise StableHloError(f"@{self.function.name}: {t} is not an element XLA holds")

    def described(self, t: IRType) -> tuple[shapes.Shape, IRType]:
        """A value's shape and element type; a scalar is a rank-0 tensor."""
        info = tensors.describe(t)
        if info is None:
            if isinstance(t, (FloatType, IntType, BoolType)):
                return (), t
            raise StableHloError(f"@{self.function.name}: {t} is not a value XLA holds")
        if not info.static:
            raise StableHloError(
                f"@{self.function.name}: {t} has a symbolic shape; XLA compiles static shapes"
            )
        return info.shape, info.dtype

    def tensor(self, shape: shapes.Shape, dtype: IRType) -> str:
        dims = "".join(f"{int(d)}x" for d in shape)
        return f"tensor<{dims}{self.element(dtype)}>"

    def type_of(self, v: Value) -> str:
        shape, dtype = self.shapes[id(v)]
        return self.tensor(shape, dtype)

    # -- values ---------------------------------------------------------------------------

    def fresh(self) -> str:
        self.counter += 1
        return f"%{self.counter}"

    def define(self, v: Value, shape: shapes.Shape, dtype: IRType, text: str) -> str:
        name = self.fresh()
        self.names[id(v)] = name
        self.shapes[id(v)] = (shape, dtype)
        self.lines.append(f"    {name} = {text}")
        return name

    def emit(self, shape: shapes.Shape, dtype: IRType, text: str) -> str:
        """A value the IR has no name for: an intermediate of one lowering."""
        name = self.fresh()
        self.lines.append(f"    {name} = {text}")
        return name

    def name(self, v: Value) -> str:
        found = self.names.get(id(v))
        if found is None:
            raise StableHloError(f"@{self.function.name}: %{v.name or '?'} was never emitted")
        return found

    def shape_of(self, v: Value) -> tuple[shapes.Shape, IRType]:
        return self.shapes[id(v)]

    # -- the function ---------------------------------------------------------------------

    def run(self) -> str:
        body = self.function.body
        if len(body.blocks) != 1:
            raise StableHloError(
                f"@{self.function.name} has control flow; XLA takes one block of values"
            )
        block = body.blocks[0]
        params = []
        for argument, (pname, t) in zip(block.arguments, self.function.params, strict=True):
            shape, dtype = self.described(t)
            self.names[id(argument)] = f"%{pname}"
            self.shapes[id(argument)] = (shape, dtype)
            params.append(f"%{pname}: {self.tensor(shape, dtype)}")
        results = ", ".join(self.tensor(*self.described(t)) for t in self.function.results)
        returned = None
        for op in block.operations:
            if op.name == "core.ret":
                returned = op
                break
            self.op(op)
        if returned is None:
            raise StableHloError(f"@{self.function.name} does not return")
        values = ", ".join(self.name(v) for v in returned.operands)
        types = ", ".join(self.type_of(v) for v in returned.operands)
        signature = f"({', '.join(params)}) -> ({results})"
        header = f"  func.func public @{_symbol(self.symbol)}{signature} {{"
        tail = f"    return {values} : {types}" if values else "    return"
        return "\n".join([header, *self.lines, tail, "  }"])

    # -- operations -----------------------------------------------------------------------

    def op(self, op: Operation) -> None:
        handler = getattr(self, f"op_{op.dialect}_{op.local_name}", None)
        if handler is None:
            raise StableHloError(f"@{self.function.name}: {op.name} has no StableHLO lowering")
        handler(op)

    def constant(self, value: object, dtype: IRType, shape: shapes.Shape = ()) -> str:
        if isinstance(dtype, BoolType):
            literal = "true" if value else "false"
        elif isinstance(dtype, FloatType):
            literal = _float_literal(float(value), dtype.width)  # type: ignore[arg-type]
        else:
            literal = str(int(value))  # type: ignore[call-overload]
        scalar = self.emit(
            (), dtype, f"stablehlo.constant dense<{literal}> : {self.tensor((), dtype)}"
        )
        if not shape:
            return scalar
        return self.broadcast(scalar, (), shape, dtype)

    def broadcast(
        self, name: str, source: shapes.Shape, target: shapes.Shape, dtype: IRType
    ) -> str:
        """`name`, of shape `source`, as a value of shape `target` (NumPy's rule)."""
        if source == target:
            return name
        offset = len(target) - len(source)
        dims = ", ".join(str(axis + offset) for axis in range(len(source)))
        return self.emit(
            target,
            dtype,
            f"stablehlo.broadcast_in_dim {name}, dims = [{dims}] : "
            f"({self.tensor(source, dtype)}) -> {self.tensor(target, dtype)}",
        )

    def operands_over(self, op: Operation, shape: shapes.Shape) -> list[str]:
        """Each operand broadcast to `shape`."""
        out = []
        for operand in op.operands:
            source, dtype = self.shape_of(operand)
            out.append(self.broadcast(self.name(operand), source, shape, dtype))
        return out

    def joined(self, op: Operation) -> tuple[shapes.Shape, IRType]:
        """The broadcast shape of `op`'s operands and its result's element type.

        Inside a fused body the values are scalars in the IR and full tensors
        here, so the shape is the operands', never the result type's.
        """
        shape: shapes.Shape = ()
        for operand in op.operands:
            shape = shapes.broadcast(shape, self.shape_of(operand)[0])
        return shape, self.described(op.results[0].type)[1]

    def elementwise(self, op: Operation, mnemonic: str) -> None:
        shape, dtype = self.joined(op)
        names = self.operands_over(op, shape)
        self.define(
            op.results[0],
            shape,
            dtype,
            f"stablehlo.{mnemonic} {', '.join(names)} : {self.tensor(shape, dtype)}",
        )

    def unary_math(
        self, op: Operation, kind: str, x: str, shape: shapes.Shape, dtype: IRType
    ) -> str:
        t = self.tensor(shape, dtype)
        if kind in _UNARY:
            return f"stablehlo.{_UNARY[kind]} {x} : {t}"
        if kind == "exp2":
            scaled = self.emit(
                shape, dtype, f"stablehlo.multiply {x}, {self.constant(_LN2, dtype, shape)} : {t}"
            )
            return f"stablehlo.exponential {scaled} : {t}"
        if kind in {"log2", "log10"}:
            logged = self.emit(shape, dtype, f"stablehlo.log {x} : {t}")
            divisor = self.constant(_LN2 if kind == "log2" else _LN10, dtype, shape)
            return f"stablehlo.divide {logged}, {divisor} : {t}"
        if kind == "trunc":
            zero = self.constant(0.0, dtype, shape)
            flags = self.tensor(shape, BoolType())
            negative = self.emit(
                shape,
                BoolType(),
                f"stablehlo.compare LT, {x}, {zero}, FLOAT : ({t}, {t}) -> {flags}",
            )
            up = self.emit(shape, dtype, f"stablehlo.ceil {x} : {t}")
            down = self.emit(shape, dtype, f"stablehlo.floor {x} : {t}")
            return (
                f"stablehlo.select {negative}, {up}, {down} : {self.tensor(shape, BoolType())}, {t}"
            )
        if kind == "erf":
            return f"chlo.erf {x} : {t} -> {t}"
        if kind == "erfc":
            erf = self.emit(shape, dtype, f"chlo.erf {x} : {t} -> {t}")
            return f"stablehlo.subtract {self.constant(1.0, dtype, shape)}, {erf} : {t}"
        raise StableHloError(f"@{self.function.name}: {op.name} {kind} has no StableHLO lowering")

    # core

    def op_core_const(self, op: Operation) -> None:
        dtype = op.results[0].type
        literal_name = self.constant(op.attributes["value"], dtype)
        self.names[id(op.results[0])] = literal_name
        self.shapes[id(op.results[0])] = ((), dtype)

    def op_core_neg(self, op: Operation) -> None:
        self.elementwise(op, "negate")

    def op_core_cmp(self, op: Operation) -> None:
        shape, _ = self.joined(op)
        dtype = self.shape_of(op.operands[0])[1]
        names = self.operands_over(op, shape)
        kind = (
            "FLOAT"
            if isinstance(dtype, FloatType)
            else ("UNSIGNED" if isinstance(dtype, BoolType) else "SIGNED")
        )
        t = self.tensor(shape, dtype)
        direction = _COMPARE[str(op.attributes["predicate"])]
        flags = self.tensor(shape, BoolType())
        self.define(
            op.results[0],
            shape,
            BoolType(),
            f"stablehlo.compare {direction}, {names[0]}, {names[1]}, {kind} : "
            f"({t}, {t}) -> {flags}",
        )

    def op_core_select(self, op: Operation) -> None:
        shape, dtype = self.joined(op)
        names = self.operands_over(op, shape)
        flags, values = self.tensor(shape, BoolType()), self.tensor(shape, dtype)
        self.define(
            op.results[0],
            shape,
            dtype,
            f"stablehlo.select {', '.join(names)} : {flags}, {values}",
        )

    def op_core_cast(self, op: Operation) -> None:
        shape, source = self.shape_of(op.operands[0])
        target = op.results[0].type
        signature = f"({self.tensor(shape, source)}) -> {self.tensor(shape, target)}"
        self.define(
            op.results[0],
            shape,
            target,
            f"stablehlo.convert {self.name(op.operands[0])} : {signature}",
        )

    def op_core_guard(self, op: Operation) -> None:
        raise StableHloError(
            f"@{self.function.name}: a guard ({op.attributes.get('kind')}) has no XLA form; "
            "XLA computes, it does not fall back"
        )

    # math and special

    def op_math_pow(self, op: Operation) -> None:
        self.elementwise(op, "power")

    # tensor

    def op_tensor_fill(self, op: Operation) -> None:
        shape, dtype = self.described(op.results[0].type)
        name = self.broadcast(self.name(op.operands[0]), (), shape, dtype)
        self.names[id(op.results[0])] = name
        self.shapes[id(op.results[0])] = (shape, dtype)

    def op_tensor_unary(self, op: Operation) -> None:
        shape, dtype = self.described(op.results[0].type)
        kind = str(op.attributes["op"])
        self.define(
            op.results[0],
            shape,
            dtype,
            self.unary_math(op, kind, self.name(op.operands[0]), shape, dtype),
        )

    def op_tensor_broadcast(self, op: Operation) -> None:
        shape, dtype = self.described(op.results[0].type)
        source, _ = self.shape_of(op.operands[0])
        name = self.broadcast(self.name(op.operands[0]), source, shape, dtype)
        self.names[id(op.results[0])] = name
        self.shapes[id(op.results[0])] = (shape, dtype)

    def op_tensor_reshape(self, op: Operation) -> None:
        shape, dtype = self.described(op.results[0].type)
        source, _ = self.shape_of(op.operands[0])
        signature = f"({self.tensor(source, dtype)}) -> {self.tensor(shape, dtype)}"
        self.define(
            op.results[0],
            shape,
            dtype,
            f"stablehlo.reshape {self.name(op.operands[0])} : {signature}",
        )

    def op_tensor_transpose(self, op: Operation) -> None:
        shape, dtype = self.described(op.results[0].type)
        source, _ = self.shape_of(op.operands[0])
        perm = ", ".join(str(int(p)) for p in op.attributes["perm"])  # type: ignore[union-attr]
        self.define(
            op.results[0],
            shape,
            dtype,
            f"stablehlo.transpose {self.name(op.operands[0])}, dims = [{perm}] : "
            f"({self.tensor(source, dtype)}) -> {self.tensor(shape, dtype)}",
        )

    def reduce(
        self,
        source_name: str,
        source: shapes.Shape,
        dtype: IRType,
        kind: str,
        axes: tuple[int, ...],
        keepdims: bool,
        target: Value,
    ) -> None:
        identity = {"add": 0.0, "mul": 1.0, "min": math.inf, "max": -math.inf}[kind]
        if not isinstance(dtype, FloatType):
            identity = {"add": 0, "mul": 1, "min": (1 << 62), "max": -(1 << 62)}[kind]
        init = self.constant(identity, dtype)
        reduced_shape = tuple(d for axis, d in enumerate(source) if axis not in axes)
        mnemonic = {"add": "add", "mul": "multiply", "min": "minimum", "max": "maximum"}[kind]
        dims = ", ".join(str(a) for a in axes)
        signature = (
            f"({self.tensor(source, dtype)}, {self.tensor((), dtype)}) -> "
            f"{self.tensor(reduced_shape, dtype)}"
        )
        text = (
            f"stablehlo.reduce({source_name} init: {init}) applies stablehlo.{mnemonic} "
            f"across dimensions = [{dims}] : {signature}"
        )
        if not keepdims:
            self.define(target, reduced_shape, dtype, text)
            return
        reduced = self.emit(reduced_shape, dtype, text)
        kept = tuple(1 if axis in axes else d for axis, d in enumerate(source))
        back = f"({self.tensor(reduced_shape, dtype)}) -> {self.tensor(kept, dtype)}"
        self.define(target, kept, dtype, f"stablehlo.reshape {reduced} : {back}")

    def op_tensor_reduce(self, op: Operation) -> None:
        source, dtype = self.shape_of(op.operands[0])
        axes = tuple(int(a) for a in op.attributes["axes"])  # type: ignore[union-attr]
        self.reduce(
            self.name(op.operands[0]),
            source,
            dtype,
            str(op.attributes["op"]),
            axes,
            bool(op.attributes.get("keepdims", False)),
            op.results[0],
        )

    def dot(self, op: Operation, lhs_contract: int, rhs_contract: int) -> None:
        shape, dtype = self.described(op.results[0].type)
        a, c = op.operands
        left, _ = self.shape_of(a)
        right, _ = self.shape_of(c)
        numbers = (
            f"#stablehlo.dot<lhs_contracting_dimensions = [{lhs_contract}], "
            f"rhs_contracting_dimensions = [{rhs_contract}]>"
        )
        signature = (
            f"({self.tensor(left, dtype)}, {self.tensor(right, dtype)}) -> "
            f"{self.tensor(shape, dtype)}"
        )
        self.define(
            op.results[0],
            shape,
            dtype,
            f'"stablehlo.dot_general"({self.name(a)}, {self.name(c)}) '
            f"{{dot_dimension_numbers = {numbers}}} : {signature}",
        )

    def op_tensor_matmul(self, op: Operation) -> None:
        self.dot(op, 1, 0)

    def op_linalg_matmul(self, op: Operation) -> None:
        self.dot(op, 1, 0)

    def op_linalg_dot(self, op: Operation) -> None:
        self.dot(op, 0, 0)

    def op_tensor_convert(self, op: Operation) -> None:
        shape, dtype = self.described(op.results[0].type)
        source, from_dtype = self.shape_of(op.operands[0])
        signature = f"({self.tensor(source, from_dtype)}) -> {self.tensor(shape, dtype)}"
        self.define(
            op.results[0],
            shape,
            dtype,
            f"stablehlo.convert {self.name(op.operands[0])} : {signature}",
        )

    def op_tensor_fused(self, op: Operation) -> None:
        """The body over the broadcast shape: each argument a full tensor."""
        body: Block = op.regions[0].blocks[0]
        iteration: shapes.Shape = ()
        for operand in op.operands:
            source, _ = self.shape_of(operand)
            iteration = shapes.broadcast(iteration, source)
        for argument, operand in zip(body.arguments, op.operands, strict=True):
            source, dtype = self.shape_of(operand)
            self.names[id(argument)] = self.broadcast(self.name(operand), source, iteration, dtype)
            self.shapes[id(argument)] = (iteration, dtype)
        produced: Value | None = None
        for inner in body.operations:
            if inner.name == "tensor.yield":
                produced = inner.operands[0]
                break
            self.op(inner)
        assert produced is not None
        kind = op.attributes.get("reduce")
        if kind is None:
            self.names[id(op.results[0])] = self.name(produced)
            self.shapes[id(op.results[0])] = self.shape_of(produced)
            return
        axes = tuple(int(a) for a in op.attributes["axes"])  # type: ignore[union-attr]
        _shape, dtype = self.shape_of(produced)
        self.reduce(
            self.name(produced),
            iteration,
            dtype,
            str(kind),
            axes,
            bool(op.attributes.get("keepdims", False)),
            op.results[0],
        )

    def op_tensor_empty(self, op: Operation) -> None:
        raise StableHloError(f"@{self.function.name}: tensor.empty has no value XLA can compute")

    def op_tensor_load(self, op: Operation) -> None:
        raise StableHloError(
            f"@{self.function.name}: tensor.load reads a buffer; XLA takes tensors as arguments"
        )

    def op_tensor_store(self, op: Operation) -> None:
        raise StableHloError(
            f"@{self.function.name}: tensor.store writes a buffer; XLA returns tensors as results"
        )


def _binary_handler(mnemonic: str):  # type: ignore[no-untyped-def]
    def handler(self: _Emitter, op: Operation) -> None:
        self.elementwise(op, mnemonic)

    return handler


def _unary_handler(kind: str):  # type: ignore[no-untyped-def]
    def handler(self: _Emitter, op: Operation) -> None:
        shape, dtype = self.shape_of(op.operands[0])
        text = self.unary_math(op, kind, self.name(op.operands[0]), shape, dtype)
        self.define(op.results[0], shape, dtype, text)

    return handler


for _name, _mnemonic in _BINARY.items():
    if _name in {"add", "sub", "mul", "div"}:
        setattr(_Emitter, f"op_core_{_name}", _binary_handler(_mnemonic))
    setattr(_Emitter, f"op_tensor_{_name}", _binary_handler(_mnemonic))
for _kind in (*_UNARY, "exp2", "log2", "log10", "trunc"):
    if _kind != "neg":
        setattr(_Emitter, f"op_math_{_kind}", _unary_handler(_kind))
for _kind in ("erf", "erfc"):
    setattr(_Emitter, f"op_special_{_kind}", _unary_handler(_kind))


def _float_literal(value: float, width: int) -> str:
    """An MLIR float literal: decimal with a point, or the bit pattern for what has none."""
    if math.isfinite(value):
        return format(value, ".17e") if width == 64 else format(value, ".9e")
    packed = struct.pack(">d", value) if width == 64 else struct.pack(">f", value)
    return "0x" + packed.hex().upper()
