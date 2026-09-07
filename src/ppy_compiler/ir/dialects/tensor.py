"""The tensor dialect: whole arrays as values, with shapes the verifier knows.

`tensor.tensor<f64, 4, 8>` is a 4-by-8 array of `f64` laid out row-major;
a trailing `layout.*` argument names another layout, and a dimension may
be a symbol or an expression in parentheses (`tensor.tensor<f64, N, (N *
2)>`). Every operation infers its result shape from its operands' -- the
rules in `ir.shape` -- and the verifier holds the written result to it.
`load` and `store` move a tensor to and from a buffer of its elements in
row-major order; `fill` makes a tensor of one scalar; everything else is
arithmetic on the values, and says nothing about memory. `lower-tensor`
decides that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .. import shape as shapes
from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Builder, Operation, Value
from ..types import BufferType, DialectType, FloatType, IndexType, IntType, IRType, is_scalar
from . import layout as layouts
from .math import UNARY as _MATH_UNARY
from .special import UNARY as _SPECIAL_UNARY

if TYPE_CHECKING:
    from ..verify import Checker

__all__ = [
    "ELEMENTWISE",
    "REDUCTIONS",
    "UNARY",
    "TensorDialect",
    "TensorInfo",
    "describe",
    "elementwise",
    "fill",
    "tensor_type",
    "unary",
]

#: Two tensors in, one out, with broadcasting. `min` and `max` are the
#: NaN-propagating pair: a NaN operand is the answer.
ELEMENTWISE = ("add", "sub", "mul", "div", "pow", "min", "max")
REDUCTIONS = ("add", "mul", "min", "max")
#: `tensor.unary {op}`: one element in, one out -- the math dialect's
#: functions, negation, and the special functions, over every element.
UNARY = (*_MATH_UNARY, "neg", *_SPECIAL_UNARY)
#: The unary operations an integer tensor has; the rest want floats.
_INTEGER_UNARY = frozenset({"neg", "abs"})


@dataclass(frozen=True, slots=True)
class TensorInfo:
    """What a tensor type says: element type, shape, layout."""

    dtype: IRType
    shape: shapes.Shape
    layout: layouts.Layout

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def static(self) -> bool:
        return shapes.is_static(self.shape)


def tensor_type(
    dtype: IRType, shape: shapes.Shape, layout: DialectType | None = None
) -> DialectType:
    """`tensor.tensor<dtype, dims..., layout>`; row-major when no layout is given."""
    args: list[IRType | int | str] = [dtype]
    for dim in shape:
        if isinstance(dim, int):
            args.append(dim)
            continue
        text = shapes.spell(dim)
        # An expression rides in parentheses, which a sum or product already has.
        if isinstance(dim, shapes.Expr) and not text.startswith("("):
            text = f"({text})"
        args.append(text)
    if layout is not None and layout != layouts.ROW_MAJOR:
        args.append(layout)
    return DialectType("tensor", "tensor", tuple(args))


def describe(t: IRType) -> TensorInfo | None:
    """The `TensorInfo` of a tensor type, or None for anything else."""
    if not isinstance(t, DialectType) or (t.dialect, t.name) != ("tensor", "tensor"):
        return None
    if verify_tensor(t) is not None:
        return None
    dtype = t.args[0]
    assert isinstance(dtype, IRType)
    rest = list(t.args[1:])
    layout = layouts.Layout()
    if rest and isinstance(rest[-1], DialectType):
        layout = layouts.describe(rest.pop())
    dims = tuple(shapes.parse_dim(d) for d in rest)  # type: ignore[arg-type]
    return TensorInfo(dtype, dims, layout)


def verify_tensor(t: DialectType) -> str | None:
    if t.name != "tensor":
        return f"tensor defines no type {t.name!r}"
    if not t.args or not isinstance(t.args[0], IRType) or not is_scalar(t.args[0]):
        return "a tensor is tensor.tensor<dtype, dims..., layout?> with a scalar dtype"
    rest = list(t.args[1:])
    if rest and isinstance(rest[-1], DialectType):
        layout = rest.pop()
        reason = layouts.verify_layout(layout)
        if reason is not None:
            return reason
        described = layouts.describe(layout)
        if described.strides is not None and len(described.strides) != len(rest):
            return f"layout gives {len(described.strides)} strides for {len(rest)} dimensions"
    for dim in rest:
        if isinstance(dim, IRType):
            return f"{dim} is not a dimension"
        try:
            parsed = shapes.parse_dim(dim)
        except shapes.ShapeError as error:
            return str(error)
        if isinstance(parsed, int) and parsed < 0:
            return "a dimension is not negative"
    return None


def _tensor(op: Operation, checker: Checker, value: Value, what: str) -> TensorInfo | None:
    info = describe(value.type)
    if info is None:
        checker.error(op, f"{what} is a tensor, not {value.type}")
    return info


def _result_shape(op: Operation, checker: Checker, dtype: IRType, expected: shapes.Shape) -> None:
    result = describe(op.results[0].type)
    if result is None:
        checker.error(op, f"{op.name} gives a tensor, not {op.results[0].type}")
        return
    if result.dtype != dtype:
        checker.error(
            op, f"{op.name} gives {dtype} elements, the result is written as {result.dtype}"
        )
    if result.shape != expected:
        checker.error(
            op,
            f"{op.name} gives shape {shapes.spell_shape(expected)}, the result is written as "
            f"{shapes.spell_shape(result.shape)}",
        )


def _verify_empty(op: Operation, checker: Checker) -> None:
    _tensor(op, checker, op.results[0], "empty")


def _verify_load(op: Operation, checker: Checker) -> None:
    info = _tensor(op, checker, op.results[0], "load")
    source = op.operands[0].type
    if info is None:
        return
    if not isinstance(source, BufferType) or source.element != info.dtype:
        checker.error(op, f"load reads a buffer<{info.dtype}>, not {source}")


def _verify_store(op: Operation, checker: Checker) -> None:
    info = _tensor(op, checker, op.operands[0], "store")
    target = op.operands[1].type
    if info is None:
        return
    if not isinstance(target, BufferType) or target.element != info.dtype:
        checker.error(op, f"store writes a buffer<{info.dtype}>, not {target}")


def _verify_elementwise(op: Operation, checker: Checker) -> None:
    a = _tensor(op, checker, op.operands[0], "the left operand")
    b = _tensor(op, checker, op.operands[1], "the right operand")
    if a is None or b is None:
        return
    if a.dtype != b.dtype:
        checker.error(
            op, f"{op.name} takes tensors of one element type, not {a.dtype} and {b.dtype}"
        )
        return
    if op.local_name in {"div", "pow"} and not isinstance(a.dtype, FloatType):
        checker.error(op, f"{op.name} takes floating-point tensors")
    try:
        shape = shapes.broadcast(a.shape, b.shape)
    except shapes.ShapeError as error:
        checker.error(op, str(error))
        return
    _result_shape(op, checker, a.dtype, shape)


def _verify_fill(op: Operation, checker: Checker) -> None:
    scalar = op.operands[0].type
    if not is_scalar(scalar):
        checker.error(op, f"tensor.fill takes a scalar, not {scalar}")
        return
    result = describe(op.results[0].type)
    if result is None:
        checker.error(op, f"tensor.fill gives a tensor, not {op.results[0].type}")
        return
    if result.dtype != scalar:
        checker.error(op, f"tensor.fill of {scalar} gives a tensor of {scalar}, not {result.dtype}")


def _verify_unary(op: Operation, checker: Checker) -> None:
    source = _tensor(op, checker, op.operands[0], "the operand")
    kind = op.attributes.get("op")
    if kind not in UNARY:
        checker.error(op, f"tensor.unary `op` is one of {', '.join(UNARY)}, not {kind!r}")
        return
    if source is None:
        return
    if kind not in _INTEGER_UNARY and not isinstance(source.dtype, FloatType):
        checker.error(op, f"tensor.unary {kind} takes a floating-point tensor, not {source.dtype}")
    _result_shape(op, checker, source.dtype, source.shape)


def _verify_broadcast(op: Operation, checker: Checker) -> None:
    source = _tensor(op, checker, op.operands[0], "broadcast")
    result = describe(op.results[0].type)
    if source is None or result is None:
        if result is None:
            checker.error(op, "broadcast gives a tensor")
        return
    try:
        shape = shapes.broadcast(source.shape, result.shape)
    except shapes.ShapeError as error:
        checker.error(op, str(error))
        return
    if shape != result.shape:
        checker.error(
            op,
            f"{shapes.spell_shape(source.shape)} does not broadcast to "
            f"{shapes.spell_shape(result.shape)}",
        )
    if result.dtype != source.dtype:
        checker.error(op, "broadcast keeps the element type")


def _verify_reshape(op: Operation, checker: Checker) -> None:
    source = _tensor(op, checker, op.operands[0], "reshape")
    result = describe(op.results[0].type)
    if source is None or result is None:
        if result is None:
            checker.error(op, "reshape gives a tensor")
        return
    try:
        shapes.reshape(source.shape, result.shape)
    except shapes.ShapeError as error:
        checker.error(op, str(error))
    if result.dtype != source.dtype:
        checker.error(op, "reshape keeps the element type")


def _ints(
    op: Operation, checker: Checker, name: str, count: int | None = None
) -> tuple[int, ...] | None:
    value = op.attributes.get(name)
    if not isinstance(value, tuple) or not all(isinstance(v, int) for v in value):
        checker.error(op, f"{op.name} needs `{name}`, a tuple of integers")
        return None
    if count is not None and len(value) != count:
        checker.error(op, f"`{name}` names {len(value)} axes for a tensor of rank {count}")
        return None
    return tuple(int(v) for v in value)


def _verify_transpose(op: Operation, checker: Checker) -> None:
    source = _tensor(op, checker, op.operands[0], "transpose")
    if source is None:
        return
    permutation = _ints(op, checker, "perm", source.rank)
    if permutation is None:
        return
    try:
        shape = shapes.transpose(source.shape, permutation)
    except shapes.ShapeError as error:
        checker.error(op, str(error))
        return
    _result_shape(op, checker, source.dtype, shape)


def _verify_slice(op: Operation, checker: Checker) -> None:
    source = _tensor(op, checker, op.operands[0], "slice")
    if source is None:
        return
    starts = _ints(op, checker, "starts", source.rank)
    stops = _ints(op, checker, "stops", source.rank)
    steps = _ints(op, checker, "steps", source.rank)
    if starts is None or stops is None or steps is None:
        return
    try:
        shape = shapes.slice_(source.shape, starts, stops, steps)
    except shapes.ShapeError as error:
        checker.error(op, str(error))
        return
    _result_shape(op, checker, source.dtype, shape)


def _verify_concat(op: Operation, checker: Checker) -> None:
    infos = [_tensor(op, checker, v, "a concat operand") for v in op.operands]
    if not infos or any(i is None for i in infos):
        if not infos:
            checker.error(op, "concat takes at least one tensor")
        return
    axis = op.attributes.get("axis")
    if not isinstance(axis, int):
        checker.error(op, "concat needs `axis`")
        return
    dtype = infos[0].dtype  # type: ignore[union-attr]
    if any(i.dtype != dtype for i in infos):  # type: ignore[union-attr]
        checker.error(op, "concat takes tensors of one element type")
        return
    try:
        shape = shapes.concat([i.shape for i in infos], axis)  # type: ignore[union-attr]
    except shapes.ShapeError as error:
        checker.error(op, str(error))
        return
    _result_shape(op, checker, dtype, shape)


def _verify_reduce(op: Operation, checker: Checker) -> None:
    source = _tensor(op, checker, op.operands[0], "reduce")
    kind = op.attributes.get("op")
    if kind not in REDUCTIONS:
        checker.error(op, f"reduce `op` is one of {', '.join(REDUCTIONS)}, not {kind!r}")
    if source is None:
        return
    axes = _ints(op, checker, "axes")
    if axes is None:
        return
    keepdims = op.attributes.get("keepdims", False)
    if not isinstance(keepdims, bool):
        checker.error(op, "reduce `keepdims` is a bool")
        return
    try:
        shape = shapes.reduce(source.shape, axes, keepdims)
    except shapes.ShapeError as error:
        checker.error(op, str(error))
        return
    _result_shape(op, checker, source.dtype, shape)


def _verify_matmul(op: Operation, checker: Checker) -> None:
    a = _tensor(op, checker, op.operands[0], "the left operand")
    b = _tensor(op, checker, op.operands[1], "the right operand")
    if a is None or b is None:
        return
    if a.dtype != b.dtype:
        checker.error(op, "matmul takes tensors of one element type")
        return
    try:
        shape = shapes.matmul(a.shape, b.shape)
    except shapes.ShapeError as error:
        checker.error(op, str(error))
        return
    _result_shape(op, checker, a.dtype, shape)


def _verify_convert(op: Operation, checker: Checker) -> None:
    source = _tensor(op, checker, op.operands[0], "convert")
    result = describe(op.results[0].type)
    if source is None or result is None:
        if result is None:
            checker.error(op, "convert gives a tensor")
        return
    if result.shape != source.shape:
        checker.error(op, "convert keeps the shape")
    if not isinstance(result.dtype, (IntType, IndexType, FloatType)):
        checker.error(op, f"convert gives numbers, not {result.dtype}")


class TensorDialect(Dialect):
    name = "tensor"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        add = registry.add_op
        add(OpSpec("tensor.empty", pure=True, verify=_verify_empty, operands=0, results=1))
        add(OpSpec("tensor.load", verify=_verify_load, operands=1, results=1))
        add(OpSpec("tensor.store", verify=_verify_store, operands=2, results=0))
        for name in ELEMENTWISE:
            add(
                OpSpec(
                    f"tensor.{name}", pure=True, verify=_verify_elementwise, operands=2, results=1
                )
            )
        add(OpSpec("tensor.fill", pure=True, verify=_verify_fill, operands=1, results=1))
        add(
            OpSpec(
                "tensor.unary",
                pure=True,
                verify=_verify_unary,
                operands=1,
                results=1,
                required_attributes=("op",),
            )
        )
        add(OpSpec("tensor.broadcast", pure=True, verify=_verify_broadcast, operands=1, results=1))
        add(OpSpec("tensor.reshape", pure=True, verify=_verify_reshape, operands=1, results=1))
        add(
            OpSpec(
                "tensor.transpose",
                pure=True,
                verify=_verify_transpose,
                operands=1,
                results=1,
                required_attributes=("perm",),
            )
        )
        add(
            OpSpec(
                "tensor.slice",
                pure=True,
                verify=_verify_slice,
                operands=1,
                results=1,
                required_attributes=("starts", "stops", "steps"),
            )
        )
        add(
            OpSpec(
                "tensor.concat",
                pure=True,
                verify=_verify_concat,
                results=1,
                required_attributes=("axis",),
            )
        )
        add(
            OpSpec(
                "tensor.reduce",
                pure=True,
                verify=_verify_reduce,
                operands=1,
                results=1,
                required_attributes=("axes", "op"),
            )
        )
        add(OpSpec("tensor.matmul", pure=True, verify=_verify_matmul, operands=2, results=1))
        add(OpSpec("tensor.convert", pure=True, verify=_verify_convert, operands=1, results=1))

    def verify_type(self, t: DialectType) -> str | None:
        return verify_tensor(t)


# -- builders -----------------------------------------------------------------------------------


def _typed(
    b: Builder,
    name: str,
    operands: tuple[Value, ...],
    result: DialectType,
    attributes=None,
    hint=None,
):  # type: ignore[no-untyped-def]
    return b.create(
        f"tensor.{name}", operands, (result,), attributes or {}, result_names=(hint,)
    ).result


def empty(b: Builder, t: DialectType, name: str | None = None) -> Value:
    return _typed(b, "empty", (), t, hint=name)


def load(b: Builder, buffer: Value, t: DialectType, name: str | None = None) -> Value:
    return _typed(b, "load", (buffer,), t, hint=name)


def store(b: Builder, tensor: Value, buffer: Value) -> Operation:
    return b.create("tensor.store", (tensor, buffer), ())


def elementwise(b: Builder, name: str, a: Value, c: Value, hint: str | None = None) -> Value:
    if name not in ELEMENTWISE:
        raise ValueError(f"no elementwise tensor operation {name!r}")
    left, right = describe(a.type), describe(c.type)
    assert left is not None and right is not None
    result = tensor_type(left.dtype, shapes.broadcast(left.shape, right.shape))
    return _typed(b, name, (a, c), result, hint=hint)


def fill(b: Builder, scalar: Value, t: DialectType, name: str | None = None) -> Value:
    """A tensor of type `t` whose every element is `scalar`."""
    return _typed(b, "fill", (scalar,), t, hint=name)


def unary(b: Builder, op: str, t: Value, name: str | None = None) -> Value:
    """`op` applied to every element of `t`."""
    if op not in UNARY:
        raise ValueError(f"no unary tensor operation {op!r}")
    info = describe(t.type)
    assert info is not None
    return _typed(b, "unary", (t,), tensor_type(info.dtype, info.shape), {"op": op}, hint=name)


def broadcast(b: Builder, t: Value, shape: shapes.Shape, name: str | None = None) -> Value:
    info = describe(t.type)
    assert info is not None
    return _typed(b, "broadcast", (t,), tensor_type(info.dtype, shape), hint=name)


def reshape(b: Builder, t: Value, shape: shapes.Shape, name: str | None = None) -> Value:
    info = describe(t.type)
    assert info is not None
    return _typed(b, "reshape", (t,), tensor_type(info.dtype, shape), hint=name)


def transpose(b: Builder, t: Value, perm: tuple[int, ...], name: str | None = None) -> Value:
    info = describe(t.type)
    assert info is not None
    result = tensor_type(info.dtype, shapes.transpose(info.shape, perm))
    return _typed(b, "transpose", (t,), result, {"perm": tuple(perm)}, hint=name)


def slice_(
    b: Builder,
    t: Value,
    starts: tuple[int, ...],
    stops: tuple[int, ...],
    steps: tuple[int, ...],
    name: str | None = None,
) -> Value:
    info = describe(t.type)
    assert info is not None
    result = tensor_type(info.dtype, shapes.slice_(info.shape, starts, stops, steps))
    return _typed(
        b,
        "slice",
        (t,),
        result,
        {"starts": tuple(starts), "stops": tuple(stops), "steps": tuple(steps)},
        hint=name,
    )


def concat(b: Builder, parts: tuple[Value, ...], axis: int, name: str | None = None) -> Value:
    infos = [describe(p.type) for p in parts]
    assert all(i is not None for i in infos)
    result = tensor_type(infos[0].dtype, shapes.concat([i.shape for i in infos], axis))  # type: ignore[union-attr]
    return _typed(b, "concat", parts, result, {"axis": axis}, hint=name)


def reduce(
    b: Builder,
    t: Value,
    axes: tuple[int, ...],
    op: str,
    keepdims: bool = False,
    name: str | None = None,
) -> Value:
    info = describe(t.type)
    assert info is not None
    result = tensor_type(info.dtype, shapes.reduce(info.shape, axes, keepdims))
    return _typed(
        b, "reduce", (t,), result, {"axes": tuple(axes), "op": op, "keepdims": keepdims}, hint=name
    )


def matmul(b: Builder, a: Value, c: Value, name: str | None = None) -> Value:
    left, right = describe(a.type), describe(c.type)
    assert left is not None and right is not None
    return _typed(
        b,
        "matmul",
        (a, c),
        tensor_type(left.dtype, shapes.matmul(left.shape, right.shape)),
        hint=name,
    )


def convert(b: Builder, t: Value, dtype: IRType, name: str | None = None) -> Value:
    info = describe(t.type)
    assert info is not None
    return _typed(b, "convert", (t,), tensor_type(dtype, info.shape), hint=name)
