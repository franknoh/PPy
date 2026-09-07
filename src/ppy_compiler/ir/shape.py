"""Shapes: dimensions that are numbers, symbols, or expressions of them.

A tensor's shape is a tuple of dimensions. A dimension is an `int`, a
symbol (`N`), or an expression over them -- `N * M`, `N + 1`, `ceil_div(N,
32)`, `max(N, M)` -- kept in a canonical form so that two spellings of one
size compare equal. Shape inference for the tensor dialect lives here:
broadcasting, matmul, reshape, transpose, slice, reduce, concat, each
answering with a shape or the reason there is none.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from functools import total_ordering

__all__ = [
    "Dim",
    "Shape",
    "ShapeError",
    "Symbol",
    "broadcast",
    "ceil_div",
    "concat",
    "is_static",
    "matmul",
    "numel",
    "parse_dim",
    "parse_shape",
    "reduce",
    "reshape",
    "slice_",
    "spell",
    "strides_of",
    "transpose",
]


class ShapeError(ValueError):
    """Shapes that do not fit an operation, with the reason."""


@total_ordering
@dataclass(frozen=True, slots=True)
class Symbol:
    """A dimension known only by name."""

    name: str

    def __str__(self) -> str:
        return self.name

    def __lt__(self, other: object) -> bool:
        return str(self) < str(other)


@dataclass(frozen=True, slots=True)
class Expr:
    """`op(args)`: `add`, `mul`, `ceil_div`, `max` over dimensions."""

    op: str
    args: tuple[Dim, ...]

    def __str__(self) -> str:
        if self.op in {"add", "mul"}:
            symbol = " + " if self.op == "add" else " * "
            return "(" + symbol.join(_spell_operand(a, self.op) for a in self.args) + ")"
        return f"{self.op}({', '.join(spell(a) for a in self.args)})"


Dim = int | Symbol | Expr
Shape = tuple[Dim, ...]


def _spell_operand(d: Dim, outer: str) -> str:
    """An operand of `outer`: a product inside a sum needs no parentheses."""
    text = spell(d)
    if isinstance(d, Expr) and d.op == "mul" and outer == "add":
        return text[1:-1]
    return text


def spell(d: Dim) -> str:
    """The canonical text of a dimension, as a type argument carries it."""
    if isinstance(d, int):
        return str(d)
    return str(d)


def is_static(shape: Shape) -> bool:
    return all(isinstance(d, int) for d in shape)


def numel(shape: Shape) -> Dim:
    total: Dim = 1
    for d in shape:
        total = mul(total, d)
    return total


# -- arithmetic on dimensions ------------------------------------------------------


def add(a: Dim, b: Dim) -> Dim:
    return _combine("add", a, b)


def mul(a: Dim, b: Dim) -> Dim:
    return _combine("mul", a, b)


def ceil_div(a: Dim, b: Dim) -> Dim:
    if isinstance(a, int) and isinstance(b, int):
        if b == 0:
            raise ShapeError("ceil_div by zero")
        return -(-a // b)
    if b == 1:
        return a
    return Expr("ceil_div", (a, b))


def maximum(a: Dim, b: Dim) -> Dim:
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b)
    if a == b:
        return a
    return Expr("max", tuple(sorted((a, b), key=spell)))


def _combine(op: str, a: Dim, b: Dim) -> Dim:
    """`a op b` flattened and folded: constants together, operands sorted."""
    identity = 0 if op == "add" else 1
    terms: list[Dim] = []
    for side in (a, b):
        if isinstance(side, Expr) and side.op == op:
            terms.extend(side.args)
        else:
            terms.append(side)
    constant = identity
    rest: list[Dim] = []
    for term in terms:
        if isinstance(term, int):
            constant = constant + term if op == "add" else constant * term
        else:
            rest.append(term)
    if op == "mul" and constant == 0:
        return 0
    if not rest:
        return constant
    if constant != identity:
        rest.append(constant)
    if len(rest) == 1:
        return rest[0]
    return Expr(op, tuple(sorted(rest, key=spell)))


# -- spelling and parsing -------------------------------------------------------------


def parse_dim(text: str | int) -> Dim:
    """A dimension from its text: `4`, `N`, `(N * M)`, `ceil_div(N, 32)`."""
    if isinstance(text, int):
        return text
    stripped = text.strip()
    try:
        tree = ast.parse(stripped, mode="eval").body
    except SyntaxError as error:
        raise ShapeError(f"{text!r} is not a dimension") from error
    return _from_ast(tree, text)


def _from_ast(node: ast.expr, text: str) -> Dim:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        if node.value < 0:
            raise ShapeError(f"a dimension is not negative: {text!r}")
        return node.value
    if isinstance(node, ast.Name):
        return Symbol(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
        left, right = _from_ast(node.left, text), _from_ast(node.right, text)
        return add(left, right) if isinstance(node.op, ast.Add) else mul(left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 2:
        left, right = (_from_ast(a, text) for a in node.args)
        if node.func.id == "ceil_div":
            return ceil_div(left, right)
        if node.func.id == "max":
            return maximum(left, right)
    raise ShapeError(f"{text!r} is not a dimension: use ints, names, `+`, `*`, `ceil_div`, `max`")


def parse_shape(items: tuple[str | int, ...]) -> Shape:
    return tuple(parse_dim(item) for item in items)


# -- inference -----------------------------------------------------------------------------


def broadcast(a: Shape, b: Shape) -> Shape:
    """NumPy broadcasting: trailing dimensions agree, or one of them is 1."""
    rank = max(len(a), len(b))
    left = (1,) * (rank - len(a)) + a
    right = (1,) * (rank - len(b)) + b
    result: list[Dim] = []
    for x, y in zip(left, right, strict=True):
        if x == 1:
            result.append(y)
        elif y in {1, x}:
            result.append(x)
        elif isinstance(x, int) and isinstance(y, int):
            raise ShapeError(f"shapes {spell_shape(a)} and {spell_shape(b)} do not broadcast")
        else:
            # Two symbols can only be broadcast if they are the same size;
            # the program asserts it by spelling one of them.
            raise ShapeError(
                f"dimensions {spell(x)} and {spell(y)} broadcast only if equal; spell one size"
            )
    return tuple(result)


def matmul(a: Shape, b: Shape) -> Shape:
    if len(a) != 2 or len(b) != 2:
        raise ShapeError("matmul takes two matrices")
    if a[1] != b[0]:
        raise ShapeError(f"matmul of {spell_shape(a)} by {spell_shape(b)}: inner dimensions differ")
    return (a[0], b[1])


def reshape(shape: Shape, new: Shape) -> Shape:
    if numel(shape) != numel(new):
        raise ShapeError(
            f"reshape of {spell_shape(shape)} to {spell_shape(new)} changes the element count"
        )
    return new


def transpose(shape: Shape, permutation: tuple[int, ...]) -> Shape:
    if sorted(permutation) != list(range(len(shape))):
        raise ShapeError(f"a transpose of rank {len(shape)} takes a permutation of its axes")
    return tuple(shape[p] for p in permutation)


def slice_(
    shape: Shape, starts: tuple[int, ...], stops: tuple[int, ...], steps: tuple[int, ...]
) -> Shape:
    if not len(starts) == len(stops) == len(steps) == len(shape):
        raise ShapeError("a slice names a start, a stop, and a step per axis")
    result: list[Dim] = []
    for dim, start, stop, step in zip(shape, starts, stops, steps, strict=True):
        if step <= 0:
            raise ShapeError("a slice step is positive")
        if start < 0 or stop < start:
            raise ShapeError("a slice has 0 <= start <= stop")
        if isinstance(dim, int) and stop > dim:
            raise ShapeError(f"a slice stops at {stop} in a dimension of {dim}")
        result.append(-(-(stop - start) // step))
    return tuple(result)


def reduce(shape: Shape, axes: tuple[int, ...], keepdims: bool) -> Shape:
    if any(axis < 0 or axis >= len(shape) for axis in axes) or len(set(axes)) != len(axes):
        raise ShapeError(f"reduce axes {axes} are not distinct axes of rank {len(shape)}")
    result: list[Dim] = []
    for index, dim in enumerate(shape):
        if index in axes:
            if keepdims:
                result.append(1)
        else:
            result.append(dim)
    return tuple(result)


def concat(shapes: list[Shape], axis: int) -> Shape:
    if not shapes:
        raise ShapeError("concat takes at least one tensor")
    rank = len(shapes[0])
    if axis < 0 or axis >= rank or any(len(s) != rank for s in shapes):
        raise ShapeError("concat takes tensors of one rank and an axis of it")
    total: Dim = 0
    for shape in shapes:
        for index, dim in enumerate(shape):
            if index != axis and dim != shapes[0][index]:
                raise ShapeError(
                    f"concat along axis {axis}: {spell_shape(shape)} differs from "
                    f"{spell_shape(shapes[0])}"
                )
        total = add(total, shape[axis])
    return tuple(total if i == axis else d for i, d in enumerate(shapes[0]))


def strides_of(shape: Shape, order: str = "row_major") -> tuple[Dim, ...]:
    """Element strides of a contiguous tensor in the given order."""
    strides: list[Dim] = [1] * len(shape)
    dims = list(range(len(shape)))
    if order == "row_major":
        dims.reverse()
    running: Dim = 1
    for index in dims:
        strides[index] = running
        running = mul(running, shape[index])
    return tuple(strides)


def spell_shape(shape: Shape) -> str:
    return "[" + ", ".join(spell(d) for d in shape) + "]"


def static_numel(shape: Shape) -> int:
    total = numel(shape)
    if not isinstance(total, int):
        raise ShapeError(f"{spell_shape(shape)} has no static size")
    return math.prod(shape) if shape else 1  # type: ignore[arg-type]
