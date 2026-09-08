"""Canonicalization patterns of the tensor dialect.

A view that changes nothing goes away: a `broadcast`, `reshape`, or `convert`
to the operand's own type, a `transpose` by the identity, a `slice` of the
whole, a `reduce` over no axes. Two transposes are one; two reshapes are one;
two negations are none. Arithmetic over `fill`s is a `fill` of the scalar
arithmetic, so a constant expression is computed once rather than per
element; and `x + fill 0` or `x * fill 1` is `x` where that is exact -- for
integers always, for floats only the multiplication, since `-0.0 + 0.0` is
`0.0`.
"""

from __future__ import annotations

from .. import shape as shapes
from ..model import Attribute, Operation, Value
from ..pattern import Pattern, PatternSet, Rewriter, RewriteResult
from ..types import FloatType
from . import tensor as tensors

__all__ = ["canonicalization_patterns", "register"]


def _constant(value: Value) -> Attribute | None:
    op = value.owner
    if isinstance(op, Operation) and op.name == "core.const":
        return op.attributes.get("value")
    return None


def _fill_constant(value: Value) -> Attribute | None:
    """The constant a `tensor.fill` of a `core.const` spreads, or None."""
    op = value.owner
    if isinstance(op, Operation) and op.name == "tensor.fill":
        return _constant(op.operands[0])
    return None


def _producer(value: Value, name: str) -> Operation | None:
    op = value.owner
    return op if isinstance(op, Operation) and op.name == name else None


class _SameTypeIsIdentity(Pattern):
    """A view to the operand's own type: `broadcast`, `reshape`, `convert`."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.name = f"{root}-identity"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        if op.operands[0].type != op.results[0].type:
            return RewriteResult.failure()
        rewriter.replace_op(op, [op.operands[0]], f"{op.name}: to its own type, removed")
        return RewriteResult.success()


class _IdentityTranspose(Pattern):
    root = "tensor.transpose"
    name = "transpose-identity"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        perm = tuple(int(p) for p in op.attributes["perm"])  # type: ignore[union-attr]
        if perm != tuple(range(len(perm))):
            return RewriteResult.failure()
        rewriter.replace_op(op, [op.operands[0]], "tensor.transpose: by the identity, removed")
        return RewriteResult.success()


class _TransposeTranspose(Pattern):
    root = "tensor.transpose"
    name = "transpose-transpose"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        inner = _producer(op.operands[0], "tensor.transpose")
        if inner is None:
            return RewriteResult.failure()
        outer = tuple(int(p) for p in op.attributes["perm"])  # type: ignore[union-attr]
        first = tuple(int(p) for p in inner.attributes["perm"])  # type: ignore[union-attr]
        composed = tuple(first[p] for p in outer)
        b = rewriter.builder(op)
        if composed == tuple(range(len(composed))):
            rewriter.replace_op(op, [inner.operands[0]], "tensor.transpose: two undo each other")
            return RewriteResult.success()
        merged = tensors.transpose(b, inner.operands[0], composed)
        rewriter.replace_op(op, [merged], "tensor.transpose: two composed into one")
        return RewriteResult.success()


class _ReshapeReshape(Pattern):
    root = "tensor.reshape"
    name = "reshape-reshape"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        inner = _producer(op.operands[0], "tensor.reshape")
        if inner is None:
            return RewriteResult.failure()
        b = rewriter.builder(op)
        merged = b.create("tensor.reshape", (inner.operands[0],), (op.results[0].type,)).result
        rewriter.replace_op(op, [merged], "tensor.reshape: two reshapes are one")
        return RewriteResult.success()


class _WholeSlice(Pattern):
    root = "tensor.slice"
    name = "slice-whole"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        if op.operands[0].type != op.results[0].type:
            return RewriteResult.failure()
        starts = tuple(int(v) for v in op.attributes["starts"])  # type: ignore[union-attr]
        steps = tuple(int(v) for v in op.attributes["steps"])  # type: ignore[union-attr]
        if any(starts) or any(step != 1 for step in steps):
            return RewriteResult.failure()
        rewriter.replace_op(op, [op.operands[0]], "tensor.slice: of the whole, removed")
        return RewriteResult.success()


class _EmptyReduce(Pattern):
    root = "tensor.reduce"
    name = "reduce-nothing"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        if tuple(op.attributes["axes"]) or op.operands[0].type != op.results[0].type:  # type: ignore[arg-type]
            return RewriteResult.failure()
        rewriter.replace_op(op, [op.operands[0]], "tensor.reduce: over no axes, removed")
        return RewriteResult.success()


class _DoubleNeg(Pattern):
    root = "tensor.unary"
    name = "tensor-neg-neg"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        inner = _producer(op.operands[0], "tensor.unary")
        if op.attributes.get("op") != "neg" or inner is None or inner.attributes.get("op") != "neg":
            return RewriteResult.failure()
        rewriter.replace_op(op, [inner.operands[0]], "tensor.unary: double negation removed")
        return RewriteResult.success()


class _UnaryOfFill(Pattern):
    """`unary op (fill c)` is `fill (op c)`: computed once, not per element."""

    root = "tensor.unary"
    name = "unary-of-fill"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        fill = _producer(op.operands[0], "tensor.fill")
        if fill is None:
            return RewriteResult.failure()
        b = rewriter.builder(op)
        kind = str(op.attributes["op"])
        scalar = tensors.scalar_unary(b, kind, fill.operands[0], _module_of(op))
        result = tensors.fill(b, scalar, op.results[0].type)  # type: ignore[arg-type]
        rewriter.replace_op(op, [result], f"tensor.unary {kind}: of a fill, computed once")
        return RewriteResult.success()


class _BinaryOfFills(Pattern):
    """`op (fill a) (fill b)` is `fill (op a b)`."""

    def __init__(self, name: str) -> None:
        self.root = f"tensor.{name}"
        self.name = f"{name}-of-fills"
        self.operation = name

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        left = _producer(op.operands[0], "tensor.fill")
        right = _producer(op.operands[1], "tensor.fill")
        if left is None or right is None:
            return RewriteResult.failure()
        b = rewriter.builder(op)
        scalar = tensors.scalar_binary(
            b, self.operation, left.operands[0], right.operands[0], _module_of(op)
        )
        result = tensors.fill(b, scalar, op.results[0].type)  # type: ignore[arg-type]
        rewriter.replace_op(op, [result], f"{op.name}: of two fills, computed once")
        return RewriteResult.success()


class _FillIdentity(Pattern):
    """`x + fill 0`, `x - fill 0`, `x * fill 1`, `x / fill 1` are `x` where exact."""

    def __init__(self, name: str, neutral: int, positions: tuple[int, ...], floats: bool):
        self.root = f"tensor.{name}"
        self.name = f"{name}-fill-identity"
        self.neutral = neutral
        self.positions = positions
        self.floats = floats

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        info = tensors.describe(op.results[0].type)
        if info is None:
            return RewriteResult.failure()
        if isinstance(info.dtype, FloatType) and not self.floats:
            return RewriteResult.failure("-0.0 + 0.0 is 0.0: floats keep their addition")
        for position in self.positions:
            constant = _fill_constant(op.operands[position])
            other = op.operands[1 - position]
            if (
                constant is not None
                and not isinstance(constant, bool)
                and constant == self.neutral
                and other.type == op.results[0].type
            ):
                rewriter.replace_op(op, [other], f"{op.name}: by the neutral element, removed")
                return RewriteResult.success()
        return RewriteResult.failure()


def _module_of(op: Operation):  # type: ignore[no-untyped-def]
    block = op.parent
    region = block.region if block is not None else None
    function = region.function if region is not None else None
    return function.module if function is not None else None


def canonicalization_patterns() -> list[Pattern]:
    return [
        _SameTypeIsIdentity("tensor.broadcast"),
        _SameTypeIsIdentity("tensor.reshape"),
        _SameTypeIsIdentity("tensor.convert"),
        _IdentityTranspose(),
        _TransposeTranspose(),
        _ReshapeReshape(),
        _WholeSlice(),
        _EmptyReduce(),
        _DoubleNeg(),
        _UnaryOfFill(),
        *(_BinaryOfFills(name) for name in tensors.ELEMENTWISE),
        _FillIdentity("add", 0, (0, 1), floats=False),
        _FillIdentity("sub", 0, (1,), floats=False),
        _FillIdentity("mul", 1, (0, 1), floats=True),
        _FillIdentity("div", 1, (1,), floats=True),
    ]


def register(patterns: PatternSet) -> None:
    for pattern in canonicalization_patterns():
        patterns.add(pattern)


# Shape inference lives in `ir.shape`; the patterns above never change a shape.
_ = shapes
