"""Canonicalization and folding patterns of the core dialect.

Each rewrite is made only where the types and the operation's own
semantics allow it. An integer `x * 0` is `0` because no overflow can
happen; a float `x * 0.0` stays, because `x` may be NaN or infinite.
`neg(neg(x))` is `x` under Python and wrap semantics but not under
`checked`, where the inner negation of the minimum is a failure the
program would have seen. Constant folding follows the overflow attribute:
a Python-semantics result that does not fit the type is left alone, a
wrapped one wraps.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable

from ..model import Attribute, Operation, Value
from ..pattern import Pattern, PatternSet, Rewriter, RewriteResult
from ..types import BoolType, FloatType, IndexType, IntType, IRType
from . import core

__all__ = ["canonicalization_patterns", "folding_patterns"]


def _constant(value: Value) -> Attribute | None:
    op = value.owner
    if isinstance(op, Operation) and op.name == "core.const":
        return op.attributes.get("value")
    return None


def _is_zero(value: Value) -> bool:
    constant = _constant(value)
    return constant is not None and not isinstance(constant, bool) and constant == 0


def _is_one(value: Value) -> bool:
    constant = _constant(value)
    return constant is not None and not isinstance(constant, bool) and constant == 1


def _int_like(t: IRType) -> bool:
    return isinstance(t, (IntType, IndexType))


class _Identity(Pattern):
    """`x op neutral -> x`, for the operand positions where it holds."""

    def __init__(self, root: str, neutral: Callable[[Value], bool], positions: tuple[int, ...]):
        self.root = root
        self.neutral = neutral
        self.positions = positions
        self.name = f"{root}-identity"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        if not _int_like(op.results[0].type):
            return RewriteResult.failure("floats keep their arithmetic")
        for position in self.positions:
            if self.neutral(op.operands[position]):
                kept = op.operands[1 - position]
                rewriter.replace_op(op, [kept], f"{op.name}: identity element removed")
                return RewriteResult.success()
        return RewriteResult.failure()


class _MulByZero(Pattern):
    root = "core.mul"
    name = "mul-by-zero"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        if not _int_like(op.results[0].type):
            return RewriteResult.failure("a float times zero may be NaN")
        for position in (0, 1):
            if _is_zero(op.operands[position]):
                zero = core.const(rewriter.builder(op), 0, op.results[0].type)
                rewriter.replace_op(op, [zero], "core.mul: multiplied by zero")
                return RewriteResult.success()
        return RewriteResult.failure()


class _DoubleNeg(Pattern):
    root = "core.neg"
    name = "neg-neg"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        inner = op.operands[0].owner
        if not isinstance(inner, Operation) or inner.name != "core.neg":
            return RewriteResult.failure()
        if _int_like(op.results[0].type) and (
            op.attributes.get("overflow") == "checked"
            or inner.attributes.get("overflow") == "checked"
        ):
            return RewriteResult.failure("a checked negation of the minimum is a failure")
        rewriter.replace_op(op, [inner.operands[0]], "core.neg: double negation removed")
        return RewriteResult.success()


def _lossless(source: IRType, target: IRType) -> bool:
    """Does every value of `source` survive a cast to `target` and back?"""
    if source == target:
        return True
    if isinstance(source, IntType) and isinstance(target, IntType):
        if source.signed == target.signed:
            return target.width >= source.width
        return (not source.signed) and target.signed and target.width > source.width
    if isinstance(source, FloatType) and isinstance(target, FloatType):
        return target.width >= source.width
    if isinstance(source, IntType) and isinstance(target, FloatType):
        return source.width < {16: 11, 32: 24, 64: 53}[target.width]
    return isinstance(source, BoolType) and isinstance(target, (IntType, FloatType))


class _CastCast(Pattern):
    root = "core.cast"
    name = "cast-cast"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        source = op.operands[0]
        if source.type == op.results[0].type:
            rewriter.replace_op(op, [source], "core.cast: identity cast removed")
            return RewriteResult.success()
        inner = source.owner
        if not isinstance(inner, Operation) or inner.name != "core.cast":
            return RewriteResult.failure()
        original = inner.operands[0]
        if original.type == op.results[0].type and _lossless(original.type, source.type):
            rewriter.replace_op(op, [original], "core.cast: lossless round trip removed")
            return RewriteResult.success()
        return RewriteResult.failure("the intermediate type loses information")


class _SelectFold(Pattern):
    root = "core.select"
    name = "select-fold"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        condition, a, b = op.operands
        if a is b:
            rewriter.replace_op(op, [a], "core.select: both arms are one value")
            return RewriteResult.success()
        chosen = _constant(condition)
        if isinstance(chosen, bool):
            rewriter.replace_op(op, [a if chosen else b], "core.select: constant condition")
            return RewriteResult.success()
        return RewriteResult.failure()


class _CmpSelf(Pattern):
    root = "core.cmp"
    name = "cmp-self"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        a, b = op.operands
        if a is not b or not (_int_like(a.type) or isinstance(a.type, BoolType)):
            return RewriteResult.failure("floats compare unequal to themselves when NaN")
        answer = op.attributes["predicate"] in {"eq", "le", "ge"}
        constant = core.const(rewriter.builder(op), answer, op.results[0].type)
        rewriter.replace_op(op, [constant], "core.cmp: a value compared to itself")
        return RewriteResult.success()


_INT_OPS: dict[str, Callable[[int, int], int]] = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "and": operator.and_,
    "or": operator.or_,
    "xor": operator.xor,
}
_FLOAT_OPS: dict[str, Callable[[float, float], float]] = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
}
_PREDICATES: dict[str, Callable[[object, object], bool]] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}


def _fit(value: int, t: IRType, overflow: str) -> int | None:
    """The constant a folded integer becomes, or None when it must not fold."""
    if isinstance(t, IndexType):
        return value
    assert isinstance(t, IntType)
    if t.fits(value):
        return value
    if overflow == "wrap":
        wrapped = value % (1 << t.width)
        if t.signed and wrapped >= 1 << (t.width - 1):
            wrapped -= 1 << t.width
        return wrapped
    return None


class _FoldBinary(Pattern):
    """Two constant operands become one constant, when the semantics agree."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.name = f"{root}-fold"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        a, b = (_constant(v) for v in op.operands)
        if a is None or b is None or isinstance(a, bool) != isinstance(b, bool):
            return RewriteResult.failure()
        t = op.results[0].type
        local = op.local_name
        folded: Attribute | None = None
        if isinstance(t, BoolType) and isinstance(a, bool) and isinstance(b, bool):
            if local in {"and", "or", "xor"}:
                folded = _INT_OPS[local](a, b)
        elif _int_like(t) and isinstance(a, int) and isinstance(b, int):
            overflow = str(op.attributes.get("overflow", "python"))
            if local in _INT_OPS:
                folded = _fit(_INT_OPS[local](a, b), t, overflow)
            elif local in {"div", "mod"} and b != 0:
                if op.attributes.get("rounding") == "floor":
                    result = a // b if local == "div" else a % b
                else:
                    quotient = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
                    result = quotient if local == "div" else a - quotient * b
                folded = _fit(result, t, overflow)
            elif local in {"shl", "shr"} and 0 <= b < 64:
                result = a << b if local == "shl" else a >> b
                folded = _fit(result, t, overflow)
        elif isinstance(t, FloatType) and t.width == 64 and local in _FLOAT_OPS:
            if local == "div" and b == 0:
                return RewriteResult.failure("division by zero stays for the guard")
            folded = float(_FLOAT_OPS[local](float(a), float(b)))
        if folded is None:
            return RewriteResult.failure("the result would not fit, or the type is not folded")
        constant = core.const(rewriter.builder(op), folded, t)
        rewriter.replace_op(op, [constant], f"{op.name}: folded {a} and {b}")
        return RewriteResult.success()


class _FoldCmp(Pattern):
    root = "core.cmp"
    name = "cmp-fold"
    benefit = 2

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        a, b = (_constant(v) for v in op.operands)
        if a is None or b is None:
            return RewriteResult.failure()
        if isinstance(a, float) and isinstance(b, float) and (math.isnan(a) or math.isnan(b)):
            return RewriteResult.failure("NaN compares by the backend's rules")
        answer = _PREDICATES[str(op.attributes["predicate"])](a, b)
        constant = core.const(rewriter.builder(op), answer, op.results[0].type)
        rewriter.replace_op(op, [constant], f"core.cmp: folded {a} and {b}")
        return RewriteResult.success()


class _FoldNeg(Pattern):
    root = "core.neg"
    name = "neg-fold"
    benefit = 2

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        a = _constant(op.operands[0])
        if a is None or isinstance(a, bool):
            return RewriteResult.failure()
        t = op.results[0].type
        folded: Attribute | None
        if _int_like(t):
            folded = _fit(-int(a), t, str(op.attributes.get("overflow", "python")))
        elif isinstance(t, FloatType) and t.width == 64:
            folded = -float(a)
        else:
            folded = None
        if folded is None:
            return RewriteResult.failure()
        constant = core.const(rewriter.builder(op), folded, t)
        rewriter.replace_op(op, [constant], f"core.neg: folded {a}")
        return RewriteResult.success()


class _FoldCast(Pattern):
    root = "core.cast"
    name = "cast-fold"
    benefit = 2

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        a = _constant(op.operands[0])
        if a is None:
            return RewriteResult.failure()
        source = op.operands[0].type
        target = op.results[0].type
        folded: Attribute | None = None
        if isinstance(target, IntType) and isinstance(a, (int, float)):
            if isinstance(a, float) and (math.isnan(a) or math.isinf(a)):
                return RewriteResult.failure("no integer for NaN or infinity")
            value = int(a)  # a float truncates toward zero, like C
            folded = (
                _fit(value, target, "wrap")
                if not isinstance(source, FloatType)
                else (value if target.fits(value) else None)
            )
        elif isinstance(target, IndexType) and isinstance(a, int):
            folded = int(a)
        elif isinstance(target, FloatType) and target.width == 64 and isinstance(a, (int, float)):
            folded = float(a)
        elif isinstance(target, BoolType) and isinstance(a, (int, float)):
            folded = bool(a)
        if folded is None:
            return RewriteResult.failure()
        constant = core.const(rewriter.builder(op), folded, target)
        rewriter.replace_op(op, [constant], f"core.cast: folded {a} to {target}")
        return RewriteResult.success()


def folding_patterns() -> list[Pattern]:
    return [
        *(_FoldBinary(f"core.{name}") for name in ("add", "sub", "mul", "div", "mod")),
        *(_FoldBinary(f"core.{name}") for name in ("and", "or", "xor", "shl", "shr")),
        _FoldCmp(),
        _FoldNeg(),
        _FoldCast(),
    ]


def canonicalization_patterns() -> list[Pattern]:
    return [
        *folding_patterns(),
        _Identity("core.add", _is_zero, (0, 1)),
        _Identity("core.sub", _is_zero, (1,)),
        _Identity("core.mul", _is_one, (0, 1)),
        _Identity("core.div", _is_one, (1,)),
        _MulByZero(),
        _DoubleNeg(),
        _CastCast(),
        _SelectFold(),
        _CmpSelf(),
    ]


def register(patterns: PatternSet) -> None:
    for pattern in canonicalization_patterns():
        patterns.add(pattern)
