"""The math dialect: elementary functions, named once for every backend.

`math.sqrt %x : f64` is what `math.sqrt(x)` in the source means; the LLVM
backend lowers it to the intrinsic of that name, a C backend to libm, a
GPU backend to its device library, an XLA backend to StableHLO. The
frontend never spells a backend's intrinsic. Every operation is pure and
takes one floating-point value (or a vector of them) and gives one back;
`pow` takes two.
"""

from __future__ import annotations

import math as _math
from typing import TYPE_CHECKING

from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Builder, Operation, Value
from ..pattern import Pattern, Rewriter, RewriteResult
from ..types import FloatType, IRType, VectorType
from . import core

if TYPE_CHECKING:
    from ..verify import Checker

__all__ = ["BINARY", "UNARY", "MathDialect", "call"]

#: One operand in, one out.
UNARY = (
    "sqrt",
    "sin",
    "cos",
    "tan",
    "exp",
    "exp2",
    "log",
    "log2",
    "log10",
    "floor",
    "ceil",
    "trunc",
    "abs",
)
#: Two operands in, one out.
BINARY = ("pow",)

_FOLD = {
    "sqrt": _math.sqrt,
    "sin": _math.sin,
    "cos": _math.cos,
    "tan": _math.tan,
    "exp": _math.exp,
    "exp2": lambda v: 2.0**v,
    "log": _math.log,
    "log2": _math.log2,
    "log10": _math.log10,
    "floor": lambda v: float(_math.floor(v)),
    "ceil": lambda v: float(_math.ceil(v)),
    "trunc": lambda v: float(_math.trunc(v)),
    "abs": abs,
}


def _floating(t: IRType) -> bool:
    return isinstance(t.element if isinstance(t, VectorType) else t, FloatType)


def _verify(op: Operation, checker: Checker) -> None:
    types = {str(v.type) for v in op.operands}
    if len(types) != 1:
        checker.error(op, f"operands differ in type: {', '.join(sorted(types))}")
        return
    t = op.operands[0].type
    if not _floating(t):
        checker.error(op, f"{op.name} takes floating-point values, not {t}")
        return
    if op.results[0].type != t:
        checker.error(op, f"result is {op.results[0].type}, expected {t}")


class MathDialect(Dialect):
    name = "math"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        for name in UNARY:
            registry.add_op(
                OpSpec(f"math.{name}", pure=True, verify=_verify, operands=1, results=1)
            )
        for name in BINARY:
            registry.add_op(
                OpSpec(f"math.{name}", pure=True, verify=_verify, operands=2, results=1)
            )

    def register_patterns(self, registry: object) -> None:
        registry.add(_Fold())  # type: ignore[attr-defined]
        registry.add(_Idempotent())  # type: ignore[attr-defined]


def call(b: Builder, name: str, *operands: Value, result_name: str | None = None) -> Value:
    """`math.<name>` over `operands`, typed as its first operand."""
    if name not in UNARY and name not in BINARY:
        raise ValueError(f"no math operation named {name!r}")
    return b.create(
        f"math.{name}", operands, (operands[0].type,), result_names=(result_name,)
    ).result


class _Fold(Pattern):
    """A math function of a constant is the constant it evaluates to, in f64."""

    root = ""
    name = "math-fold"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        if op.dialect != "math" or op.results[0].type != FloatType(64):
            return RewriteResult.failure()
        values = []
        for operand in op.operands:
            owner = operand.owner
            if not isinstance(owner, Operation) or owner.name != "core.const":
                return RewriteResult.failure()
            values.append(float(owner.attributes["value"]))  # type: ignore[arg-type]
        local = op.local_name
        try:
            if local == "pow":
                folded = _math.pow(values[0], values[1])
            else:
                folded = float(_FOLD[local](values[0]))
        except (ValueError, OverflowError, ZeroDivisionError):
            return RewriteResult.failure("the domain error is the program's to see")
        constant = core.const(rewriter.builder(op), folded, op.results[0].type)
        rewriter.replace_op(op, [constant], f"math.{local}: folded {values}")
        return RewriteResult.success()


class _Idempotent(Pattern):
    """`abs(abs(x))`, `floor(floor(x))`, `ceil(ceil(x))`, `trunc(trunc(x))`: once is enough."""

    root = ""
    name = "math-idempotent"

    def match_and_rewrite(self, op: Operation, rewriter: Rewriter) -> RewriteResult:
        if op.name not in {"math.abs", "math.floor", "math.ceil", "math.trunc"}:
            return RewriteResult.failure()
        inner = op.operands[0].owner
        if not isinstance(inner, Operation) or inner.name != op.name:
            return RewriteResult.failure()
        rewriter.replace_op(op, [op.operands[0]], f"{op.name}: applied twice is applied once")
        return RewriteResult.success()
