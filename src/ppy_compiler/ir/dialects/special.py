"""The special dialect: the functions beyond `math`, named once for every backend.

`special.erf`, `erfc`, `gamma`, `gammaln`, `ndtr`, `logit`, and the
Bessel functions `bessel_j0`, `bessel_j1`, `bessel_y0`, `bessel_y1`
take one floating-point value; `bessel_jn` and `bessel_yn` take the
integer order first. Every operation is pure. The LLVM and C backends
lower them to libm (`erf`, `tgamma`, `lgamma`, `j0`, ...), `ndtr` and
`logit` to the expressions they are; the SciPy plugin maps
`scipy.special` onto them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Builder, Operation, Value
from ..types import FloatType, IndexType, IntType

if TYPE_CHECKING:
    from ..verify import Checker

__all__ = ["ORDERED", "UNARY", "SpecialDialect", "call"]

#: One floating-point operand in, one out; and libm's name for each.
UNARY = {
    "erf": "erf",
    "erfc": "erfc",
    "gamma": "tgamma",
    "gammaln": "lgamma",
    "bessel_j0": "j0",
    "bessel_j1": "j1",
    "bessel_y0": "y0",
    "bessel_y1": "y1",
    "ndtr": "",
    "logit": "",
}
#: An integer order, then the value.
ORDERED = {"bessel_jn": "jn", "bessel_yn": "yn"}


def _verify_unary(op: Operation, checker: Checker) -> None:
    t = op.operands[0].type
    if not isinstance(t, FloatType):
        checker.error(op, f"{op.name} takes a floating-point value, not {t}")
        return
    if op.results[0].type != t:
        checker.error(op, f"{op.name} gives {t}, not {op.results[0].type}")


def _verify_ordered(op: Operation, checker: Checker) -> None:
    order, value = (v.type for v in op.operands)
    if not isinstance(order, (IntType, IndexType)):
        checker.error(op, f"{op.name} takes an integer order first, not {order}")
    if not isinstance(value, FloatType):
        checker.error(op, f"{op.name} takes a floating-point value, not {value}")
        return
    if op.results[0].type != value:
        checker.error(op, f"{op.name} gives {value}, not {op.results[0].type}")


class SpecialDialect(Dialect):
    name = "special"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        for name in UNARY:
            registry.add_op(
                OpSpec(f"special.{name}", pure=True, verify=_verify_unary, operands=1, results=1)
            )
        for name in ORDERED:
            registry.add_op(
                OpSpec(f"special.{name}", pure=True, verify=_verify_ordered, operands=2, results=1)
            )


def call(b: Builder, name: str, *operands: Value, result_name: str | None = None) -> Value:
    if name not in UNARY and name not in ORDERED:
        raise ValueError(f"no special function named {name!r}")
    return b.create(
        f"special.{name}", operands, (operands[-1].type,), result_names=(result_name,)
    ).result
