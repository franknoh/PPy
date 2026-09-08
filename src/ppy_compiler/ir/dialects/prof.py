"""The prof dialect: the counters a profiling run increments (spec 84).

`prof.hit {counter = N}` adds one to counter `N` whenever the block it
sits in runs; `prof.hit_if %c {counter = N}` adds one when `%c` is true,
which counts a conditional branch's taken edge without touching the graph.
The `instrument-profile` pass places them, the backends lower them to
adds on the module's counter array, and `ppy run --profile` reads the
array when the program ends. Neither has a result and neither is pure, so
no pass moves or drops one; a profile-guided build carries none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Builder, Operation, Value
from ..types import BoolType

if TYPE_CHECKING:
    from ..verify import Checker

__all__ = ["ProfDialect", "hit", "hit_if"]


def _verify_counter(op: Operation, checker: Checker) -> None:
    counter = op.attributes.get("counter")
    if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
        checker.error(op, f"{op.name} needs a non-negative integer `counter`")


def _verify_hit_if(op: Operation, checker: Checker) -> None:
    _verify_counter(op, checker)
    if not isinstance(op.operands[0].type, BoolType):
        checker.error(op, "prof.hit_if counts a bool condition")


class ProfDialect(Dialect):
    name = "prof"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        registry.add_op(
            OpSpec(
                "prof.hit",
                verify=_verify_counter,
                operands=0,
                results=0,
                required_attributes=("counter",),
            )
        )
        registry.add_op(
            OpSpec(
                "prof.hit_if",
                verify=_verify_hit_if,
                operands=1,
                results=0,
                required_attributes=("counter",),
            )
        )


def hit(b: Builder, counter: int) -> Operation:
    """One more for `counter` each time the block runs."""
    return b.create("prof.hit", (), (), {"counter": counter})


def hit_if(b: Builder, condition: Value, counter: int) -> Operation:
    """One more for `counter` each time `condition` holds."""
    return b.create("prof.hit_if", (condition,), (), {"counter": counter})
