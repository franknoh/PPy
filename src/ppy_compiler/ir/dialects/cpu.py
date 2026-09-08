"""The cpu dialect: what a CPU is told beyond the program's meaning.

`cpu.prefetch %p {rw, locality}` asks for a cache line ahead of its use;
`cpu.pause` is the spin-wait hint a busy loop puts between its polls.
Neither changes a value; a backend without the instruction drops the hint.
A function compiled for a feature set carries `cpu.features` as an
attribute, and the boundary refuses to bind it on a machine without them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Builder, Operation, Value
from ..types import PtrType

if TYPE_CHECKING:
    from ..verify import Checker

__all__ = ["CpuDialect", "pause", "prefetch"]


def _verify_prefetch(op: Operation, checker: Checker) -> None:
    if not isinstance(op.operands[0].type, PtrType):
        checker.error(op, f"prefetch takes a pointer, not {op.operands[0].type}")
    if op.attributes.get("rw", "read") not in {"read", "write"}:
        checker.error(op, "prefetch `rw` is `read` or `write`")
    locality = op.attributes.get("locality", 3)
    if not isinstance(locality, int) or not 0 <= locality <= 3:
        checker.error(op, "prefetch `locality` is 0 (none) to 3 (keep)")


class CpuDialect(Dialect):
    name = "cpu"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        registry.add_op(OpSpec("cpu.prefetch", verify=_verify_prefetch, operands=1, results=0))
        registry.add_op(OpSpec("cpu.pause", operands=0, results=0))


def prefetch(b: Builder, pointer: Value, *, rw: str = "read", locality: int = 3) -> Operation:
    return b.create("cpu.prefetch", (pointer,), (), {"rw": rw, "locality": locality})


def pause(b: Builder) -> Operation:
    return b.create("cpu.pause", (), ())
