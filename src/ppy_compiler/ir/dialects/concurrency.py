"""The concurrency dialect: threads, and what keeps them apart.

`spawn @f(%args...) : i64` starts a thread running the IR function `@f`
with those arguments and hands back its handle; `join %h : i64` waits for
it and gives the status `@f` returned -- zero, or the fallback status a
guard failed with, which the joiner then fails with too. The
synchronization objects are memory the program owns, not opaque handles:
a mutex is one `i64` slot (zero unlocked), a condition is one `i64` slot
counting notifications, a barrier is two `i64` slots (arrivals, then the
generation), and every backend implements them the same way over the
atomic dialect, so a program built for one runs exactly like one built for
another. `thread_id : i64` is the running thread's identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Builder, Operation, SymbolRef, Value
from ..types import I64, IntType, PtrType

if TYPE_CHECKING:
    from ..verify import Checker

__all__ = [
    "ConcurrencyDialect",
    "barrier",
    "condition_notify",
    "condition_wait",
    "join",
    "mutex_lock",
    "mutex_unlock",
    "spawn",
    "thread_id",
]


def _slot(op: Operation, checker: Checker, index: int, what: str) -> None:
    pointer = op.operands[index].type
    if not (isinstance(pointer, PtrType) and pointer.pointee == I64 and pointer.mutable):
        checker.error(op, f"{what} is a mutable pointer to i64, not {pointer}")


def _verify_spawn(op: Operation, checker: Checker) -> None:
    callee = op.attributes.get("callee")
    if not isinstance(callee, SymbolRef):
        checker.error(op, "spawn needs a `callee` symbol")
        return
    if checker.module is None:
        return
    target = checker.module.functions.get(callee.name)
    if target is None:
        checker.error(op, f"spawn of @{callee.name}, which is not a function of this module")
        return
    given = tuple(v.type for v in op.operands)
    expected = tuple(t for _name, t in target.params)
    if given != expected:
        checker.error(op, f"spawn of @{callee.name} passes {given}, it takes {expected}")
    if target.results:
        checker.error(
            op, f"a spawned function returns nothing; @{callee.name} returns {target.results}"
        )
    if op.results[0].type != I64:
        checker.error(op, "spawn gives an i64 handle")


def _verify_join(op: Operation, checker: Checker) -> None:
    if op.operands[0].type != I64 or op.results[0].type != I64:
        checker.error(op, "join takes the i64 handle and gives the thread's i64 status")


def _verify_mutex(op: Operation, checker: Checker) -> None:
    _slot(op, checker, 0, "a mutex")


def _verify_wait(op: Operation, checker: Checker) -> None:
    _slot(op, checker, 0, "a condition")
    _slot(op, checker, 1, "a mutex")


def _verify_notify(op: Operation, checker: Checker) -> None:
    _slot(op, checker, 0, "a condition")


def _verify_barrier(op: Operation, checker: Checker) -> None:
    _slot(op, checker, 0, "a barrier")
    if not isinstance(op.operands[1].type, IntType) or op.operands[1].type != I64:
        checker.error(op, f"a barrier's party count is i64, not {op.operands[1].type}")


def _verify_thread_id(op: Operation, checker: Checker) -> None:
    if op.results[0].type != I64:
        checker.error(op, "thread_id gives an i64")


class ConcurrencyDialect(Dialect):
    name = "concurrency"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        add = registry.add_op
        add(
            OpSpec(
                "concurrency.spawn",
                verify=_verify_spawn,
                results=1,
                required_attributes=("callee",),
            )
        )
        add(OpSpec("concurrency.join", verify=_verify_join, operands=1, results=1))
        add(OpSpec("concurrency.mutex_lock", verify=_verify_mutex, operands=1, results=0))
        add(OpSpec("concurrency.mutex_unlock", verify=_verify_mutex, operands=1, results=0))
        add(OpSpec("concurrency.condition_wait", verify=_verify_wait, operands=2, results=0))
        add(OpSpec("concurrency.condition_notify", verify=_verify_notify, operands=1, results=0))
        add(OpSpec("concurrency.barrier", verify=_verify_barrier, operands=2, results=0))
        add(OpSpec("concurrency.thread_id", verify=_verify_thread_id, operands=0, results=1))


def spawn(
    b: Builder, callee: str, arguments: tuple[Value, ...], *, name: str | None = None
) -> Value:
    return b.create(
        "concurrency.spawn", arguments, (I64,), {"callee": SymbolRef(callee)}, result_names=(name,)
    ).result


def join(b: Builder, handle: Value, *, name: str | None = None) -> Value:
    return b.create("concurrency.join", (handle,), (I64,), result_names=(name,)).result


def mutex_lock(b: Builder, mutex: Value) -> Operation:
    return b.create("concurrency.mutex_lock", (mutex,), ())


def mutex_unlock(b: Builder, mutex: Value) -> Operation:
    return b.create("concurrency.mutex_unlock", (mutex,), ())


def condition_wait(b: Builder, condition: Value, mutex: Value) -> Operation:
    return b.create("concurrency.condition_wait", (condition, mutex), ())


def condition_notify(b: Builder, condition: Value) -> Operation:
    return b.create("concurrency.condition_notify", (condition,), ())


def barrier(b: Builder, slots: Value, parties: Value) -> Operation:
    return b.create("concurrency.barrier", (slots, parties), ())


def thread_id(b: Builder, *, name: str | None = None) -> Value:
    return b.create("concurrency.thread_id", (), (I64,), result_names=(name,)).result
