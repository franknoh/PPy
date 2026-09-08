"""The async dialect: coroutines as explicit state, and the IO that suspends them (spec 76).

An `async` IR function -- one carrying `ppy.async = true` -- is lowered
from a Python `async def` and returns its value through a `future<T>`.
Before the lowering it awaits: `async.create @f(%args)` starts another
coroutine and hands back its future, `async.await %fut` waits for one, and
`sleep`, `accept`, `connect`, `read`, and `write` are the operations of the
runtime that complete later, each a future; `start` runs a created coroutine
without waiting for it, as a task. `listen`, `port`, and `close` are
immediate. The `LowerAsync` pass then makes every coroutine a state
machine: a starter that allocates a frame of `slots` words, stores the
arguments, and `spawn`s it; and a resume function that reads its state,
runs to its next `suspend`, and at the end `complete`s or `fail`s its
future. `result` reads the value the awaited future carried. The runtime
(`ppy_runtime.aio`) owns frames and futures and drives the loop; a
backend lowers the low-level operations to its calls, and never invents
blocking (spec 78).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Builder, IRFunction, Operation, SymbolRef, Value
from ..types import F64, I64, VOID, FutureType, IntType, IRType, PtrType, VoidType, is_scalar

if TYPE_CHECKING:
    from ..verify import Checker

__all__ = [
    "IO_FUTURES",
    "AsyncDialect",
    "accept",
    "await_",
    "close",
    "complete",
    "connect",
    "create",
    "fail",
    "frame_new",
    "is_async",
    "listen",
    "mark_async",
    "port",
    "read",
    "result",
    "sleep",
    "spawn",
    "start",
    "suspend",
    "write",
]

#: The runtime operations that complete later, and what their futures carry.
IO_FUTURES: dict[str, tuple[tuple[IRType, ...], IRType]] = {
    "sleep": ((F64,), VOID),
    "accept": ((I64,), I64),
    "connect": ((PtrType(IntType(8, False), mutable=False), I64, I64), I64),
    "read": ((I64, PtrType(IntType(8, False)), I64), I64),
    "write": ((I64, PtrType(IntType(8, False), mutable=False), I64), I64),
}
#: The header words of every frame, the runtime's: its future, the awaited one, the state.
HEADER_SLOTS = 3


def is_async(function: IRFunction) -> bool:
    return bool(function.attributes.get("ppy.async", False))


def mark_async(function: IRFunction) -> IRFunction:
    function.attributes["ppy.async"] = True
    return function


def _in_async(op: Operation, checker: Checker) -> bool:
    function = checker.function
    if function is None or is_async(function):
        return True
    checker.error(op, f"{op.name} suspends; only an async function (`ppy.async`) may")
    return False


def _future_of(op: Operation, checker: Checker, inner: IRType) -> None:
    result = op.results[0].type
    if not isinstance(result, FutureType) or result.inner != inner:
        checker.error(op, f"{op.name} gives future<{inner}>, not {result}")


def _verify_create(op: Operation, checker: Checker) -> None:
    callee = op.attributes.get("callee")
    if not isinstance(callee, SymbolRef):
        checker.error(op, "create needs a `callee` symbol")
        return
    module = checker.module
    if module is None:
        return
    target = module.functions.get(callee.name)
    if target is None:
        checker.error(op, f"create of undefined @{callee.name}")
        return
    if not is_async(target):
        checker.error(op, f"@{callee.name} is not async; call it")
        return
    given = tuple(v.type for v in op.operands)
    expected = tuple(t for _name, t in target.params)
    if given != expected:
        checker.error(
            op,
            f"@{callee.name} takes ({', '.join(map(str, expected))}), "
            f"created with ({', '.join(map(str, given))})",
        )
    inner = target.results[0] if target.results else VOID
    _future_of(op, checker, inner)


def _verify_await(op: Operation, checker: Checker) -> None:
    if not _in_async(op, checker):
        return
    future = op.operands[0].type
    if not isinstance(future, FutureType):
        checker.error(op, f"await takes a future, not {future}")
        return
    if isinstance(future.inner, VoidType):
        if op.results:
            checker.error(op, "awaiting future<void> gives nothing")
    elif len(op.results) != 1 or op.results[0].type != future.inner:
        checker.error(op, f"awaiting {future} gives {future.inner}")


def _fits(given: IRType, wanted: IRType) -> bool:
    """A pointer fits by what it points at and whether it may write; its space is any."""
    if isinstance(wanted, PtrType):
        return (
            isinstance(given, PtrType)
            and given.pointee == wanted.pointee
            and (given.mutable or not wanted.mutable)
        )
    return given == wanted


def _verify_io(name: str):  # type: ignore[no-untyped-def]
    operands, inner = IO_FUTURES[name]

    def verify(op: Operation, checker: Checker) -> None:
        given = tuple(v.type for v in op.operands)
        if len(given) != len(operands) or not all(
            _fits(g, w) for g, w in zip(given, operands, strict=True)
        ):
            checker.error(
                op,
                f"async.{name} takes ({', '.join(map(str, operands))}), "
                f"not ({', '.join(map(str, given))})",
            )
        _future_of(op, checker, inner)

    return verify


def _verify_listen(op: Operation, checker: Checker) -> None:
    given = tuple(v.type for v in op.operands)
    wanted = (PtrType(IntType(8, False), mutable=False), I64, I64, I64)
    if len(given) != 4 or not all(_fits(g, w) for g, w in zip(given, wanted, strict=True)):
        checker.error(op, f"async.listen takes (host, length, port, backlog), not {given}")
    if op.results[0].type != I64:
        checker.error(op, "async.listen gives the socket as i64")


def _verify_start(op: Operation, checker: Checker) -> None:
    if not isinstance(op.operands[0].type, FutureType):
        checker.error(op, f"start takes a future, not {op.operands[0].type}")


def _verify_port(op: Operation, checker: Checker) -> None:
    if op.operands[0].type != I64 or op.results[0].type != I64:
        checker.error(op, "async.port takes a socket (i64) and gives its port (i64)")


def _verify_close(op: Operation, checker: Checker) -> None:
    if op.operands[0].type != I64:
        checker.error(op, f"async.close takes a socket (i64), not {op.operands[0].type}")


def _frame(op: Operation, checker: Checker, operand: Value) -> bool:
    if operand.type != PtrType(I64):
        checker.error(op, f"{op.name} works on a frame (ptr<i64>), not {operand.type}")
        return False
    return True


def _verify_frame_new(op: Operation, checker: Checker) -> None:
    slots = op.attributes.get("slots")
    if isinstance(slots, bool) or not isinstance(slots, int) or slots < HEADER_SLOTS:
        checker.error(op, f"a frame has at least {HEADER_SLOTS} slots, not {slots!r}")
    if op.results[0].type != PtrType(I64):
        checker.error(op, "frame_new gives ptr<i64>")


def _verify_spawn(op: Operation, checker: Checker) -> None:
    callee = op.attributes.get("callee")
    if not isinstance(callee, SymbolRef):
        checker.error(op, "spawn needs the resume function as `callee`")
        return
    _frame(op, checker, op.operands[0])
    if not isinstance(op.results[0].type, FutureType):
        checker.error(op, f"spawn gives a future, not {op.results[0].type}")
    module = checker.module
    if module is None:
        return
    target = module.functions.get(callee.name)
    if target is None:
        checker.error(op, f"spawn of undefined @{callee.name}")
    elif tuple(t for _n, t in target.params) != (PtrType(I64),) or target.results:
        checker.error(op, f"@{callee.name} resumes a frame: it takes ptr<i64> and returns nothing")


def _verify_suspend(op: Operation, checker: Checker) -> None:
    if _frame(op, checker, op.operands[0]) and not isinstance(op.operands[1].type, FutureType):
        checker.error(op, f"suspend waits for a future, not {op.operands[1].type}")


def _verify_result(op: Operation, checker: Checker) -> None:
    if _frame(op, checker, op.operands[0]) and not is_scalar(op.results[0].type):
        checker.error(op, f"result reads a scalar, not {op.results[0].type}")


def _verify_complete(op: Operation, checker: Checker) -> None:
    if not _frame(op, checker, op.operands[0]):
        return
    if len(op.operands) > 1 and not is_scalar(op.operands[1].type):
        checker.error(op, f"complete carries a scalar, not {op.operands[1].type}")


def _verify_fail(op: Operation, checker: Checker) -> None:
    _frame(op, checker, op.operands[0])
    if not isinstance(op.attributes.get("label"), str):
        checker.error(op, "fail names what failed in `label`")
    code = op.attributes.get("code")
    if isinstance(code, bool) or not isinstance(code, int) or code < 1:
        checker.error(op, "fail numbers its site in `code`, from 1")


class AsyncDialect(Dialect):
    name = "async"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        add = registry.add_op
        add(
            OpSpec(
                "async.create",
                verify=_verify_create,
                operands=None,
                results=1,
                required_attributes=("callee",),
            )
        )
        add(OpSpec("async.await", verify=_verify_await, operands=1, results=None))
        for name, (operands, _inner) in IO_FUTURES.items():
            add(OpSpec(f"async.{name}", verify=_verify_io(name), operands=len(operands), results=1))
        add(OpSpec("async.listen", verify=_verify_listen, operands=4, results=1))
        add(OpSpec("async.start", verify=_verify_start, operands=1, results=0))
        add(OpSpec("async.port", pure=True, verify=_verify_port, operands=1, results=1))
        add(OpSpec("async.close", verify=_verify_close, operands=1, results=0))
        add(
            OpSpec(
                "async.frame_new",
                verify=_verify_frame_new,
                operands=0,
                results=1,
                required_attributes=("slots",),
            )
        )
        add(
            OpSpec(
                "async.spawn",
                verify=_verify_spawn,
                operands=1,
                results=1,
                required_attributes=("callee",),
            )
        )
        add(OpSpec("async.suspend", verify=_verify_suspend, operands=2, results=0))
        add(OpSpec("async.result", verify=_verify_result, operands=1, results=1))
        add(OpSpec("async.complete", verify=_verify_complete, operands=None, results=0))
        add(
            OpSpec(
                "async.fail",
                verify=_verify_fail,
                operands=1,
                results=0,
                required_attributes=("label", "code"),
            )
        )


# -- builders --------------------------------------------------------------------


def create(
    b: Builder, callee: str, arguments: tuple[Value, ...], inner: IRType, name=None
) -> Value:  # type: ignore[no-untyped-def]
    """Start coroutine `@callee(arguments)`; its future carries `inner`."""
    return b.create(
        "async.create",
        arguments,
        (FutureType(inner),),
        {"callee": SymbolRef(callee)},
        result_names=(name,),
    ).result


def await_(b: Builder, future: Value, name: str | None = None) -> Value | None:
    """Wait for `future`; its value, or None for a future<void>."""
    assert isinstance(future.type, FutureType)
    if isinstance(future.type.inner, VoidType):
        b.create("async.await", (future,))
        return None
    return b.create("async.await", (future,), (future.type.inner,), result_names=(name,)).result


def _io(b: Builder, name: str, operands: tuple[Value, ...], hint: str | None) -> Value:
    _operands, inner = IO_FUTURES[name]
    return b.create(f"async.{name}", operands, (FutureType(inner),), result_names=(hint,)).result


def sleep(b: Builder, seconds: Value, name: str | None = None) -> Value:
    return _io(b, "sleep", (seconds,), name)


def accept(b: Builder, socket: Value, name: str | None = None) -> Value:
    return _io(b, "accept", (socket,), name)


def connect(b: Builder, host: Value, length: Value, port: Value, name: str | None = None) -> Value:
    return _io(b, "connect", (host, length, port), name)


def read(b: Builder, socket: Value, buffer: Value, count: Value, name: str | None = None) -> Value:
    return _io(b, "read", (socket, buffer, count), name)


def write(b: Builder, socket: Value, buffer: Value, count: Value, name: str | None = None) -> Value:
    return _io(b, "write", (socket, buffer, count), name)


def listen(
    b: Builder, host: Value, length: Value, port: Value, backlog: Value, name: str | None = None
) -> Value:
    """A listening socket bound to `host:port`, as i64; negative when it failed."""
    return b.create(
        "async.listen", (host, length, port, backlog), (I64,), result_names=(name,)
    ).result


def start(b: Builder, future: Value) -> Operation:
    """Run the coroutine behind `future` from now on, without waiting for it."""
    return b.create("async.start", (future,))


def port(b: Builder, socket: Value, name: str | None = None) -> Value:
    """The port a socket is bound to, as i64; negative when it failed."""
    return b.create("async.port", (socket,), (I64,), result_names=(name,)).result


def close(b: Builder, socket: Value) -> Operation:
    return b.create("async.close", (socket,))


def frame_new(b: Builder, slots: int, name: str | None = None) -> Value:
    """A frame of `slots` words, the first `HEADER_SLOTS` the runtime's."""
    return b.create(
        "async.frame_new", (), (PtrType(I64),), {"slots": slots}, result_names=(name,)
    ).result


def spawn(b: Builder, resume: str, frame: Value, inner: IRType, name: str | None = None) -> Value:
    """Schedule `frame` to run through `@resume`; the future its completion fulfils."""
    return b.create(
        "async.spawn",
        (frame,),
        (FutureType(inner),),
        {"callee": SymbolRef(resume)},
        result_names=(name,),
    ).result


def suspend(b: Builder, frame: Value, future: Value) -> Operation:
    """Resume `frame` once `future` completes; the resume function returns after this."""
    return b.create("async.suspend", (frame, future))


def result(b: Builder, frame: Value, t: IRType, name: str | None = None) -> Value:
    """What the future `frame` last awaited carried."""
    return b.create("async.result", (frame,), (t,), result_names=(name,)).result


def complete(b: Builder, frame: Value, value: Value | None = None) -> Operation:
    return b.create("async.complete", (frame,) if value is None else (frame, value))


def fail(b: Builder, frame: Value, label: str, code: int) -> Operation:
    """Guard `label` (site `code`) failed: the future fails, and nothing falls back."""
    return b.create("async.fail", (frame,), (), {"label": label, "code": code})
