"""`ppy.concurrent`: threads, and the memory that keeps them apart.

`spawn(f, *args)` starts `f(*args)` on a new thread and hands back its
handle; `join(handle)` waits for it. The synchronization objects are
memory the program owns, so they cross into native code as pointers: a
mutex is one `int` slot (zero unlocked), a condition is one `int` slot
counting notifications, a barrier is two `int` slots. `lock`, `unlock`,
`wait`, `notify`, and `barrier` work on them the same way on every path
-- spinning on atomic operations -- so a program synchronizes identically
under CPython and compiled. `thread_id()` names the running thread.

```python
from ppy import concurrent, native

def fill(p: native.ptr[int], begin: int, end: int) -> None:
    for i in range(begin, end):
        native.store(native.offset(p, i), i * i)

def main() -> int:
    data = native.stack_alloc[int](8)
    first = concurrent.spawn(fill, data, 0, 4)
    second = concurrent.spawn(fill, data, 4, 8)
    concurrent.join(first)
    concurrent.join(second)
    return native.load(native.offset(data, 7))
```
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from . import atomic as _atomic
from ._native_api import Pointer

__all__ = ["barrier", "join", "lock", "notify", "spawn", "thread_id", "unlock", "wait"]


class Thread:
    """A spawned thread: the handle `join` takes."""

    __slots__ = ("_error", "_thread")

    def __init__(self, function: Callable[..., Any], arguments: tuple[Any, ...]) -> None:
        self._error: BaseException | None = None

        def run() -> None:
            try:
                function(*arguments)
            except BaseException as error:  # noqa: BLE001 - re-raised by join
                self._error = error

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def __repr__(self) -> str:
        return f"concurrent.Thread({self._thread.ident})"


def spawn(function: Callable[..., Any], *arguments: Any) -> Thread:
    """Run `function(*arguments)` on a new thread."""
    if not callable(function):
        raise TypeError("concurrent.spawn takes a function")
    return Thread(function, arguments)


def join(handle: Thread) -> None:
    """Wait for the thread; what it raised is raised here."""
    if not isinstance(handle, Thread):
        raise TypeError("concurrent.join takes the handle spawn gave")
    handle._thread.join()
    if handle._error is not None:
        raise handle._error


def _slot(pointer: Any, what: str) -> Pointer[Any]:
    if not isinstance(pointer, Pointer) or pointer.element is not int or not pointer.mutable:
        raise TypeError(f"concurrent.{what} takes a native.ptr[int]")
    return pointer


def lock(mutex: Pointer[Any]) -> None:
    """Take the mutex: spin until its slot goes from 0 to 1."""
    slot = _slot(mutex, "lock")
    while True:
        _found, taken = _atomic.compare_exchange(slot, 0, 1, "acq_rel")
        if taken:
            return
        threading.Event().wait(0)


def unlock(mutex: Pointer[Any]) -> None:
    _atomic.store(_slot(mutex, "unlock"), 0, "release")


def wait(condition: Pointer[Any], mutex: Pointer[Any]) -> None:
    """Release the mutex, wait for a notification after this call, take it back."""
    slot = _slot(condition, "wait")
    generation = _atomic.load(slot, "acquire")
    unlock(mutex)
    while _atomic.load(slot, "acquire") == generation:
        threading.Event().wait(0)
    lock(mutex)


def notify(condition: Pointer[Any]) -> None:
    """Wake every waiter on the condition."""
    _atomic.fetch_add(_slot(condition, "notify"), 1, "acq_rel")


def barrier(slots: Pointer[Any], parties: int) -> None:
    """Wait until `parties` threads have arrived; `slots` is two `int`s."""
    arrivals = _slot(slots, "barrier")
    generation_slot = Pointer(arrivals.memory, arrivals.index + 1, arrivals.element)
    generation = _atomic.load(generation_slot, "acquire")
    arrived = _atomic.fetch_add(arrivals, 1, "acq_rel") + 1
    if arrived == parties:
        _atomic.store(arrivals, 0, "release")
        _atomic.fetch_add(generation_slot, 1, "acq_rel")
        return
    while _atomic.load(generation_slot, "acquire") == generation:
        threading.Event().wait(0)


def thread_id() -> int:
    return threading.get_ident()
