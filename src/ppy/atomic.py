"""`ppy.atomic`: shared memory, one operation at a time.

Every function takes a `native.ptr[T]` to the slot and a memory order,
`"seq_cst"` unless said otherwise (`"relaxed"`, `"acquire"`, `"release"`,
`"acq_rel"`). The compiler lowers each to the atomic instruction of that
order; the reference implementation here serializes them under one lock,
which is what an atomic operation on an interpreter with a global lock
already is.

```python
from ppy import atomic, native

counter = native.stack_alloc[int](1)
atomic.fetch_add(counter, 1)
old, swapped = atomic.compare_exchange(counter, 1, 5)
```
"""

from __future__ import annotations

import threading
from typing import Any

from ._native_api import Pointer

__all__ = [
    "ORDERS",
    "compare_exchange",
    "exchange",
    "fence",
    "fetch_add",
    "fetch_and",
    "fetch_or",
    "fetch_sub",
    "fetch_xor",
    "load",
    "store",
]

ORDERS = ("relaxed", "acquire", "release", "acq_rel", "seq_cst")
_LOCK = threading.Lock()


def _check(pointer: Any, order: str, what: str) -> None:
    if not isinstance(pointer, Pointer):
        raise TypeError(f"atomic.{what} takes a native.ptr")
    if order not in ORDERS:
        raise ValueError(f"atomic.{what}: order is one of {', '.join(ORDERS)}, not {order!r}")


def _wrap(pointer: Pointer[Any], value: Any) -> Any:
    """What the slot holds after a write of `value`: the width's wrap."""
    if pointer.element in {float, float} or pointer.memory.typecode in "fd":
        return float(value)
    width = 8 * pointer.memory.itemsize
    signed = pointer.memory.typecode.islower()
    value = int(value) & ((1 << width) - 1)
    if signed and value >= 1 << (width - 1):
        value -= 1 << width
    return value


def load(pointer: Pointer[Any], order: str = "seq_cst") -> Any:
    _check(pointer, order, "load")
    with _LOCK:
        return pointer.memory[pointer.index]


def store(pointer: Pointer[Any], value: Any, order: str = "seq_cst") -> None:
    _check(pointer, order, "store")
    with _LOCK:
        pointer.memory[pointer.index] = _wrap(pointer, value)


def exchange(pointer: Pointer[Any], value: Any, order: str = "seq_cst") -> Any:
    _check(pointer, order, "exchange")
    with _LOCK:
        old = pointer.memory[pointer.index]
        pointer.memory[pointer.index] = _wrap(pointer, value)
        return old


def compare_exchange(
    pointer: Pointer[Any], expected: Any, desired: Any, order: str = "seq_cst"
) -> tuple[Any, bool]:
    """(the value found, whether it was `expected` and is now `desired`)."""
    _check(pointer, order, "compare_exchange")
    with _LOCK:
        old = pointer.memory[pointer.index]
        if old == expected:
            pointer.memory[pointer.index] = _wrap(pointer, desired)
            return old, True
        return old, False


def _fetch(name: str, pointer: Pointer[Any], value: Any, order: str, combine: Any) -> Any:
    _check(pointer, order, name)
    with _LOCK:
        old = pointer.memory[pointer.index]
        pointer.memory[pointer.index] = _wrap(pointer, combine(old, value))
        return old


def fetch_add(pointer: Pointer[Any], value: int, order: str = "seq_cst") -> int:
    return _fetch("fetch_add", pointer, value, order, lambda a, b: a + b)


def fetch_sub(pointer: Pointer[Any], value: int, order: str = "seq_cst") -> int:
    return _fetch("fetch_sub", pointer, value, order, lambda a, b: a - b)


def fetch_and(pointer: Pointer[Any], value: int, order: str = "seq_cst") -> int:
    return _fetch("fetch_and", pointer, value, order, lambda a, b: a & b)


def fetch_or(pointer: Pointer[Any], value: int, order: str = "seq_cst") -> int:
    return _fetch("fetch_or", pointer, value, order, lambda a, b: a | b)


def fetch_xor(pointer: Pointer[Any], value: int, order: str = "seq_cst") -> int:
    return _fetch("fetch_xor", pointer, value, order, lambda a, b: a ^ b)


def fence(order: str = "seq_cst") -> None:
    if order not in ORDERS or order == "relaxed":
        raise ValueError(f"atomic.fence: order is one of {', '.join(ORDERS[1:])}, not {order!r}")
    with _LOCK:
        pass
