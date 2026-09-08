"""Arrow arrays across the native boundary: the C Data Interface, without a copy.

An Arrow array is handed to native code as the `ArrowArray` struct the
Arrow C Data Interface defines -- length, null count, offset, and the
buffers -- which `arrow.import` in the IR reads directly. No `PyObject`
crosses the boundary (spec 54). The producer keeps ownership: the struct's
`release` callback is called when the borrow ends, and never twice.

```python
from ppy_runtime.arrow import exported

with exported(array) as struct:      # the address of an ArrowArray
    native(struct, ...)
```
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = ["ArrowArray", "ArrowSchema", "exported", "release"]

_RELEASE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class ArrowArray(ctypes.Structure):
    """`struct ArrowArray` of the Arrow C Data Interface."""

    _fields_ = [
        ("length", ctypes.c_int64),
        ("null_count", ctypes.c_int64),
        ("offset", ctypes.c_int64),
        ("n_buffers", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("buffers", ctypes.POINTER(ctypes.c_void_p)),
        ("children", ctypes.c_void_p),
        ("dictionary", ctypes.c_void_p),
        ("release", _RELEASE),
        ("private_data", ctypes.c_void_p),
    ]


class ArrowSchema(ctypes.Structure):
    """`struct ArrowSchema` of the Arrow C Data Interface."""

    _fields_ = [
        ("format", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
        ("metadata", ctypes.c_char_p),
        ("flags", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("children", ctypes.c_void_p),
        ("dictionary", ctypes.c_void_p),
        ("release", _RELEASE),
        ("private_data", ctypes.c_void_p),
    ]


def release(struct: ArrowArray | ArrowSchema) -> None:
    """Give the producer its memory back, once; a released struct is inert."""
    if struct.release:
        struct.release(ctypes.addressof(struct))


@contextmanager
def exported(array: Any) -> Iterator[int]:
    """Borrow `array` (a `pyarrow.Array`) as an `ArrowArray` struct: its address.

    The array's buffers are shared, not copied; the struct is released when
    the block ends, whatever happened inside it.
    """
    struct = ArrowArray()
    schema = ArrowSchema()
    array._export_to_c(ctypes.addressof(struct), ctypes.addressof(schema))
    try:
        yield ctypes.addressof(struct)
    finally:
        release(struct)
        release(schema)
