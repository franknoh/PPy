"""Allocating a buffer, in the one spelling every path understands.

`ppy.buffer[int](n)` is `array.array("q", bytes(8 * n))` under CPython and a
zeroed native allocation in a standalone build, where there is no
`array.array` to make. Writing it this way is what lets the same source run
both places.
"""

from __future__ import annotations

import array as _array

from ._markers import i8, u8

#: The `array` type code for each element type, and how wide one is. A byte
#: element is a byte in memory; everything else is a machine word.
_CODES = {int: ("q", 8), float: ("d", 8), i8: ("b", 1), u8: ("B", 1)}


class _Allocating:
    """One `ppy.buffer[T]`, waiting for how many elements to make."""

    __slots__ = ("_spec",)

    def __init__(self, spec) -> None:  # type: ignore[no-untyped-def]
        self._spec = spec

    def __repr__(self) -> str:
        return f"ppy.buffer[{getattr(self._spec, '__name__', self._spec)!r}]"

    def __call__(self, count: int):  # type: ignore[no-untyped-def]
        described = _CODES.get(self._spec)
        if described is None:
            raise TypeError(f"{self._spec!r} is not an element type a buffer can hold")
        code, width = described
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError("a buffer is made with how many elements it holds")
        if count < 0:
            raise ValueError("a buffer cannot hold fewer than no elements")
        return _array.array(code, bytes(width * count))


class _TypedBuffer:
    """`ppy.buffer[T](n)`: `n` elements of `T`, all zero.

    ```python
    tree = ppy.buffer[int](1 << 18)
    weights = ppy.buffer[float](rows * cols)
    text = ppy.buffer[ppy.i8](length)     # one byte per element
    ```
    """

    __slots__ = ()

    def __getitem__(self, spec) -> _Allocating:  # type: ignore[no-untyped-def]
        return _Allocating(spec)

    def __call__(self, *_args, **_keywords):  # type: ignore[no-untyped-def]
        raise TypeError("`ppy.buffer` needs its element type: `ppy.buffer[int](n)`")


buffer = _TypedBuffer()
