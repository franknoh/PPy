"""`ppy.simd`: a few scalars operated on at once.

`simd.Vector[T, N]` is `N` lanes of `T`. `splat[T, N](x)` fills one,
`load[T, N](p)` reads one from a `native.ptr[T]`, `store(v, p)` writes it
back, `extract(v, i)` and `insert(v, i, x)` reach one lane, `shuffle(a,
b, mask)` builds a vector from the lanes of two, and `reduce_add`,
`reduce_min`, `reduce_max` fold one to a scalar in lane order. `+ - * /`,
the comparisons, `& | ^`, and `select(mask, a, b)` work lane by lane;
integer lanes wrap at their width, which is what the machine's lanes do.

```python
from ppy import native, simd

def dot4(a: native.ptr[float], b: native.ptr[float]) -> float:
    return simd.reduce_add(simd.load[float, 4](a) * simd.load[float, 4](b))
```

Under CPython every operation runs here, lane by lane, on the same
`native.Pointer` memory; the compiler lowers them to vector instructions.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterator
from typing import Annotated, Any

from ._markers import f32, f64, i8, i16, i32, i64, u8, u16, u32, u64
from ._native_api import Pointer

__all__ = [
    "Vector",
    "extract",
    "insert",
    "load",
    "reduce_add",
    "reduce_max",
    "reduce_min",
    "select",
    "shuffle",
    "splat",
    "store",
]

#: Each lane type's width and signedness; floats have no wrap.
_LANES: dict[Any, tuple[int, bool] | None] = {
    int: (64, True),
    bool: None,
    float: None,
    i8: (8, True),
    u8: (8, False),
    i16: (16, True),
    u16: (16, False),
    i32: (32, True),
    u32: (32, False),
    i64: (64, True),
    u64: (64, False),
    f32: None,
    f64: None,
}


def _wrap(element: Any, value: Any) -> Any:
    """`value` as the lane holds it: wrapped to the width, or a float, or a bool."""
    if element is bool:
        return bool(value)
    layout = _LANES.get(element)
    if layout is None:
        return float(value)
    width, signed = layout
    value = int(value) & ((1 << width) - 1)
    if signed and value >= 1 << (width - 1):
        value -= 1 << width
    return value


class _VectorSpec:
    __slots__ = ("count", "element")

    def __init__(self, element: Any, count: int) -> None:
        self.element = element
        self.count = count


class Vector:
    """`N` lanes of one scalar type: the reference implementation."""

    __slots__ = ("element", "lanes")

    def __init__(self, element: Any, lanes: tuple[Any, ...]) -> None:
        if element not in _LANES:
            raise TypeError(f"{element!r} is not a lane type")
        if not lanes:
            raise ValueError("a vector has at least one lane")
        self.element = element
        self.lanes = tuple(_wrap(element, v) for v in lanes)

    def __class_getitem__(cls, item: Any) -> Any:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("simd.Vector takes two parameters: Vector[T, N]")
        element, count = item
        if element not in _LANES or not isinstance(count, int) or count < 1:
            raise TypeError("simd.Vector[T, N]: T a scalar, N a positive count")
        return Annotated[Vector, _VectorSpec(element, count)]

    @property
    def count(self) -> int:
        return len(self.lanes)

    def __len__(self) -> int:
        return len(self.lanes)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.lanes)

    def __repr__(self) -> str:
        name = getattr(self.element, "__name__", repr(self.element))
        return f"simd.Vector[{name}, {len(self.lanes)}]{self.lanes}"

    def __eq__(self, other: object) -> Any:
        if isinstance(other, Vector):
            return self._compare(other, operator.eq)
        return NotImplemented

    def __ne__(self, other: object) -> Any:
        if isinstance(other, Vector):
            return self._compare(other, operator.ne)
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def _lanes_of(self, other: Any, what: str) -> tuple[Any, ...]:
        if not isinstance(other, Vector):
            raise TypeError(f"vector {what} takes two vectors")
        if other.element is not self.element or len(other.lanes) != len(self.lanes):
            raise TypeError(f"vector {what} takes two vectors of one type")
        return other.lanes

    def _arith(self, other: Any, function: Callable[[Any, Any], Any], what: str) -> Vector:
        if self.element is bool:
            raise TypeError(f"vector {what} takes numbers, not bools")
        lanes = self._lanes_of(other, what)
        return Vector(
            self.element, tuple(function(a, b) for a, b in zip(self.lanes, lanes, strict=True))
        )

    def _compare(self, other: Any, function: Callable[[Any, Any], bool]) -> Vector:
        lanes = self._lanes_of(other, "comparison")
        return Vector(bool, tuple(function(a, b) for a, b in zip(self.lanes, lanes, strict=True)))

    def _bitwise(self, other: Any, function: Callable[[Any, Any], Any], what: str) -> Vector:
        if _LANES.get(self.element) is None and self.element is not bool:
            raise TypeError(f"vector {what} takes integers or bools")
        lanes = self._lanes_of(other, what)
        return Vector(
            self.element, tuple(function(a, b) for a, b in zip(self.lanes, lanes, strict=True))
        )

    def __add__(self, other: Any) -> Vector:
        return self._arith(other, operator.add, "+")

    def __sub__(self, other: Any) -> Vector:
        return self._arith(other, operator.sub, "-")

    def __mul__(self, other: Any) -> Vector:
        return self._arith(other, operator.mul, "*")

    def __truediv__(self, other: Any) -> Vector:
        if _LANES.get(self.element) is not None:
            raise TypeError("vector / takes floats; integer lanes have no division yet")
        return self._arith(other, operator.truediv, "/")

    def __neg__(self) -> Vector:
        if self.element is bool:
            raise TypeError("vector - takes numbers, not bools")
        return Vector(self.element, tuple(-v for v in self.lanes))

    def __and__(self, other: Any) -> Vector:
        return self._bitwise(other, operator.and_, "&")

    def __or__(self, other: Any) -> Vector:
        return self._bitwise(other, operator.or_, "|")

    def __xor__(self, other: Any) -> Vector:
        return self._bitwise(other, operator.xor, "^")

    def __lt__(self, other: Any) -> Vector:
        return self._compare(other, operator.lt)

    def __le__(self, other: Any) -> Vector:
        return self._compare(other, operator.le)

    def __gt__(self, other: Any) -> Vector:
        return self._compare(other, operator.gt)

    def __ge__(self, other: Any) -> Vector:
        return self._compare(other, operator.ge)


class _Splat:
    """`simd.splat[T, N](x)`: every lane `x`."""

    __slots__ = ()

    def __getitem__(self, item: Any) -> Callable[[Any], Vector]:
        element, count = _parameters(item, "splat")
        return lambda value: Vector(element, (value,) * count)


class _Load:
    """`simd.load[T, N](p)`: the `N` elements at `p`."""

    __slots__ = ()

    def __getitem__(self, item: Any) -> Callable[[Pointer[Any]], Vector]:
        element, count = _parameters(item, "load")

        def read(pointer: Pointer[Any]) -> Vector:
            if not isinstance(pointer, Pointer):
                raise TypeError("simd.load reads through a native.ptr")
            if pointer.element is not element:
                raise TypeError(
                    f"simd.load[{element!r}, {count}] through a pointer to {pointer.element!r}"
                )
            lanes = tuple(pointer.memory[pointer.index + i] for i in range(count))
            return Vector(element, lanes)

        return read


def _parameters(item: Any, what: str) -> tuple[Any, int]:
    if not isinstance(item, tuple) or len(item) != 2:
        raise TypeError(f"simd.{what}[T, N] takes the lane type and the count")
    element, count = item
    if element not in _LANES or not isinstance(count, int) or count < 1:
        raise TypeError(f"simd.{what}[T, N]: T a scalar, N a positive count")
    return element, count


splat = _Splat()
load = _Load()


def store(vector: Vector, pointer: Pointer[Any]) -> None:
    """Write the lanes to the `count` elements at `pointer`."""
    if not isinstance(vector, Vector) or not isinstance(pointer, Pointer):
        raise TypeError("simd.store takes a vector and a native.ptr")
    if not pointer.mutable:
        raise TypeError("simd.store cannot write through a const_ptr")
    if pointer.element is not vector.element:
        raise TypeError(
            f"simd.store of {vector.element!r} lanes through a pointer to {pointer.element!r}"
        )
    for offset, value in enumerate(vector.lanes):
        pointer.memory[pointer.index + offset] = int(value) if vector.element is bool else value


def extract(vector: Vector, index: int) -> Any:
    if not isinstance(vector, Vector):
        raise TypeError("simd.extract takes a vector")
    if not 0 <= index < len(vector.lanes):
        raise IndexError(f"lane {index} of {len(vector.lanes)}")
    return vector.lanes[index]


def insert(vector: Vector, index: int, value: Any) -> Vector:
    if not isinstance(vector, Vector):
        raise TypeError("simd.insert takes a vector")
    if not 0 <= index < len(vector.lanes):
        raise IndexError(f"lane {index} of {len(vector.lanes)}")
    lanes = list(vector.lanes)
    lanes[index] = value
    return Vector(vector.element, tuple(lanes))


def shuffle(a: Vector, b: Vector, mask: tuple[int, ...]) -> Vector:
    """Lanes `mask[i]` of `a` followed by `b`: `0..N-1` from `a`, `N..2N-1` from `b`."""
    if not isinstance(a, Vector) or not isinstance(b, Vector):
        raise TypeError("simd.shuffle takes two vectors")
    pool = a.lanes + a._lanes_of(b, "shuffle")
    if not mask or any(not isinstance(m, int) or not 0 <= m < len(pool) for m in mask):
        raise IndexError(f"a shuffle lane is in [0, {len(pool)})")
    return Vector(a.element, tuple(pool[m] for m in mask))


def _reduce(vector: Vector, what: str) -> Any:
    if not isinstance(vector, Vector):
        raise TypeError(f"simd.reduce_{what} takes a vector")
    if vector.element is bool:
        raise TypeError(f"simd.reduce_{what} takes numbers, not bools")
    accumulator = vector.lanes[0]
    for lane in vector.lanes[1:]:
        if what == "add":
            accumulator = _wrap(vector.element, accumulator + lane)
        elif what == "min":
            accumulator = min(accumulator, lane)
        else:
            accumulator = max(accumulator, lane)
    return accumulator


def reduce_add(vector: Vector) -> Any:
    """The lanes summed in order, first to last."""
    return _reduce(vector, "add")


def reduce_min(vector: Vector) -> Any:
    return _reduce(vector, "min")


def reduce_max(vector: Vector) -> Any:
    return _reduce(vector, "max")


def select(mask: Vector, a: Vector, b: Vector) -> Vector:
    """Lane by lane, `a` where `mask` is true, else `b`."""
    if not isinstance(mask, Vector) or mask.element is not bool:
        raise TypeError("simd.select takes a bool vector mask")
    if not isinstance(a, Vector):
        raise TypeError("simd.select takes two vectors to choose from")
    lanes = a._lanes_of(b, "select")
    if len(mask.lanes) != len(a.lanes):
        raise TypeError("simd.select: the mask has the vectors' lane count")
    return Vector(
        a.element, tuple(x if m else y for m, x, y in zip(mask.lanes, a.lanes, lanes, strict=True))
    )
