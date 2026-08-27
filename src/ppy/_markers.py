"""PPY annotation markers: fixed-width numerics, containers, and refinements.

Every marker is an ordinary `typing.Annotated` alias, so a `.ppy` module keeps
working under plain CPython and under any Python-aware editor.
"""

from __future__ import annotations

from typing import Annotated, Any, TypeVar

__all__ = [
    "IntWidth",
    "FloatWidth",
    "ArraySpec",
    "VectorSpec",
    "BufferSpec",
    "Range",
    "Length",
    "NoAlias",
    "Shape",
    "DType",
    "Contiguous",
    "Array",
    "Vector",
    "Buffer",
    "i8", "i16", "i32", "i64",
    "u8", "u16", "u32", "u64",
    "f16", "f32", "f64",
    "NUMERIC_MARKERS",
]

_T = TypeVar("_T")


class _Meta:
    __slots__ = ()
    _fields: tuple[str, ...] = ()

    def __repr__(self) -> str:
        args = ", ".join(f"{f}={getattr(self, f)!r}" for f in self._fields)
        return f"ppy.{type(self).__name__}({args})"

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return all(getattr(self, f) == getattr(other, f) for f in self._fields)

    def __hash__(self) -> int:
        return hash((type(self).__name__, *(getattr(self, f) for f in self._fields)))


class IntWidth(_Meta):
    """Fixed-width integer representation request and range contract."""

    __slots__ = ("bits", "signed")
    _fields = ("bits", "signed")

    def __init__(self, bits: int, signed: bool) -> None:
        self.bits = bits
        self.signed = signed

    @property
    def low(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def high(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1

    @property
    def name(self) -> str:
        return f"{'i' if self.signed else 'u'}{self.bits}"


class FloatWidth(_Meta):
    """Explicit floating-point precision contract."""

    __slots__ = ("bits",)
    _fields = ("bits",)

    def __init__(self, bits: int) -> None:
        self.bits = bits

    @property
    def name(self) -> str:
        return f"f{self.bits}"


class ArraySpec(_Meta):
    __slots__ = ("element", "length")
    _fields = ("element", "length")

    def __init__(self, element: Any, length: Any) -> None:
        self.element = element
        self.length = length


class VectorSpec(_Meta):
    __slots__ = ("element",)
    _fields = ("element",)

    def __init__(self, element: Any) -> None:
        self.element = element


class BufferSpec(_Meta):
    __slots__ = ("element",)
    _fields = ("element",)

    def __init__(self, element: Any) -> None:
        self.element = element


class Range(_Meta):
    """Refinement: `low <= value <= high`."""

    __slots__ = ("low", "high")
    _fields = ("low", "high")

    def __init__(self, low: int | float, high: int | float) -> None:
        self.low = low
        self.high = high


class Length(_Meta):
    """Refinement: `len(value) == size`."""

    __slots__ = ("size",)
    _fields = ("size",)

    def __init__(self, size: int) -> None:
        self.size = size


class NoAlias(_Meta):
    """Caller obligation: this argument does not alias any other argument."""

    __slots__ = ()
    _fields = ()


class Shape(_Meta):
    """Refinement: array shape, with `str` entries naming symbolic dimensions."""

    __slots__ = ("dims",)
    _fields = ("dims",)

    def __init__(self, *dims: int | str) -> None:
        self.dims = tuple(dims)


class DType(_Meta):
    """Refinement: the array's element type, named as the library spells it."""

    __slots__ = ("name",)
    _fields = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


class Contiguous(_Meta):
    """Refinement: the buffer is C-contiguous."""

    __slots__ = ()
    _fields = ()


class Array:
    """`Array[T, N]`: fixed-length homogeneous value container (tuple-like)."""

    def __class_getitem__(cls, params: Any) -> Any:
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("ppy.Array requires two parameters: Array[T, N]")
        element, length = params
        return Annotated[tuple[element, ...], ArraySpec(element, length)]


class Vector:
    """`Vector[T]`: dynamic-length homogeneous mutable container (list-like)."""

    def __class_getitem__(cls, element: Any) -> Any:
        if isinstance(element, tuple):
            raise TypeError("ppy.Vector requires one parameter: Vector[T]")
        return Annotated[list[element], VectorSpec(element)]


class Buffer:
    """`Buffer[T]`: contiguous borrowed buffer view (buffer protocol)."""

    def __class_getitem__(cls, element: Any) -> Any:
        if isinstance(element, tuple):
            raise TypeError("ppy.Buffer requires one parameter: Buffer[T]")
        return Annotated[memoryview, BufferSpec(element)]


i8 = Annotated[int, IntWidth(8, True)]
i16 = Annotated[int, IntWidth(16, True)]
i32 = Annotated[int, IntWidth(32, True)]
i64 = Annotated[int, IntWidth(64, True)]
u8 = Annotated[int, IntWidth(8, False)]
u16 = Annotated[int, IntWidth(16, False)]
u32 = Annotated[int, IntWidth(32, False)]
u64 = Annotated[int, IntWidth(64, False)]
f16 = Annotated[float, FloatWidth(16)]
f32 = Annotated[float, FloatWidth(32)]
f64 = Annotated[float, FloatWidth(64)]

NUMERIC_MARKERS: dict[str, Any] = {
    "i8": i8, "i16": i16, "i32": i32, "i64": i64,
    "u8": u8, "u16": u16, "u32": u32, "u64": u64,
    "f16": f16, "f32": f32, "f64": f64,
}
