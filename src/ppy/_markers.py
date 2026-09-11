"""PPY annotation markers: fixed-width numerics, containers, and refinements.

Every marker is an ordinary `typing.Annotated` alias, so a `.ppy` module keeps
working under plain CPython and under any Python-aware editor.
"""

from __future__ import annotations

from typing import Annotated, Any

__all__ = [
    "NUMERIC_MARKERS",
    "Array",
    "ArraySpec",
    "Borrowed",
    "Buffer",
    "BufferSpec",
    "Contiguous",
    "DType",
    "Dynamic",
    "FloatWidth",
    "IntWidth",
    "Length",
    "Mut",
    "NoAlias",
    "Owned",
    "Range",
    "Shape",
    "Vector",
    "VectorSpec",
    "assume",
    "check",
    "f16",
    "f32",
    "f64",
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
]


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

    __slots__ = ("high", "low")
    _fields = ("low", "high")

    def __init__(self, low: float, high: float) -> None:
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


class _Ownership(_Meta):
    """How a parameter holds what it is handed; the subclass says which way."""

    __slots__ = ()
    _fields = ()
    mode = ""

    def __class_getitem__(cls, item: Any) -> Any:
        return Annotated[item, cls()]


class Owned(_Ownership):
    """`Owned[T]`: the callee takes the value and may keep, store, or return it."""

    __slots__ = ()
    mode = "owned"


class Borrowed(_Ownership):
    """`Borrowed[T]`: the callee reads the value for the call and no longer;
    it may not return it, store it where it outlives the call, or mutate it."""

    __slots__ = ()
    mode = "borrowed"


class Mut(_Ownership):
    """`Mut[T]`: a borrow the callee may write through, for the call and no longer."""

    __slots__ = ()
    mode = "mut"


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

#: An explicit Python-dynamic boundary. `Dynamic` is `Any` at runtime, but
#: spelling it says the dynamism is a decision, not an inference failure.
Dynamic = Any


def _describe(target: Any) -> str:
    return getattr(target, "__name__", None) or repr(target)


_BUFFER_FORMATS: dict[Any, frozenset[str]] = {
    int: frozenset({"q", "l8"}),
    float: frozenset({"d"}),
    bool: frozenset({"?"}),
    ("i", 8): frozenset({"b"}),
    ("u", 8): frozenset({"B"}),
    ("i", 16): frozenset({"h"}),
    ("u", 16): frozenset({"H"}),
    ("i", 32): frozenset({"i", "l4"}),
    ("u", 32): frozenset({"I", "L4"}),
    ("i", 64): frozenset({"q", "l8"}),
    ("u", 64): frozenset({"Q", "L8"}),
    ("f", 32): frozenset({"f"}),
    ("f", 64): frozenset({"d"}),
}
#: The largest finite value a narrower float holds; wider than this is not that float.
_FLOAT_LIMITS = {16: 65504.0, 32: 3.4028234663852886e38}
#: Contracts between a caller and a callee that no one value can bear witness to.
_UNCHECKABLE = (NoAlias, Owned, Borrowed, Mut)


class _Validation:
    """The runtime validation `ppy.check[T]` does: sound, for every `T` it accepts.

    A value is checked all the way down -- a `list[int]` element by element,
    a `dict[str, float]` key and value, a tuple field by field, a dataclass
    field by field -- and every PPy refinement in an `Annotated` is checked
    too: an `i8` is an `int` from -128 to 127, an `Array[int, 3]` is a tuple
    of three ints, a `Buffer[float]` is a contiguous one-dimensional buffer
    of doubles, a `Range`, `Length`, `Shape`, `DType`, or `Contiguous` is
    what it says of the value. What comes out is what typed code was
    promised. A `T` this cannot validate soundly -- a callable's signature,
    an iterator, a protocol, or a contract between caller and callee such as
    `Owned[T]`, `Borrowed[T]`, `Mut[T]`, or `NoAlias` that no single value
    can bear witness to -- is refused outright rather than checked in part;
    `ppy.assume[T]` is the unchecked crossing.
    """

    __slots__ = ("target",)

    def __init__(self, target: Any) -> None:
        self.target = target

    def __call__(self, value: Any) -> Any:
        self._validate(self.target, value, "value")
        return value

    def __repr__(self) -> str:
        return f"ppy.check[{self.target!r}]"

    def _validate(self, target: Any, value: Any, where: str) -> None:
        import dataclasses
        import types
        import typing

        if target is Any or target is object:
            return
        if target is None or target is type(None):
            if value is not None:
                raise TypeError(f"{where}: expected None, got {type(value).__name__}")
            return
        origin = typing.get_origin(target)
        arguments = typing.get_args(target)
        if origin is typing.Annotated:
            self._annotated(target, arguments[0], arguments[1:], value, where)
            return
        if origin is typing.Union or isinstance(target, types.UnionType):
            for member in arguments:
                try:
                    self._validate(member, value, where)
                    return
                except TypeError as error:
                    if str(error).startswith("ppy.check cannot validate"):
                        raise
                    continue
            raise TypeError(f"{where}: expected {target!r}, got {type(value).__name__}")
        if origin is typing.Literal:
            if value not in arguments:
                raise TypeError(f"{where}: expected one of {arguments!r}, got {value!r}")
            return
        if origin is None:
            if not isinstance(target, type):
                raise TypeError(f"ppy.check cannot validate against {target!r}")
            if target is float and isinstance(value, int) and not isinstance(value, bool):
                # Typed code accepts an int where a float is expected; the check does too.
                return
            if not isinstance(value, target):
                raise TypeError(
                    f"{where}: expected {_describe(target)}, got {type(value).__name__}"
                )
            if dataclasses.is_dataclass(target):
                try:  # a field annotation may be a string, under the annotations future
                    hints = typing.get_type_hints(target, include_extras=True)
                except Exception as error:  # pylint: disable=broad-exception-caught
                    raise TypeError(
                        f"ppy.check cannot validate against {target!r}: {error}"
                    ) from error
                for field in dataclasses.fields(target):
                    self._validate(
                        hints.get(field.name, field.type),
                        getattr(value, field.name),
                        f"{where}.{field.name}",
                    )
            return
        if origin in (list, set, frozenset, collections_deque()):
            self._sequence(origin, arguments, value, where)
            return
        if origin is tuple:
            self._tuple(arguments, value, where)
            return
        if origin is dict:
            self._dict(arguments, value, where)
            return
        if isinstance(origin, type) and not arguments:
            if not isinstance(value, origin):
                raise TypeError(
                    f"{where}: expected {_describe(origin)}, got {type(value).__name__}"
                )
            return
        raise TypeError(f"ppy.check cannot validate against {target!r}")

    # -- Annotated: every PPy refinement checked, the uncheckable refused ------------------

    def _annotated(self, target: Any, base: Any, metadata: tuple, value: Any, where: str) -> None:
        buffer = None
        for item in metadata:
            if isinstance(item, _UNCHECKABLE):
                raise TypeError(
                    f"ppy.check cannot validate against {target!r}: {type(item).__name__} is a "
                    "contract between a caller and a callee, not a property of one value; "
                    "`ppy.assume[T](value)` is the unchecked crossing"
                )
            if isinstance(item, BufferSpec):
                buffer = item
            elif isinstance(item, ArraySpec) and (
                not isinstance(item.length, int) or isinstance(item.length, bool)
            ):
                raise TypeError(
                    f"ppy.check cannot validate against Array[..., {item.length!r}]: "
                    "the length is not a number"
                )
            elif isinstance(item, _Meta) and not isinstance(item, _CHECKED_META):
                raise TypeError(
                    f"ppy.check cannot validate against {target!r}: {item!r} has no runtime check"
                )
        if buffer is not None:
            # The representation is the buffer protocol, of which `memoryview` is
            # the spelling: an `array.array("q")` is a `Buffer[int]` as much as a view is.
            self._buffer(buffer, value, where)
        else:
            self._validate(base, value, where)
        for item in metadata:
            if isinstance(item, IntWidth):
                self._int_width(item, value, where)
            elif isinstance(item, FloatWidth):
                self._float_width(item, value, where)
            elif isinstance(item, ArraySpec):
                self._array(item, value, where)
            elif isinstance(item, Range):
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"{where}: {item!r} needs a number, got {type(value).__name__}")
                if not item.low <= value <= item.high:
                    raise TypeError(f"{where}: {value!r} is outside {item!r}")
            elif isinstance(item, Length):
                self._length(item.size, value, where, item)
            elif isinstance(item, Shape):
                self._shape(item, value, where)
            elif isinstance(item, DType):
                self._dtype(item, value, where)
            elif isinstance(item, Contiguous):
                self._contiguous(value, where)

    def _int_width(self, width: IntWidth, value: Any, where: str) -> None:
        if not isinstance(value, int):
            raise TypeError(f"{where}: expected {width.name}, got {type(value).__name__}")
        if not width.low <= value <= width.high:
            raise TypeError(
                f"{where}: {value!r} does not fit {width.name} ({width.low}..{width.high})"
            )

    def _float_width(self, width: FloatWidth, value: Any, where: str) -> None:
        limit = _FLOAT_LIMITS.get(width.bits)
        if limit is None:
            return
        import math

        magnitude = abs(float(value))
        if math.isinf(magnitude) or math.isnan(magnitude):
            return
        if magnitude > limit:
            raise TypeError(
                f"{where}: {value!r} is wider than {width.name} holds (|x| <= {limit!r})"
            )

    def _array(self, spec: ArraySpec, value: Any, where: str) -> None:
        self._length(spec.length, value, where, spec)

    def _length(self, size: int, value: Any, where: str, spec: Any) -> None:
        try:
            found = len(value)
        except TypeError as error:
            raise TypeError(f"{where}: {spec!r} needs a value with a length") from error
        if found != size:
            raise TypeError(f"{where}: expected a length of {size}, got {found}")

    def _buffer(self, spec: BufferSpec, value: Any, where: str) -> None:
        formats = _BUFFER_FORMATS.get(_buffer_key(spec.element))
        if formats is None:
            raise TypeError(
                f"ppy.check cannot validate against Buffer[{_describe(spec.element)}]: "
                "the element has no buffer format"
            )
        try:
            view = memoryview(value)
        except TypeError as error:
            raise TypeError(
                f"{where}: expected a buffer of {_describe(spec.element)}, "
                f"got {type(value).__name__}"
            ) from error
        with view:
            if view.ndim != 1 or not view.c_contiguous:
                raise TypeError(f"{where}: a Buffer is one contiguous dimension")
            spelled = view.format.lstrip("@=<>!")
            if spelled in ("l", "L"):
                spelled += str(view.itemsize)
            if spelled not in formats:
                raise TypeError(
                    f"{where}: a Buffer[{_describe(spec.element)}] holds "
                    f"{min(formats, key=len)!r} elements, this holds {view.format!r}"
                )

    def _shape(self, spec: Shape, value: Any, where: str) -> None:
        shape = getattr(value, "shape", None)
        if shape is None:
            raise TypeError(f"{where}: {spec!r} needs a value with a shape")
        shape = tuple(shape)
        if len(shape) != len(spec.dims):
            raise TypeError(f"{where}: expected {len(spec.dims)} dimension(s), got shape {shape!r}")
        bound: dict[str, int] = {}
        for wanted, found in zip(spec.dims, shape, strict=True):
            if isinstance(wanted, str):
                if bound.setdefault(wanted, found) != found:
                    raise TypeError(
                        f"{where}: dimension {wanted!r} is {bound[wanted]} and {found} in {shape!r}"
                    )
            elif wanted != found:
                raise TypeError(f"{where}: expected shape {spec.dims!r}, got {shape!r}")

    def _dtype(self, spec: DType, value: Any, where: str) -> None:
        dtype = getattr(value, "dtype", None)
        if dtype is None:
            raise TypeError(f"{where}: {spec!r} needs a value with a dtype")
        names = {str(dtype), str(dtype).rpartition(".")[2], str(getattr(dtype, "name", ""))}
        if spec.name not in names:
            raise TypeError(f"{where}: expected dtype {spec.name!r}, got {dtype!s}")

    def _contiguous(self, value: Any, where: str) -> None:
        flags = getattr(value, "flags", None)
        if flags is not None:
            try:
                contiguous = bool(flags["C_CONTIGUOUS"])
            except (KeyError, TypeError):
                contiguous = None
        elif hasattr(value, "is_contiguous"):
            contiguous = bool(value.is_contiguous())
        elif isinstance(value, memoryview):
            contiguous = value.c_contiguous
        else:
            try:
                with memoryview(value) as view:
                    contiguous = view.c_contiguous
            except TypeError:
                contiguous = None
        if contiguous is None:
            raise TypeError(f"{where}: cannot tell whether a {type(value).__name__} is contiguous")
        if not contiguous:
            raise TypeError(f"{where}: not C-contiguous")

    # -- containers ---------------------------------------------------------------------

    def _sequence(self, origin: Any, arguments: tuple, value: Any, where: str) -> None:
        if len(arguments) > 1:
            raise TypeError(f"ppy.check cannot validate against {origin.__name__}{arguments!r}")
        if not isinstance(value, origin):
            raise TypeError(f"{where}: expected {_describe(origin)}, got {type(value).__name__}")
        if not arguments:
            return
        for index, item in enumerate(value):
            self._validate(arguments[0], item, f"{where}[{index}]")

    def _tuple(self, arguments: tuple, value: Any, where: str) -> None:
        if not isinstance(value, tuple):
            raise TypeError(f"{where}: expected tuple, got {type(value).__name__}")
        if not arguments:
            return
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            for index, item in enumerate(value):
                self._validate(arguments[0], item, f"{where}[{index}]")
            return
        if arguments == ((),):
            arguments = ()
        if len(value) != len(arguments):
            raise TypeError(f"{where}: expected a tuple of {len(arguments)}, got {len(value)}")
        for index, (item, expected) in enumerate(zip(value, arguments, strict=True)):
            self._validate(expected, item, f"{where}[{index}]")

    def _dict(self, arguments: tuple, value: Any, where: str) -> None:
        if arguments and len(arguments) != 2:
            raise TypeError(f"ppy.check cannot validate against dict{arguments!r}")
        if not isinstance(value, dict):
            raise TypeError(f"{where}: expected dict, got {type(value).__name__}")
        if not arguments:
            return
        for key, item in value.items():
            self._validate(arguments[0], key, f"{where} key {key!r}")
            self._validate(arguments[1], item, f"{where}[{key!r}]")


#: The refinements the validation checks; any other PPy metadata is refused.
_CHECKED_META = (
    IntWidth,
    FloatWidth,
    ArraySpec,
    VectorSpec,
    BufferSpec,
    Range,
    Length,
    Shape,
    DType,
    Contiguous,
)


def _buffer_key(element: Any) -> Any:
    """What `_BUFFER_FORMATS` is keyed by for a buffer element: a type or a (kind, bits)."""
    import typing

    if typing.get_origin(element) is typing.Annotated:
        base, *metadata = typing.get_args(element)
        for item in metadata:
            if isinstance(item, IntWidth):
                return ("i" if item.signed else "u", item.bits)
            if isinstance(item, FloatWidth):
                return ("f", item.bits)
        return base
    return element


def collections_deque() -> Any:
    import collections

    return collections.deque


class _Check:
    """`ppy.check[T](value)`: validate a dynamic value against `T`, all the way down."""

    __slots__ = ()

    def __getitem__(self, target: Any) -> _Validation:
        return _Validation(target)

    def __repr__(self) -> str:
        return "ppy.check"


class _Assumption:
    """`ppy.assume[T]`: the value as it is, with `T` taken on the programmer's word."""

    __slots__ = ("target",)

    def __init__(self, target: Any) -> None:
        self.target = target

    def __call__(self, value: Any) -> Any:
        return value

    def __repr__(self) -> str:
        return f"ppy.assume[{self.target!r}]"


class _Assume:
    """`ppy.assume[T](value)`: an unchecked type assertion.

    Nothing is validated: the value comes back as it went in, and typed code
    takes it for a `T` because the programmer said so. Where the word is
    wrong the program is wrong in the way an unchecked cast makes it wrong --
    native code reading an `int` that is a `str`. `ppy.check[T]` is the
    validated crossing; this is the escape hatch, and it should look like one.
    """

    __slots__ = ()

    def __getitem__(self, target: Any) -> _Assumption:
        return _Assumption(target)

    def __repr__(self) -> str:
        return "ppy.assume"


check = _Check()
assume = _Assume()

NUMERIC_MARKERS: dict[str, Any] = {
    "i8": i8,
    "i16": i16,
    "i32": i32,
    "i64": i64,
    "u8": u8,
    "u16": u16,
    "u32": u32,
    "u64": u64,
    "f16": f16,
    "f32": f32,
    "f64": f64,
}
