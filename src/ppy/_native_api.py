"""`ppy.native`: the directive, and the namespace of typed native memory.

Under plain CPython every one of these has a reference implementation over
`array` memory, so a program that uses them runs unchanged on every path;
the compiler lowers them to pointer operations in native code. What the
reference implementation cannot do -- reinterpret memory the machine
could -- fails clearly rather than approximately.

```python
from ppy import native

@native
def fill(p: native.ptr[float], n: int) -> None:
    for i in range(n):
        native.store(native.offset(p, i), 0.5 * i)
```
"""

from __future__ import annotations

import array as _array
import ctypes
from collections.abc import Callable
from typing import Annotated, Any, TypeVar

from ._directives import Directive, attach
from ._directives import native as _directive
from ._markers import f32, f64, i8, i16, i32, i64, u8, u16, u32, u64

__all__ = ["Pointer", "native"]

_T = TypeVar("_T")

#: Each element type's `array` code and width. `int` is a machine word.
_LAYOUT: dict[Any, tuple[str, int]] = {
    int: ("q", 8),
    float: ("d", 8),
    bool: ("B", 1),
    i8: ("b", 1),
    u8: ("B", 1),
    i16: ("h", 2),
    u16: ("H", 2),
    i32: ("i", 4),
    u32: ("I", 4),
    i64: ("q", 8),
    u64: ("Q", 8),
    f32: ("f", 4),
    f64: ("d", 8),
}
_C_TYPES: dict[Any, Any] = {
    int: ctypes.c_int64,
    float: ctypes.c_double,
    bool: ctypes.c_bool,
    i8: ctypes.c_int8,
    u8: ctypes.c_uint8,
    i16: ctypes.c_int16,
    u16: ctypes.c_uint16,
    i32: ctypes.c_int32,
    u32: ctypes.c_uint32,
    i64: ctypes.c_int64,
    u64: ctypes.c_uint64,
    f32: ctypes.c_float,
    f64: ctypes.c_double,
    type(None): None,
}


def _layout(element: Any) -> tuple[str, int]:
    described = _LAYOUT.get(element)
    if described is None:
        raise TypeError(f"{element!r} is not an element type native memory holds")
    return described


class Pointer[T]:
    """A typed pointer into `array` memory: the reference implementation.

    `memory` is the array, `index` the element the pointer names. A pointer
    from `stack_alloc` owns its memory; `offset` makes another pointer into
    the same memory.
    """

    __slots__ = ("element", "index", "memory", "mutable")

    def __init__(self, memory: Any, index: int, element: Any, *, mutable: bool = True) -> None:
        self.memory = memory
        self.index = index
        self.element = element
        self.mutable = mutable

    def __repr__(self) -> str:
        name = getattr(self.element, "__name__", repr(self.element))
        return f"native.ptr[{name}]@{self.index}"

    def address(self) -> int:
        """The machine address, for handing to C."""
        _code, width = _layout(self.element)
        buffer = (ctypes.c_char * (len(self.memory) * width)).from_buffer(self.memory)
        return ctypes.addressof(buffer) + self.index * width


class _PointerType:
    """`native.ptr[T]` as an annotation: `Annotated[Pointer, element]`."""

    __slots__ = ("mutable",)

    def __init__(self, mutable: bool) -> None:
        self.mutable = mutable

    def __getitem__(self, element: Any) -> Any:
        return Annotated[Pointer, _PointerSpec(element, self.mutable)]

    def __repr__(self) -> str:
        return "native.ptr" if self.mutable else "native.const_ptr"


class _PointerSpec:
    __slots__ = ("element", "mutable")

    def __init__(self, element: Any, mutable: bool) -> None:
        self.element = element
        self.mutable = mutable


class _Sized:
    """`native.sizeof[T]()` and `native.alignof[T]()`."""

    __slots__ = ("_what",)

    def __init__(self, what: str) -> None:
        self._what = what

    def __getitem__(self, element: Any) -> Callable[[], int]:
        if element is Pointer or getattr(element, "__origin__", None) is Pointer:
            return lambda: 8
        _code, width = _layout(element)
        return lambda: width


class _StackAlloc:
    """`native.stack_alloc[T](n)`: `n` zeroed elements the function owns."""

    __slots__ = ()

    def __getitem__(self, element: Any) -> Callable[[int], Pointer[Any]]:
        code, width = _layout(element)

        def allocate(count: int) -> Pointer[Any]:
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("a stack allocation is made with how many elements it holds")
            return Pointer(_array.array(code, bytes(width * count)), 0, element)

        return allocate


class _Cast:
    """`native.cast[U](p)`: the same memory read as `U`, width permitting."""

    __slots__ = ()

    def __getitem__(self, element: Any) -> Callable[[Pointer[Any]], Pointer[Any]]:
        code, width = _layout(element)

        def cast(pointer: Pointer[Any]) -> Pointer[Any]:
            _from_code, from_width = _layout(pointer.element)
            view = memoryview(pointer.memory)
            try:
                recast = view.cast("B").cast(code)
            except TypeError as error:
                raise TypeError(f"cannot view {pointer.element!r} memory as {element!r}") from error
            index = pointer.index * from_width // width
            return Pointer(recast, index, element, mutable=pointer.mutable)

        return cast


def _load(pointer: Pointer[Any]) -> Any:
    if not isinstance(pointer, Pointer):
        raise TypeError("native.load reads through a native.ptr")
    value = pointer.memory[pointer.index]
    if pointer.element is bool:
        return bool(value)  # type: ignore[return-value]
    return value


def _store(pointer: Pointer[Any], value: Any) -> None:
    if not isinstance(pointer, Pointer):
        raise TypeError("native.store writes through a native.ptr")
    if not pointer.mutable:
        raise TypeError("native.store cannot write through a const_ptr")
    pointer.memory[pointer.index] = int(value) if pointer.element is bool else value  # type: ignore[call-overload]


def _offset(pointer: Pointer[Any], count: int) -> Pointer[Any]:
    if not isinstance(pointer, Pointer):
        raise TypeError("native.offset moves a native.ptr")
    return Pointer(pointer.memory, pointer.index + count, pointer.element, mutable=pointer.mutable)


class _Extern:
    """`@native.extern("sin", library="m")`: a C function, called through
    ctypes under CPython and directly in native code."""

    __slots__ = ("convention", "library", "pure", "symbol")

    def __init__(self, symbol: str, library: str | None, pure: bool, convention: str) -> None:
        self.symbol = symbol
        self.library = library
        self.pure = pure
        self.convention = convention

    def __call__(self, stub: Callable[..., Any]) -> Callable[..., Any]:
        symbol, library = self.symbol, self.library
        annotations = dict(getattr(stub, "__annotations__", {}))
        returns = annotations.pop("return", None)
        prototype: Any = None

        def resolve() -> Any:
            nonlocal prototype
            if prototype is not None:
                return prototype
            handle = _open_library(library)
            function = getattr(handle, symbol)
            function.argtypes = [_c_type(t) for t in annotations.values()]
            function.restype = _c_type(returns)
            prototype = function
            return function

        def call(*arguments: Any) -> Any:
            function = resolve()
            converted = [a.address() if isinstance(a, Pointer) else a for a in arguments]
            return function(*converted)

        call.__name__ = stub.__name__
        call.__qualname__ = stub.__qualname__
        call.__doc__ = stub.__doc__
        call.__annotations__ = dict(getattr(stub, "__annotations__", {}))
        call.__wrapped__ = stub  # type: ignore[attr-defined]
        options = {"symbol": symbol, "pure": self.pure, "convention": self.convention}
        if library is not None:
            options["library"] = library
        return attach(call, Directive("native.extern", options))


def _open_library(library: str | None) -> Any:
    """The shared library `library` names: by the loader's search, by its
    conventional file name (`libm.so`), or by path; None is the process."""
    if library is None:
        return ctypes.CDLL(None)
    from ctypes import util as _ctypes_util

    candidates: list[str] = []
    found = _ctypes_util.find_library(library)
    if found:
        candidates.append(found)
    if "/" in library or library.endswith((".so", ".dylib", ".dll")):
        candidates.append(library)
    else:
        candidates.extend(f"lib{library}{suffix}" for suffix in (".so", ".dylib", ".dll"))
        candidates.append(library)
    errors: list[str] = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError as error:
            errors.append(str(error))
    raise OSError(f"no shared library for {library!r}: {'; '.join(errors)}")


def _c_type(annotation: Any) -> Any:
    if annotation is None:
        return None
    if annotation in _C_TYPES:
        return _C_TYPES[annotation]
    if getattr(annotation, "__origin__", None) is Pointer or annotation is Pointer:
        return ctypes.c_void_p
    metadata = getattr(annotation, "__metadata__", ())
    if any(isinstance(m, _PointerSpec) for m in metadata):
        return ctypes.c_void_p
    origin = getattr(annotation, "__origin__", None)
    if origin is not None and origin in _C_TYPES:
        return _C_TYPES[origin]
    raise TypeError(f"{annotation!r} has no C type")


class _Native:
    """The `ppy.native` object: a directive when applied, a namespace otherwise."""

    ptr = _PointerType(mutable=True)
    const_ptr = _PointerType(mutable=False)
    sizeof = _Sized("size")
    alignof = _Sized("alignment")
    stack_alloc = _StackAlloc()
    cast = _Cast()
    load = staticmethod(_load)
    store = staticmethod(_store)
    offset = staticmethod(_offset)
    Pointer = Pointer

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _directive(*args, **kwargs)

    @staticmethod
    def extern(
        symbol: str, *, library: str | None = None, pure: bool = False, convention: str = "c"
    ) -> _Extern:
        """Bind a stub to a C symbol; `pure=True` declares it has no effects."""
        return _Extern(symbol, library, pure, convention)

    @staticmethod
    def export(name: str | None = None) -> Callable[[_T], _T]:
        """Give a function a public C symbol in the built library."""

        def bind(obj: _T) -> _T:
            return attach(
                obj, Directive("native.export", {"name": name or getattr(obj, "__name__", "")})
            )

        return bind

    def __repr__(self) -> str:
        return "ppy.native"


native = _Native()
