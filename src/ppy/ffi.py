"""`ppy.ffi`: binding C functions, one layer over `ppy.native.extern`.

```python
from ppy import ffi, native

libm = ffi.library("m")


@ffi.bind(libm, symbol="sin", pure=True)
def sin(x: float) -> float: ...
```

A binding is a stub whose annotations are the C signature; under CPython
the call goes through ctypes, in native code straight to the symbol. The
markers say what a plain signature cannot: `ffi.nullable[T]` for a pointer
that may be null, `ffi.LengthOf("xs")` for an integer that is the length
of another parameter, and `ppy.Owned`/`ppy.Borrowed` for who keeps memory.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from ._native_api import native

__all__ = ["LengthOf", "Library", "bind", "library", "nullable"]


class Library:
    """A shared library by name (`"m"`), or by path."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"ffi.library({self.name!r})"


def library(name: str) -> Library:
    return Library(name)


def bind(
    lib: Library | None = None,
    *,
    symbol: str | None = None,
    pure: bool = False,
    convention: str = "c",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """`@ffi.bind(lib, symbol="sin")`: the stub is the C function."""

    def decorate(stub: Callable[..., Any]) -> Callable[..., Any]:
        name = symbol or stub.__name__
        return native.extern(
            name, library=lib.name if lib is not None else None, pure=pure, convention=convention
        )(stub)

    return decorate


class LengthOf:
    """Marker: this integer is the length of the named buffer parameter."""

    __slots__ = ("parameter",)

    def __init__(self, parameter: str) -> None:
        self.parameter = parameter

    def __repr__(self) -> str:
        return f"ffi.LengthOf({self.parameter!r})"


class _Nullable:
    """`ffi.nullable[native.ptr[T]]`: the pointer may be null."""

    __slots__ = ()

    def __getitem__(self, pointer: Any) -> Any:
        return Annotated[pointer | None, _NullableSpec()]


class _NullableSpec:
    __slots__ = ()


nullable = _Nullable()
