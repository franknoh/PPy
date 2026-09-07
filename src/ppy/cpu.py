"""`ppy.cpu`: the CPU as a facade, with no instruction named.

```python
from ppy import cpu

@cpu.target("avx2", "fma")
def dot(a: native.ptr[float], b: native.ptr[float], n: int) -> float: ...

if "avx2" in cpu.features():
    ...
lanes = cpu.vector_width[float]()
cpu.prefetch(p)
cpu.pause()
```

`features()` is what this machine has, in LLVM's names; the compiler folds
a membership test on it to a constant, so a program takes the branch for
the machine it is compiled on. `vector_width[T]()` is how many `T` a
vector register holds here. `prefetch` and `pause` are hints: no value
changes, and under CPython they do nothing. A function under
`@cpu.target(...)` is compiled with those features on, and the boundary
binds it only on a machine that has them; elsewhere its Python definition
runs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from ppy_runtime import _cpu

from ._directives import Directive, attach
from ._markers import f32, f64, i8, i16, i32, i64, u8, u16, u32, u64

__all__ = ["features", "pause", "prefetch", "target", "vector_width"]

_T = TypeVar("_T")

_BITS: dict[Any, int] = {
    int: 64,
    float: 64,
    bool: 8,
    i8: 8,
    u8: 8,
    i16: 16,
    u16: 16,
    i32: 32,
    u32: 32,
    i64: 64,
    u64: 64,
    f32: 32,
    f64: 64,
}


def features() -> tuple[str, ...]:
    """This machine's CPU features, sorted, in LLVM's spelling."""
    return _cpu.features()


class _VectorWidth:
    """`cpu.vector_width[T]()`: the lanes of `T` in one vector register."""

    __slots__ = ()

    def __getitem__(self, element: Any) -> Callable[[], int]:
        bits = _BITS.get(element)
        if bits is None:
            raise TypeError(f"{element!r} is not a scalar a vector register holds")
        return lambda: _cpu.vector_width(bits)


vector_width = _VectorWidth()


def prefetch(pointer: Any, *, write: bool = False, locality: int = 3) -> None:
    """Ask for the cache line at `pointer`; under CPython, nothing to ask."""
    del pointer, write, locality


def pause() -> None:
    """The spin-wait hint; under CPython, nothing to hint."""


def target(*names: str) -> Callable[[_T], _T]:
    """Compile the function with these CPU features on (`"avx2"`, `"fma"`)."""
    if not names or not all(isinstance(n, str) and n for n in names):
        raise TypeError("cpu.target takes the feature names, as strings")

    def bind(obj: _T) -> _T:
        return attach(obj, Directive("cpu.target", {"features": tuple(names)}))

    return bind
