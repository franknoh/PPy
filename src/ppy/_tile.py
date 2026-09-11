"""`ppy.tile`: the reference implementation of tile kernels.

A tile kernel is a function that runs once per *program* of a launch and
works on tiles -- vectors of `BLOCK` elements -- rather than on one element
per thread: `tile.arange(BLOCK)` is the lane index, `tile.load` gathers a
tile through a pointer, arithmetic is lane by lane, `tile.sum` and its
kind reduce a tile to one number, and `tile.store` scatters one back. Under
CPython a launch runs the programs one after another, here, with a `Tile`
that is a Python list per lane: exact and slow, the reference the compiled
kernel is held to. The compiler lowers a tile kernel to a block of threads
that each hold a slice of every tile, and reduces across the block with
shuffles and shared memory.
"""

# sum, max, and min are the tile vocabulary, like the builtins they shadow.
# pylint: disable=redefined-builtin

from __future__ import annotations

import builtins
import threading
from collections.abc import Callable, Iterator
from typing import Any

from ._native_api import Pointer

__all__ = [
    "Tile",
    "arange",
    "launch",
    "load",
    "max",
    "min",
    "num_programs",
    "program_id",
    "store",
    "sum",
    "where",
]

_current = threading.local()


class _Program:
    __slots__ = ("count", "index")

    def __init__(self, index: int, count: int) -> None:
        self.index = index
        self.count = count


def _program() -> _Program:
    program = getattr(_current, "program", None)
    if program is None:
        raise RuntimeError("`ppy.tile` positions are known only inside a launched kernel")
    return program


class Tile:
    """`BLOCK` lanes of one scalar kind: `int`, `float`, or `bool`."""

    __slots__ = ("kind", "lanes")

    def __init__(self, kind: type, lanes: list) -> None:
        self.kind = kind
        self.lanes = lanes

    def __len__(self) -> int:
        return len(self.lanes)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.lanes)

    def __repr__(self) -> str:
        return f"tile.Tile[{self.kind.__name__}]({self.lanes!r})"

    # -- lane by lane --------------------------------------------------------------

    def _zip(self, other: Any, operate: Callable[[Any, Any], Any], kind: type | None) -> Tile:
        if isinstance(other, Tile):
            if len(other) != len(self):
                raise ValueError("tiles of different sizes have no lanes in common")
            lanes = [operate(a, b) for a, b in zip(self.lanes, other.lanes, strict=True)]
        else:
            lanes = [operate(a, other) for a in self.lanes]
        return Tile(kind or _kind_of(lanes[0]) if lanes else self.kind, lanes)

    def _rzip(self, other: Any, operate: Callable[[Any, Any], Any]) -> Tile:
        lanes = [operate(other, a) for a in self.lanes]
        return Tile(_kind_of(lanes[0]) if lanes else self.kind, lanes)

    def __add__(self, other: Any) -> Tile:
        return self._zip(other, lambda a, b: a + b, None)

    def __radd__(self, other: Any) -> Tile:
        return self._rzip(other, lambda a, b: a + b)

    def __sub__(self, other: Any) -> Tile:
        return self._zip(other, lambda a, b: a - b, None)

    def __rsub__(self, other: Any) -> Tile:
        return self._rzip(other, lambda a, b: a - b)

    def __mul__(self, other: Any) -> Tile:
        return self._zip(other, lambda a, b: a * b, None)

    def __rmul__(self, other: Any) -> Tile:
        return self._rzip(other, lambda a, b: a * b)

    def __truediv__(self, other: Any) -> Tile:
        return self._zip(other, lambda a, b: a / b, float)

    def __rtruediv__(self, other: Any) -> Tile:
        return self._rzip(other, lambda a, b: a / b)

    def __floordiv__(self, other: Any) -> Tile:
        return self._zip(other, lambda a, b: a // b, None)

    def __mod__(self, other: Any) -> Tile:
        return self._zip(other, lambda a, b: a % b, None)

    def __neg__(self) -> Tile:
        return Tile(self.kind, [-a for a in self.lanes])

    def __lt__(self, other: Any) -> Tile:
        return self._zip(other, lambda a, b: a < b, bool)

    def __le__(self, other: Any) -> Tile:
        return self._zip(other, lambda a, b: a <= b, bool)

    def __gt__(self, other: Any) -> Tile:
        return self._zip(other, lambda a, b: a > b, bool)

    def __ge__(self, other: Any) -> Tile:
        return self._zip(other, lambda a, b: a >= b, bool)

    def __eq__(self, other: object) -> Tile:  # type: ignore[override]
        return self._zip(other, lambda a, b: a == b, bool)

    def __ne__(self, other: object) -> Tile:  # type: ignore[override]
        return self._zip(other, lambda a, b: a != b, bool)

    __hash__ = None  # type: ignore[assignment]

    def __and__(self, other: Any) -> Tile:
        return self._zip(other, lambda a, b: a & b, None)

    def __or__(self, other: Any) -> Tile:
        return self._zip(other, lambda a, b: a | b, None)

    def __xor__(self, other: Any) -> Tile:
        return self._zip(other, lambda a, b: a ^ b, None)

    def __invert__(self) -> Tile:
        if self.kind is bool:
            return Tile(bool, [not a for a in self.lanes])
        return Tile(self.kind, [~a for a in self.lanes])


def _kind_of(value: Any) -> type:
    if isinstance(value, bool):
        return bool
    if isinstance(value, int):
        return int
    return float


# -- where a program is ------------------------------------------------------------


def program_id() -> int:
    """Which program of the launch this is."""
    return _program().index


def num_programs() -> int:
    return _program().count


def arange(block: int) -> Tile:
    """The lane indices `0 .. block - 1`; `block` names the kernel's tile size."""
    if not isinstance(block, int) or isinstance(block, bool) or block < 1:
        raise TypeError("`tile.arange(BLOCK)` takes a positive int")
    return Tile(int, list(range(block)))


# -- memory --------------------------------------------------------------------------


def _pointer(pointer: Any, what: str) -> Pointer[Any]:
    if not isinstance(pointer, Pointer):
        raise TypeError(f"{what} goes through a `native.ptr`")
    return pointer


def load(pointer: Any, offsets: Any, mask: Any = None, other: Any = 0) -> Any:
    """The elements at `pointer + offsets`: a tile for a tile of offsets, one for one.

    Where `mask` is false the lane is `other` and nothing is read.
    """
    source = _pointer(pointer, "`tile.load`")
    if not isinstance(offsets, Tile):
        return source.memory[source.index + int(offsets)]
    if mask is not None and not (isinstance(mask, Tile) and mask.kind is bool):
        raise TypeError("`tile.load` takes a tile of bools as its mask")
    kind = float if source.element is float else (bool if source.element is bool else int)
    lanes = []
    for lane, offset in enumerate(offsets.lanes):
        if mask is not None and not mask.lanes[lane]:
            lanes.append(kind(other))
        else:
            lanes.append(source.memory[source.index + offset])
    return Tile(kind, lanes)


def store(pointer: Any, offsets: Any, value: Any, mask: Any = None) -> None:
    """Write `value` at `pointer + offsets`: a tile lane by lane, one element for one."""
    target = _pointer(pointer, "`tile.store`")
    if not target.mutable:
        raise TypeError("`tile.store` cannot write through a `const_ptr`")
    if not isinstance(offsets, Tile):
        if isinstance(value, Tile):
            raise TypeError("`tile.store` of a tile takes a tile of offsets")
        target.memory[target.index + int(offsets)] = value
        target.touch()
        return
    if mask is not None and not (isinstance(mask, Tile) and mask.kind is bool):
        raise TypeError("`tile.store` takes a tile of bools as its mask")
    lanes = value.lanes if isinstance(value, Tile) else [value] * len(offsets)
    for lane, offset in enumerate(offsets.lanes):
        if mask is None or mask.lanes[lane]:
            target.memory[target.index + offset] = lanes[lane]
    target.touch()


def where(mask: Any, a: Any, b: Any) -> Tile:
    """Lane by lane, `a` where `mask` holds and `b` where it does not."""
    if not isinstance(mask, Tile) or mask.kind is not bool:
        raise TypeError("`tile.where` takes a tile of bools first")
    first = a.lanes if isinstance(a, Tile) else [a] * len(mask)
    second = b.lanes if isinstance(b, Tile) else [b] * len(mask)
    lanes = [x if m else y for m, x, y in zip(mask.lanes, first, second, strict=True)]
    return Tile(_kind_of(lanes[0]) if lanes else float, lanes)


# -- reductions -------------------------------------------------------------------------


def _reduce(value: Any, what: str, combine: Callable[[list], Any]) -> Any:
    if not isinstance(value, Tile):
        raise TypeError(f"`tile.{what}` takes a tile")
    if not value.lanes:
        raise ValueError(f"`tile.{what}` of an empty tile")
    return combine(value.lanes)


def sum(value: Any) -> Any:
    return _reduce(value, "sum", builtins.sum)


def max(value: Any) -> Any:
    return _reduce(value, "max", builtins.max)


def min(value: Any) -> Any:
    return _reduce(value, "min", builtins.min)


# -- the launch --------------------------------------------------------------------------


def launch(function: Callable[..., Any], programs: int, arguments: tuple) -> None:
    """Run `function(*arguments)` once per program, in order, here."""
    if not callable(function):
        raise TypeError("tile.launch takes a kernel function")
    if not isinstance(programs, int) or isinstance(programs, bool) or programs < 1:
        raise ValueError("tile.launch: the number of programs is a positive int")
    for index in range(programs):
        _current.program = _Program(index, programs)
        try:
            function(*arguments)
        finally:
            _current.program = None


def dispatch(function: Callable[..., Any], programs: int, arguments: tuple) -> None:
    """The kernel the build staged behind `function`, on the device; or its definition here."""
    kernel = getattr(function, "__ppy_kernel__", None)
    definition = getattr(function, "__ppy_fallback__", function)
    if kernel is None or not getattr(kernel, "threads", 0):
        launch(definition, programs, arguments)
        return
    binding = getattr(function, "__ppy_binding__", None)
    try:
        kernel.launch((int(programs), 1, 1), (int(kernel.threads), 1, 1), arguments)
    except RuntimeError as error:
        if binding is not None:
            binding.last_error = f"{type(error).__name__}: {error}"
            binding.fallbacks += 1
        launch(definition, programs, arguments)
        return
    if binding is not None:
        binding.calls += 1
