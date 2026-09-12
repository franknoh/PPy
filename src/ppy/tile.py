"""`ppy.tile`: kernels over tiles, lowered to a block of threads, written as CUDA.

```python
from ppy import native, tile


@tile.kernel
def saxpy(n: int, a: float, x: native.const_ptr[float], y: native.ptr[float]) -> None:
    offsets = tile.program_id() * 256 + tile.arange(256)
    mask = offsets < n
    xs = tile.load(x, offsets, mask)
    ys = tile.load(y, offsets, mask)
    tile.store(y, offsets, a * xs + ys, mask)


def run(n: int, a: float, x: native.const_ptr[float], y: native.ptr[float]) -> None:
    tile.launch(saxpy, (n + 255) // 256, n, a, x, y)
```

A `@kernel` runs once per *program* of a launch and works on tiles: `BLOCK`
lanes of one scalar, where `BLOCK` is the count `tile.arange` names, one per
kernel, a power of two of at least 32. `program_id()` is which program this
is, `arange(BLOCK)` the lane index, `load(p, offsets, mask, other)` a gather
and `store(p, offsets, value, mask)` a scatter, arithmetic and comparisons
are lane by lane with a scalar broadcast, `where(mask, a, b)` chooses per
lane, and `sum`, `max`, and `min` reduce a tile to one number. There is no
thread to name, no shared memory to lay out, and no shuffle to write: the
compiler gives each program a block of threads that each hold a slice of
every tile, and reduces across the block with shuffles and shared memory
on its own. `launch(kernel, programs, *args)` runs it and waits. Under
CPython the launch runs the programs here, one after another, exactly.
"""

# sum, max, and min are the tile vocabulary, like the builtins they shadow.
# pylint: disable=redefined-builtin

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import _tile
from ._directives import _flexible
from ._tile import Tile, arange, load, max, min, num_programs, program_id, store, sum, where

__all__ = [
    "Tile",
    "arange",
    "compiled",
    "kernel",
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

kernel = _flexible("tile.kernel")


def launch(function: Callable[..., Any], programs: int, /, *arguments: Any) -> None:
    """Run kernel `function(*arguments)` over `programs` programs, and wait."""
    _tile.dispatch(function, programs, arguments)


def compiled(function: Callable[..., Any]) -> bool:
    """Whether a launch of `function` runs on the device here: a kernel the build staged."""
    return getattr(function, "__ppy_kernel__", None) is not None
