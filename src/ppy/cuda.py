"""`ppy.cuda`: kernels written in Python, lowered to the gpu dialect, written as CUDA.

```python
from ppy import cuda, native


@cuda.kernel
def saxpy(n: int, a: float, x: native.const_ptr[float], y: native.ptr[float]) -> None:
    i = cuda.global_id()
    if i < n:
        slot = native.offset(y, i)
        native.store(slot, a * native.load(native.offset(x, i)) + native.load(slot))


def run(n: int, a: float, x: native.const_ptr[float], y: native.ptr[float]) -> None:
    cuda.launch(saxpy, (n + 255) // 256, 256, n, a, x, y)
```

`@kernel` marks a function of scalars and pointers that returns nothing
and runs once per thread of a launch; `@device` one a kernel calls. Inside,
`thread_id()`, `block_id()`, `block_dim()`, `grid_dim()`, and `global_id()`
say where a thread is, `syncthreads()` and `syncwarp()` wait for the block
and the warp, `shared[T, N]()` and `local[T, N]()` are memory of the block
and of the thread, and the `shfl` family trades a scalar across the warp.
`launch(kernel, grid, block, *args)` runs a kernel and waits. Under CPython
the launch runs the grid here, on threads that know their position -- the
reference, exact and slow (spec 72); the compiler lowers the same calls to
the gpu dialect, and `ppy emit cuda` writes them as CUDA C++.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import _gpu
from ._directives import _flexible

__all__ = [
    "block_dim",
    "block_id",
    "compiled",
    "device",
    "global_id",
    "grid_dim",
    "kernel",
    "launch",
    "local",
    "shared",
    "shfl",
    "shfl_down",
    "shfl_up",
    "shfl_xor",
    "syncthreads",
    "syncwarp",
    "thread_id",
    "warp_size",
]

kernel = _flexible("cuda.kernel")
device = _flexible("cuda.device")


def launch(function: Callable[..., Any], grid: Any, block: Any, /, *arguments: Any) -> None:
    """Run kernel `function(*arguments)` over `grid` blocks of `block` threads, and wait."""
    _gpu.dispatch("cuda", function, grid, block, arguments)


def compiled(function: Callable[..., Any]) -> bool:
    """Whether a launch of `function` runs on the device here: a kernel the build staged."""
    return getattr(function, "__ppy_kernel__", None) is not None


def warp_size() -> int:
    """The lanes of a warp: 32."""
    return _gpu.WIDTHS["cuda"]


thread_id = _gpu.thread_id
block_id = _gpu.block_id
block_dim = _gpu.block_dim
grid_dim = _gpu.grid_dim
global_id = _gpu.global_id
syncthreads = _gpu.syncthreads
syncwarp = _gpu.syncwarp
shared = _gpu.shared
local = _gpu.local
shfl = _gpu.shfl
shfl_up = _gpu.shfl_up
shfl_down = _gpu.shfl_down
shfl_xor = _gpu.shfl_xor
