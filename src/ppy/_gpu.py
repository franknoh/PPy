"""The reference behind `ppy.cuda` and `ppy.hip`: a kernel run on CPU threads.

A launch runs the grid one block at a time and a block's threads together,
each a Python thread that knows its position, so `syncthreads` is a real
barrier and a shuffle really trades values between the lanes of a warp.
It is exact and slow: the reference a compiled kernel is held to. The
compiler lowers the same calls to the gpu dialect (`docs/internals/ir.md`), and the
source backends write them as CUDA or HIP.
"""

from __future__ import annotations

import array
import threading
from collections.abc import Callable
from typing import Any

from ._native_api import Pointer, _layout

__all__ = [
    "WIDTHS",
    "block_dim",
    "block_id",
    "dispatch",
    "extent",
    "global_id",
    "grid_dim",
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
]

#: How many lanes a subgroup has: a CUDA warp, an AMD wavefront.
WIDTHS = {"cuda": 32, "hip": 64}
_AXES = {"x": 0, "y": 1, "z": 2}

_current = threading.local()


class _Warp:
    __slots__ = ("barrier", "size", "table")

    def __init__(self, size: int) -> None:
        self.size = size
        self.barrier = threading.Barrier(size)
        self.table: list[Any] = [None] * size


class _Block:
    """One block's shared state: its barrier, its warps, its shared memory."""

    def __init__(self, threads: int, width: int) -> None:
        self.barrier = threading.Barrier(threads)
        self.lock = threading.Lock()
        self.width = width
        self.warps = [_Warp(min(width, threads - start)) for start in range(0, threads, width)]
        self.shared: dict[int, tuple[Any, int, Pointer[Any]]] = {}

    def shared_memory(self, ordinal: int, element: Any, count: int) -> Pointer[Any]:
        """The block's `ordinal`-th shared declaration: one array, every thread's."""
        with self.lock:
            found = self.shared.get(ordinal)
            if found is None:
                code, width = _layout(element)
                found = (
                    element,
                    count,
                    Pointer(array.array(code, bytes(width * count)), 0, element),
                )
                self.shared[ordinal] = found
        if found[0] is not element or found[1] != count:
            raise RuntimeError(
                "every thread of a block declares the same shared memory in the same order"
            )
        return found[2]


class _Position:
    __slots__ = ("block", "block_dim", "context", "grid_dim", "linear", "shared_calls", "thread")

    def __init__(
        self,
        thread: tuple[int, int, int],
        block: tuple[int, int, int],
        block_dim: tuple[int, int, int],
        grid_dim: tuple[int, int, int],
        context: _Block,
        linear: int,
    ) -> None:
        self.thread = thread
        self.block = block
        self.block_dim = block_dim
        self.grid_dim = grid_dim
        self.context = context
        self.linear = linear
        self.shared_calls = 0


def _position(what: str) -> _Position:
    found = getattr(_current, "position", None)
    if found is None:
        raise RuntimeError(f"{what} is read inside a kernel; no kernel is running")
    return found


def _axis(dim: str, what: str) -> int:
    index = _AXES.get(dim)
    if index is None:
        raise ValueError(f'{what} takes an axis, "x", "y", or "z"')
    return index


def extent(spec: Any, what: str) -> tuple[int, int, int]:
    sizes: tuple[Any, ...] | None
    if isinstance(spec, int) and not isinstance(spec, bool):
        sizes = (spec,)
    elif isinstance(spec, tuple):
        sizes = spec
    else:
        sizes = None
    if (
        sizes is None
        or not 1 <= len(sizes) <= 3
        or any(not isinstance(s, int) or isinstance(s, bool) or s < 1 for s in sizes)
    ):
        raise ValueError(f"{what} is a positive int or a tuple of up to three")
    padded = (*sizes, 1, 1)
    return int(padded[0]), int(padded[1]), int(padded[2])


# -- the launch ------------------------------------------------------------------


def launch(api: str, function: Callable[..., Any], grid: Any, block: Any, arguments: tuple) -> None:
    """Run `function(*arguments)` once per thread of `grid` blocks of `block` threads."""
    if not callable(function):
        raise TypeError(f"{api}.launch takes a kernel function")
    grid3 = extent(grid, f"{api}.launch: the grid")
    block3 = extent(block, f"{api}.launch: the block")
    threads = block3[0] * block3[1] * block3[2]
    for bz in range(grid3[2]):
        for by in range(grid3[1]):
            for bx in range(grid3[0]):
                _run_block(function, arguments, (bx, by, bz), block3, grid3, threads, WIDTHS[api])


def dispatch(
    api: str, function: Callable[..., Any], grid: Any, block: Any, arguments: tuple
) -> None:
    """The kernel the build staged behind `function`, on the device; or its definition here.

    A device that refuses -- the driver gone, memory short -- leaves the
    arguments as they were, and the reference launch answers instead; the
    binding records why.
    """
    kernel = getattr(function, "__ppy_kernel__", None)
    definition = getattr(function, "__ppy_fallback__", function)
    if kernel is None:
        launch(api, definition, grid, block, arguments)
        return
    binding = getattr(function, "__ppy_binding__", None)
    try:
        kernel.launch(
            extent(grid, f"{api}.launch: the grid"),
            extent(block, f"{api}.launch: the block"),
            arguments,
        )
    except RuntimeError as error:
        if binding is not None:
            binding.last_error = f"{type(error).__name__}: {error}"
            binding.fallbacks += 1
        launch(api, definition, grid, block, arguments)
        return
    if binding is not None:
        binding.calls += 1


def _run_block(
    function: Callable[..., Any],
    arguments: tuple,
    block: tuple[int, int, int],
    block_dim: tuple[int, int, int],
    grid_dim: tuple[int, int, int],
    threads: int,
    width: int,
) -> None:
    context = _Block(threads, width)
    failures: list[BaseException] = []
    lock = threading.Lock()

    def worker(position: _Position) -> None:
        _current.position = position
        try:
            function(*arguments)
        except threading.BrokenBarrierError:
            pass  # another thread failed first; its error is the one raised
        except BaseException as error:  # noqa: BLE001 - re-raised by the launch
            with lock:
                failures.append(error)
            context.barrier.abort()
            for warp in context.warps:
                warp.barrier.abort()
        finally:
            _current.position = None

    workers = []
    linear = 0
    for tz in range(block_dim[2]):
        for ty in range(block_dim[1]):
            for tx in range(block_dim[0]):
                position = _Position((tx, ty, tz), block, block_dim, grid_dim, context, linear)
                workers.append(threading.Thread(target=worker, args=(position,), daemon=True))
                linear += 1
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    if failures:
        raise failures[0]


# -- where a thread is -----------------------------------------------------------


def thread_id(dim: str = "x") -> int:
    """This thread's index within its block."""
    return _position("thread_id").thread[_axis(dim, "thread_id")]


def block_id(dim: str = "x") -> int:
    """This block's index within the grid."""
    return _position("block_id").block[_axis(dim, "block_id")]


def block_dim(dim: str = "x") -> int:
    """How many threads a block has."""
    return _position("block_dim").block_dim[_axis(dim, "block_dim")]


def grid_dim(dim: str = "x") -> int:
    """How many blocks the grid has."""
    return _position("grid_dim").grid_dim[_axis(dim, "grid_dim")]


def global_id(dim: str = "x") -> int:
    """This thread's index within the grid: block times block size, plus thread."""
    position = _position("global_id")
    axis = _axis(dim, "global_id")
    return position.block[axis] * position.block_dim[axis] + position.thread[axis]


# -- synchronization and memory ------------------------------------------------


def syncthreads() -> None:
    """Every thread of the block arrives before any leaves."""
    _position("syncthreads").context.barrier.wait()


def syncwarp() -> None:
    position = _position("syncwarp")
    position.context.warps[position.linear // position.context.width].barrier.wait()


class _Memory:
    """`shared[T, N]()` and `local[T, N]()`: memory of the block, or of the thread."""

    __slots__ = ("_shared",)

    def __init__(self, shared: bool) -> None:
        self._shared = shared

    def __getitem__(self, spec: Any) -> Callable[[], Pointer[Any]]:
        what = "shared" if self._shared else "local"
        if not isinstance(spec, tuple) or len(spec) != 2:
            raise TypeError(f"{what}[T, N] takes the element type and the count")
        element, count = spec
        code, width = _layout(element)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"{what}[T, N] takes a positive count")

        def allocate() -> Pointer[Any]:
            position = _position(what)
            if not self._shared:
                return Pointer(array.array(code, bytes(width * count)), 0, element)
            ordinal = position.shared_calls
            position.shared_calls += 1
            return position.context.shared_memory(ordinal, element, count)

        return allocate

    def __repr__(self) -> str:
        return "shared" if self._shared else "local"


shared = _Memory(shared=True)
local = _Memory(shared=False)


def _shuffle(value: Any, source_of: Callable[[int, int], int], what: str) -> Any:
    position = _position(what)
    context = position.context
    warp = context.warps[position.linear // context.width]
    lane = position.linear % context.width
    warp.table[lane] = value
    warp.barrier.wait()
    source = source_of(lane, warp.size)
    result = warp.table[source] if 0 <= source < warp.size else value
    warp.barrier.wait()
    return result


def shfl(value: Any, lane: int) -> Any:
    """`value` as lane `lane` of this warp holds it."""
    return _shuffle(value, lambda _mine, size: lane % size, "shfl")


def shfl_up(value: Any, delta: int) -> Any:
    """`value` from the lane `delta` below; the lowest lanes keep their own."""
    return _shuffle(value, lambda mine, _size: mine - delta, "shfl_up")


def shfl_down(value: Any, delta: int) -> Any:
    """`value` from the lane `delta` above; the highest lanes keep their own."""
    return _shuffle(value, lambda mine, _size: mine + delta, "shfl_down")


def shfl_xor(value: Any, mask: int) -> Any:
    """`value` from the lane whose index is this one's xor `mask`."""
    return _shuffle(value, lambda mine, _size: mine ^ mask, "shfl_xor")
