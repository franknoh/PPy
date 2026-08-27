"""Parallel execution of native regions (spec 17).

A generated kernel touches no Python object, so `ctypes` releases the GIL for
the duration of the call and worker threads make real progress. Only a loop
whose iterations are provably independent is split; a reduction is split only
where the program has permitted reassociation.
"""

from __future__ import annotations

import ctypes
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

__all__ = ["WorkerPool", "pool", "chunk_bounds", "MIN_PARALLEL_ELEMENTS"]

#: Below this element count the thread hand-off costs more than it saves.
MIN_PARALLEL_ELEMENTS = 1 << 15

_lock = threading.Lock()
_pool: "WorkerPool | None" = None


@dataclass(slots=True)
class WorkerPool:
    """A PPY-owned worker pool sized to avoid oversubscribing library pools."""

    threads: int
    executor: ThreadPoolExecutor

    def map_chunks(self, work: Callable[[int, int], object], length: int) -> list[object]:
        bounds = chunk_bounds(length, self.threads)
        if len(bounds) == 1:
            start, stop = bounds[0]
            return [work(start, stop)]
        futures = [self.executor.submit(work, start, stop) for start, stop in bounds]
        return [future.result() for future in futures]

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)


def _requested_threads(setting: str | int) -> int:
    if isinstance(setting, int) and setting > 0:
        return setting
    available = os.cpu_count() or 1
    # Leave the accelerator and BLAS pools room rather than compete with them
    # (spec 17.3).
    reserved = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
    if reserved > 0:
        return max(1, min(available, reserved))
    return max(1, available)


def pool(threads: str | int = "auto") -> WorkerPool:
    """The process-wide PPY worker pool, created on first use."""
    global _pool
    with _lock:
        if _pool is None:
            count = _requested_threads(threads)
            _pool = WorkerPool(
                threads=count,
                executor=ThreadPoolExecutor(max_workers=count, thread_name_prefix="ppy-worker"),
            )
        return _pool


def chunk_bounds(length: int, threads: int) -> list[tuple[int, int]]:
    """Split `[0, length)` into contiguous chunks, one per worker."""
    if length < MIN_PARALLEL_ELEMENTS or threads <= 1:
        return [(0, length)]
    count = min(threads, max(1, length // (MIN_PARALLEL_ELEMENTS // 4)))
    size = -(-length // count)
    bounds = []
    start = 0
    while start < length:
        bounds.append((start, min(start + size, length)))
        start += size
    return bounds


def offset(pointer, element_size: int, index: int):  # type: ignore[no-untyped-def]
    """A pointer to element `index` of the same buffer."""
    address = ctypes.cast(pointer, ctypes.c_void_p).value or 0
    return ctypes.cast(address + index * element_size, type(pointer))
