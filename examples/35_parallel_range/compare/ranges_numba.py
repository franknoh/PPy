"""The four kernels with Numba: `prange` under `parallel=True`; timed warm."""

import time

import numpy as np
from numba import njit, prange

N = 8_000_000


@njit(parallel=True)
def squares(out):
    for i in prange(len(out)):
        out[i] = i * i
    return out[len(out) - 1]


@njit(parallel=True)
def dot(a, b):
    total = 0.0
    for i in prange(len(a)):
        total += a[i] * b[i]
    return total


@njit(parallel=True, fastmath=True)
def dot_relaxed(a, b):
    total = 0.0
    for i in prange(len(a)):
        total += a[i] * b[i]
    return total


@njit(parallel=True)
def count_odd(xs, scale):
    hits = 0
    for i in prange(len(xs)):
        if xs[i] % 2 == 1:
            hits += scale
    return hits


@njit
def fill(out, base):
    for i in range(len(out)):
        out[i] = base + i
    acc = 0.0
    for j in range(len(out)):
        acc += out[j]
    return acc


def timed(label, run, repeats=5):
    run()
    best = 1e9
    for _ in range(repeats):
        started = time.perf_counter()
        run()
        best = min(best, (time.perf_counter() - started) * 1000.0)
    print(f"# {label}: {best:.2f} ms")


def main():
    i = np.arange(N, dtype=np.float64)
    a = 0.001 * i
    b = 1.0 / (1 + i)
    ints = (np.arange(N, dtype=np.int64) % 3).astype(np.int64)
    room = np.empty(N, dtype=np.int64)
    floats = np.empty(N, dtype=np.float64)
    print(squares(room), f"{dot(a, b):.6f}", f"{dot_relaxed(a, b):.3f}", count_odd(ints, 3), fill(floats, 0.5))
    timed("squares", lambda: squares(room))
    timed("dot", lambda: dot(a, b))
    timed("dot_relaxed", lambda: dot_relaxed(a, b))
    timed("count_odd", lambda: count_odd(ints, 3))
    timed("fill", lambda: fill(floats, 0.5))


main()
