"""The same kernels under Numba: `@njit` over NumPy arrays."""

import time

import numpy as np
from numba import njit


@njit(cache=False)
def total(values):
    result = 0.0
    for i in range(len(values)):
        result += values[i]
    return result


@njit(cache=False)
def dot(a, b):
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


@njit(cache=False, fastmath=True)
def dot_relaxed(a, b):
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


@njit(cache=False)
def digest(values, modulus):
    result = 0
    for i in range(len(values)):
        result += values[i] % modulus
    return result


def timed(label, run, rounds):
    best = 1e9
    answer = None
    for _ in range(5):
        started = time.perf_counter()
        for _i in range(rounds):
            answer = run()
        best = min(best, (time.perf_counter() - started) / rounds)
    print(f"# {label}: {best * 1000:.4f} ms")
    return answer


def main():
    size = 8192
    x = np.arange(size, dtype=np.float64) * 0.001
    y = np.arange(size, dtype=np.float64) * 0.002
    counts = np.arange(size, dtype=np.int64) * 7919
    for _ in range(50):
        total(x)
        dot(x, y)
        dot_relaxed(x, y)
        digest(counts, 1000003)
    timed("total", lambda: total(x), 2000)
    timed("dot", lambda: dot(x, y), 2000)
    timed("dot_relaxed", lambda: dot_relaxed(x, y), 2000)
    timed("digest", lambda: digest(counts, 1000003), 2000)
    print(round(total(x), 6), round(dot(x, y), 6), round(dot_relaxed(x, y), 3), digest(counts, 1000003))

main()
