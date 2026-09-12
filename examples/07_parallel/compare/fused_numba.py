"""The same expression under Numba: an explicit loop in `@njit(parallel=True)` with `prange`."""

import time

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def fused(a, b):
    out = np.empty_like(a)
    for i in prange(a.shape[0]):
        out[i] = (a[i] * b[i] + a[i]) * (b[i] - a[i]) + a[i] * 0.5 - b[i] * 0.25
    return out


@njit
def sum_of_squares(a):
    total = 0.0
    for i in range(a.shape[0]):
        total += a[i] * a[i]
    return total


def timed(label, run):
    best = 1e9
    answer = None
    for _ in range(5):
        started = time.perf_counter()
        answer = run()
        best = min(best, time.perf_counter() - started)
    print(f"# {label}: {best * 1000:.2f} ms")
    return answer


def main():
    x = np.linspace(0.0, 10.0, 8_000_000)
    y = np.linspace(1.0, 5.0, 8_000_000)
    fused(x, y)
    sum_of_squares(x)
    out = timed("fused", lambda: fused(x, y))
    print(f"{float(out[7]):.12f} {float(out[-1]):.12f}")
    print(f"{timed('sum of squares', lambda: sum_of_squares(x)):.3f}")


main()
