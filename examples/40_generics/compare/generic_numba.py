"""The same work under Numba: an `@njit` function is generic by dispatch, compiled once
per argument types it meets, which is the same instance-per-type-tuple rule."""

import time

from numba import njit


@njit
def largest(a, b):
    return a if a > b else b


@njit
def clamp(x, lo, hi):
    return largest(lo, x) if x < hi else hi


@njit
def sweep(n):
    total = 0.0
    for i in range(n):
        total += clamp(i * 0.25, 1.0, 10.0) + largest(i, 3)
    return total


def main():
    sweep(1000)
    best = 1e9
    for _ in range(5):
        started = time.perf_counter()
        total = sweep(8_000_000)
        best = min(best, (time.perf_counter() - started) * 1000.0)
    print(f"# sweep, eight million clamps and comparisons: {best:.2f} ms")
    print(f"{total:.1f}")


main()
