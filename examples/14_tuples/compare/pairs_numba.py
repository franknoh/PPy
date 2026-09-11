"""The same work under Numba: tuples are native there too, in `@njit` functions."""

import time

from numba import njit


@njit
def divmod_pair(a, b):
    return (a // b, a % b)


@njit
def walk(start, count):
    x, y = start
    for i in range(count):
        x, y = (x + float(i)) / 2.0, (y + 1.0) / 2.0
    return x + y


def main():
    divmod_pair(1, 7)
    walk((0.0, 0.0), 1000)
    best_calls = 1e9
    best_loop = 1e9
    for _ in range(5):
        started = time.perf_counter()
        acc = 0
        for i in range(1_000_000):
            q, r = divmod_pair(i, 7)
            acc += q - r
        best_calls = min(best_calls, (time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        far = walk((0.0, 0.0), 8_000_000)
        best_loop = min(best_loop, (time.perf_counter() - started) * 1000.0)
    print(f"# divmod_pair, a million calls from Python: {best_calls:.2f} ms")
    print(f"# walk, eight million steps from a tuple natively: {best_loop:.2f} ms")
    print(f"{acc} {far:.3f}")


main()
