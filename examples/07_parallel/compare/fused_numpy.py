"""The fused expression as NumPy evaluates it: five array operations, four temporaries."""

import time

import numpy as np


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
    out = timed("fused", lambda: (x * y + x) * (y - x) + x * 0.5 - y * 0.25)
    print(f"{float(out[7]):.12f} {float(out[-1]):.12f}")
    print(f"{timed('sum of squares', lambda: float(np.sum(x * x))):.3f}")


main()
