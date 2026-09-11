"""The same kernels as NumPy array expressions: no loop to write."""

import time

import numpy as np


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
    timed("total", lambda: float(np.sum(x)), 2000)
    timed("dot", lambda: float(np.dot(x, y)), 2000)
    timed("dot_relaxed", lambda: float(np.dot(x, y)), 2000)
    timed("digest", lambda: int((counts % 1000003).sum()), 2000)
    print(
        round(float(np.sum(x)), 6),
        round(float(np.dot(x, y)), 6),
        round(float(np.dot(x, y)), 3),
        int((counts % 1000003).sum()),
    )

main()
