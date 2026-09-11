"""The same two expressions as NumPy evaluates them: a temporary per operation."""

import time

import numpy as np


def normalize(x):
    scale = np.sqrt(np.sum(x * x))
    return x / scale


def blend(a, b):
    return np.sin(a) * 2.0 + np.cos(b)


def timed(label, run):
    best = 1e9
    answer = None
    for _ in range(5):
        started = time.perf_counter()
        answer = run()
        best = min(best, (time.perf_counter() - started) * 1000.0)
    print(f"# {label}: {best:.2f} ms")
    return answer


def main():
    values = np.linspace(0.0, 10.0, 8_000_000)
    other = np.linspace(1.0, 5.0, 8_000_000)
    normalized = timed("normalize", lambda: normalize(values))
    mixed = timed("blend", lambda: blend(values, other))
    print(f"{float(normalized[1]):.9f} {float(normalized[-1]):.9f}")
    print(f"{float(mixed[1]):.9f} {float(mixed[-1]):.9f}")


main()
