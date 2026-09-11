"""The four kernels as NumPy array expressions: no loop to write, a temporary per step."""

import os
import time

# BLAS threads pinned to the performance cores: on a hybrid CPU the default
# pool lands `dot` on the efficiency cores now and then, which doubles a
# three-millisecond row and makes the run unstable.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np  # noqa: E402

N = 8_000_000


def squares(out):
    out[:] = np.arange(len(out), dtype=np.int64) ** 2
    return int(out[-1])


def dot(a, b):
    return float(np.dot(a, b))


def count_odd(xs, scale):
    return int(np.count_nonzero(xs % 2 == 1)) * scale


def fill(out, base):
    out[:] = base + np.arange(len(out), dtype=np.float64)
    return float(out.sum())


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
    print(squares(room), round(dot(a, b), 6), round(dot(a, b), 3), count_odd(ints, 3), fill(floats, 0.5))
    timed("squares", lambda: squares(room))
    timed("dot_relaxed", lambda: dot(a, b))
    timed("count_odd", lambda: count_odd(ints, 3))
    timed("fill", lambda: fill(floats, 0.5))


main()
