"""The four kernels with Taichi: kernels over fields, the outermost loop parallel; timed warm."""

import time

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu, default_ip=ti.i64, default_fp=ti.f64)

N = 8_000_000
a_f = ti.field(ti.f64, shape=N)
b_f = ti.field(ti.f64, shape=N)
ints = ti.field(ti.i64, shape=N)
room = ti.field(ti.i64, shape=N)
floats = ti.field(ti.f64, shape=N)


@ti.kernel
def squares() -> ti.i64:
    for i in range(N):
        room[i] = ti.i64(i) * ti.i64(i)
    return room[N - 1]


@ti.kernel
def dot() -> ti.f64:
    total = 0.0
    for i in range(N):
        total += a_f[i] * b_f[i]
    return total


@ti.kernel
def count_odd(scale: ti.i64) -> ti.i64:
    hits = 0
    for i in range(N):
        if ints[i] % 2 == 1:
            hits += scale
    return hits


@ti.kernel
def fill(base: ti.f64) -> ti.f64:
    for i in range(N):
        floats[i] = base + i
    acc = 0.0
    ti.loop_config(serialize=True)
    for j in range(N):
        acc += floats[j]
    return acc


def timed(label, run, repeats=5):
    run()
    ti.sync()
    best = 1e9
    for _ in range(repeats):
        started = time.perf_counter()
        run()
        ti.sync()
        best = min(best, (time.perf_counter() - started) * 1000.0)
    print(f"# {label}: {best:.2f} ms")


def main():
    i = np.arange(N, dtype=np.float64)
    a_f.from_numpy(0.001 * i)
    b_f.from_numpy(1.0 / (1 + i))
    ints.from_numpy((np.arange(N, dtype=np.int64) % 3).astype(np.int64))
    print(squares(), f"{dot():.6f}", f"{dot():.3f}", count_odd(3), fill(0.5))
    timed("squares", squares)
    timed("dot", dot)
    timed("count_odd", lambda: count_odd(3))
    timed("fill", lambda: fill(0.5))


main()
