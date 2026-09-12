"""saxpy over sixteen million doubles and a per-block max: Taichi on its CUDA backend."""

import contextlib
import os
import sys
import time

import numpy as np

os.environ.setdefault("LD_LIBRARY_PATH", "/usr/lib/wsl/lib")  # where WSL keeps libcuda
with contextlib.redirect_stdout(sys.stderr):  # the banners are not answers
    import taichi as ti

    ti.init(arch=ti.cuda, default_fp=ti.f64, default_ip=ti.i64)

N = 1 << 24
x = ti.field(ti.f64, shape=N)
y = ti.field(ti.f64, shape=N)
values = ti.field(ti.f64, shape=N)
out = ti.field(ti.f64, shape=N // 64)


@ti.func
def fma(a, x, y):
    return a * x + y


@ti.kernel
def saxpy(a: ti.f64):
    for i in x:
        y[i] = fma(a, x[i], y[i])


@ti.kernel
def block_max():
    # Taichi has no thread, block, or shared memory to name: the outer loop is
    # parallel over blocks, and the max of each block is an inner serial loop.
    for b in out:
        best = values[b * 64]
        for k in range(1, 64):
            candidate = values[b * 64 + k]
            best = candidate if candidate > best else best
        out[b] = best


def timed(label, run, repeats=10):
    run()
    ti.sync()
    best = 1e9
    for _ in range(repeats):
        started = time.perf_counter()
        run()
        ti.sync()
        best = min(best, (time.perf_counter() - started) * 1000.0)
    print(f"# {label}: {best:.3f} ms")


def main():
    hx = np.arange(N, dtype=np.float64)
    hy = np.ones(N, dtype=np.float64)
    hv = np.array([float((i * 37) % 101) for i in range(N)])
    x.from_numpy(hx)
    y.from_numpy(hy)
    values.from_numpy(hv)
    saxpy(2.0)
    block_max()
    print(float(y.to_numpy().sum()), float(out.to_numpy().max()))
    timed("saxpy", lambda: saxpy(2.0))
    timed("block_max", block_max)
    hy = y.to_numpy()

    def saxpy_copying():
        x.from_numpy(hx)
        y.from_numpy(hy)
        saxpy(2.0)
        hy[:] = y.to_numpy()

    def block_max_copying():
        values.from_numpy(hv)
        block_max()
        return out.to_numpy()

    timed("saxpy with copies", saxpy_copying)
    timed("block_max with copies", block_max_copying)


main()
