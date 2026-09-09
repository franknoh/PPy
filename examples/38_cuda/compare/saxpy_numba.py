"""saxpy over sixteen million doubles and a per-block max: Numba's CUDA target; timed warm."""

import time

import numpy as np
from numba import cuda

N = 1 << 24


@cuda.jit(device=True)
def fma(a, x, y):
    return a * x + y


@cuda.jit
def saxpy(n, a, x, y):
    i = cuda.grid(1)
    if i < n:
        y[i] = fma(a, x[i], y[i])


@cuda.jit
def block_max(x, out):
    parked = cuda.shared.array(64, dtype=np.float64)
    tid = cuda.threadIdx.x
    parked[tid] = x[cuda.grid(1)]
    cuda.syncthreads()
    mine = parked[tid]
    other = cuda.shfl_xor_sync(0xFFFFFFFF, mine, 1)
    mine = other if other > mine else mine
    parked[tid] = mine
    cuda.syncthreads()
    if tid == 0:
        best = parked[0]
        for k in range(1, cuda.blockDim.x):
            best = parked[k] if parked[k] > best else best
        out[cuda.blockIdx.x] = best


def timed(label, run, repeats=10):
    run()
    cuda.synchronize()
    best = 1e9
    for _ in range(repeats):
        started = time.perf_counter()
        run()
        cuda.synchronize()
        best = min(best, (time.perf_counter() - started) * 1000.0)
    print(f"# {label}: {best:.3f} ms")


def main():
    x = cuda.to_device(np.arange(N, dtype=np.float64))
    y = cuda.to_device(np.ones(N, dtype=np.float64))
    values = cuda.to_device(np.array([float((i * 37) % 101) for i in range(N)]))
    out = cuda.device_array(N // 64, dtype=np.float64)
    saxpy[(N + 255) // 256, 256](N, 2.0, x, y)
    block_max[N // 64, 64](values, out)
    print(float(y.copy_to_host().sum()), float(out.copy_to_host().max()))
    timed("saxpy", lambda: saxpy[(N + 255) // 256, 256](N, 2.0, x, y))
    timed("block_max", lambda: block_max[N // 64, 64](values, out))
    hx, hy, hv = x.copy_to_host(), y.copy_to_host(), values.copy_to_host()

    def saxpy_copying():
        dx, dy = cuda.to_device(hx), cuda.to_device(hy)
        saxpy[(N + 255) // 256, 256](N, 2.0, dx, dy)
        dy.copy_to_host(hy)

    def block_max_copying():
        dv = cuda.to_device(hv)
        block_max[N // 64, 64](dv, out)
        return out.copy_to_host()

    timed("saxpy with copies", saxpy_copying)
    timed("block_max with copies", block_max_copying)


main()
