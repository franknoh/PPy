"""saxpy over sixteen million doubles and a per-block max: CuPy; timed warm, device-synchronized."""

import time

import cupy as cp

N = 1 << 24
block_max = cp.RawKernel(
    r"""
extern "C" __global__ void block_max(const double *x, double *out) {
    __shared__ double parked[64];
    int tid = threadIdx.x;
    parked[tid] = x[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();
    double mine = parked[tid];
    double other = __shfl_xor_sync(0xffffffffu, mine, 1);
    mine = other > mine ? other : mine;
    parked[tid] = mine;
    __syncthreads();
    if (tid == 0) {
        double best = parked[0];
        for (int k = 1; k < blockDim.x; k++) {
            best = parked[k] > best ? parked[k] : best;
        }
        out[blockIdx.x] = best;
    }
}
""",
    "block_max",
)
saxpy = cp.ElementwiseKernel("float64 a, float64 x, float64 y", "float64 z", "z = a * x + y", "saxpy")


def timed(label, run, repeats=10):
    run()
    cp.cuda.Device().synchronize()
    best = 1e9
    for _ in range(repeats):
        started = time.perf_counter()
        run()
        cp.cuda.Device().synchronize()
        best = min(best, (time.perf_counter() - started) * 1000.0)
    print(f"# {label}: {best:.3f} ms")


def main():
    x = cp.arange(N, dtype=cp.float64)
    y = cp.ones(N, dtype=cp.float64)
    values = cp.asarray([float((i * 37) % 101) for i in range(N)])
    out = cp.zeros(N // 64, dtype=cp.float64)
    saxpy(2.0, x, y, y)
    block_max((N // 64,), (64,), (values, out))
    print(float(y.sum()), float(out.max()))
    timed("saxpy", lambda: saxpy(2.0, x, y, y))
    timed("block_max", lambda: block_max((N // 64,), (64,), (values, out)))
    # The same launches the way a host-array launch means them: arrays copied in, results out.
    import numpy as np

    hx, hy, hv = x.get(), y.get(), values.get()

    def saxpy_copying():
        dx, dy = cp.asarray(hx), cp.asarray(hy)
        saxpy(2.0, dx, dy, dy)
        hy[:] = dy.get()

    def block_max_copying():
        dv = cp.asarray(hv)
        block_max((N // 64,), (64,), (dv, out))
        return out.get()

    timed("saxpy with copies", saxpy_copying)
    timed("block_max with copies", block_max_copying)


main()
