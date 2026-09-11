"""saxpy over sixteen million doubles and a per-block max: Triton kernels over CuPy memory."""

import time

import cupy as cp
import triton
import triton.language as tl

N = 1 << 24


class Pointer:
    """What a Triton launch needs of an array: its device pointer and its dtype."""

    def __init__(self, array):
        self.array = array
        self.dtype = array.dtype

    def data_ptr(self):
        return self.array.data.ptr


@triton.jit
def fma(a, x, y):
    return a * x + y


@triton.jit
def saxpy(n, a, x_ptr, y_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, fma(a, x, y), mask=mask)


@triton.jit
def block_max(x_ptr, out_ptr, BLOCK: tl.constexpr):
    # One program per block of 64: the block's values are a vector, and the
    # max is a reduction over it -- there is no thread, no shared memory, no shuffle.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x_ptr + offsets)
    tl.store(out_ptr + tl.program_id(0), tl.max(values, axis=0))


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


def run_saxpy(x, y):
    saxpy[((N + 255) // 256,)](N, 2.0, Pointer(x), Pointer(y), BLOCK=256)


def run_block_max(values, out):
    block_max[(N // 64,)](Pointer(values), Pointer(out), BLOCK=64)


def main():
    x = cp.arange(N, dtype=cp.float64)
    y = cp.ones(N, dtype=cp.float64)
    values = cp.asarray([float((i * 37) % 101) for i in range(N)])
    out = cp.zeros(N // 64, dtype=cp.float64)
    run_saxpy(x, y)
    run_block_max(values, out)
    print(float(y.sum()), float(out.max()))
    timed("saxpy", lambda: run_saxpy(x, y))
    timed("block_max", lambda: run_block_max(values, out))
    hx, hy, hv = x.get(), y.get(), values.get()

    def saxpy_copying():
        dx, dy = cp.asarray(hx), cp.asarray(hy)
        run_saxpy(dx, dy)
        hy[:] = dy.get()

    def block_max_copying():
        dv = cp.asarray(hv)
        run_block_max(dv, out)
        return out.get()

    timed("saxpy with copies", saxpy_copying)
    timed("block_max with copies", block_max_copying)


main()
