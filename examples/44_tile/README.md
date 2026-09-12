# Tile kernels

A kernel that works on tiles rather than threads: `tile.arange(256)` is the
lane index, `tile.load` gathers a tile, arithmetic is lane by lane,
`tile.sum` and its kind reduce a tile to one number, and `tile.store`
scatters one back. There is no thread to name, no shared memory to lay
out, and no shuffle to write; the compiler gives each program a block of
threads that each hold a slice of every tile and reduces across the block
on its own.

## A program owns a tile

```python
@tile.kernel
def saxpy(n: int, a: float, x: native.const_ptr[float], y: native.ptr[float]) -> None:
    offsets = tile.program_id() * 256 + tile.arange(256)
    mask = offsets < n
    xs = tile.load(x, offsets, mask)
    ys = tile.load(y, offsets, mask)
    tile.store(y, offsets, a * xs + ys, mask)
```

`tile.launch(saxpy, (n + 255) // 256, n, a, x, y)` runs one program per
256 elements. `offsets < n` is a tile of bools, and a masked load reads
nothing off the end and gives `0` -- or the `other` you pass -- in those
lanes; a masked store writes nothing there. A scalar beside a tile is
broadcast: `a * xs` scales every lane. The same file runs on CPython, where
a launch runs the programs one after another and a tile is a list per lane
-- the reference the device is held to.

## A reduction is one call

```python
@tile.kernel
def block_max(x: native.const_ptr[float], out: native.ptr[float]) -> None:
    pid = tile.program_id()
    values = tile.load(x, pid * 64 + tile.arange(64))
    tile.store(out, pid, tile.max(values))
```

`tile.max(values)` is the block's max in one call. Natively the program is a
block of threads -- as many as the tile has lanes, up to 256, so a 1024
tile is four lanes per thread -- each reduces its own lanes, a shuffle tree
reduces the warp, the warps meet in shared memory, and every thread ends
with the same number; `tile.store(out, pid, ...)` of one element is written
once. `row_stats` does a mean and a variance the same way, with
`tile.where(mask, ..., 0.0)` keeping the padding out of the sum. What the
[CUDA example](../38_cuda/README.md) spells with `cuda.shared`,
`cuda.syncthreads`, and `cuda.shfl_xor` is here what the compiler writes.

## What a tile kernel may hold

A tile is `int`, `float`, or `bool` lanes, and one kernel has one block
size, a power of two of at least 32, named by its `tile.arange`. The
operators are `+ - * / // %`, the comparisons, `& | ^ ~` on ints and bools,
`tile.where`, and `tile.sum`, `tile.max`, `tile.min`; `E1644` names a
misuse. `ppy emit cuda` writes a tile kernel as CUDA C++ like any other,
and `ppy emit ptx` as PTX.

## Compared with Triton and Taichi

The same two kernels over sixteen million doubles, as programs over tiles
in the three tools that have them, in [`compare/`](compare/):
[`tiles_bench.ppy`](compare/tiles_bench.ppy), [`tiles_triton.py`](compare/tiles_triton.py),
[`tiles_taichi.py`](compare/tiles_taichi.py). Milliseconds, best of warm
launches with the device synchronized, over five processes. The
thread-level ports of the same kernels -- CuPy, Numba, Mojo, CUDA C -- are
the [CUDA example](../38_cuda/README.md)'s comparison. The block max, as
each spells it:

**PPY** -- a program loads its tile and reduces it; the same file runs on
CPython:

```python
@tile.kernel
def block_max(x: native.const_ptr[float], out: native.ptr[float]) -> None:
    pid = tile.program_id()
    values = tile.load(x, pid * 64 + tile.arange(64))
    tile.store(out, pid, tile.max(values))
```

**Triton** -- the same shape, `tl.program_id`, `tl.arange`, `tl.load`,
`tl.max`; it is launched over CuPy memory through a six-line `data_ptr()`
wrapper, and its compiler tiles the work across a warp group:

```python
@triton.jit
def block_max(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + tl.program_id(0), tl.max(tl.load(x_ptr + offsets), axis=0))
```

**Taichi** -- no program either: arrays are `ti.field`s, the outer loop of a
kernel is parallel over the blocks and the max of each is an inner serial
loop; `ti.init(arch=ti.cuda, default_fp=ti.f64)` makes the answers match,
and on WSL the driver goes on the library path by hand:

```python
@ti.kernel
def block_max():
    for b in out:
        best = values[b * 64]
        for k in range(1, 64):
            candidate = values[b * 64 + k]
            best = candidate if candidate > best else best
        out[b] = best
```

<!-- compare:start -->
| | PPY `tile.launch` | Triton | Taichi |
|---|---:|---:|---:|
| saxpy, arrays on the device | **0.72 ± 0.09** | 0.77 ± 0.10 | 0.89 ± 0.11 |
| block max, arrays on the device | 0.41 ± 0.02 | 0.81 ± 0.02 | **0.31 ± 0.03** |
| saxpy, arrays copied in and out per launch | **41.11 ± 0.67** | 51.73 ± 6.92 | 163.66 ± 7.49 |
| block max, array copied in per launch | 14.87 ± 0.40 | **11.38 ± 0.36** | 29.90 ± 2.21 |
<!-- compare:end -->

On saxpy the three are the memory bandwidth of the device. On the block
max they are three lowerings of one idea: PPY gives the 64-lane tile a
block of 64 threads and reduces it with a shuffle tree, Triton gives it a
warp group and its own reduction, and Taichi one thread per block that
loops over the 64 -- no exchange at all, which is why it is the fastest
here and why it would not be on a wider tile. The copying rows are the
other memory model, an array sent in and brought back on every launch;
PPY's driver reads the host array in place, Triton goes through CuPy's
`asarray`, and Taichi's `from_numpy` copies twice.

NVIDIA GeForce RTX 5080 Laptop GPU, driver 610.71, CUDA 13.3; Triton 3.8.0,
Taichi 1.7.4 on CPython 3.12.13; PPY on CPython 3.13.13.

## Run it

```bash
python  tiles.ppy
ppy run tiles.ppy
ppy emit cuda tiles.ppy
```

<!-- outputs:start -->
## What it prints

**`python  tiles.ppy`**

```text
1.0 1999.0
[100.0, 98.0, 100.0, 100.0]
49.8008 852.8236
# kernels compiled for a device here: False
```

**`ppy run tiles.ppy`**

```text
1.0 1999.0
[100.0, 98.0, 100.0, 100.0]
49.8008 852.8236
# kernels compiled for a device here: True
```

**`ppy emit cuda tiles.ppy`**

*386 lines: [outputs/03-ppy-emit-cuda-tiles-ppy.txt](outputs/03-ppy-emit-cuda-tiles-ppy.txt)*

<!-- outputs:end -->
