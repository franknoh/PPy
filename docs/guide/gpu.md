# GPU kernels: `ppy.cuda` and `ppy.hip`

```python
from ppy import cuda, native


@cuda.kernel
def saxpy(n: int, a: float, x: native.const_ptr[float], y: native.ptr[float]) -> None:
    i = cuda.global_id()
    if i < n:
        slot = native.offset(y, i)
        native.store(slot, a * native.load(native.offset(x, i)) + native.load(slot))


def run(n: int, a: float, x: native.const_ptr[float], y: native.ptr[float]) -> None:
    cuda.launch(saxpy, (n + 255) // 256, 256, n, a, x, y)
```

`@cuda.kernel` marks a kernel: a function of scalars and pointers that
returns nothing and runs once per thread of a launch; `@cuda.device` marks
a function a kernel calls. Inside, `cuda.thread_id()`, `block_id()`,
`block_dim()`, `grid_dim()`, and `global_id()` (the block's index times its
size, plus the thread's) say where a thread is, each along `"x"` unless
told `"y"` or `"z"`; `syncthreads()` waits for the block and `syncwarp()`
for the warp; `shared[T, N]()` is `N` elements of `T` the block shares and
`local[T, N]()` this thread's own, both `native.ptr[T]`; `shfl(v, lane)`,
`shfl_up(v, d)`, `shfl_down(v, d)`, and `shfl_xor(v, m)` trade a scalar
across the warp, whose size `warp_size()` gives. `cuda.launch(kernel,
grid, block, *args)` runs the kernel over `grid` blocks of `block` threads
-- each an `int` or a tuple of up to three -- and waits. `ppy.hip` is the
same vocabulary spelled `hip`, with a 64-lane wavefront; a kernel marked
by either is one kernel, and `ppy emit cuda` and `ppy emit hip` write it
either way. `E1644` names a misuse.

Under CPython a launch runs the grid one block at a time and a block's
threads together, each a Python thread that knows its position, so
barriers and shuffles are real: the reference, exact and slow. The
compiler lowers a kernel to the gpu dialect of the IR
([The IR](../internals/ir.md)) -- inside device code `int` arithmetic
wraps and nothing guards, as on the device -- and refuses what has no
device form: a list or a buffer parameter, a returned value, a call to a
host function. The CPU backends leave device code alone, and a function
that launches stays in Python until the launch runtime; `ppy emit cuda`
and `ppy emit hip` write the kernels, the device functions, and the host
functions with their launches as one CUDA or HIP C++ unit.

Under `ppy run`, a kernel is compiled to PTX -- the gpu dialect as LLVM IR
for NVPTX, libdevice for the math library, `ppy emit nvvm-ir` and `ppy
emit ptx` show the two -- and `cuda.launch` runs it through the CUDA
driver where one is present: scalars by value, a `native` pointer's whole
array copied to the device and, when the pointer is mutable, back, so a
launch means what the reference launch means. `cuda.compiled(kernel)`
says whether that is so here. Where the driver, a device, or the NVPTX
backend is missing, the reference launch runs and `W2008` says why.
`PPY_CUDA_ARCH` names the architecture the PTX is written for (`sm_70`
unless set; a driver compiles PTX forward). An AMD card has no launch
runtime in PPy: `ppy.cuda` and `ppy.tile` launch through the CUDA driver
only, so on a ROCm machine `compiled` is `False` and the reference launch
runs; HIP is `ppy emit hip`, the source for `hipcc`.

`cuda.device_alloc[T](n)` is `n` zeroed elements of `T` that live on the
device between launches: a `native.ptr[T]` like `stack_alloc`'s, so the same
loops fill and read it and `native.offset` keeps its kind. The host reads and
writes through a mirror, and whichever side wrote last holds the truth: a
launch takes the device address and copies nothing, a host read after a
launch brings the array back once, a host write before a launch sends it
once. Without a device -- and under `hip`, which has no launch runtime yet
-- there is only the mirror, and every path reads and writes it directly, so
the program means the same thing everywhere. Under `ppy run` a function
that allocates device memory stays in Python, as one that launches does;
`ppy emit cuda` and `ppy emit hip` write it into the host function as a
managed allocation (`cudaMallocManaged`), which the host reads and writes
as the mirror is, freed when the function returns. A built artifact carries its
kernels: `ppy build` writes each staged payload beside the manifest, and
the launcher binds it without the compiler -- as it does an `@xla.jit`
function's StableHLO.

Examples: [CUDA](../howto/38_cuda.md).

## Tile kernels

`ppy.tile` is the other way to write a kernel: a program owns a tile of
`BLOCK` lanes rather than a thread owning one element, and the block's
threads, shared memory, and shuffles are the compiler's to write.

```python
from ppy import native, tile


@tile.kernel
def block_max(x: native.const_ptr[float], out: native.ptr[float]) -> None:
    pid = tile.program_id()
    values = tile.load(x, pid * 64 + tile.arange(64))
    tile.store(out, pid, tile.max(values))


def run(blocks: int, x: native.const_ptr[float], out: native.ptr[float]) -> None:
    tile.launch(block_max, blocks, x, out)
```

`tile.arange(BLOCK)` names the kernel's block size -- one per kernel, a power
of two of at least 32 -- and is the lane index; `tile.program_id()` and
`tile.num_programs()` say where a program is. `tile.load(p, offsets, mask,
other)` gathers a tile of `int`, `float`, or `bool` elements through a
pointer, reading nothing where `mask` is false and giving `other` there;
`tile.store(p, offsets, value, mask)` scatters one, and with a single
offset and a scalar writes one element. Arithmetic (`+ - * / // %`), the
comparisons, and `& | ^ ~` are lane by lane, a scalar beside a tile is
broadcast, `tile.where(mask, a, b)` chooses per lane, and `tile.sum`,
`tile.max`, `tile.min` reduce a tile to a scalar. `tile.launch(kernel,
programs, *args)` runs it and waits, on the device where the build staged
the kernel and through the reference launch otherwise.

Natively a program is a block of up to 256 threads; a tile is a vector of
`BLOCK / threads` lanes per thread, strided across the block so a load is
coalesced, a gather or scatter is that many lane loads or stores, and a
reduction is each thread's lanes, then a shuffle tree across the warp, then
the warps through shared memory, every thread ending with the same number.
`ppy emit cuda` and `ppy emit ptx` write a tile kernel like any other; the
`tile.compiled` check says whether a launch runs on the device here. The
[tile example](../howto/44_tile.md) measures the two kernels against Triton
and Taichi.
