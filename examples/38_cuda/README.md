# GPU kernels

A saxpy and a block reduction with shared memory and a warp shuffle, written
once in `ppy.cuda`: the reference launch under CPython, PTX through the CUDA
driver under `ppy run` where a device is present, and CUDA or HIP source
from `ppy emit`. The printed totals are identical.

## Kernels, device functions, and where a thread is

```python
@cuda.kernel
def saxpy(n: int, a: float, x: native.const_ptr[float], y: native.ptr[float]) -> None:
    i = cuda.global_id()
    if i < n:
        slot = native.offset(y, i)
        native.store(slot, fma(a, native.load(native.offset(x, i)), native.load(slot)))
```

`@cuda.kernel` marks a function of scalars and pointers that returns nothing
and runs once per thread of a launch; `@cuda.device` marks `fma`, a function
a kernel calls. `thread_id`, `block_id`, `block_dim`, and `global_id` say
where a thread is. `cuda.launch(kernel, grid, block, *args)` runs the kernel
over `grid` blocks of `block` threads and waits; a `native` pointer's whole
array goes to the device and, when the pointer is mutable, comes back, so a
launch means what the reference launch means. `x` and `y` are made with
`cuda.device_alloc[float](n)` instead: memory that lives on the device
between launches, filled and read through the same `native.store` and
`native.load`, so the launch passes an address and copies nothing.

## Shared memory, a barrier, a shuffle

`block_max` parks each thread's value in `cuda.shared[float, 64]()`, waits
at `syncthreads`, trades with its neighbour through `shfl_xor(mine, 1)`,
and lets thread 0 finish the reduction. Under CPython the launch runs a
block's threads together, each a Python thread that knows its position, so
the barrier and the shuffle are real. Inside device code `int` arithmetic
wraps and nothing guards, as on the device; the gpu dialect's verifier
refuses what has no device form — a list or buffer parameter, a returned
value, a call to a host function.

## Where it ran

`cuda.compiled(saxpy)` says whether PTX ran; without a driver, a device, or
the NVPTX backend the reference launch runs and `W2008` says why. The line
that prints it starts with `# `, the mark for output that may differ by
machine. `PPY_CUDA_ARCH` picks the architecture the PTX is written for
(`sm_70` unless set). A built artifact carries its kernels: `ppy build`
stages each PTX beside the manifest and the launcher binds it without the
compiler.

## Compared with CuPy, Numba, Triton, Taichi, Mojo, and CUDA C

The same two kernels over sixteen million doubles, each written the way its
tool wants it, in [`compare/`](compare/): [`saxpy_bench.ppy`](compare/saxpy_bench.ppy),
[`saxpy_cupy.py`](compare/saxpy_cupy.py), [`saxpy_numba.py`](compare/saxpy_numba.py),
[`saxpy_triton.py`](compare/saxpy_triton.py), [`saxpy_taichi.py`](compare/saxpy_taichi.py),
[`saxpy.mojo`](compare/saxpy.mojo), and [`saxpy.cu`](compare/saxpy.cu). Each times
the best of warm launches with the device synchronized, over five processes;
milliseconds. [`examples/compare.py`](../compare.py) held all seven to the
same two answers first.

| | PPY `cuda.launch` | CuPy | Numba CUDA | Triton | Taichi | Mojo | CUDA C |
|---|---:|---:|---:|---:|---:|---:|---:|
| saxpy, arrays on the device | 0.66 ± 0.03 | 0.67 ± 0.02 | 0.69 ± 0.02 | 0.70 ± 0.03 | 0.69 ± 0.08 | 0.52 ± 0.02 | **0.51 ± 0.00** |
| block max, arrays on the device | 2.01 ± 0.02 | 1.96 ± 0.01 | 2.00 ± 0.02 | 0.79 ± 0.02 | **0.25 ± 0.01** | 1.96 ± 0.02 | 1.88 ± 0.00 |
| saxpy, arrays copied in and out per launch | 172.53 ± 1.62 | 47.46 ± 1.54 | 35.58 ± 0.28 | 46.27 ± 0.56 | 146.83 ± 9.47 | **27.25 ± 0.41** | 29.95 ± 0.34 |
| block max, array copied in per launch | 83.04 ± 0.45 | 12.39 ± 0.54 | 13.41 ± 0.10 | 10.86 ± 0.07 | 26.15 ± 0.31 | **10.79 ± 0.20** | 10.83 ± 0.18 |

What each port asked for:

- **PPY** is the source above: a Python function with `cuda.global_id()`,
  shared memory, a barrier, a shuffle, launched with `cuda.launch`; the same
  file runs on CPython through the reference launch.
- **CuPy** is one line for saxpy, an `ElementwiseKernel`; its block max is
  CUDA C in a string handed to `RawKernel`.
- **Numba CUDA** reads like PPY: `@cuda.jit`, `cuda.grid(1)`,
  `cuda.shared.array`, `cuda.syncthreads()`, `cuda.shfl_xor_sync`.
- **Triton** has no thread to name: a program owns a block of 64 as one
  vector, and the block max is `tl.max` over it, with no shared memory and
  no shuffle to write. Its launcher wants a `data_ptr()` and a `dtype`, which
  a six-line wrapper gives it over CuPy memory.
- **Taichi** has no thread, block, or shared memory either: the outer loop of
  a kernel is parallel over the blocks and the max of each block is an inner
  serial loop over a `ti.field`. `ti.init(arch=ti.cuda, default_fp=ti.f64)`
  is what makes the answers match, and on WSL the driver has to be put on
  the library path by hand.
- **Mojo** writes the kernel as a `def` with `global_idx`, `thread_idx`, a
  `stack_allocation` in `AddressSpace.SHARED`, `barrier()` from MAX's
  `max.gpu.sync`, and `shuffle_xor` -- which has no `Float64` form, so the
  double crosses as its bits. Kernel arguments must be fixed-width
  (`Int64`, not `Int`), a parameter may not be called `out`, and the launch is
  `DeviceContext.enqueue_function` with `grid_dim` and `block_dim`.
- **CUDA C** is the kernel the others are approximating, timed with CUDA
  events around the launch alone, so its rows carry no host-side overhead.

With the arrays on the device, a launch is the kernel: PPY, CuPy, Numba,
Mojo, and CUDA C run the same block max the same way, and on saxpy the
Python-hosted ports sit about 0.15 ms above Mojo and CUDA C, which is what
a launch through the interpreter costs. Triton and Taichi are faster on it because
they were written without the shared-memory exchange -- one vector
reduction per block is a different kernel, and the fair statement is that
their programming models do not ask for the exchange at all. The copying
rows are the other memory model, a `native.stack_alloc` array sent in and
brought back on every launch; there PPY's copies are four to six times
slower than CuPy's, Numba's, Triton's, and Mojo's for the same traffic, and
Taichi's `from_numpy` is slower still. A program keeps its data on the
device by allocating it there.

NVIDIA GeForce RTX 5080 Laptop GPU, driver 610.71, CUDA 13.3; CuPy 14.2.0,
Numba 0.67.0, Triton 3.8.0, Taichi 1.7.4 on CPython 3.12.13; Mojo 1.0.0
with MAX 26.5; nvcc 13.3; PPY on CPython 3.13.13.

## Run it

```bash
python  saxpy.ppy
ppy run saxpy.ppy
ppy emit cuda saxpy.ppy
ppy emit hip saxpy.ppy
ppy emit ptx saxpy.ppy
ppy inspect saxpy.ppy --stage gpu
```

<!-- outputs:start -->
## What it prints

**`python  saxpy.ppy`**

```text
90000.0
100.0 98.0
# kernels compiled for a device here: False
```

**`ppy run saxpy.ppy`**

```text
90000.0
100.0 98.0
# kernels compiled for a device here: True
```

**`ppy emit cuda saxpy.ppy`**

<details markdown="1">
<summary>79 lines</summary>

```text
/* saxpy: generated by ppy, CUDA C++ */
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

static inline int ppy_ovf_add_i64(int64_t a, int64_t b, int64_t *out) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_add_overflow(a, b, out);
#else
    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) {
        return 1;
    }
    *out = a + b;
    return 0;
#endif
}

extern "C" {

static __device__ double ppy_saxpy_fma(double a, double x, double y);
__global__ void ppy_saxpy_saxpy(int64_t n, double a, const double *x, double *y);
__global__ void ppy_saxpy_block_max(const double *x, double *out);
int32_t ppy_saxpy_run(int64_t n, double a, const double *x, double *y, int64_t *out);

static __device__ double ppy_saxpy_fma(double a, double x, double y) {
    return a * x + y;
}

__global__ void ppy_saxpy_saxpy(int64_t n, double a, const double *x, double *y) {
    int64_t t1 = (int64_t)blockIdx.x;
    int64_t t2 = (int64_t)blockDim.x;
    int64_t t3 = (int64_t)threadIdx.x;
    int64_t i = int64_t(uint64_t(t1) * uint64_t(t2) + uint64_t(t3));
    if (i < n) {
        double *slot = y + i;
        *slot = ppy_saxpy_fma(a, x[i], *slot);
    }
}

__global__ void ppy_saxpy_block_max(const double *x, double *out) {
    __shared__ double shared[64];

    double *parked = shared;
    int64_t tid = (int64_t)threadIdx.x;
    int64_t t1 = (int64_t)blockIdx.x;
    int64_t t2 = (int64_t)blockDim.x;
    int64_t t3 = (int64_t)threadIdx.x;
    parked[tid] = x[int64_t(uint64_t(t1) * uint64_t(t2) + uint64_t(t3))];
    __syncthreads();
    double mine = parked[tid];
    double other = __shfl_xor_sync(0xffffffffu, mine, (int)1);
    mine = other > mine ? other : mine;
    parked[tid] = mine;
    __syncthreads();
    if (tid == 0) {
        double best = *parked;
        int64_t t4 = (int64_t)blockDim.x;
        int64_t k = 1;
        while (k < t4) {
            double candidate = parked[k];
            best = candidate > best ? candidate : best;
            k = int64_t(uint64_t(k) + 1u);
        }
        int64_t t5 = (int64_t)blockIdx.x;
        out[t5] = best;
    }
}

int32_t ppy_saxpy_run(int64_t n, double a, const double *x, double *y, int64_t *out) {
    int64_t t1;
    if (ppy_ovf_add_i64(n, 255, &t1)) return 1; /* arith.ok */
    ppy_saxpy_saxpy<<<dim3((unsigned)(t1 / 256 - (t1 % 256 < 0)), (unsigned)1, (unsigned)1), dim3((unsigned)256, (unsigned)1, (unsigned)1)>>>(n, a, x, y);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1; /* launch.ok */
    *out = 0;
    return 0;
}

} /* extern "C" */
```

</details>

**`ppy emit hip saxpy.ppy`**

<details markdown="1">
<summary>79 lines</summary>

```text
/* saxpy: generated by ppy, HIP C++ */
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

static inline int ppy_ovf_add_i64(int64_t a, int64_t b, int64_t *out) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_add_overflow(a, b, out);
#else
    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) {
        return 1;
    }
    *out = a + b;
    return 0;
#endif
}

extern "C" {

static __device__ double ppy_saxpy_fma(double a, double x, double y);
__global__ void ppy_saxpy_saxpy(int64_t n, double a, const double *x, double *y);
__global__ void ppy_saxpy_block_max(const double *x, double *out);
int32_t ppy_saxpy_run(int64_t n, double a, const double *x, double *y, int64_t *out);

static __device__ double ppy_saxpy_fma(double a, double x, double y) {
    return a * x + y;
}

__global__ void ppy_saxpy_saxpy(int64_t n, double a, const double *x, double *y) {
    int64_t t1 = (int64_t)blockIdx.x;
    int64_t t2 = (int64_t)blockDim.x;
    int64_t t3 = (int64_t)threadIdx.x;
    int64_t i = int64_t(uint64_t(t1) * uint64_t(t2) + uint64_t(t3));
    if (i < n) {
        double *slot = y + i;
        *slot = ppy_saxpy_fma(a, x[i], *slot);
    }
}

__global__ void ppy_saxpy_block_max(const double *x, double *out) {
    __shared__ double shared[64];

    double *parked = shared;
    int64_t tid = (int64_t)threadIdx.x;
    int64_t t1 = (int64_t)blockIdx.x;
    int64_t t2 = (int64_t)blockDim.x;
    int64_t t3 = (int64_t)threadIdx.x;
    parked[tid] = x[int64_t(uint64_t(t1) * uint64_t(t2) + uint64_t(t3))];
    __syncthreads();
    double mine = parked[tid];
    double other = __shfl_xor(mine, (int)1);
    mine = other > mine ? other : mine;
    parked[tid] = mine;
    __syncthreads();
    if (tid == 0) {
        double best = *parked;
        int64_t t4 = (int64_t)blockDim.x;
        int64_t k = 1;
        while (k < t4) {
            double candidate = parked[k];
            best = candidate > best ? candidate : best;
            k = int64_t(uint64_t(k) + 1u);
        }
        int64_t t5 = (int64_t)blockIdx.x;
        out[t5] = best;
    }
}

int32_t ppy_saxpy_run(int64_t n, double a, const double *x, double *y, int64_t *out) {
    int64_t t1;
    if (ppy_ovf_add_i64(n, 255, &t1)) return 1; /* arith.ok */
    ppy_saxpy_saxpy<<<dim3((unsigned)(t1 / 256 - (t1 % 256 < 0)), (unsigned)1, (unsigned)1), dim3((unsigned)256, (unsigned)1, (unsigned)1)>>>(n, a, x, y);
    if (hipDeviceSynchronize() != hipSuccess) return 1; /* launch.ok */
    *out = 0;
    return 0;
}

} /* extern "C" */
```

</details>

**`ppy emit ptx saxpy.ppy`**

<details markdown="1">
<summary>146 lines</summary>

```text
//
// Generated by LLVM NVPTX Back-End
//

.version 7.0
.target sm_70
.address_size 64

	// .globl	ppy_saxpy_saxpy
// ppy_saxpy_block_max_shared1 has been demoted

.visible .entry ppy_saxpy_saxpy(
	.param .u64 ppy_saxpy_saxpy_param_0,
	.param .f64 ppy_saxpy_saxpy_param_1,
	.param .u64 .ptr .align 1 ppy_saxpy_saxpy_param_2,
	.param .u64 .ptr .align 1 ppy_saxpy_saxpy_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<4>;
	.reg .b64 	%rd<17>;

	ld.param.b64 	%rd5, [ppy_saxpy_saxpy_param_0];
	ld.param.b64 	%rd6, [ppy_saxpy_saxpy_param_3];
	cvta.to.global.u64 	%rd1, %rd6;
	ld.param.b64 	%rd7, [ppy_saxpy_saxpy_param_2];
	cvta.to.global.u64 	%rd2, %rd7;
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mul.wide.u32 	%rd8, %r1, %r2;
	mov.u32 	%r3, %tid.x;
	cvt.u64.u32 	%rd9, %r3;
	add.s64 	%rd3, %rd8, %rd9;
	setp.ge.s64 	%p1, %rd3, %rd5;
	@%p1 bra 	$L__BB0_2;
	ld.param.b64 	%rd4, [ppy_saxpy_saxpy_param_1];
	shl.b64 	%rd10, %rd3, 3;
	add.s64 	%rd11, %rd1, %rd10;
	add.s64 	%rd12, %rd2, %rd10;
	ld.global.b64 	%rd13, [%rd12];
	ld.global.b64 	%rd14, [%rd11];
	mul.rn.f64 	%rd15, %rd4, %rd13;
	add.rn.f64 	%rd16, %rd15, %rd14;
	st.global.b64 	[%rd11], %rd16;
$L__BB0_2:
	ret;

}
	// .globl	ppy_saxpy_block_max
.visible .entry ppy_saxpy_block_max(
	.param .u64 .ptr .align 1 ppy_saxpy_block_max_param_0,
	.param .u64 .ptr .align 1 ppy_saxpy_block_max_param_1
)
{
	.reg .pred 	%p<13>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<40>;
	// demoted variable
	.shared .align 8 .b8 ppy_saxpy_block_max_shared1[512];
	ld.param.b64 	%rd6, [ppy_saxpy_block_max_param_0];
	cvta.to.global.u64 	%rd7, %rd6;
	ld.param.b64 	%rd8, [ppy_saxpy_block_max_param_1];
	cvta.to.global.u64 	%rd1, %rd8;
	mov.u32 	%r1, %tid.x;
	mul.wide.u32 	%rd9, %r1, 8;
	mov.b64 	%rd10, ppy_saxpy_block_max_shared1;
	add.s64 	%rd11, %rd10, %rd9;
	mov.u32 	%r2, %ctaid.x;
	mov.u32 	%r3, %ntid.x;
	mul.wide.u32 	%rd12, %r2, %r3;
	shl.b64 	%rd13, %rd12, 3;
	add.s64 	%rd14, %rd7, %rd13;
	add.s64 	%rd15, %rd14, %rd9;
	ld.global.b64 	%rd16, [%rd15];
	st.shared.b64 	[%rd11], %rd16;
	bar.sync 	0;
	ld.shared.b64 	%rd17, [%rd11];
	cvt.u32.u64 	%r4, %rd17;
	shfl.sync.bfly.b32 	%r5, %r4, 1, 31, -1;
	{ .reg .b32 tmp; mov.b64 {tmp, %r6}, %rd17; }
	shfl.sync.bfly.b32 	%r7, %r6, 1, 31, -1;
	cvt.u64.u32 	%rd18, %r7;
	shl.b64 	%rd19, %rd18, 32;
	cvt.u64.u32 	%rd20, %r5;
	or.b64 	%rd21, %rd19, %rd20;
	setp.lt.f64 	%p1, %rd17, %rd21;
	selp.f64 	%rd22, %rd21, %rd17, %p1;
	st.shared.b64 	[%rd11], %rd22;
	bar.sync 	0;
	setp.ne.b32 	%p2, %r1, 0;
	@%p2 bra 	$L__BB1_9;
	cvt.u64.u32 	%rd2, %r2;
	cvt.u64.u32 	%rd3, %r3;
	cvt.u32.u64 	%r8, %rd3;
	ld.shared.b64 	%rd39, [ppy_saxpy_block_max_shared1];
	setp.lt.u32 	%p3, %r8, 2;
	@%p3 bra 	$L__BB1_8;
	add.s64 	%rd4, %rd3, -1;
	and.b64 	%rd37, %rd4, 3;
	add.s32 	%r9, %r8, -2;
	setp.lt.u32 	%p4, %r9, 3;
	mov.b64 	%rd36, 1;
	@%p4 bra 	$L__BB1_6;
	add.s64 	%rd34, %rd10, 16;
	and.b64 	%rd5, %rd4, -4;
	mov.b64 	%rd35, 0;
$L__BB1_4:
	ld.shared.b64 	%rd23, [%rd34+-8];
	setp.gt.f64 	%p5, %rd23, %rd39;
	selp.f64 	%rd24, %rd23, %rd39, %p5;
	ld.shared.b64 	%rd25, [%rd34];
	setp.gt.f64 	%p6, %rd25, %rd24;
	selp.f64 	%rd26, %rd25, %rd24, %p6;
	ld.shared.b64 	%rd27, [%rd34+8];
	setp.gt.f64 	%p7, %rd27, %rd26;
	selp.f64 	%rd28, %rd27, %rd26, %p7;
	ld.shared.b64 	%rd29, [%rd34+16];
	setp.gt.f64 	%p8, %rd29, %rd28;
	selp.f64 	%rd39, %rd29, %rd28, %p8;
	add.s64 	%rd35, %rd35, 4;
	add.s64 	%rd34, %rd34, 32;
	setp.ne.b64 	%p9, %rd5, %rd35;
	@%p9 bra 	$L__BB1_4;
	setp.eq.b64 	%p10, %rd37, 0;
	add.s64 	%rd36, %rd35, 1;
	@%p10 bra 	$L__BB1_8;
$L__BB1_6:
	shl.b64 	%rd30, %rd36, 3;
	add.s64 	%rd38, %rd10, %rd30;
$L__BB1_7:
	.pragma "nounroll";
	ld.shared.b64 	%rd31, [%rd38];
	setp.gt.f64 	%p11, %rd31, %rd39;
	selp.f64 	%rd39, %rd31, %rd39, %p11;
	add.s64 	%rd38, %rd38, 8;
	add.s64 	%rd37, %rd37, -1;
	setp.ne.b64 	%p12, %rd37, 0;
	@%p12 bra 	$L__BB1_7;
$L__BB1_8:
	shl.b64 	%rd32, %rd2, 3;
	add.s64 	%rd33, %rd1, %rd32;
	st.global.b64 	[%rd33], %rd39;
$L__BB1_9:
	ret;

}
```

</details>

**`ppy inspect saxpy.ppy --stage gpu`**

<details markdown="1">
<summary>165 lines</summary>

```text
; ---- saxpy [gpu] ----
func @saxpy_fma(%a: f64, %x: f64, %y: f64) -> f64 attrs {effects = [], gpu.kind = "device", ppy.abi = "ppy", ppy.qualname = "saxpy.fma", ppy.releases_gil = true, ppy.symbol = "ppy_saxpy_fma"} loc("examples/38_cuda/saxpy.ppy":5:0) {
^entry:
    %a_addr = core.alloca : ptr<f64, stack> loc("examples/38_cuda/saxpy.ppy":5:0)
    core.store %a, %a_addr
    %x_addr = core.alloca : ptr<f64, stack>
    core.store %x, %x_addr
    %y_addr = core.alloca : ptr<f64, stack>
    core.store %y, %y_addr
    %0 = core.load %a_addr : f64 loc("examples/38_cuda/saxpy.ppy":6:4)
    %1 = core.load %x_addr : f64
    %2 = core.mul %0, %1 : f64
    %3 = core.load %y_addr : f64
    %4 = core.add %2, %3 : f64
    core.ret %4
}

func @saxpy_saxpy(%n: i64, %a: f64, %x: ptr<f64, generic, const>, %y: ptr<f64>) -> () attrs {effects = ["read_memory", "write_memory"], gpu.kind = "kernel", ppy.abi = "ppy", ppy.qualname = "saxpy.saxpy", ppy.releases_gil = true, ppy.symbol = "ppy_saxpy_saxpy"} loc("examples/38_cuda/saxpy.ppy":10:0) {
^entry:
    %n_addr = core.alloca : ptr<i64, stack> loc("examples/38_cuda/saxpy.ppy":10:0)
    core.store %n, %n_addr
    %a_addr = core.alloca : ptr<f64, stack>
    core.store %a, %a_addr
    %x_addr = core.alloca : ptr<ptr<f64, generic, const>, stack>
    core.store %x, %x_addr
    %y_addr = core.alloca : ptr<ptr<f64>, stack>
    core.store %y, %y_addr
    %0 = gpu.block_id.x : index loc("examples/38_cuda/saxpy.ppy":11:4)
    %1 = gpu.block_dim.x : index
    %2 = core.mul %0, %1 {overflow = "wrap"} : index
    %3 = gpu.thread_id.x : index
    %4 = core.add %2, %3 {overflow = "wrap"} : index
    %5 = core.cast %4 : i64
    %i_addr = core.alloca : ptr<i64, stack>
    core.store %5, %i_addr
    %6 = core.load %i_addr : i64 loc("examples/38_cuda/saxpy.ppy":12:4)
    %7 = core.load %n_addr : i64
    %n_entry = core.load %n_addr : i64
    %8 = core.cmp.lt %6, %7 : bool
    %slot_addr = core.alloca : ptr<ptr<f64>, stack>
    core.cond_br %8, ^then1, ^else2 loc("examples/38_cuda/saxpy.ppy":12:4)
^then1:
    %9 = core.load %y_addr : ptr<f64> loc("examples/38_cuda/saxpy.ppy":13:8)
    %10 = core.load %i_addr : i64
    %11 = core.ptr_offset %9, %10 : ptr<f64>
    core.store %11, %slot_addr
    %12 = core.load %slot_addr : ptr<f64> loc("examples/38_cuda/saxpy.ppy":14:8)
    %13 = core.load %a_addr : f64
    %14 = core.load %x_addr : ptr<f64, generic, const>
    %15 = core.load %i_addr : i64
    %16 = core.ptr_offset %14, %15 : ptr<f64, generic, const>
    %17 = core.load %16 : f64
    %18 = core.load %slot_addr : ptr<f64>
    %19 = core.load %18 : f64
    %20 = core.call %13, %17, %19 {callee = @saxpy_fma} : f64
    core.store %20, %12
    core.br ^endif3
^else2:
    core.br ^endif3 loc("examples/38_cuda/saxpy.ppy":14:8)
^endif3:
    core.ret loc("examples/38_cuda/saxpy.ppy":14:8)
}

func @saxpy_block_max(%x: ptr<f64, generic, const>, %out: ptr<f64>) -> () attrs {effects = ["alloc", "may_raise", "read_memory", "sync", "write_memory"], gpu.kind = "kernel", ppy.abi = "ppy", ppy.qualname = "saxpy.block_max", ppy.releases_gil = true, ppy.symbol = "ppy_saxpy_block_max"} loc("examples/38_cuda/saxpy.ppy":18:0) {
^entry:
    %x_addr = core.alloca : ptr<ptr<f64, generic, const>, stack> loc("examples/38_cuda/saxpy.ppy":18:0)
    core.store %x, %x_addr
    %out_addr = core.alloca : ptr<ptr<f64>, stack>
    core.store %out, %out_addr
    %0 = gpu.shared_alloc {count = 64} : ptr<f64, shared> loc("examples/38_cuda/saxpy.ppy":19:4)
    %parked_addr = core.alloca : ptr<ptr<f64, shared>, stack>
    core.store %0, %parked_addr
    %1 = gpu.thread_id.x : index loc("examples/38_cuda/saxpy.ppy":20:4)
    %2 = core.cast %1 : i64
    %tid_addr = core.alloca : ptr<i64, stack>
    core.store %2, %tid_addr
    %3 = core.load %parked_addr : ptr<f64, shared> loc("examples/38_cuda/saxpy.ppy":21:4)
    %4 = core.load %tid_addr : i64
    %5 = core.ptr_offset %3, %4 : ptr<f64, shared>
    %6 = core.load %x_addr : ptr<f64, generic, const>
    %7 = gpu.block_id.x : index
    %8 = gpu.block_dim.x : index
    %9 = core.mul %7, %8 {overflow = "wrap"} : index
    %10 = gpu.thread_id.x : index
    %11 = core.add %9, %10 {overflow = "wrap"} : index
    %12 = core.cast %11 : i64
    %13 = core.ptr_offset %6, %12 : ptr<f64, generic, const>
    %14 = core.load %13 : f64
    core.store %14, %5
    gpu.barrier loc("examples/38_cuda/saxpy.ppy":22:4)
    %15 = core.load %parked_addr : ptr<f64, shared> loc("examples/38_cuda/saxpy.ppy":23:4)
    %16 = core.load %tid_addr : i64
    %17 = core.ptr_offset %15, %16 : ptr<f64, shared>
    %18 = core.load %17 : f64
    %mine_addr = core.alloca : ptr<f64, stack>
    core.store %18, %mine_addr
    %19 = core.load %mine_addr : f64 loc("examples/38_cuda/saxpy.ppy":24:4)
    %20 = core.const 1 : i64
    %21 = gpu.subgroup_shuffle.xor %19, %20 : f64
    %other_addr = core.alloca : ptr<f64, stack>
    core.store %21, %other_addr
    %22 = core.load %other_addr : f64 loc("examples/38_cuda/saxpy.ppy":25:4)
    %23 = core.load %mine_addr : f64
    %24 = core.cmp.gt %22, %23 : bool
    %25 = core.load %other_addr : f64
    %26 = core.load %mine_addr : f64
    %27 = core.select %24, %25, %26 : f64
    core.store %27, %mine_addr
    %28 = core.load %parked_addr : ptr<f64, shared> loc("examples/38_cuda/saxpy.ppy":26:4)
    %29 = core.load %tid_addr : i64
    %30 = core.ptr_offset %28, %29 : ptr<f64, shared>
    %31 = core.load %mine_addr : f64
    core.store %31, %30
    gpu.barrier loc("examples/38_cuda/saxpy.ppy":27:4)
    %32 = core.load %tid_addr : i64 loc("examples/38_cuda/saxpy.ppy":28:4)
    %33 = core.const 0 : i64
    %34 = core.cmp.eq %32, %33 : bool
    %best_addr = core.alloca : ptr<f64, stack>
    %35 = core.const 1 : i64
    %36 = core.const 1 : i64
    %k_addr = core.alloca : ptr<i64, stack>
    %candidate_addr = core.alloca : ptr<f64, stack>
    core.cond_br %34, ^then1, ^else2 loc("examples/38_cuda/saxpy.ppy":28:4)
^then1:
    %37 = core.load %parked_addr : ptr<f64, shared> loc("examples/38_cuda/saxpy.ppy":29:8)
    %38 = core.load %37 : f64
    core.store %38, %best_addr
    %39 = gpu.block_dim.x : index loc("examples/38_cuda/saxpy.ppy":30:8)
    %40 = core.cast %39 : i64
    core.store %35, %k_addr
    core.br ^for.head6
^else2:
    core.br ^endif3 loc("examples/38_cuda/saxpy.ppy":33:8)
^endif3:
    core.ret loc("examples/38_cuda/saxpy.ppy":33:8)
^for.head6:
    %41 = core.load %k_addr : i64 loc("examples/38_cuda/saxpy.ppy":30:8)
    %42 = core.cmp.lt %41, %40 : bool
    core.cond_br %42, ^for.body7, ^for.end9
^for.body7:
    %43 = core.load %parked_addr : ptr<f64, shared> loc("examples/38_cuda/saxpy.ppy":31:12)
    %44 = core.load %k_addr : i64
    %45 = core.ptr_offset %43, %44 : ptr<f64, shared>
    %46 = core.load %45 : f64
    core.store %46, %candidate_addr
    %47 = core.load %candidate_addr : f64 loc("examples/38_cuda/saxpy.ppy":32:12)
    %48 = core.load %best_addr : f64
    %49 = core.cmp.gt %47, %48 : bool
    %50 = core.load %candidate_addr : f64
    %51 = core.load %best_addr : f64
    %52 = core.select %49, %50, %51 : f64
    core.store %52, %best_addr
    %53 = core.load %k_addr : i64
    %54 = core.add %53, %36 {overflow = "wrap"} : i64
    core.store %54, %k_addr
    core.br ^for.head6
^for.end9:
    %55 = core.load %out_addr : ptr<f64> loc("examples/38_cuda/saxpy.ppy":33:8)
    %56 = gpu.block_id.x : index
    %57 = core.cast %56 : i64
    %58 = core.ptr_offset %55, %57 : ptr<f64>
    %59 = core.load %best_addr : f64
    core.store %59, %58
    core.br ^endif3
}
```

</details>

<!-- outputs:end -->

Read on: [GPU kernels](../../docs/guide/gpu.md) ·
[The IR: the gpu dialect](../../docs/internals/ir.md)

`saxpy.ppy` is hand-written; there is no `.py` source and no conversion step.
