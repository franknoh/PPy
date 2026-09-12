# Buffers and JIT

Borrowed memory, reassociation, and specialization: three decisions that set
how fast a numeric function runs, each made once in a signature or a
decorator.

## `Buffer[T]` is borrowed, a list is copied

```python
@ppy.pure
@ppy.opt(3)
def total_view(values: Buffer[float]) -> float:
    result: float = 0.0
    for i in range(len(values)):
        result += values[i]
    return result
```

`total_list` takes a `list[float]` and is copied into a buffer on every
call; `total_view` takes a `Buffer[float]` — a `memoryview` over an
`array.array` — and reads the caller's memory in place. Same loop, same
sum, about 10× apart on 8,192 elements, because one of them pays for 8,192
boxed floats per call. A buffer is borrowed unless the signature says
otherwise, so the callee may not keep it (`E1612`) or write through it
without `Mut[...]` (`E1613`).

## Reassociation is opt-in

`dot_strict` and `dot_relaxed` are the same dot product. The strict one
keeps CPython's accumulation order and matches it bit for bit; the
`@ppy.fastmath` one lets LLVM vectorize the reduction and differs in the
last bits — the program prints the gap. The choice is per function, and the
default is the exact one.

## `@ppy.jit` specializes on the values it sees

```python
@ppy.jit(threshold=4, max_specializations=4)
@ppy.pure
@ppy.opt(3)
def digest_jit(values: Buffer[int], modulus: int) -> int:
```

After four calls with the same `modulus`, a version specialized to that
constant is compiled; `% 1000003` becomes multiply-and-shift instead of a
division. The guard on the value is in the generated C wrapper, and a call
with a different modulus falls back to the generic version or compiles
another specialization, up to four.

## Compared with Numba, Cython, NumPy, and C

The three kernels over 8,192 elements -- a sum, a dot product in order and
reassociated, a modular digest -- each call timed as the mean of 2,000
calls, best of five rounds, milliseconds per call over five processes. The
programs are in [`compare/`](compare/): [`kernels_bench.ppy`](compare/kernels_bench.ppy),
[`kernels_numba.py`](compare/kernels_numba.py), [`kernels_cython.pyx`](compare/kernels_cython.pyx),
[`kernels_numpy.py`](compare/kernels_numpy.py), [`kernels.c`](compare/kernels.c).
At this size a call is a few microseconds, so the table is as much about the
call boundary as about the loop.

**PPY** -- the loop as written, a `Buffer[float]` borrowed from an
`array.array`, `@ppy.fastmath` where reassociation is allowed:

```python
@ppy.pure
@ppy.opt(3)
def dot(a: Buffer[float], b: Buffer[float]) -> float:
    result: float = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result
```

**Numba** -- the same loop under `@njit`, no annotations, NumPy arrays,
`fastmath=True` for the relaxed dot; its dispatcher types the arguments on
every call:

```python
@njit
def dot(a, b):
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result
```

**Cython** -- typed memoryviews and `cdef` locals in a `.pyx`, built into an
extension by `cythonize -i`; `boundscheck=False` is what makes it a C loop,
and there is no relaxed dot to write:

```cython
cpdef double dot(double[::1] a, double[::1] b):
    cdef double result = 0.0
    cdef Py_ssize_t i
    for i in range(a.shape[0]):
        result += a[i] * b[i]
    return result
```

**NumPy** -- no loop: `np.dot` is BLAS, `np.sum` pairwise, and
`(counts % m).sum()` builds a temporary. **C** is the loop with no Python
around it, timed inside the process: the floor.

<!-- compare:start -->
| | PPY `ppy run` | Numba `@njit` | Cython | NumPy | C (the loop alone) |
|---|---:|---:|---:|---:|---:|
| total | 0.0039 ± 0.0000 | 0.0039 ± 0.0000 | 0.0045 ± 0.0008 | **0.0024 ± 0.0003** | 0.0038 ± 0.0000 |
| dot | 0.0039 ± 0.0000 | 0.0039 ± 0.0001 | 0.0049 ± 0.0007 | **0.0015 ± 0.0002** | 0.0038 ± 0.0000 |
| dot_relaxed | **0.0011 ± 0.0000** | 0.0011 ± 0.0000 | 0.0046 ± 0.0009 | 0.0015 ± 0.0002 | 0.0038 ± 0.0000 |
| digest | 0.0080 ± 0.0000 | 0.0121 ± 0.0001 | 0.0106 ± 0.0029 | 0.0205 ± 0.0026 | **0.0049 ± 0.0000** |
<!-- compare:end -->

The ordered sums sit on the C loop in PPY, Numba, and Cython alike, a
microsecond of call above it; the reassociated dot is where
`@ppy.fastmath` and Numba's `fastmath=True` let the vectorizer in and
Cython, with no such switch, stays scalar. NumPy's `dot` is BLAS and the
fastest row, its `digest` pays for an 8,192-element temporary and is the
slowest. The C `digest` is the loop's floor; what stands between it and
the others is the price of a call through the interpreter.

Intel Core Ultra 9 386H; Numba 0.67.0, Cython 3.3.0, NumPy 2.5.3 on
CPython 3.12.13, gcc 13.3, PPY on CPython 3.13.13, from a checkout on a
native filesystem.

## Run it

```bash
python  buffers_and_jit.ppy
ppy     buffers_and_jit.ppy
ppy run buffers_and_jit.ppy
```

<!-- outputs:start -->
## What it prints

**`python  buffers_and_jit.ppy`**

```text
list[float] copied        104.387 us   
Buffer[float] borrowed    158.837 us   0.66x, same sum: True
dot, strict order         277.649 us   
dot, @ppy.fastmath        257.140 us   1.08x, differs by 0.00e+00
digest, generic           263.962 us   
digest, @ppy.jit          260.730 us   1.01x, same: True
```

**`ppy     buffers_and_jit.ppy`**

```text
list[float] copied        102.127 us   
Buffer[float] borrowed    162.701 us   0.63x, same sum: True
dot, strict order         261.217 us   
dot, @ppy.fastmath        259.543 us   1.01x, differs by 0.00e+00
digest, generic           270.856 us   
digest, @ppy.jit          268.401 us   1.01x, same: True
```

**`ppy run buffers_and_jit.ppy`**

```text
list[float] copied         11.790 us   
Buffer[float] borrowed      3.726 us   3.16x, same sum: True
dot, strict order           4.196 us   
dot, @ppy.fastmath          0.849 us   4.94x, differs by 4.07e-10
digest, generic            11.656 us   
digest, @ppy.jit            8.874 us   1.31x, same: True
```

<!-- outputs:end -->

Read on: [Directives and markers](../../docs/guide/directives.md) ·
[Algorithms](../15_algorithms/README.md)

`buffers_and_jit.ppy` is hand-written; there is no `.py` source and no
conversion step.
