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
| total | 0.0037 ± 0.0000 | 0.0037 ± 0.0001 | 0.0037 ± 0.0000 | **0.0021 ± 0.0001** | 0.0037 ± 0.0001 |
| dot | 0.0037 ± 0.0000 | 0.0038 ± 0.0001 | 0.0037 ± 0.0001 | **0.0013 ± 0.0001** | 0.0036 ± 0.0000 |
| dot_relaxed | **0.0010 ± 0.0000** | 0.0011 ± 0.0000 | 0.0037 ± 0.0000 | 0.0013 ± 0.0000 | 0.0036 ± 0.0000 |
| digest | 0.0075 ± 0.0001 | 0.0116 ± 0.0001 | 0.0081 ± 0.0000 | 0.0209 ± 0.0006 | **0.0047 ± 0.0000** |
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
list[float] copied         97.306 us   
Buffer[float] borrowed    153.282 us   0.63x, same sum: True
dot, strict order         244.091 us   
dot, @ppy.fastmath        238.020 us   1.03x, differs by 0.00e+00
digest, generic           239.351 us   
digest, @ppy.jit          237.332 us   1.01x, same: True
```

**`ppy     buffers_and_jit.ppy`**

```text
list[float] copied         92.352 us   
Buffer[float] borrowed    146.138 us   0.63x, same sum: True
dot, strict order         238.365 us   
dot, @ppy.fastmath        236.017 us   1.01x, differs by 0.00e+00
digest, generic           246.093 us   
digest, @ppy.jit          245.886 us   1.00x, same: True
```

**`ppy run buffers_and_jit.ppy`**

```text
list[float] copied         10.684 us   
Buffer[float] borrowed      3.485 us   3.07x, same sum: True
dot, strict order           3.541 us   
dot, @ppy.fastmath          1.186 us   2.99x, differs by 4.07e-10
digest, generic            10.894 us   
digest, @ppy.jit            7.474 us   1.46x, same: True
```

<!-- outputs:end -->

Read on: [Directives and markers](../../docs/guide/directives.md) ·
[Algorithms](../15_algorithms/README.md)

`buffers_and_jit.ppy` is hand-written; there is no `.py` source and no
conversion step.
