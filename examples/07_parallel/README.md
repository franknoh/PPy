# Parallel fused kernels

`@ppy.parallel` splits a fused NumPy loop across the worker pool. The
output is bit-identical to the serial kernel and to NumPy, because an
elementwise loop has no order to lose; a reduction does, and the compiler
will not split one that reassociates unless the function says so.

## The decorator asks, the analysis answers

```python
@ppy.pure
@ppy.parallel
@ppy.opt(3)
def parallel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b + a) * (b - a) + a * 0.5 - b * 0.25
```

Eight million elements, five operations, one fused loop chunked across
`[tool.ppy.parallel] threads` workers and joined. `serial` is the same
expression without the decorator, and the program checks `np.array_equal`
across serial, parallel, and NumPy — it prints `bit-identical: True` on
every path. A loop the analysis cannot prove splittable stays serial and
says why in an optimization remark.

## A sum keeps its order unless you let it go

```python
@ppy.pure
@ppy.fastmath
def relaxed_total(a: np.ndarray) -> float:
    return np.sum(a * a)


@ppy.pure
def strict_total(a: np.ndarray) -> float:
    return np.sum(a * a)
```

Splitting a floating-point sum changes where the rounding happens, so
`strict_total` is left in NumPy's order and equals NumPy's number exactly.
`@ppy.fastmath` permits the reassociation: `relaxed_total` vectorizes and
splits, and lands within 1e-3 of the strict answer on eight million squares.
The choice is made per function, and the default is the exact one.

`parallel.range` is the spelling for a loop the program itself declares
splittable — pointers, buffers, and reductions included
([parallel range](../35_parallel_range/README.md)). `@ppy.parallel` stays the
switch for fused NumPy loops like these.

## Compared with NumPy, numexpr, Numba, and JAX

The fused expression and the sum of squares over eight million doubles, in
[`compare/`](compare/): [`fused_bench.ppy`](compare/fused_bench.ppy),
[`fused_numpy.py`](compare/fused_numpy.py), [`fused_numexpr.py`](compare/fused_numexpr.py),
[`fused_numba.py`](compare/fused_numba.py), and [`fused_jax.py`](compare/fused_jax.py).
Each prints two elements of the result to twelve digits and the sum to
three, then the best of five calls; [`examples/compare.py`](../compare.py)
runs each five times and reports the mean and standard deviation across
processes, in milliseconds. All five print the same answers: the elementwise
result is bit-identical everywhere, and the sums agree to the digits shown.

| | PPY | NumPy | numexpr | Numba `prange` | JAX `jit` |
|---|---:|---:|---:|---:|---:|
| fused, serial | 20.33 ± 0.33 | — | — | — | — |
| fused | 12.54 ± 0.42 | 56.68 ± 1.49 | 6.52 ± 0.43 | **4.89 ± 1.64** | 6.07 ± 0.68 |
| sum of squares | 18.45 ± 0.39 | 12.63 ± 0.49 | 6.59 ± 0.15 | 4.13 ± 0.03 | **0.72 ± 0.06** |
| sum of squares, relaxed | 7.03 ± 0.21 | — | — | — | — |

What each port asked for:

- **PPY** is the source above: the NumPy expression in a function, `@ppy.parallel`
  to split it, `@ppy.fastmath` to let the sum reassociate. The serial row is
  the same expression fused into one loop with no decorator; `strict_total`
  is NumPy's sum in NumPy's order.
- **NumPy** is the expression as written: five operations, four temporaries
  of 64 MB each, one thread.
- **numexpr** is the expression as a string, compiled to its virtual machine
  and evaluated in chunks across threads; `sum(x * x)` likewise.
- **Numba** is an explicit loop under `@njit(parallel=True)` with `prange`,
  writing into `np.empty_like`; the sum is a serial `@njit` loop, and it
  agrees with the others only to the digits printed.
- **JAX** is the expression under `jax.jit` on the CPU, fused by XLA and
  run on its thread pool, with `jax_enable_x64` so the answers match.

The serial fused loop is well ahead of NumPy's four temporaries, and the
parallel one is the slowest of the four threaded kernels here: Numba, JAX,
and numexpr split the same work across the same cores in half the time.
The ordered sum is NumPy's own order and pays for it; the relaxed one
vectorizes, and JAX's reduction, which reassociates by default, is the row
to compare it with.

Intel Core Ultra 9 386H (16 threads), threads backend; NumPy 2.5.3, numexpr 2.14.2, Numba 0.67.0 on
CPython 3.12.13, JAX 0.11.1 on CPython 3.13.13, PPY on CPython 3.13.13,
from a checkout on a native filesystem.

## Run it

```bash
python  parallel.ppy
ppy     parallel.ppy
ppy run parallel.ppy
```

<!-- outputs:start -->
## What it prints

**`python  parallel.ppy`**

```text
fused serial         87.2 ms   sample=-0.249979000059
fused parallel       99.3 ms   sample=-0.249979000059
numpy               100.8 ms   sample=-0.249979000059
bit-identical: True
strict == numpy: True
relaxed close   : True
```

**`ppy     parallel.ppy`**

```text
fused serial         84.3 ms   sample=-0.249979000059
fused parallel       87.1 ms   sample=-0.249979000059
numpy                89.4 ms   sample=-0.249979000059
bit-identical: True
strict == numpy: True
relaxed close   : True
```

**`ppy run parallel.ppy`**

```text
fused serial         20.3 ms   sample=-0.249979000059
fused parallel       13.6 ms   sample=-0.249979000059
numpy                21.6 ms   sample=-0.249979000059
bit-identical: True
strict == numpy: True
relaxed close   : True
```

<!-- outputs:end -->

Read on: [Parallel loops](../../docs/guide/parallel.md) ·
[NumPy fusion](../05_numpy/README.md)

`parallel.ppy` is hand-written; there is no `.py` source and no conversion step.
