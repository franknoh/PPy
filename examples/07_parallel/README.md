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
[`fused_numba.py`](compare/fused_numba.py), [`fused_jax.py`](compare/fused_jax.py).
Milliseconds, best of five calls, over five processes.

**PPY** -- the NumPy expression in a function, `@ppy.parallel` to split it;
the serial row is the same expression fused into one loop with no
decorator, and `strict_total` is NumPy's sum in NumPy's order:

```python
@ppy.pure
@ppy.parallel
@ppy.opt(3)
def parallel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b + a) * (b - a) + a * 0.5 - b * 0.25
```

**NumPy** is the expression as written -- five operations, four 64 MB
temporaries, one thread. **numexpr** is the expression as a string,
compiled to its own virtual machine and evaluated in chunks across threads:

```python
ne.evaluate("(x * y + x) * (y - x) + x * 0.5 - y * 0.25", local_dict={"x": x, "y": y})
```

**Numba** is an explicit loop under `@njit(parallel=True)` writing into
`np.empty_like`:

```python
@njit(parallel=True)
def fused(a, b):
    out = np.empty_like(a)
    for i in prange(a.shape[0]):
        out[i] = (a[i] * b[i] + a[i]) * (b[i] - a[i]) + a[i] * 0.5 - b[i] * 0.25
    return out
```

**JAX** is the expression under `jax.jit` on the CPU, fused by XLA and run
on its thread pool, with `jax_enable_x64`:

```python
@jax.jit
def fused(a, b):
    return (a * b + a) * (b - a) + a * 0.5 - b * 0.25
```

<!-- compare:start -->
| | PPY | NumPy | numexpr | Numba `prange` | JAX `jit` |
|---|---:|---:|---:|---:|---:|
| fused, serial | 11.88 ± 0.30 | — | — | — | — |
| fused | 4.31 ± 0.12 | 56.63 ± 1.32 | 7.10 ± 0.27 | **3.76 ± 0.51** | 6.83 ± 0.57 |
| sum of squares | 13.28 ± 0.35 | 12.37 ± 0.25 | 6.84 ± 0.11 | 4.15 ± 0.11 | **0.75 ± 0.04** |
| sum of squares, relaxed | 4.53 ± 0.10 | — | — | — | — |
<!-- compare:end -->

Fusing the expression is what removes NumPy's four temporaries: the serial
fused loop reads the two inputs once and writes the output once, and
splitting that loop across the cores is a memory-bandwidth problem that
PPY, Numba, JAX, and numexpr solve the same way, within a couple of
milliseconds of each other. The fused kernel also checks its own result in
the same loop -- one add-reduction of non-finite elements the vectorizer
keeps in a register, one guard after -- which is how it keeps NumPy's
floating-point reporting without a second pass over 64 MB. The ordered sum
is NumPy's order and NumPy's time; the relaxed one vectorizes on one
thread, and JAX's reduction, reassociated across its pool, is the row that
shows what the same permission buys with threads.

Intel Core Ultra 9 386H (16 threads), threads backend; NumPy 2.5.3, numexpr
2.14.2, Numba 0.67.0 on CPython 3.12.13, JAX 0.11.1 on CPython 3.13.13,
PPY on CPython 3.13.13, from a checkout on a native filesystem.

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
fused serial        114.8 ms   sample=-0.249979000059
fused parallel      123.4 ms   sample=-0.249979000059
numpy               121.4 ms   sample=-0.249979000059
bit-identical: True
strict == numpy: True
relaxed close   : True
```

**`ppy     parallel.ppy`**

```text
fused serial        113.8 ms   sample=-0.249979000059
fused parallel      127.6 ms   sample=-0.249979000059
numpy               132.2 ms   sample=-0.249979000059
bit-identical: True
strict == numpy: True
relaxed close   : True
```

**`ppy run parallel.ppy`**

```text
fused serial         14.9 ms   sample=-0.249979000059
fused parallel        5.6 ms   sample=-0.249979000059
numpy                18.3 ms   sample=-0.249979000059
bit-identical: True
strict == numpy: True
relaxed close   : True
```

<!-- outputs:end -->

Read on: [Parallel loops](../../docs/guide/parallel.md) ·
[NumPy fusion](../05_numpy/README.md)

`parallel.ppy` is hand-written; there is no `.py` source and no conversion step.
