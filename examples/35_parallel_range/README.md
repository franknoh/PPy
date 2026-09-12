# Parallel ranges

`for i in parallel.range(n)` says the iterations may run at once. Under
CPython they run in order, which is one of the orders allowed; natively the
body becomes a function over a chunk of the range, and the build's parallel
backend decides how the chunks run.

## Writes split freely; a float sum keeps its order

```python
def squares(out: native.ptr[int], n: int) -> int:
    for i in parallel.range(n):
        native.store(native.offset(out, i), i * i)
    return native.load(native.offset(out, n - 1))


def dot(a: Buffer[float], b: Buffer[float]) -> float:
    total = 0.0
    for i in parallel.range(len(a)):
        total += a[i] * b[i]
    return total
```

A loop that writes through a pointer or a buffer splits without ceremony.
One `+=` into an outer `int` or `float` is a reduction, lowered as a
`parallel.reduce`: an integer reduction (`count_odd`) splits freely; a
floating-point one keeps its order, because the answer would change
otherwise — `dot` prints twelve digits and they are the same on every
path. `@ppy.fastmath` permits the reassociation, which is why `dot_relaxed`
prints six.

## What the body may not do, and the backends

Assign a variable that lives outside the loop, `break`, or `return`; a
function that does is refused with the reason (`E1650`). `@ppy.parallel` on
a function asks the same of its outermost `range` loops without spelling
`parallel.range`; a loop that does not pass the analysis stays serial and
says why in an optimization remark — `fill` has two, and `--report-opt`
names what happened to each. `[tool.ppy.parallel] backend` chooses
`threads`, `serial`, `simd`, or `openmp` (for `ppy emit c`), and every choice
gives the same answer.

## Compared with Numba, Taichi, Mojo, and NumPy

The same four kernels over eight million elements, each written the way its
tool wants it, in [`compare/`](compare/): [`ranges_bench.ppy`](compare/ranges_bench.ppy)
(this example's kernels, with a relaxed `dot` beside the ordered one),
[`ranges_numba.py`](compare/ranges_numba.py), [`ranges_taichi.py`](compare/ranges_taichi.py),
[`ranges.mojo`](compare/ranges.mojo), [`ranges_numpy.py`](compare/ranges_numpy.py).
Milliseconds, best of five warm calls, over five processes.

**PPY** -- `parallel.range`, typed parameters, the same file on CPython;
`dot` keeps a serial sum's order by design and `@ppy.fastmath` is the
relaxed one:

```python
def dot(a: Buffer[float], b: Buffer[float]) -> float:
    total = 0.0
    for i in parallel.range(len(a)):
        total += a[i] * b[i]
    return total
```

**Numba** -- `@njit(parallel=True)` and `prange`; a `prange` sum
reassociates on its own, so there is no ordered `dot` to write, and
integers wrap silently:

```python
@njit(parallel=True)
def dot(a, b):
    total = 0.0
    for i in prange(len(a)):
        total += a[i] * b[i]
    return total
```

**Taichi** -- arrays are `ti.field`s declared at module level, the outermost
loop of a kernel is parallel by itself, and the defaults overflow in 32 bits
and accumulate in `f32` silently; `ti.init(default_ip=ti.i64, default_fp=ti.f64)`
is what makes the answers match:

```python
@ti.kernel
def dot() -> ti.f64:
    total = 0.0
    for i in range(N):
        total += a_f[i] * b_f[i]
    return total
```

**Mojo** -- a different language: `List[Float64]`, `mut` parameters,
`Int64(i)` where an index meets an element, `parallelize` from
`max.algorithm` taking a function of one index, so a reduction is partial
sums per worker written by hand. **NumPy** has no loop to write and no way
to keep one: `dot` is BLAS, `count_odd` builds two temporaries to count.

<!-- compare:start -->
| | PPY `ppy run` | Numba `prange` | Taichi | Mojo | NumPy |
|---|---:|---:|---:|---:|---:|
| squares | 1.34 ± 0.03 | **0.93 ± 0.06** | 1.16 ± 0.04 | 1.05 ± 0.05 | 13.82 ± 0.52 |
| dot, in order | **6.39 ± 0.40** | — | — | 14.93 ± 1.30 | — |
| dot, reassociated | 1.80 ± 0.13 | 1.34 ± 0.04 | 1.50 ± 0.04 | 1.52 ± 0.10 | **1.33 ± 0.05** |
| count_odd | 0.99 ± 0.04 | **0.70 ± 0.04** | 1.23 ± 0.06 | 1.37 ± 0.19 | 27.64 ± 1.35 |
| fill, serial | 6.17 ± 0.11 | 7.51 ± 0.04 | **4.02 ± 0.38** | 19.57 ± 1.09 | 15.43 ± 0.28 |
<!-- compare:end -->

On the elementwise kernels the four compiled tools are within a few tenths
of a millisecond, which is the memory bandwidth of the machine, and NumPy's
temporaries are ten times behind. The reductions are where the models
differ: PPY's ordered `dot` is the only one that promises a serial sum's
answer, and it is a serial sum; its relaxed `dot`, Numba's and Taichi's are
the same vectorized tree. The serial `fill` is the row that measures a plain
loop with no threads, and Mojo's plain `List` loop is the slow one there.

Intel Core Ultra 9 386H (16 threads), threads backend; Numba 0.67.0, Taichi
1.7.4, NumPy 2.5.3 on CPython 3.12.13, Mojo 1.0.0, PPY on CPython 3.13.13,
from a checkout on a native filesystem.

## Run it

```bash
python  ranges.ppy
ppy run ranges.ppy
ppy build ranges.ppy --report-opt
ppy emit c ranges.ppy
```

<!-- outputs:start -->
## What it prints

**`python  ranges.ppy`**, **`ppy run ranges.ppy`**

```text
9999800001
99.987909853872
99.987910
99999
500000.0
```

**`ppy build ranges.ppy --report-opt`**

<details markdown="1">
<summary>166 lines</summary>

```text
optimization report: PPy (O2, ir road)
module ranges
  ranges.count_odd: native, bound to Python
  ranges.dot: native, bound to Python
  ranges.dot_relaxed: native, bound to Python
  ranges.fill: native, native callers only
  ranges.squares: native, native callers only
  ranges.main: Python -- has effects that must run on CPython: IO
  block merged: 18
  dead code removed: 114
  note: 10
  parallel loop emitted: 12
    - `ranges.squares`: the loop over `i` is a parallel loop
    - `ranges.dot`: the loop over `i` is a parallel add reduction into `total` (kept in order: `@ppy.fastmath` would let it split)
    - `ranges.dot_relaxed`: the loop over `i` is a parallel add reduction into `total`
    - `ranges.count_odd`: the loop over `i` is a parallel add reduction into `hits`
    - `ranges.fill`: the loop over `i` is a parallel loop
    - `ranges.fill`: the loop over `j` is a parallel add reduction into `acc` (kept in order: `@ppy.fastmath` would let it split)
    - @ranges_count_odd__par4: core.cmp: folded 2 and 0
    - @ranges_count_odd__par4: core.guard: condition always holds
    - simplify-cfg: ^for.guards1 merged into ^entry
    - simplify-cfg: ^for.setup2 merged into ^entry
    - simplify-cfg: ^for.latch5 merged into ^for.body4
    - simplify-cfg: ^for.guards1 merged into ^entry
    - simplify-cfg: ^for.setup2 merged into ^entry
    - simplify-cfg: ^for.latch5 merged into ^for.body4
    - simplify-cfg: ^for.guards1 merged into ^entry
    - simplify-cfg: ^for.setup2 merged into ^entry
    - simplify-cfg: ^for.latch5 merged into ^for.body4
    - simplify-cfg: ^for.guards1 merged into ^entry
    - simplify-cfg: ^for.setup2 merged into ^entry
    - simplify-cfg: ^for.latch5 merged into ^endif9
    - simplify-cfg: ^for.guards1 merged into ^entry
    - simplify-cfg: ^for.setup2 merged into ^entry
    - simplify-cfg: ^for.latch5 merged into ^for.body4
    - simplify-cfg: ^for.guards1 merged into ^entry
    - simplify-cfg: ^for.setup2 merged into ^entry
    - simplify-cfg: ^for.latch5 merged into ^for.body4
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.select removed
    - dce: unused core.cmp removed
    - dce: unused core.select removed
    - dce: unused core.cmp removed
    - dce: unused core.select removed
    - dce: unused core.cmp removed
    - dce: unused core.select removed
    - dce: unused core.cmp removed
    - dce: unused core.select removed
    - dce: unused core.cmp removed
    - dce: unused core.select removed
    - dce: unused core.cmp removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.sub removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.sub removed
    - dce: unused core.const removed
    - parallel loop over @ranges_squares__par1 lowered to 16 chunks
    - parallel reduction over @ranges_dot__par2 lowered to one chunk
    - parallel reduction over @ranges_dot_relaxed__par3 lowered to 16 chunks
    - parallel reduction over @ranges_count_odd__par4 lowered to 16 chunks
    - parallel loop over @ranges_fill__par5 lowered to 16 chunks
    - parallel reduction over @ranges_fill__par6 lowered to one chunk
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.mul: identity element removed
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.mul: multiplied by zero
    - @ranges_squares: core.add: identity element removed
    - @ranges_squares: core.sub: folded 16 and 1
    - @ranges_squares: core.sub: identity element removed
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.mul: identity element removed
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.mul: multiplied by zero
    - @ranges_dot_relaxed: core.add: identity element removed
    - @ranges_dot_relaxed: core.sub: folded 16 and 1
    - @ranges_dot_relaxed: core.sub: identity element removed
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.mul: identity element removed
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.mul: multiplied by zero
    - @ranges_count_odd: core.add: identity element removed
    - @ranges_count_odd: core.sub: folded 16 and 1
    - @ranges_count_odd: core.sub: identity element removed
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.mul: identity element removed
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.mul: multiplied by zero
    - @ranges_fill: core.add: identity element removed
    - @ranges_fill: core.sub: folded 16 and 1
    - @ranges_fill: core.sub: identity element removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
    - dce: unused core.const removed
```

</details>

**`ppy emit c ranges.ppy`**

*1267 lines: [outputs/03-ppy-emit-c-ranges-ppy.txt](outputs/03-ppy-emit-c-ranges-ppy.txt)*

<!-- outputs:end -->

Read on: [Parallel loops](../../docs/guide/parallel.md) ·
[Parallel fused kernels](../07_parallel/README.md)

`ranges.ppy` is hand-written; there is no `.py` source and no conversion step.
