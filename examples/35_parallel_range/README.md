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
[`ranges.mojo`](compare/ranges.mojo), and [`ranges_numpy.py`](compare/ranges_numpy.py).
Each program prints its answers, then the best of five warm calls per kernel;
[`examples/compare.py`](../compare.py) runs each program five times and reports
the mean and standard deviation across processes, in milliseconds. All five
print the same answers. The ordered sums agree to the last digit; the
reassociated ones differ from them in the thirteenth.

| kernel | PPY `ppy run` | Numba `prange` | Taichi | Mojo | NumPy |
|---|---:|---:|---:|---:|---:|
| squares | 1.31 ± 0.07 | **0.84 ± 0.22** | 1.02 ± 0.03 | 1.00 ± 0.06 | 13.47 ± 0.51 |
| dot, in order | **6.14 ± 0.26** | — | — | 15.28 ± 0.29 | — |
| dot, reassociated | 1.68 ± 0.11 | 1.07 ± 0.15 | 1.46 ± 0.03 | 1.54 ± 0.08 | **1.04 ± 0.05** |
| count_odd | 0.99 ± 0.07 | **0.71 ± 0.59** | 1.21 ± 0.03 | 1.44 ± 0.27 | 28.10 ± 0.84 |
| fill, serial | 6.15 ± 0.15 | 7.54 ± 0.08 | **4.20 ± 0.46** | 20.04 ± 0.21 | 15.32 ± 0.81 |

What each port asked for:

- **PPY** is the source above: typed parameters, `parallel.range`, and the
  same file runs on CPython. `dot` keeps the order of a serial sum by
  design, which is why it is the slow row; `@ppy.fastmath` is the relaxed one.
- **Numba** is `@njit(parallel=True)` and `prange` over NumPy arrays, with no
  annotations. A `prange` sum reassociates on its own -- the plain reduction
  measures 1.46 ± 0.80 ms, `fastmath=True` the 1.07 in the table -- so there
  is no ordered `dot` to write. Integers wrap at 64 bits silently.
- **Taichi** wants the arrays as `ti.field`s declared with their shapes at
  module level and the kernels written over them; the outermost loop of a
  kernel is parallel on its own, and the ordered second loop of `fill` needs
  `ti.loop_config(serialize=True)`. With the defaults, `squares` overflowed
  in 32-bit and `dot` accumulated in `f32`, both silently:
  `ti.init(default_ip=ti.i64, default_fp=ti.f64)` is what makes the answers
  match.
- **Mojo** is a different language -- Mojo 1.0's `def`, `var`, `mut`
  parameters, `List[Int64]`, `Int64(i)` where the index meets the element --
  compiled to a binary. `parallelize` now lives in `max.algorithm`, the MAX
  package, and takes a function of one index, so a reduction is partial sums
  per worker written by hand. This is the straightforward `List` port, without
  `UnsafePointer` or SIMD types; its serial loops are the slowest here.
- **NumPy** has no loop to write, and no way to keep one: each step is an
  array expression with a temporary, `dot` is BLAS, and `count_odd` builds
  two eight-million-element temporaries to count.

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

**`python  ranges.ppy`**

```text
9999800001
99.987909853872
99.987910
99999
500000.0
```

**`ppy run ranges.ppy`**

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

*1267 lines: [outputs/04-ppy-emit-c-ranges-ppy.txt](outputs/04-ppy-emit-c-ranges-ppy.txt)*

<!-- outputs:end -->

Read on: [Parallel loops](../../docs/guide/parallel.md) ·
[Parallel fused kernels](../07_parallel/README.md)

`ranges.ppy` is hand-written; there is no `.py` source and no conversion step.
