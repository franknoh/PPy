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

*3134 lines: [outputs/04-ppy-emit-c-ranges-ppy.txt](outputs/04-ppy-emit-c-ranges-ppy.txt)*

<!-- outputs:end -->

Read on: [Parallel loops](../../docs/guide/parallel.md) ·
[Parallel fused kernels](../07_parallel/README.md)

`ranges.ppy` is hand-written; there is no `.py` source and no conversion step.
