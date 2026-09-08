# The same fused kernel, on every core

`@ppy.parallel` on a fused NumPy expression splits its loop across the
worker pool. The output is bit-identical to the serial kernel and to NumPy,
because an elementwise loop has no order to lose. A reduction does, and the
compiler will not split one that reassociates unless you say so.

## Mark it splittable; the compiler still proves it

```python
@ppy.pure
@ppy.parallel
@ppy.opt(3)
def parallel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b + a) * (b - a) + a * 0.5 - b * 0.25
```

The decorator asks; the analysis answers. Eight million elements, five
operations, one fused loop chunked across `[tool.ppy.parallel] threads`
workers and joined. `serial` is the same expression without the decorator,
and the program checks `np.array_equal` across serial, parallel, and NumPy —
it prints `bit-identical: True` on every path.

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
The choice is the function's, per function, and the checker never makes it
for you.

`parallel.range` is the 0.2.0 spelling for a loop the program itself
declares splittable — pointers, buffers, and reductions included
([parallel range](../35_parallel_range/README.md)). `@ppy.parallel` stays the
switch for fused NumPy loops like these.

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
fused serial        104.9 ms   sample=-0.249979000059
fused parallel      116.6 ms   sample=-0.249979000059
numpy               122.1 ms   sample=-0.249979000059
bit-identical: True
strict == numpy: True
relaxed close   : True
```

**`ppy     parallel.ppy`**

```text
fused serial        108.6 ms   sample=-0.249979000059
fused parallel      109.2 ms   sample=-0.249979000059
numpy               105.1 ms   sample=-0.249979000059
bit-identical: True
strict == numpy: True
relaxed close   : True
```

**`ppy run parallel.ppy`**

```text
fused serial         23.2 ms   sample=-0.249979000059
fused parallel       15.4 ms   sample=-0.249979000059
numpy                29.3 ms   sample=-0.249979000059
bit-identical: True
strict == numpy: True
relaxed close   : True
```

<!-- outputs:end -->

## Read on

- [Parallel loops](../../docs/guide/parallel.md) — `parallel.range`, reductions, and the backends.
- [NumPy fusion](../05_numpy/README.md) — the loop this one splits.

`parallel.ppy` is hand-written; there is no `.py` source and no conversion step.
