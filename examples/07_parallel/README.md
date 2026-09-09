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
