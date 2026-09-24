# Parallel loops: `ppy.parallel.range`

`for i in parallel.range(...)` says the iterations of a loop may run at once.
This page covers how such a loop runs, what its body may do, and how
reductions work.

## Example

```python
from ppy import Buffer, native, parallel


def squares(out: native.ptr[int], n: int) -> None:
    for i in parallel.range(n):
        native.store(native.offset(out, i), i * i)


def dot(a: Buffer[float], b: Buffer[float]) -> float:
    total = 0.0
    for i in parallel.range(len(a)):
        total += a[i] * b[i]
    return total
```

## How it runs

Under CPython the iterations run in order, which is one of the orders
allowed. Natively the body becomes a function over a chunk of the range, and
a `parallel.for` in the IR runs it.

The build's `[tool.ppy.parallel] backend` decides how:

| backend | what it does |
|---|---|
| `threads` | splits the range across the worker count and joins |
| `serial` | runs it on the calling thread |
| `simd` | hands the serial loop to the vectorizer |
| `openmp` | spells it as OpenMP regions in `ppy emit c` |

Each choice gives the same answer. A range too small to be worth the threads
runs serially whatever the setting.

## Rules for the body

The body may read anything the function holds and write through pointers and
buffers. It may not:

- assign a variable that lives outside the loop
- `break`
- `return`

A function that does is refused with the reason.

## Reductions

One `+=` or `*=` into an outer `int` or `float` is a reduction, lowered as a
`parallel.reduce`.

- An integer reduction splits freely.
- A floating-point reduction keeps its order, because the answer would change
  otherwise. `@ppy.fastmath` on the function permits the reassociation.

## Guard failures

A thread whose chunk fails a guard fails the whole loop, and the function
falls back to Python as a whole.

## `@ppy.parallel`

`@ppy.parallel` on a function asks the same of its outermost `range` loops
without spelling `parallel.range`. Each loop that passes the analysis becomes
parallel. Each that does not stays serial and says why in an optimization
remark. The fused NumPy loops `@ppy.parallel` already split are unchanged.

## Diagnostics

`E1650` names a misuse of `parallel.range` itself.

Examples: [Parallel range](../howto/35_parallel_range.md),
[Parallel fused kernels](../howto/07_parallel.md).
