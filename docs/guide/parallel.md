# Parallel loops: `ppy.parallel.range`

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

`for i in parallel.range(...)` says the iterations may run at once. Under
CPython they run in order, which is one of the orders allowed. Natively
the body becomes a function over a chunk of the range and a `parallel.for`
in the IR runs it; the build's `[tool.ppy.parallel] backend` decides how
-- `threads` splits the range across the worker count and joins,
`serial` runs it on the calling thread, `simd` hands the serial loop to
the vectorizer, `openmp` spells it as OpenMP regions in `ppy emit c` --
and every choice gives the same answer. A range too small to be worth the
threads runs serially whatever the setting.

The body may read anything the function holds and write through pointers
and buffers; it may not assign a variable that lives outside the loop,
`break`, or `return`, and a function that does is refused with the reason.
One `+=` or `*=` into an outer `int` or `float` is a reduction, lowered as
a `parallel.reduce`: an integer reduction splits freely, a floating-point
one keeps its order -- the answer would change otherwise -- unless the
function is `@ppy.fastmath`, which permits the reassociation. A thread
whose chunk fails a guard fails the whole loop, and the function falls
back to Python as a whole.

`@ppy.parallel` on a function asks the same of its outermost `range`
loops without spelling `parallel.range`: each that passes the analysis
becomes parallel, and each that does not stays serial and says why in an
optimization remark. `E1650` names a misuse of `parallel.range` itself.
The fused NumPy loops `@ppy.parallel` already split are unchanged.

Examples: [Parallel range](../howto/35_parallel_range.md),
[Parallel fused kernels](../howto/07_parallel.md).
