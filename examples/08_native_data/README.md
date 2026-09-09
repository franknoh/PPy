# Native data

Which Python values cross the boundary as machine values, and which stay
boxed. Scalars, fixed-size tuples, and all-scalar classes are handed over
flat; a `list[float]` is copied into a buffer on the way in; everything else
stays on the Python side. `ppy explain` says which is which, and why.

## Tuples

```python
@ppy.pure
def norm2(point: tuple[f64, f64, f64]) -> f64:
    return point[0] * point[0] + point[1] * point[1] + point[2] * point[2]


@ppy.pure
def centroid(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
```

A `tuple[f64, f64, f64]` is three doubles in the ABI. Returning a tuple is
two result slots, not a heap object; `centroid` allocates nothing on the
native path. `Array[int, 3]` is the same idea for a small fixed container.

## Lists

`dot` takes two `list[float]` and `total` a `list[i64]`. Native code cannot
walk a Python list, so a homogeneous list is copied into a contiguous
buffer at the call, which is why a borrowed
[`Buffer[T]`](../12_buffers_and_jit/README.md) is the faster spelling.
`total([10**30, 1])` shows the guard on the element: the first value does
not fit an `i64`, the boundary refuses, and the Python body prints the exact
sum.

```bash
ppy explain native_data.ppy:dot     # the representation chosen for every parameter
```

## Run it

```bash
python  native_data.ppy
ppy     native_data.ppy
ppy run native_data.ppy
```

<!-- outputs:start -->
## What it prints

**`python  native_data.ppy`**

```text
32.0
15
1000000000000000000000000000001
14.0 (2.0, 3.0)
383
```

**`ppy     native_data.ppy`**

```text
32.0
15
1000000000000000000000000000001
14.0 (2.0, 3.0)
383
```

**`ppy run native_data.ppy`**

```text
32.0
15
1000000000000000000000000000001
14.0 (2.0, 3.0)
383
```

<!-- outputs:end -->

Read on: [Tuples](../14_tuples/README.md) ·
[Value classes](../13_value_classes/README.md) ·
[Input and native lowering](../../docs/guide/native-lowering.md)

`native_data.ppy` is hand-written; there is no `.py` source and no conversion
step.
