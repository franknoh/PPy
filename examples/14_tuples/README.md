# Tuples as scalar atoms

A `tuple[float, float]` is two doubles. Passed to a native function it is
two arguments; returned, it is two result slots. `midpoint` allocates
nothing, `divmod_pair` returns a quotient and a remainder without a heap
object between them, and the boundary boxes the pair back into a Python
tuple only on the way out.

## Fixed length, scalar elements

```python
@ppy.pure
@ppy.opt(3)
def divmod_pair(a: int, b: int) -> tuple[int, int]:
    return (a // b, a % b)
```

The rule is exact: a tuple of known length whose elements are scalars
flattens. A homogeneous `tuple[int, ...]` has no known length and stays
boxed; so does a tuple wider than the ABI allows. `divmod_pair(-17, 5)` is
`(-4, 3)` — floor division and a divisor-signed remainder, the same on all
three paths.

## Run it

```bash
python  tuples.ppy
ppy     tuples.ppy
ppy run tuples.ppy
```

<!-- outputs:start -->
## What it prints

**`python  tuples.ppy`**

```text
(2.0, 3.0)
25.0
(3, 2) (-4, 3)
```

**`ppy     tuples.ppy`**

```text
(2.0, 3.0)
25.0
(3, 2) (-4, 3)
```

**`ppy run tuples.ppy`**

```text
(2.0, 3.0)
25.0
(3, 2) (-4, 3)
```

<!-- outputs:end -->

## Read on

- [Native data](../08_native_data/README.md) — tuples beside lists, arrays, and value classes.
- [Numerics](../11_numerics/README.md) — why `-17 // 5` is `-4`.

`tuples.ppy` is hand-written; there is no `.py` source and no conversion step.
