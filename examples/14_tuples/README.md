# Tuples

A tuple of known length and scalar elements is passed and returned unboxed.
`tuple[float, float]` is two doubles in the ABI; passed to a native function
it is two arguments, returned it is two result slots, and the boundary boxes
the pair back into a Python tuple only on the way out.

```python
@ppy.pure
@ppy.opt(3)
def divmod_pair(a: int, b: int) -> tuple[int, int]:
    return (a // b, a % b)
```

`midpoint` allocates nothing on the native path; `divmod_pair` returns a
quotient and a remainder without a heap object between them. The rule is
exact: a homogeneous `tuple[int, ...]` has no known length and stays boxed,
and so does a tuple wider than the ABI allows. `divmod_pair(-17, 5)` is
`(-4, 3)` — floor division and a divisor-signed remainder — on all three
paths.

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

Read on: [Native data](../08_native_data/README.md) ·
[Numerics](../11_numerics/README.md)

`tuples.ppy` is hand-written; there is no `.py` source and no conversion step.
