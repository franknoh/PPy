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
`(-4, 3)`: floor division and a divisor-signed remainder, as Python's.

## Compared with Numba

A million `divmod_pair` calls from Python, and a native loop that unpacks a
tuple parameter and walks eight million steps, in [`compare/`](compare/):
[`pairs_bench.ppy`](compare/pairs_bench.ppy) -- under `ppy run` and, the
same file, under `python` -- and [`pairs_numba.py`](compare/pairs_numba.py).
Milliseconds, best of five, over five processes.

Tuples are native in both; PPY spells the types, Numba infers them at the
first call:

```python
@ppy.pure
@ppy.opt(3)
def walk(start: tuple[float, float], count: int) -> float:
    x, y = start
    for i in range(count):
        x, y = (x + float(i)) / 2.0, (y + 1.0) / 2.0
    return x + y
```

```python
@njit
def walk(start, count):
    x, y = start
    for i in range(count):
        x, y = (x + float(i)) / 2.0, (y + 1.0) / 2.0
    return x + y
```

<!-- compare:start -->
| | PPY `ppy run` | CPython, the same file | Numba `@njit` |
|---|---:|---:|---:|
| divmod_pair, a million calls from Python | **32.78 ± 1.40** | 47.86 ± 0.33 | 107.03 ± 5.77 |
| walk, eight million steps from a tuple natively | 11.26 ± 0.12 | 316.30 ± 2.14 | **11.25 ± 0.19** |
<!-- compare:end -->

The loop is the same machine code on both: two doubles in registers,
nothing allocated. The call from Python is where they differ, and the
difference is the boundary: PPY's generated wrapper unpacks two ints and
boxes a pair on the way out, Numba's dispatcher types the arguments on
every call. What does not lower is a tuple handed from one native function
to another: `midpoint(point, ...)` in a native loop keeps the loop in
Python (`ppy explain` says `a tuple result cannot be forwarded between
native calls yet`), so the loop here takes the tuple as a parameter and
rebuilds it in place.

Intel Core Ultra 9 386H; Numba 0.67.0 on CPython 3.12.13, PPY on CPython
3.14.5, from a checkout on a native filesystem.

## Run it

```bash
python  tuples.ppy
ppy     tuples.ppy
ppy run tuples.ppy
```

<!-- outputs:start -->
## What it prints

**`python  tuples.ppy`**, **`ppy     tuples.ppy`**, **`ppy run tuples.ppy`**

```text
(2.0, 3.0)
25.0
(3, 2) (-4, 3)
```

<!-- outputs:end -->

Read on: [Native data](../08_native_data/README.md) ·
[Numerics](../11_numerics/README.md)

`tuples.ppy` is hand-written; there is no `.py` source and no conversion step.
