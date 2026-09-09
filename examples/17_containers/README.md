# Containers

Element types inferred from first use, and the difference between mutating a
container the function made and one it was given.

## Element types from first use

```python
@ppy.pure
def grow(count: int) -> list[int]:
    out = []
    for i in range(count):
        out.append(i * i)
    return out
```

`out = []` has no element type until `out.append(i * i)` makes it a
`list[int]`; `seen = set()` becomes a `set[str]` at `seen.add(value)`;
`counts` is declared `dict[int, int]` and `counts.get(value, 0) + 1` checks
against it. None of these functions is native — a dict or a set has no
native form — but all of them are strict, typed, and pure.

## Local mutation is pure, shared mutation is not

`flatten` extends a list it created, so it is pure. Had it extended `rows`,
the write would be an effect on an argument and `@ppy.pure` would fail with
`E1601`. The distinction is by alias, not by name: `ys = xs; ys.append(1)`
mutates `xs` whatever it is called, and the analysis follows the alias to
say so. The same alias map is what lets `ppy convert` declare a read-only
parameter as `Sequence[T]` rather than `list[T]`.

## Run it

```bash
python  containers.ppy
ppy     containers.ppy
ppy run containers.ppy
```

<!-- outputs:start -->
## What it prints

**`python  containers.ppy`**

```text
[(1, 1), (2, 2), (3, 3)]
[0, 1, 4, 9, 16, 25]
3
[1, 2, 3, 4, 5]
```

**`ppy     containers.ppy`**

```text
[(1, 1), (2, 2), (3, 3)]
[0, 1, 4, 9, 16, 25]
3
[1, 2, 3, 4, 5]
```

**`ppy run containers.ppy`**

```text
[(1, 1), (2, 2), (3, 3)]
[0, 1, 4, 9, 16, 25]
3
[1, 2, 3, 4, 5]
```

<!-- outputs:end -->

Read on: [Conversion and inference](../../docs/internals/conversion.md) ·
[Effects and purity](../../docs/guide/effects.md)

`containers.ppy` is hand-written; there is no `.py` source and no conversion
step.
