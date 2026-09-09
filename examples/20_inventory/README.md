# Inventory

Untyped Python that converts cleanly, with no hand editing afterwards.
`inventory.py` has no annotations; `inventory.ppy` beside it has every
parameter and return typed, `Sequence` where the body only reads,
`@ppy.pure` where the checker proved it — and `examples/verify_conversions.py`
regenerates it on every run to keep that claim true.

## What the converter recovered

```python
@ppy.pure
def total_value(items: Sequence[tuple[str, int, float]]) -> float:
    total: float = 0.0
    for item in items:
        total += line_value(item)
    return total
```

- `list[tuple[str, int, float]]` for every `items` parameter, from the call
  sites; widened to `Sequence[...]` where the function only reads it.
- The return type of every function, propagated along the call graph:
  `summary` returns a three-tuple because that is what both of its `return`
  statements build.
- `@ppy.pure` on the six functions the checker proved pure, and on nothing
  it could not.
- `items: list[tuple[str, int, float]]` on the local in `main`, because
  `write-local-annotations` is on by default.

## What the input had to get right

An earlier version of this file had one `calc(items, kind)` that did three
jobs behind an integer flag. It returned `int | float`, so nothing
downstream could be typed and the conversion produced four errors.
Splitting it into `total_value` and `total_units` made the file
convertible. No tool makes that decision: it is a design change, not a
mechanical one, and the converter renames nothing and splits nothing.

## Run it

```bash
ppy convert inventory.py --promote-buffers --dry-run
python  inventory.ppy
ppy run inventory.ppy
```

<!-- outputs:start -->
## What it prints

**`ppy convert inventory.py --promote-buffers --dry-run`**

<details markdown="1">
<summary>81 lines</summary>

```text
# ---- ./inventory.ppy ----
import math
from collections.abc import Sequence

import ppy


@ppy.pure
def line_value(item: tuple[str, int, float]) -> float:
    return item[1] * item[2]


@ppy.pure
def total_value(items: Sequence[tuple[str, int, float]]) -> float:
    total: float = 0.0
    for item in items:
        total += line_value(item)
    return total


@ppy.pure
def total_units(items: Sequence[tuple[str, int, float]]) -> int:
    total: int = 0
    for item in items:
        total += item[1]
    return total


@ppy.pure
def discounted(price: float, discount: float) -> float:
    return price * (1.0 - discount)


def apply_discount(
    items: Sequence[tuple[str, int, float]],
    discount: float,
) -> list[tuple[str, int, float]]:
    out: list[tuple[str, int, float]] = []
    for item in items:
        out.append((item[0], item[1], discounted(item[2], discount)))
    return out


def below_stock(items: Sequence[tuple[str, int, float]], threshold: int) -> list[str]:
    low: list[str] = []
    for item in items:
        if item[1] < threshold:
            low.append(item[0])
    return low


@ppy.pure
def summary(items: Sequence[tuple[str, int, float]]) -> tuple[float, float, float]:
    value: float = total_value(items)
    count: int = len(items)
    if not count:
        return (0.0, 0.0, 0.0)
    return (value, value / count, math.sqrt(value))


@ppy.pure
def labeled(name: str, value: float) -> str:
    return name + "=" + str(round(value, 2))


def main() -> None:
    items: list[tuple[str, int, float]] = [
        ("bolt", 120, 0.25),
        ("nut", 340, 0.1),
        ("washer", 80, 0.05),
        ("bracket", 15, 12.5),
    ]

    value, average, root = summary(items)
    print(labeled("total", value), labeled("avg", average), labeled("root", root))
    print(apply_discount(items, 0.2)[0])
    print(below_stock(items, 100))
    print("units:", total_units(items))


main()
```

</details>

**`python  inventory.ppy`**

```text
total=255.5 avg=63.88 root=15.98
('bolt', 120, 0.2)
['washer', 'bracket']
units: 555
```

**`ppy run inventory.ppy`**

```text
total=255.5 avg=63.88 root=15.98
('bolt', 120, 0.2)
['washer', 'bracket']
units: 555
```

<!-- outputs:end -->

Read on: [Conversion and inference](../../docs/internals/conversion.md) ·
[Inference](../23_inference/README.md)

Generated, not hand-written: `inventory.ppy` is exactly what
`ppy convert inventory.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run.
