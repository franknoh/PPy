# Untyped Python in, strict PPY out

`inventory.py` has no annotations. `inventory.ppy` beside it has every
parameter typed, every return typed, `Sequence` where the body only reads,
`Final`-quality constants, and `@ppy.pure` on the six functions the checker
could prove — and not one character of it was typed by hand.
`examples/verify_conversions.py` regenerates it on every run to keep that
claim true.

## What `ppy convert` recovered

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


```

*81 lines in all — [full output](outputs/01-ppy-convert-inventory-py-promote-buffers.txt).*

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

## Read on

- [Conversion and inference](../../docs/internals/conversion.md) — where the types come from.
- [Inference](../23_inference/README.md) — three more untyped modules, one thing each.

Generated, not hand-written: `inventory.ppy` is exactly what
`ppy convert inventory.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run.
