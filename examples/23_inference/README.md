# Inference

Three untyped modules, converted with no hand editing afterwards. Each `.ppy`
here is exactly what `ppy convert` writes.

## Provenance

Generated, not hand-written. Each of these is exactly what the converter
writes, and `verify_conversions.py` regenerates it to prove it:

- `pipeline.ppy` &larr; `ppy convert pipeline.py`
- `shapes.ppy` &larr; `ppy convert shapes.py`
- `stats.ppy` &larr; `ppy convert stats.py`

## What each one exercises

`stats.py` — call-graph propagation. Nothing is annotated; `mean`, `variance`,
`standardize`, and `report` all get parameter and return types from a single
module-level list of floats.

`shapes.py` — classes and optionals. Instance field types come from what
`__init__` assigns. `widest` is defined before `Rect`, so its annotation is
quoted. It returns `Rect | None`, and `label` accepts that union.

`pipeline.py` — everything at once:

- a call chain several functions deep
- instance fields from `__init__`
- a forward reference to a class defined later
- a container's element type from what is appended to it
- `shifted`, which nothing calls, typed from the arithmetic in its body
- `@ppy.pure` where the checker proved it

## What the converter will not do

Renaming, splitting a function that does several jobs, and restructuring an
algorithm are design decisions, not mechanical ones. When a buffer promotion is
blocked it names the blocker rather than guessing:

```
remark[R3003]: `raw` could be a borrowed buffer, but `raw` is sliced, which
               copies; indexing it element by element instead would let the
               memory be borrowed
```

## Run it

```bash
ppy convert pipeline.py --dry-run
python  pipeline.py
ppy run pipeline.ppy
```

<!-- outputs:start -->
## What it prints

**`ppy convert pipeline.py --dry-run`**

```text
# ---- ./pipeline.ppy ----
import math
from collections.abc import Sequence

import ppy


class Summary:
    def __init__(self, mean: float, deviation: float, count: int) -> None:
        self.mean: float = mean
        self.deviation: float = deviation
        self.count: int = count

    def scaled(self, factor: float) -> 'Summary':
        return Summary(self.mean * factor, self.deviation * factor, self.count)

    def shifted(self, offset: float) -> 'Summary':
        return Summary(self.mean + offset, self.deviation, self.count)

    @ppy.pure
    def describe(self) -> str:
        return f"n={self.count} mean={self.mean:.3f} sd={self.deviation:.3f}"


@ppy.pure
def clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


@ppy.pure
def normalize(value: float, mean: float, spread: float) -> float:
    return clamp((value - mean) / spread, -3.0, 3.0)


def summarize(readings: Sequence[float]) -> Summary | None:
    count: int = len(readings)
… 34 more lines
```

**`python  pipeline.py`**

```text
n=8 mean=5.000 sd=2.000 first=-1.5
empty
n=8 mean=10.000 sd=4.000
```

**`ppy run pipeline.ppy`**

```text
n=8 mean=5.000 sd=2.000 first=-1.5
empty
n=8 mean=10.000 sd=4.000
```

<!-- outputs:end -->
