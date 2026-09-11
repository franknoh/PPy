# Migration

A small legacy telemetry script, dynamic in the usual harmless ways —
`importlib.import_module`, a `globals()` write, `setattr`/`getattr` on
constant names — run through `ppy migrate` instead of `ppy convert`.

## Before and after

```python
math = importlib.import_module("math")
globals()["SCALE"] = 4
setattr(first, "flagged", True)
print(getattr(first, "flagged"), scaled, spread(scaled))
```

becomes

```python
import math
SCALE: Final[int] = 4
first.flagged = True
print(first.flagged, scaled, spread(scaled))
```

`ppy convert` would refuse the original. `ppy migrate` rewrites each site
to the static form it always meant, proving the rewrite equivalent before
making it, and removes the `import importlib` that fed the first line once
nothing uses it. Then the ordinary conversion runs over the rewritten code:
fourteen annotations, `Final` on the constant, `Sequence` where a parameter
is only read, `@ppy.pure` on `spread`. The result passes `ppy check` under
`strict = true` with nothing left over — this migration ends where `ppy
convert` starts.

## What a migration reports

`ppy migrate --report migration.json` writes the full accounting: what was
`AUTOFIXED`, what `REQUIRES_REWRITE`, what needs a `DYNAMIC_BOUNDARY`, and
whether the output is `strict_ready`. `--diff` shows the rewrite before it
happens. On a real codebase the unit of migration is a kernel file, not the
repository; [Migrating a real project](../../docs/internals/migrating.md)
says which files to hand it.

## Run it

```bash
ppy migrate legacy.py --diff
python  legacy.ppy
ppy run legacy.ppy
```

<!-- outputs:start -->
## What it prints

**`ppy migrate legacy.py --diff`**

<details markdown="1">
<summary>52 lines</summary>

```text
--- ./legacy.py
+++ ./legacy.ppy
@@ -1,32 +1,34 @@
 """A little legacy telemetry script, dynamic in all the usual harmless ways."""
+import math
+from collections.abc import Sequence
+from typing import Final
 
-import importlib
+import ppy
 
-math = importlib.import_module("math")
-
-globals()["SCALE"] = 4
+SCALE: Final[int] = 4
 
 
 class Reading:
-    def __init__(self, value):
-        self.value = value
-        self.flagged = False
+    def __init__(self, value: float) -> None:
+        self.value: float = value
+        self.flagged: bool = False
 
 
-def normalize(readings):
-    out = []
+def normalize(readings: Sequence[Reading]) -> list[float]:
+    out: list[float] = []
     for reading in readings:
         out.append(reading.value * SCALE)
     return out
 
 
-def spread(values):
+@ppy.pure
+def spread(values: Sequence[float]) -> int:
     return math.ceil(max(values) - min(values))
 
 
-samples = [Reading(0.5), Reading(1.25), Reading(2.0)]
-first = samples[0]
-setattr(first, "flagged", True)
+samples: list[Reading] = [Reading(0.5), Reading(1.25), Reading(2.0)]
+first: Reading = samples[0]
+first.flagged = True
 
-scaled = normalize(samples)
-print(getattr(first, "flagged"), scaled, spread(scaled))
+scaled: list[float] = normalize(samples)
+print(first.flagged, scaled, spread(scaled))
```

</details>

**`python  legacy.ppy`**, **`ppy run legacy.ppy`**

```text
True [2.0, 5.0, 8.0] 6
```

<!-- outputs:end -->

Read on: [CLI: `ppy migrate`](../../docs/cli.md) ·
[Dynamic boundaries](../16_dynamic/README.md)

Generated, not hand-written: `legacy.ppy` is exactly what `ppy migrate
legacy.py` writes, and `examples/verify_conversions.py` checks that on every
run.
