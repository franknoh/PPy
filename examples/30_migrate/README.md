# Five dynamic idioms, rewritten and proven equivalent

`legacy.py` is a small telemetry script that is dynamic in all the usual
harmless ways: `importlib.import_module("math")`, `globals()["SCALE"] = 4`,
`setattr(first, "flagged", True)`, `getattr(first, "flagged")`. `ppy
convert` would refuse it. `ppy migrate` rewrites each site to the static
form it always meant, proves the rewrite equivalent first, and then
staticizes as usual — and the result checks strict with nothing left over.

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

The `import importlib` that fed the first line is removed as freight with no
cargo. Then the ordinary conversion runs over the rewritten code: fourteen
annotations, `Final` on the constant, `Sequence` where a parameter is only
read, `@ppy.pure` on `spread`. This migration ends where `ppy convert`
starts.

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
```

*52 lines in all — [full output](outputs/01-ppy-migrate-legacy-py-diff.txt).*

**`python  legacy.ppy`**

```text
True [2.0, 5.0, 8.0] 6
```

**`ppy run legacy.ppy`**

```text
True [2.0, 5.0, 8.0] 6
```

<!-- outputs:end -->

## Read on

- [CLI: `ppy migrate`](../../docs/cli.md) — the passes and the report's categories.
- [Dynamic boundaries](../16_dynamic/README.md) — what stays dynamic on purpose.

Generated, not hand-written: `legacy.ppy` is exactly what `ppy migrate
legacy.py` writes, and `examples/verify_conversions.py` checks that on every
run.
