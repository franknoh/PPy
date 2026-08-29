# Migration

A little legacy telemetry script, dynamic in all the usual harmless ways —
`importlib.import_module`, `globals()` writes, `setattr`/`getattr` on
constant names — run through `ppy migrate` instead of `ppy convert`.

## Provenance

Generated, not hand-written. `legacy.ppy` is exactly what
`ppy migrate legacy.py` writes; `verify_conversions.py` regenerates it to
prove that.

## What it shows

- The migration passes rewrite what was static all along: `setattr(first,
  "flagged", True)` becomes `first.flagged = True`, `globals()["SCALE"] = 4`
  becomes `SCALE = 4`, and `math = importlib.import_module("math")` becomes
  `import math` — after which the `import importlib` that fed it is removed
  as freight with no cargo. Five sites, each proven equivalent before it is
  touched.
- Staticization then runs as usual over the rewritten code: fourteen
  annotations, `Final` on the constant, `@ppy.pure` where proven.
- The result passes `ppy check` under `strict = true` with nothing left
  over — this migration ends where `ppy convert` starts.
- `ppy migrate --report migration.json` writes the full accounting;
  `ppy migrate --diff` shows it before it happens.

## Run it

```bash
python legacy.ppy    # plain CPython
ppy legacy.ppy       # optimized backend
ppy run legacy.ppy   # the LLVM path
```
