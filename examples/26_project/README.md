# A project is one call graph, and one program

`src/app.ppy` calls `geometry.perimeter` from `src/geometry.ppy`. `ppy
check src` analyzes both as one call graph — a function's parameter types
may come from call sites in another file — and `ppy build` links both
modules' IR into one program, inlines across the seam, and emits one
object. A change to `geometry.ppy` invalidates `app.ppy`'s cache entry
because the dependency digest is part of its key, and nothing else.

## Two modules, one graph

```python
@ppy.pure
def perimeter(points: list[tuple[float, float]]) -> float:
    total: float = 0.0
    for index in range(len(points)):
        a = points[index]
        b = points[(index + 1) % len(points)]
        total += distance(a[0], a[1], b[0], b[1])
    return total
```

`distance` takes four `f64`; `perimeter` calls it from a loop over tuples.
Both lower, and after linking the call is a direct native call — the
whole-program pass inlines a small callee across modules, makes functions
Python never binds private, and drops what nothing reaches.

```bash
ppy emit linked-ir src/app.ppy     # the whole program, modules linked and optimized as one
```

## Layout

`pyproject.toml` at the root names `src` as the source root; `app.ppy`
calls `ppy.install()` before importing its sibling so the same file runs
under plain `python` too. `ppy convert src/` and `ppy check src/` take the
directory and work across it.

## Run it

```bash
ppy check src
ppy run   src/app.ppy
```

<!-- outputs:start -->
## What it prints

**`ppy check src`**

*(prints nothing; exits 0)*

**`ppy run   src/app.ppy`**

```text
4.0
5.0
```

<!-- outputs:end -->

## Read on

- [Interop](../24_interop/README.md) — the import hook on its own.
- [The IR: linking and the program](../../docs/internals/ir.md) — how modules become one object.
- [Configuration](../../docs/reference/config.md) — `source-roots` and the cache keys.

`src/app.ppy` and `src/geometry.ppy` are hand-written; there is no `.py`
source and no conversion step.
