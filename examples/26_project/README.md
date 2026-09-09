# A multi-module project

Two modules analyzed as one call graph and built as one program.
`src/app.ppy` calls `geometry.perimeter` from `src/geometry.ppy`; `ppy check
src` types both together, and `ppy build` links both modules' IR into one
object.

## One graph

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

A function's parameter types may come from call sites in another file.
`distance` takes four `f64`; `perimeter` calls it from a loop over tuples.
Both lower, and after linking the call is a direct native call: the
whole-program pass inlines a small callee across modules, makes functions
Python never binds private, and drops what nothing reaches.

```bash
ppy emit linked-ir src/app.ppy     # the whole program, modules linked and optimized as one
```

## Cache keys and layout

A change to `geometry.ppy` invalidates `app.ppy`'s cache entry because the
dependency digest is part of its key, and nothing else. `pyproject.toml` at
the root names `src` as the source root; `app.ppy` calls `ppy.install()`
before importing its sibling so the same file runs under plain `python`
too. `ppy convert src/` and `ppy check src/` take the directory and work
across it.

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

Read on: [Interop](../24_interop/README.md) ·
[The IR: linking and the program](../../docs/internals/ir.md) ·
[Configuration](../../docs/reference/config.md)

`src/app.ppy` and `src/geometry.ppy` are hand-written; there is no `.py`
source and no conversion step.
