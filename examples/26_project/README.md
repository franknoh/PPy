# Project layout

A multi-module project, analyzed as one call graph rather than file by file.

## Provenance

Hand-written. `src/app.ppy`, `src/geometry.ppy` are written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- `ppy convert src/` and `ppy check src/` work across modules.
- A module's cache key includes the hashes of the modules it depends on, so a
  change to one invalidates exactly what it should.
- Cross-module inference: a function's parameter types come from call sites in
  other files.
- Since 0.2.0 a package builds as one program: each module's IR links into
  one, whole-program optimization inlines across the seams, and one object
  comes out. `ppy emit linked-ir src/app.ppy` prints the linked program.

## Run it

```bash
ppy check src
ppy run   src/app.ppy
```
