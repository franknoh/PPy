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
