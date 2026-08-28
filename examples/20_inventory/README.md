# Inventory

Untyped Python that converts cleanly, with no hand editing afterwards.

`inventory.ppy` is exactly what `ppy convert inventory.py --promote-buffers`
writes. `verify_conversions.py` regenerates it and fails on any difference.

## Provenance

Generated, not hand-written. Each of these is exactly what the converter
writes, and `verify_conversions.py` regenerates it to prove it:

- `inventory.ppy` &larr; `ppy convert inventory.py --promote-buffers`

## What the converter recovered

- `list[tuple[str, int, float]]` for every `items` parameter, from the call sites
- the return type of every function, propagated along the call graph
- `@ppy.pure` on the functions the checker proved pure
- `ITEMS: list[tuple[str, int, float]]` on the module-level binding

## Why the input looks the way it does

An earlier version of this file had one `calc(items, kind)` that did three jobs
behind an integer flag. It returned `int | float`, so nothing downstream could
be typed and conversion produced four errors. Splitting it into `total_value`
and `total_units` is what makes the file convertible, and no tool can make that
decision: it is a design change, not a mechanical one.

## Run it

```bash
python  inventory.py
ppy     inventory.ppy
ppy run inventory.ppy
```
