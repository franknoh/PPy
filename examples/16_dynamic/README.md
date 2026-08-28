# Dynamic boundaries

The escape hatch, and what it costs.

## Provenance

Hand-written. `dynamic.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- `ppy.dynamic` works as a decorator and as a context manager.
- Inside a boundary, unknown attributes and computed `getattr` are permitted and typed `Any`.
- A boundary is an optimization barrier, and the project can forbid them entirely.

## Run it

```bash
python  dynamic.ppy
ppy     dynamic.ppy
ppy run dynamic.ppy
```
