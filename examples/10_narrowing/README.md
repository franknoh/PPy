# Narrowing

Every form the checker understands.

## Provenance

Hand-written. `narrowing.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- `is None`, `isinstance`, `match` patterns, boolean operators, and walrus bindings all narrow.
- A `match` case sees the subject with earlier cases subtracted; a guarded case rules nothing out.

## Run it

```bash
python  narrowing.ppy
ppy     narrowing.ppy
ppy run narrowing.ppy
```
