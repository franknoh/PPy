# Arbitrary precision

Integers stay Python integers.

## Provenance

Hand-written. `arbitrary_precision.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- A value that fits a machine word uses one; a value that does not falls back to CPython's arbitrary precision.
- The fallback is a guard on the native path, not a different answer.

## Run it

```bash
python  arbitrary_precision.ppy
ppy     arbitrary_precision.ppy
ppy run arbitrary_precision.ppy
```
