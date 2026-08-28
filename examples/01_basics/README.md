# Basics

Fixed-width integer markers, purity contracts, and per-function optimization levels.

## Provenance

Hand-written. `basics.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- `ppy.i64` pins a parameter to a machine word, so the compiler does not have to prove the range.
- `@ppy.pure` is a contract the checker verifies, not a hint.
- `@ppy.opt(3)` raises the optimization level for one function.

## Run it

```bash
python  basics.ppy
ppy     basics.ppy
ppy run basics.ppy
```
