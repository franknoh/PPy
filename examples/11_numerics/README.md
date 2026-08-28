# Numerics

Where PPY refuses to differ from CPython.

## Provenance

Hand-written. `numerics.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- Floor division rounds toward negative infinity and the remainder takes the divisor's sign.
- An expression that overflows a machine word hands the call back to CPython rather than wrapping.

## Run it

```bash
python  numerics.ppy
ppy     numerics.ppy
ppy run numerics.ppy
```
