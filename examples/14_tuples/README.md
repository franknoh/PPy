# Tuples

Fixed tuples flatten into scalar atoms.

## Provenance

Hand-written. `tuples.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- A tuple of known length and scalar elements is passed and returned unboxed.
- A homogeneous or oversized tuple stays boxed.

## Run it

```bash
python  tuples.ppy
ppy     tuples.ppy
ppy run tuples.ppy
```
