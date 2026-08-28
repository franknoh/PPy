# Exceptions

Exception behavior is part of the contract.

## Provenance

Hand-written. `errors.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- `ZeroDivisionError` and `IndexError` are raised at the same points on every path.
- The optimizer preserves exception ordering unless it can prove the code cannot raise.

## Run it

```bash
python  errors.ppy
ppy     errors.ppy
ppy run errors.ppy
```
