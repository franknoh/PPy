# Classes

Ordinary classes, with fields the checker knows.

## Provenance

Hand-written. `classes.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- Field types come from annotations or from what `__init__` assigns.
- Methods are resolved statically, so a typo is a compile-time error.

## Run it

```bash
python  classes.ppy
ppy     classes.ppy
ppy run classes.ppy
```
