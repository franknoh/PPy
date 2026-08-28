# Strings

String work stays on CPython, and says so.

## Provenance

Hand-written. `strings.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- String methods are typed, so the code checks; they have no native lowering, so they stay boxed.
- `ppy explain` reports which calls kept the function boxed.

## Run it

```bash
python  strings.ppy
ppy     strings.ppy
ppy run strings.ppy
```
