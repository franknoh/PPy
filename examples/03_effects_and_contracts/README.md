# Effects and contracts

What `@ppy.pure` refuses, and why.

## Provenance

Hand-written. `effects.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- Effects are a set, not a bit: I/O, global writes, randomness, and time are tracked separately.
- Declaring a contract the code does not satisfy is an error, with the offending effect named.

## Run it

```bash
python  effects.ppy
ppy     effects.ppy
ppy run effects.ppy
```
