# Effects and contracts

What `@ppy.pure` refuses, and why.

## Provenance

Hand-written. `effects.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- Effects are a set, not a bit: I/O, global writes, randomness, and time are tracked separately.
- Declaring a contract the code does not satisfy is an error, with the offending effect named.
- A `@ppy.dynamic` boundary suspends dynamic-feature checks, not type declarations: the `eval` result crosses back into the declared `-> int` through `ppy.check[int]`, which validates at runtime.

## Run it

```bash
python  effects.ppy
ppy     effects.ppy
ppy run effects.ppy
```
