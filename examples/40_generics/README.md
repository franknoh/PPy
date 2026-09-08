# Generics

Type parameters the way Python 3.12 spells them, inferred at each call and
checked against their bounds; native code monomorphizes, one instance per
tuple of type arguments.

## Provenance

Hand-written. `generic.ppy` is written directly; there is no `.py` source
and no conversion step involved.

## What it shows

- `largest(1, 2)` is an `int` and `largest(1.5, 2.5)` a `float`: the call
  infers `T`, checks it against `int | float` (`E1721` otherwise), and has
  the declared return type with `T` substituted.
- A bound that is a `Protocol` lends its methods: `label` may call `name()`
  because `Named` says so, a class whose members cover the protocol's is an
  instance of it, and an unbounded `T` has no operators at all.
- `sweep` is native and calls `clamp` and `largest` with `float` and `int`
  arguments: each generic is lowered once per tuple of type arguments under a
  name that spells them (`ppy emit ir` shows them, marked `ppy.generic`), and
  the calls go straight to those instances. The generics keep their Python
  bodies for every other caller.
- `[tool.ppy.generics]` bounds the process -- `max-specializations` per
  generic and `max-depth` of a type argument -- and a generic that calls
  itself with its own parameter wrapped in a type is refused outright.

## Run it

```bash
python  generic.ppy
ppy run generic.ppy
ppy emit ir generic.ppy   # the monomorphized instances
```
