# Value classes

An all-scalar dataclass has no boxed representation.

## Provenance

Hand-written. `value_classes.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- Fields flatten into ABI atoms, so a call passes machine values rather than an object pointer.
- A class too wide for the ABI limit stays boxed.
- Native code dispatches an operator on a value class statically: `a + b` calls
  the class's own `__add__`, lowered like any native function, never Python's
  dynamic dispatch. Generics over value classes are in [40_generics](../40_generics/).

## Run it

```bash
python  value_classes.ppy
ppy     value_classes.ppy
ppy run value_classes.ppy
```
