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

<!-- outputs:start -->
## What it prints

**`python  value_classes.ppy`**

```text
14.0 25.0 25.0 7.0
25.0
# native: False
distance2 (8 float ops)      57.6 ns/call
steps     (200 iters)      4308.2 ns/call
```

**`ppy     value_classes.ppy`**

```text
14.0 25.0 25.0 7.0
25.0
# native: False
distance2 (8 float ops)      56.9 ns/call
steps     (200 iters)      4280.1 ns/call
```

**`ppy run value_classes.ppy`**

```text
14.0 25.0 25.0 7.0
25.0
# native: False
distance2 (8 float ops)      56.4 ns/call
steps     (200 iters)       316.6 ns/call
```

<!-- outputs:end -->
