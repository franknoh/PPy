# XLA

`@xla.jit` marks a function of floats, ints, and bools whose body is one
block of arithmetic and math for XLA: the compiler emits StableHLO, the
build stages it, and the PJRT bridge compiles and runs it on a device.

## Provenance

Hand-written. `device_math.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- `f` is what XLA takes: arithmetic, `math.sin`, a `select`; `ppy emit
  stablehlo` prints the module, `ppy inspect --stage stablehlo` the same.
- `branchy` is not, yet: a branch is control flow, so the compiler reports
  `W2007` and the function runs as written -- correct, and honest about it.
- Under plain CPython, and wherever no device is present, every function
  runs as written. `xla.devices()` is what the bridge sees, and the line that
  prints it starts with `# `, the mark for what may differ between machines.
- XLA computes `sin` with its own library, so the last bits of a result can
  differ from CPython's `math`; the printed digits are rounded so that the
  three paths compare equal where they should.

## Run it

```bash
python  device_math.ppy
ppy run device_math.ppy               # through PJRT where the bridge is installed (uv sync --group jax)
ppy emit stablehlo device_math.ppy    # the StableHLO for `f`
```
