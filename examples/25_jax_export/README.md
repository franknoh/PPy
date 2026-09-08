# JAX export

Build-time export of a staged function to StableHLO.

## Provenance

Hand-written. `model.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- A `@jax.jit` function whose inputs carry `ppy.Shape` and `ppy.DType` can be
  exported ahead of time, so the trace is not repeated at startup.
- `ppy.Shape` may name symbolic dimensions, so one artifact serves every batch
  size.
- Export imports and runs project code, so it happens only when the project sets
  both `build-execution = "allow"` and `[tool.ppy.plugins.jax]
  allow-build-export = true`. With either removed the functions stay ordinary
  jitted calls and `ppy` reports which, and why.
- `ppy.xla` ([39_xla](../39_xla/)) is the JAX-free form: a scalar function
  marked `@xla.jit` is emitted as StableHLO by the compiler itself and run
  through PJRT, with no trace and no JAX in the compiler.

## Run it

```bash
ppy build model.ppy
ppy run   model.ppy
```
