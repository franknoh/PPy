# XLA: `ppy.xla`

`ppy.xla` compiles a function of scalars to StableHLO and runs it through
XLA.

```python
import math

from ppy import xla


@xla.jit
def f(x: float, y: float) -> float:
    product = math.sin(x) * y
    return (product + (x if product > 0.0 else -x)) / 2.0
```

`@xla.jit` (or `xla.compile(f)`) marks a function of floats, ints, and bools
whose body is one block of arithmetic and math for XLA.

## How it runs

1. The compiler lowers the function to the IR and emits StableHLO.
   `ppy emit stablehlo` shows the text.
2. The build stages the StableHLO.
3. At run time the PJRT bridge compiles the module once, caching the
   executable by the module's digest, the bindings' version, and the device.
4. Each call runs on the device.

Under plain CPython, and wherever there is no device, the function runs as
written.

## Devices

`xla.devices()`, `xla.default_device()`, and `xla.device_put(x)` ask the
bridge. Without one they answer nothing, `None`, and the value itself.

The bridge compiles for the platform JAX would pick: the GPU where a CUDA or
ROCm plugin is installed, the CPU otherwise. `PPY_XLA_PLATFORM` names one
explicitly (`cpu`, `gpu`).

## Limitations

- A branch, a loop, a guard, or a buffer parameter is not yet what XLA takes.
  The function is reported (`W2007`) and stays where it is.
- XLA computes `sin` and its kin with its own library, so the last bits of a
  result can differ from CPython's `math`. The arithmetic is IEEE either way.
- The bridge compiles and runs through XLA's own bindings. Placing a NumPy
  array on the device goes through JAX's `device_put`, while that is the one
  public way to reach the client XLA compiled for. So the bridge needs JAX
  installed. The compiler that wrote the StableHLO does not.

## Relation to `@ppy.jax`

`@ppy.jax` ([JAX export](../howto/25_jax_export.md)) works in the other
direction: a `@jax.jit` function exported at build time, traced by JAX.
`ppy.xla` is the compiler writing StableHLO itself.

Examples: [XLA](../howto/39_xla.md).
