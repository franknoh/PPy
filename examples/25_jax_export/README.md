# JAX export

Build-time export of a `@jax.jit` function to StableHLO. JAX traces a
jitted function on its first call, every time the process starts; when the
function's inputs are fully described, `ppy build` can trace it once and
stage the result in the artifact.

## Describe the input, and the trace can move

```python
Batch = Annotated[jax.Array, ppy.Shape("B", 4), ppy.DType("float32")]


@jax.jit
def score(x: Batch) -> jax.Array:
    return jnp.sum(jnp.tanh(normalize(x)), axis=-1)
```

`ppy.Shape` and `ppy.DType` on the annotation are what make the function
exportable. `"B"` is a symbolic dimension: `jax.export` serializes the
function for any `B`, so one artifact serves every batch size, and the
runtime executes it through PJRT. A call whose input does not match the
description falls back to the ordinary jitted call. A function that is
differentiated is not exported, and the build says why: a serialized export
carries no VJP.

## Off until the project opts in

Export imports and runs project code at build time. It happens only when
the project sets both `[tool.ppy] build-execution = "allow"` and
`[tool.ppy.plugins.jax] allow-build-export = true`, as this folder's
`pyproject.toml` does. With either missing, the functions stay ordinary
jitted calls and `ppy` reports which, and why.

The JAX-free form is [`ppy.xla`](../39_xla/README.md): a scalar function
marked `@xla.jit` is emitted as StableHLO by the compiler itself and run
through PJRT, with no trace and no JAX in the compiler.

## Run it

```bash
python  model.ppy
ppy run model.ppy
```

<!-- outputs:start -->
## What it prints

**`python  model.ppy`**, **`ppy run model.ppy`**

```text
cpu
[0.0, 0.0, 0.0]
```

<!-- outputs:end -->

Read on: [Plugins: JAX](../../docs/internals/plugins.md) ·
[XLA](../../docs/guide/xla.md)

`model.ppy` is hand-written; there is no `.py` source and no conversion step.
