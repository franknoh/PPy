# Flax

An MLP regression trained with Flax (linen) and optax, converted from
ordinary Python and checked under `strict = true` with nothing extra
installed or configured. The jax plugin models the Flax and optax surface,
so strict mode has a signature for every call.

## What the conversion typed, and what it left alone

```python
@ppy.pure
def target_curve(x: jax.Array) -> jax.Array:
    return 3.0 * x * x - 2.0 * x + 0.5
```

The untyped helpers got their types from their call sites — `target_curve`
receives a `jax.Array` because that is what flows into it — and `@ppy.pure`
where the checker proved it. `train_step`, decorated with
`@partial(jax.jit, static_argnums=(0, 1))`, and `__call__`, decorated with
`@nn.compact`, keep their author-written signatures: a `partial` of a
vouched decorator counts as that decorator, and the policy leaves a
transformed function's annotations as they are.

## A base class only the plugin knows

`class Mlp(nn.Module)` inherits `init` and `apply` from a base that is not
in the project. The checker resolves them through the class's external MRO,
which the plugin supplies, so `model.init(key, xs)` and `model.apply(p, xs)`
type-check. Layer constructors (`nn.Dense`), activations (`nn.relu`),
`optax.adam`, and `tx.update`'s `(updates, state)` pair all have signatures.
Parameter pytrees are an explicit `Any` boundary, declared in the source,
not an inferred one. All three paths train to the same loss on the same
device, matching the original `.py`.

## Run it

```bash
python  train.ppy
ppy run train.ppy
```

<!-- outputs:start -->
## What it prints

**`python  train.ppy`**

```text
device=cpu
loss 3.0810 -> 0.0105
learned
```

**`ppy run train.ppy`**

```text
device=cpu
loss 3.0810 -> 0.0105
learned
```

<!-- outputs:end -->

Read on: [Training with JAX](../22_training_jax/README.md) ·
[Plugins: JAX](../../docs/internals/plugins.md)

Generated, not hand-written: `train.ppy` is exactly what `ppy convert
train.py` writes, and `examples/verify_conversions.py` checks that on every
run.
