# Flax

An MLP regression trained with Flax (linen) and optax, converted from
ordinary Python and checked under `strict = true` — the jax plugin covers the
Flax and optax surface, so nothing extra is installed or configured.

## Provenance

Generated, not hand-written. `train.ppy` is exactly what
`ppy convert train.py` writes; `verify_conversions.py` regenerates it to
prove that.

## What it shows

- The untyped helpers gain types from their call sites — `target_curve` gets
  `jax.Array` because that is what flows into it — and `@ppy.pure` where the
  checker proved it.
- `class Mlp(nn.Module)` inherits `init`/`apply` from a base only the plugin
  knows; the checker resolves them through the class's external MRO.
- Layer constructors (`nn.Dense`), activations (`nn.relu`), optimizers
  (`optax.adam`), and `tx.update`'s `(updates, state)` pair all have
  signatures under strict mode; parameter pytrees are an explicit `Any`
  boundary, not an inferred one.
- `@nn.compact` and `@partial(jax.jit, ...)` decorated functions keep their
  author-written signatures.
- All three paths train to the same loss on the same device, matching the
  original `.py`.

## Run it

```bash
python train.ppy    # plain CPython
ppy train.ppy       # optimized backend
ppy run train.ppy   # native-eligible helpers through LLVM
```
