# Training a JAX MLP

The same trainer as [21_training_torch](../21_training_torch/README.md)
with JAX, converted by `ppy convert` with no hand editing. Preprocessing
goes from about 63 ms to under 1 ms. The training step stays where it was —
about 25 ms for 100 steps either way — because XLA compiled it on its first
call and there is no per-operator Python overhead left to remove.

## Borrowed buffers, again

```python
def standardize(raw: Buffer[float], out: Buffer[float], rows: int, cols: int) -> float:
```

The converter declared both parameters `Buffer[float]` because the body
indexes them and never slices, and the values feeding them became
`array.array`. The loop lowers natively.

## `@jax.jit` is already the fast path

```python
@jax.jit
def train_step(x, y, w1, b1, w2, b2, rate) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    grads = gradients_of(x, y, w1, b1, w2, b2)
    return (w1 - rate * grads[0], b1 - rate * grads[1], w2 - rate * grads[2], b2 - rate * grads[3])
```

`ppy build` can export a `@jax.jit` function whose inputs carry `ppy.Shape`
and `ppy.DType` to StableHLO ahead of time, which saves the trace rather
than the kernel ([JAX export](../25_jax_export/README.md)). It declines to
export a function that is differentiated, and says why: a serialized
`jax.export` artifact carries no VJP, so routing `forward_loss` through one
would break `jax.grad` on a program that runs correctly under plain CPython.

## What the input had to get right

- `standardize` indexes rather than slices, as in the torch example.
- `train_step` takes and returns the parameters positionally rather than
  rebinding one list. Rebinding a variable from the return value of the
  function it is passed to makes its type self-referential, and inference
  gives up.
- `forward_loss` is called directly as well as through `jax.grad`. A
  function only ever reached through a higher-order transform has no call
  site to infer from.

## Run it

```bash
python  train.ppy
ppy run train.ppy
```

<!-- outputs:start -->
## What it prints

**`python  train.ppy`**

```text
# device: cpu
# native prep: False
prep      70.4 ms   checksum=-21433.891867
train    107.6 ms   loss 1.0663 -> 1.0109
```

**`ppy run train.ppy`**

```text
# device: cpu
# native prep: False
prep       1.1 ms   checksum=-21433.891867
train    103.6 ms   loss 1.0663 -> 1.0109
```

<!-- outputs:end -->

Read on: [Plugins: JAX](../../docs/internals/plugins.md) ·
[Flax](../29_flax/README.md)

Generated, not hand-written: `train.ppy` is exactly what
`ppy convert train.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run.
