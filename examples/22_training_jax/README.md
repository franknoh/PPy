# Training a JAX MLP

The same shape as `21_training_torch`, with JAX.

`train.ppy` is exactly what `ppy convert train.py --promote-buffers` writes.

## Provenance

Generated, not hand-written. Each of these is exactly what the converter
writes, and `verify_conversions.py` regenerates it to prove it:

- `train.ppy` &larr; `ppy convert train.py --promote-buffers`

## Measured

| | plain | ppy run |
|---|---:|---:|
| preprocessing | 62.8 ms | **0.9 ms** |
| 100 training steps | 24.9 ms | 24.2 ms |

`checksum` and the loss trajectory are identical on both paths.

## Where the speedup comes from, and where it does not

Preprocessing, for the same reason as the torch example: borrowed buffers turn
the loop into native code.

Training does not get faster, and should not. XLA already compiles the step on
first call, so there is no per-operator Python overhead left to remove.

## Ahead-of-time export

`ppy build` can export a `@jax.jit` function whose inputs are fully described by
`ppy.Shape` and `ppy.DType`, which saves the trace rather than the kernel. It is
declined for a function that is differentiated, and says why: a serialized
`jax.export` artifact carries no VJP, so routing one would break code that runs
correctly under plain CPython.

## What the input had to get right

- `standardize` indexes rather than slices, as in the torch example.
- `train_step` takes and returns the parameters positionally rather than
  rebinding one list. Rebinding a variable from the return value of the function
  it is passed to makes its type self-referential, and inference gives up.
- `forward_loss` is called directly as well as through `jax.grad`. A function
  only ever reached through a higher-order transform has no call site to infer
  from.

## Run it

```bash
python    train.py
ppy check train.ppy
ppy build train.ppy
ppy run   train.ppy
```

<!-- outputs:start -->
## What it prints

**`python    train.py`**

```text
# device: cpu
# native prep: False
prep      54.9 ms   checksum=-21433.891867
train    103.9 ms   loss 1.0663 -> 1.0109
```

**`ppy check train.ppy`**

*(prints nothing; exits 0)*

**`ppy build train.ppy`**

*(prints nothing; exits 0)*

**`ppy run   train.ppy`**

```text
# device: cpu
# native prep: False
prep       0.9 ms   checksum=-21433.891867
train    105.3 ms   loss 1.0663 -> 1.0109
```

<!-- outputs:end -->
