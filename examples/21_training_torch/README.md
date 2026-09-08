# A torch training script, converted, and where the time went

`train.py` is an ordinary PyTorch script: standardize 20,000 rows of
features in a Python loop, then run 100 steps of a two-layer MLP.
`train.ppy` is what `ppy convert` wrote from it. The preprocessing loop
became native and went from about 54 ms to under 1 ms; the training loop
became one ATen region and got about 20% faster on the CPU. The checksum
and the loss trajectory are identical on every path, which is the part that
matters.

## The loop is where the speedup is

```python
def standardize(raw: Buffer[float], out: Buffer[float], rows: int, cols: int) -> float:
```

`standardize` reads and writes by index only, so `--promote-buffers`
declared its parameters `Buffer[float]` and rewrote the values feeding them
into `array.array`. The loop lowers to native code writing into memory the
caller owns, and the 20,000×16 standardization that dominated the script's
Python time disappears from the profile.

## The training step is one region, and that is all it can be

`forward_loss` is a single function of curated tensor operations —
`matmul`, `add`, `relu`, `sub`, `mul`, `mean` — so the torch plugin compiles
it into one C++ ATen region. Every `at::` call still goes through the
dispatcher, `.backward()` sees the same graph, and the loss trajectory is
bit-identical. On small CPU tensors that removes one Python round trip per
operator, roughly 20%; on an accelerator it removes nothing measurable,
because kernel launch latency dominates. This README does not claim
otherwise.

## What the input had to get right

- `standardize` indexes rather than slices. A slice copies, so a sliced
  parameter cannot be borrowed, and the converter says so:
  `remark[R3003]: raw is sliced, which copies`.
- `descend` narrows `parameter.grad` before using it. It is `Tensor | None`
  until a backward pass fills it, and the unnarrowed version is a latent
  crash that `ppy check` reports.
- `forward_loss` is one function. Split across two, every operator pays a
  Python round trip and no region forms.

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
# aten region: False
prep      68.5 ms   checksum=-21282.454900
train   3632.8 ms   loss 1.0250 -> 1.0019
```

**`ppy run train.ppy`**

```text
# device: cpu
# native prep: False
# aten region: True
prep       1.2 ms   checksum=-21282.454900
train   3441.9 ms   loss 1.0250 -> 1.0019
```

<!-- outputs:end -->

## Read on

- [PyTorch regions](../09_torch/README.md) — the region on its own, with the guard.
- [Training with JAX](../22_training_jax/README.md) — the same script where XLA already owns the step.
- [Plugins: PyTorch](../../docs/internals/plugins.md) — the curated ops and what fuses.

Generated, not hand-written: `train.ppy` is exactly what
`ppy convert train.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run.
