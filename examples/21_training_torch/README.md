# Training a torch MLP

An ordinary PyTorch training script — standardize 20,000 rows of features
in a Python loop, then run 100 steps of a two-layer MLP — converted by
`ppy convert` with no hand editing. The preprocessing loop went from about
54 ms to under 1 ms; the training step became one ATen region and got about
20% faster on the CPU. The checksum and the loss trajectory are identical on
every path.

## Where the speedup is

```python
def standardize(raw: Buffer[float], out: Buffer[float], rows: int, cols: int) -> float:
```

`standardize` reads and writes by index only, so `--promote-buffers`
declared its parameters `Buffer[float]` and rewrote the values feeding them
into `array.array`. The loop lowers to native code writing into memory the
caller owns, and the 20,000×16 standardization that dominated the script's
Python time disappears from the profile.

## Where it is not

`forward_loss` is a single function of curated tensor operations —
`matmul`, `add`, `relu`, `sub`, `mul`, `mean` — so the torch plugin compiles
it into one C++ ATen region. Every `at::` call still goes through the
dispatcher, `.backward()` sees the same graph, and the loss trajectory is
bit-identical. On small CPU tensors that removes one Python round trip per
operator, roughly 20%; on an accelerator it removes nothing measurable,
because kernel launch latency dominates.

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
prep      65.0 ms   checksum=-21282.454900
train    159.7 ms   loss 1.0250 -> 1.0019
```

**`ppy run train.ppy`**

```text
# device: cpu
# native prep: False
# aten region: True
prep       0.9 ms   checksum=-21282.454900
train    187.7 ms   loss 1.0250 -> 1.0019
```

<!-- outputs:end -->

Read on: [PyTorch regions](../09_torch/README.md) ·
[Training with JAX](../22_training_jax/README.md) ·
[Plugins: PyTorch](../../docs/internals/plugins.md)

Generated, not hand-written: `train.ppy` is exactly what
`ppy convert train.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run.
