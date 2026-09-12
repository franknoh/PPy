# Training a torch MLP

An ordinary PyTorch training script — standardize 20,000 rows of features
in a Python loop, then run 100 steps of a two-layer MLP — converted by
`ppy convert` with no hand editing. The preprocessing loop went from about
70 ms to under 1 ms; the training step became one ATen region.

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
dispatcher and `.backward()` sees the same graph. What the region removes
is the Python round trip between operators; over a 20,000-row batch the
matmul is the time and the round trips are not, so the step runs at
PyTorch's speed, and on an accelerator kernel launch latency dominates the
same way.

## What the input had to get right

- `standardize` indexes rather than slices. A slice copies, so a sliced
  parameter cannot be borrowed, and the converter says so:
  `remark[R3003]: raw is sliced, which copies`.
- `descend` narrows `parameter.grad` before using it. It is `Tensor | None`
  until a backward pass fills it, and the unnarrowed version is a latent
  crash that `ppy check` reports.
- `forward_loss` is one function. Split across two, every operator pays a
  Python round trip and no region forms.

## Compared with PyTorch as it is usually written

The standardization of 20,000×16 rows, and one forward pass of the step's
operators, in [`compare/`](compare/):
[`train_bench.ppy`](compare/train_bench.ppy) -- under `ppy run` and, the
same file, under `python` -- [`train_eager.py`](compare/train_eager.py), and
[`train_compile.py`](compare/train_compile.py). Milliseconds, best of five
(the forward pass best of two hundred), over five processes; eight PyTorch
threads.

**PPY** standardizes in the loop the script was written with, over
`Buffer[float]`; **PyTorch** is how the same preprocessing is written for
PyTorch, vectorized over the batch, with `roll` for the interaction term:

```python
def standardize(raw: Buffer[float], out: Buffer[float], rows: int, cols: int) -> float:
    total: float = 0.0
    for row in range(rows):
        base: int = row * cols
        target: int = row * cols * 2
        sum_: float = 0.0
        for i in range(cols):
            sum_ += raw[base + i]
        mean: float = sum_ / cols
        ...
```

```python
def standardize(raw: torch.Tensor) -> tuple[torch.Tensor, float]:
    mean = raw.mean(dim=1, keepdim=True)
    deviation = torch.sqrt(((raw - mean) ** 2).mean(dim=1, keepdim=True)) + 1e-8
    z = (raw - mean) / deviation
    interaction = z * z.roll(-1, dims=1)
    out = torch.cat([z, interaction], dim=1)
    return out, float(out.sum())
```

`forward_loss` is the same six operators on every side: one ATen region
under PPY, eager under PyTorch, and under `torch.compile` in the third
column.

<!-- compare:start -->
| | PPY `ppy run` | CPython, the same file | PyTorch, vectorized | `torch.compile` |
|---|---:|---:|---:|---:|
| standardize, 20000 rows | 1.27 ± 0.28 | 72.14 ± 2.32 | 1.06 ± 0.26 | **0.89 ± 0.23** |
| forward pass, per call | 0.13 ± 0.01 | 0.12 ± 0.01 | 0.13 ± 0.00 | **0.12 ± 0.00** |
<!-- compare:end -->

The loop as the script wrote it, compiled, is level with the vectorized
rewrite: both are one pass over the rows, and the rewrite costs seven
tensor temporaries the loop never makes. Under `python` the same loop is
the cost the conversion removed. The forward pass is PyTorch's on every
side -- the matmul over 20,000 rows is the time, and the Python round
trips between six operators are not -- so the region neither gains nor
loses there, and `torch.compile`'s guards are a few microseconds on top.

Intel Core Ultra 9 386H; PyTorch 2.14.0 (CPU) on CPython 3.13.13, PPY
against the same PyTorch on CPython 3.14.5, from a checkout on a native
filesystem.

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
prep      66.7 ms   checksum=-21282.454900
train    142.6 ms   loss 1.0250 -> 1.0019
```

**`ppy run train.ppy`**

```text
# device: cpu
# native prep: True
# aten region: True
prep       1.1 ms   checksum=-21282.454900
train    320.7 ms   loss 1.0250 -> 1.0019
```

<!-- outputs:end -->

Read on: [PyTorch regions](../09_torch/README.md) ·
[Training with JAX](../22_training_jax/README.md) ·
[Plugins: PyTorch](../../docs/internals/plugins.md)

Generated, not hand-written: `train.ppy` is exactly what
`ppy convert train.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run.
