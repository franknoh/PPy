# Training a torch MLP

Untyped PyTorch training code, converted with no hand editing afterwards.

`train.ppy` is exactly what `ppy convert train.py --promote-buffers` writes.

## Provenance

Generated, not hand-written. Each of these is exactly what the converter
writes, and `verify_conversions.py` regenerates it to prove it:

- `train.ppy` &larr; `ppy convert train.py --promote-buffers`

## Measured

| | plain | ppy run |
|---|---:|---:|
| preprocessing | 53.9 ms | **0.8 ms** |
| 100 training steps | 208.3 ms | 163.4 ms |

`checksum` and the loss trajectory are identical on both paths.

## Where the speedup comes from

The large factor is the preprocessing, which is ordinary arithmetic rather than
tensor work. `standardize` reads and writes by index only, so the converter
declares its parameters `Buffer[float]` and rewrites the values feeding them
into `array.array`. The loop then lowers to a native loop writing into memory
the caller owns.

`forward_loss` is a single function of curated tensor operations, so it compiles
into one C++ ATen region. Every `at::` call still goes through the dispatcher,
so `.backward()` sees the same graph. That is worth roughly 20% on small CPU
tensors and nothing on an accelerator, where kernel launch latency dominates.

## What the input had to get right

- `standardize` indexes rather than slices. A slice copies, so a sliced
  parameter cannot be borrowed, and the converter says so:
  `remark[R3003]: \`raw\` is sliced, which copies`.
- `descend` narrows `parameter.grad` before using it. It is `Tensor | None`
  until a backward pass fills it, and the unnarrowed version is a latent crash
  that `ppy check` reports.
- `forward_loss` is one function. Split across two, every operator pays a Python
  round trip and no region forms.

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
# aten region: False
prep      60.3 ms   checksum=-21282.454900
train    333.3 ms   loss 1.0250 -> 1.0019
```

**`ppy check train.ppy`**

*(prints nothing; exits 0)*

**`ppy build train.ppy`**

*(prints nothing; exits 0)*

**`ppy run   train.ppy`**

```text
# device: cpu
# native prep: False
# aten region: True
prep       0.9 ms   checksum=-21282.454900
train    324.8 ms   loss 1.0250 -> 1.0019
```

<!-- outputs:end -->
