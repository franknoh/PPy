# PyTorch ATen regions

A function of curated tensor ops becomes one C++ function.

## Provenance

Hand-written. `torch_region.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- Every `at::` call still goes through the dispatcher, so autograd and device keys are unchanged.
- A tensor subclass or a `__torch_function__` override trips the guard and the Python body runs.
- On an accelerator this changes nothing measurable: kernel launch latency dominates.

## Run it

```bash
python  torch_region.ppy
ppy     torch_region.ppy
ppy run torch_region.ppy
```

<!-- outputs:start -->
## What it prints

**`python  torch_region.ppy`**

```text
torch 2.14.0+cpu | cuda False
# region active: False
layer on cpu              2.015 us/call   sample=596.816528
autograd survives the region: True
```

**`ppy     torch_region.ppy`**

```text
torch 2.14.0+cpu | cuda False
# region active: True
layer on cpu              1.711 us/call   sample=596.816528
autograd survives the region: True
```

**`ppy run torch_region.ppy`**

```text
torch 2.14.0+cpu | cuda False
# region active: True
layer on cpu              1.769 us/call   sample=596.816528
autograd survives the region: True
```

<!-- outputs:end -->
