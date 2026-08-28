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
