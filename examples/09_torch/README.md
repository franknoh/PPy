# PyTorch ATen regions

A function whose body is entirely curated tensor operations compiles into
one C++ region that calls ATen directly: one Python round trip per call
instead of one per operator. `layer` is a matmul, an add, and a ReLU; on an
8×32 CPU input the region is worth about 15% per call here, and autograd is
unchanged.

## Through the dispatcher

```python
@ppy.opt(3)
def layer(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.relu(torch.add(torch.matmul(x, weight), bias))
```

The region does not reimplement `matmul`. Every `at::` call inside it goes
through PyTorch's dispatcher, so device selection, dtype promotion, and
autograd behave as before — `residual(tracked, tracked).sum().backward()`
fills `tracked.grad` on every path, and the program prints so. What the
region removes is the Python interpreter between operators.

## What the guard refuses, and what the region costs

A tensor subclass, a `__torch_function__` override, or a device the region
was not built for fails the guard, and the Python body runs. Building the
region needs a C++ compiler and `ninja`; `toolchain_ready()` says what is
missing. On an accelerator the region changes nothing measurable — kernel
launch latency dominates the Python overhead it removes — so the program
measures on the CPU, and on `cuda` too when one is present.

A built artifact carries its regions: `ppy build` copies the extension
beside the manifest, and the launcher loads it with no compiler in the
process ([torchrun](../31_torchrun/README.md)).

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
layer on cpu              1.973 us/call   sample=596.816528
autograd survives the region: True
```

**`ppy     torch_region.ppy`**

```text
torch 2.14.0+cpu | cuda False
# region active: True
layer on cpu              1.716 us/call   sample=596.816528
autograd survives the region: True
```

**`ppy run torch_region.ppy`**

```text
torch 2.14.0+cpu | cuda False
# region active: True
layer on cpu              1.866 us/call   sample=596.816528
autograd survives the region: True
```

<!-- outputs:end -->

Read on: [Plugins: PyTorch](../../docs/internals/plugins.md) ·
[Training with torch](../21_training_torch/README.md)

`torch_region.ppy` is hand-written; there is no `.py` source and no conversion
step.
