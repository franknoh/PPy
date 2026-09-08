# A PyTorch layer as one C++ call

`layer` is three tensor operations: a matmul, an add, a ReLU. Under PyTorch
that is three trips from Python into ATen and back. Under PPY it is one:
the function's body is entirely curated tensor ops, so the plugin compiles
it into a single C++ region that calls ATen directly. On an 8×32 input on
the CPU that is about 15% per call here. Autograd does not notice.

## Through the dispatcher, on purpose

```python
@ppy.opt(3)
def layer(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.relu(torch.add(torch.matmul(x, weight), bias))
```

The region does not reimplement `matmul`. Every `at::` call inside it goes
through PyTorch's dispatcher, so device selection, dtype promotion, and
autograd behave exactly as before — `residual(tracked, tracked).sum().backward()`
fills `tracked.grad` on every path, and the program prints so. What the
region removes is the Python interpreter between operators: one round trip
per call instead of one per operator.

## What the guard refuses

A tensor subclass, a `__torch_function__` override, or a device the region
was not built for fails the guard, and the Python body runs. Building the
region needs a C++ compiler and `ninja`; `toolchain_ready()` says what is
missing. On an accelerator the region changes nothing measurable — kernel
launch latency dominates the Python overhead it removes — so the numbers
worth having are CPU numbers, and this program prints those.

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
layer on cpu              2.350 us/call   sample=596.816528
autograd survives the region: True
```

**`ppy     torch_region.ppy`**

```text
torch 2.14.0+cpu | cuda False
# region active: True
layer on cpu              2.298 us/call   sample=596.816528
autograd survives the region: True
```

**`ppy run torch_region.ppy`**

```text
torch 2.14.0+cpu | cuda False
# region active: True
layer on cpu              2.367 us/call   sample=596.816528
autograd survives the region: True
```

<!-- outputs:end -->

## Read on

- [Plugins: PyTorch](../../docs/internals/plugins.md) — the 55 curated ops, the guard, and what fuses.
- [Training with torch](../21_training_torch/README.md) — a whole training step, measured honestly.

`torch_region.ppy` is hand-written; there is no `.py` source and no conversion
step.
