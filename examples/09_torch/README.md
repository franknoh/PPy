# PyTorch ATen regions

A function whose body is entirely curated tensor operations compiles into
one C++ region that calls ATen directly: one Python round trip per call
instead of one per operator. `layer` is a matmul, an add, and a ReLU; the
region runs them as PyTorch would, and autograd is unchanged.

## Through the dispatcher

```python
@ppy.opt(3)
def layer(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.relu(torch.add(torch.matmul(x, weight), bias))
```

The region does not reimplement `matmul`. Every `at::` call inside it goes
through PyTorch's dispatcher, so device selection, dtype promotion, and
autograd behave as before — `residual(tracked, tracked).sum().backward()`
fills `tracked.grad`. What the region removes is the Python interpreter
between operators.

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

## Compared with PyTorch eager and `torch.compile`

The same `layer` on an 8×32 input, twenty thousand calls, in
[`compare/`](compare/): [`layer_bench.ppy`](compare/layer_bench.ppy),
[`layer_eager.py`](compare/layer_eager.py),
[`layer_compile.py`](compare/layer_compile.py). Milliseconds per call, to
four places, best of five rounds, over five processes; one PyTorch thread.

**PPY** is the function as written, and `ppy run` compiles it into one ATen
region; **PyTorch eager** is the same function with no decorator, three
dispatches from Python; **`torch.compile`** is the same function under the
decorator, traced by Dynamo and written by Inductor:

```python
@ppy.opt(3)
def layer(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.relu(torch.add(torch.matmul(x, weight), bias))
```

```python
@torch.compile
def layer(x, weight, bias):
    return torch.relu(torch.add(torch.matmul(x, weight), bias))
```

<!-- compare:start -->
| | PPY ATen region | PyTorch eager | `torch.compile` |
|---|---:|---:|---:|
| layer, per call | **0.0019 ± 0.0000** | 0.0020 ± 0.0000 | 0.0087 ± 0.0002 |
<!-- compare:end -->

Three operators on a tensor this small cost about two microseconds either
way: PyTorch's eager dispatch is cheap enough that the Python round trips
the region removes are within the noise of the ATen calls themselves. The
region's value is not this number; it is that the function keeps its
source, its autograd, and its dispatcher, and that a built artifact carries
it with no compiler in the process. `torch.compile` pays for its guards on
every call, which on an 8×32 input is more than the work.

Intel Core Ultra 9 386H; PyTorch 2.14.0 (CPU) on CPython 3.13.13, PPY
against the same PyTorch on CPython 3.14.5, from a checkout on a native
filesystem.

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
layer on cpu              2.530 us/call   sample=596.816528
autograd survives the region: True
```

**`ppy     torch_region.ppy`**

```text
torch 2.14.0+cpu | cuda False
# region active: True
layer on cpu              2.069 us/call   sample=596.816528
autograd survives the region: True
```

**`ppy run torch_region.ppy`**

```text
torch 2.14.0+cpu | cuda False
# region active: True
layer on cpu              2.118 us/call   sample=596.816528
autograd survives the region: True
```

<!-- outputs:end -->

Read on: [Plugins: PyTorch](../../docs/internals/plugins.md) ·
[Training with torch](../21_training_torch/README.md)

`torch_region.ppy` is hand-written; there is no `.py` source and no conversion
step.
