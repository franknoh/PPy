# Multi-GPU JAX training

A data-parallel MLP over a mesh of every accelerator in the machine: the
batch sharded across the devices, the parameters replicated, the gradient
summed across devices by the collective `jax.jit` inserts, and the input
standardized natively by PPy first, as the
[single-device trainer](../22_training_jax/README.md) does.

## The sharding

```python
mesh = Mesh(devices, ("batch",))
by_batch = NamedSharding(mesh, PartitionSpec("batch"))
everywhere = NamedSharding(mesh, PartitionSpec())
x = jax.device_put(x_host, by_batch)
w1 = jax.device_put(jax.random.normal(k1, (COLS * 2, HIDDEN)) * 0.1, everywhere)
```

`train_step` is the same `jax.value_and_grad` step as the single-device
trainer; with `x` and `y` sharded over the batch axis and the parameters
replicated, the mean loss is a reduction across devices, and XLA inserts
the all-reduce that sums each device's gradient contribution. The program
runs the same hundred steps on one device afterwards and holds the two
runs to one answer: the loss must fall, the two final losses must agree,
and the largest difference between any two corresponding parameters is
printed.

## What counts as an accelerator

Fewer than two accelerators is reported, not worked around: the program
prints `requires >= 2 physical accelerator devices; found N` and exits.
A CPU is not a GPU, and a virtual device
(`XLA_FLAGS=--xla_force_host_platform_device_count=2`) is not a second
card. `--allow-cpu-devices` lets a test that made virtual CPU devices on
purpose exercise the sharding here (`tests/test_multi_device_jax.py`
does); it is never a hardware pass. The real run is
[`scripts/cloud/runpod_matrix.py --multigpu`](../../scripts/cloud/runpod_matrix.py),
which rents two GPUs, runs this program on them under `ppy run` and under
`python`, and brings the transcript back; the
[hardware validation](../../docs/internals/hardware-validation.md) page
records the last one.

## Run it

```bash
python  train.ppy
ppy run train.ppy
```

<!-- outputs:start -->
## What it prints

**`python  train.ppy`**, **`ppy run train.ppy`**

```text
requires >= 2 physical accelerator devices; found 0
```

<!-- outputs:end -->

Read on: [Training a JAX MLP](../22_training_jax/README.md) ·
[Hardware validation](../../docs/internals/hardware-validation.md)

`train.ppy` is hand-written; there is no `.py` source and no conversion step.
