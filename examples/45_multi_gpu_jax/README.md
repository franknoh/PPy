# Multi-GPU JAX training

This trainer runs a data-parallel MLP over a mesh of every accelerator in the
machine, and holds the result to a single-device run. The batch is sharded
across the devices, the parameters are replicated, and the gradient is summed
across devices by the collective `jax.jit` inserts. PPy standardizes the
input natively first, as the [single-device trainer](../22_training_jax/README.md)
does.

## Run it

On a machine with fewer than two accelerators the program says so and exits;
see [What counts as an accelerator](#what-counts-as-an-accelerator).

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

## The sharding

```python
mesh = Mesh(devices, ("batch",))
by_batch = NamedSharding(mesh, PartitionSpec("batch"))
everywhere = NamedSharding(mesh, PartitionSpec())
x = jax.device_put(x_host, by_batch)
w1 = jax.device_put(jax.random.normal(k1, (COLS * 2, HIDDEN)) * 0.1, everywhere)
```

`train_step` is the same `jax.value_and_grad` step as the single-device
trainer. With `x` and `y` sharded over the batch axis and the parameters
replicated, the mean loss is a reduction across devices, and XLA inserts the
all-reduce that sums each device's gradient contribution.

Afterwards the program runs the same hundred steps on one device and holds
the two runs to one answer:

- The loss must fall.
- The two final losses must agree.
- The largest difference between any two corresponding parameters is
  printed.

The timed loop dispatches its hundred steps without a host synchronization
between them. The first and the last loss stay on the device until
`block_until_ready()` ends the timing, so `train` measures the steps rather
than a wait for the device after each one.

## What counts as an accelerator

With fewer than two accelerators, the program does not work around it: it
prints `requires >= 2 physical accelerator devices; found N` and exits. A CPU
is not a GPU, and a virtual device
(`XLA_FLAGS=--xla_force_host_platform_device_count=2`) is not a second card.

`--allow-cpu-devices` lets a test that made virtual CPU devices on purpose
exercise the sharding here (`tests/test_multi_device_jax.py` does). It is
never a hardware pass.

The real run is
[`scripts/cloud/runpod_matrix.py --multigpu`](../../scripts/cloud/runpod_matrix.py).
It rents two GPUs, runs this program on them under `ppy run` and under
`python`, and brings the transcript back. The
[hardware validation](../../docs/internals/hardware-validation.md) page
records the last one.

Read on: [Training a JAX MLP](../22_training_jax/README.md) ·
[Hardware validation](../../docs/internals/hardware-validation.md)

`train.ppy` is hand-written; there is no `.py` source and no conversion step.
