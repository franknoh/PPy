# A trainer under `torchrun` and `accelerate launch`

A plain PyTorch trainer whose kernels are `.ppy` modules. `import ppy` is
the whole integration: each rank finds the kernels' native build in the
project cache, and preprocessing that took 110 ms per rank in Python takes
1.6 ms. No bootstrap, no launcher of its own, no `ppy run`.

```python
import ppy  # first: the import hook, serving .ppy modules from their native build
import torch

import features  # features.ppy: the preprocessing loops, native
import model  # model.ppy: the model as one ATen region, native
```

## The program does not know how it was started

`train.py` reads the environment every launcher agrees on (`RANK`,
`WORLD_SIZE`, `LOCAL_RANK`) and runs the same under `python`, `torchrun`,
and `accelerate launch`. The launcher starts ordinary interpreters, and each
one's `import ppy` finds the kernels' native build.

## The two kernels

`features.ppy` holds the per-batch arithmetic — standardizing rows,
appending interactions, bucketing the result — as loops over borrowed
buffers. That is where the time goes on the Python side of a trainer, and
where the native path wins. `model.ppy` imports torch: its `forward_loss`
is one function of curated tensor operations, compiled into an ATen region
that ships beside the manifest and loads without the compiler in the
process. `.backward()` sees the same graph, because every `at::` call still
goes through the dispatcher.

## Warm the cache before the launch

The first process to import a kernel builds it into `.ppy-cache/`; every
process after that finds the build. Under a launcher the ranks start
together, so without a warm cache each builds the same artifact and the
first to finish is kept — correct, but paid for N times. `ppy build --warm .`
before the launch builds every kernel once, and no rank builds anything. A
cold first build of the torch region takes tens of seconds, once.

## Measured

CPU, 2 ranks, each on 20,000 rows × 16 columns; 100 steps of a 32-unit MLP.

| per rank | `PPY_IMPORT=python` | `import ppy` |
|---|---:|---:|
| preprocessing (`standardize` + `bucketize`) | 109.9 ms | **1.6 ms** |
| 100 training steps (`forward_loss` region) | 399 ms | 250–600 ms |

The preprocessing is the point. The region removes
four Python round trips per step, which
[21_training_torch](../21_training_torch/README.md) measures at about 20%
in isolation; here the step is dominated by the tensor work and the
run-to-run noise of a two-rank CPU launch is wider than the gain. On an
accelerator the region changes nothing measurable. `PPY_IMPORT=python` runs
the same files as plain Python; `PPY_QUIET=1` silences the per-rank notes.

## Run it

```bash
ppy build --warm .
python train.py
torchrun --standalone --nproc_per_node=2 train.py
accelerate launch --multi_gpu --num_processes 2 train.py
PPY_IMPORT=python python train.py
ppy check .
python features.ppy && ppy run features.ppy
python model.ppy    && ppy run model.ppy
```

<!-- outputs:start -->
## What it prints

**`ppy build --warm .`**, **`ppy check .`**

*(prints nothing; exits 0)*

**`python train.py`**

```text
# rank 0/1 device=cpu native=True region=True loader=GeneratedLoader
rank 0: prep       1.9 ms   checksum=-21015.470416 outside=4103
rank 0: train  16513.7 ms   loss 1.0412 -> 1.0107
```

**`torchrun --standalone --nproc_per_node=2 train.py`**

```text
# rank 0/2 device=cpu native=True region=True loader=GeneratedLoader
rank 0: prep       1.9 ms   checksum=-21015.470416 outside=4103
rank 0: train    783.5 ms   loss 1.0412 -> 1.0123
# rank 1/2 device=cpu native=True region=True loader=GeneratedLoader
rank 1: prep       1.9 ms   checksum=-20883.104755 outside=4005
rank 1: train    781.9 ms   loss 1.0486 -> 1.0239
```

**`accelerate launch --multi_gpu --num_processes 2 train.py`**

*not run here: `accelerate` is not installed*

**`PPY_IMPORT=python python train.py`**

```text
# rank 0/1 device=cpu native=False region=False loader=PPySourceLoader
rank 0: prep     125.0 ms   checksum=-21015.470416 outside=4103
rank 0: train  12706.6 ms   loss 1.0412 -> 1.0107
```

**`python features.ppy && ppy run features.ppy`**

```text
checksum=-2083.056718 outside=134 counts=[452, 3853, 11555, 16790, 16661, 11022, 3174, 359]
checksum=-2083.056718 outside=134 counts=[452, 3853, 11555, 16790, 16661, 11022, 3174, 359]
```

**`python model.ppy    && ppy run model.ppy`**

```text
# aten region: False
loss=4.584410
# aten region: True
loss=4.584410
```

<!-- outputs:end -->

`accelerate` is not a dependency of this repository; `--multi_gpu` is what
makes it launch several processes, and the script runs them on the CPU when
there is no CUDA device.

Read on: [Interop](../24_interop/README.md) ·
[CLI: `ppy build --warm`](../../docs/cli.md) ·
[Migrating a real project](../../docs/internals/migrating.md)

`features.ppy`, `model.ppy`, and `train.py` are hand-written; there is no
conversion step.
