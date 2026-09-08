# A trainer under `torchrun` and `accelerate launch`

The same program, the same kernels, native on every rank -- with `import ppy`
as the whole integration.

```python
import ppy  # first: the import hook, serving .ppy modules from their native build
import torch

import features  # features.ppy: the preprocessing loops, native
import model  # model.ppy: the model as one ATen region, native
```

## Provenance

Hand-written. `features.ppy`, `model.ppy`, and `train.py` are written
directly; there is no conversion step involved.

## What it shows

- `train.py` is plain Python and knows nothing about how it was started. It
  reads the environment every launcher agrees on (`RANK`, `WORLD_SIZE`,
  `LOCAL_RANK`) and runs the same under `python`, `torchrun`, and
  `accelerate launch`. There is no bootstrap, no hook of its own, and no
  `ppy run`: the launcher starts ordinary interpreters, and each one's
  `import ppy` finds the kernels' native build.
- `features.ppy` holds the per-batch arithmetic -- standardizing rows,
  appending interactions, bucketing the result -- as loops over borrowed
  buffers. That is where the time goes on the Python side of a trainer, and
  where the native path wins.
- `model.ppy` imports torch. Until now that alone kept a module on the
  in-process JIT; its ATen region is compiled into the artifact instead, the
  extension ships beside the manifest, and the runtime loads it without the
  compiler. `.backward()` sees the same graph: every `at::` call still goes
  through the dispatcher.
- The first process to import a kernel builds it into `.ppy-cache/`; every
  process after that finds the build. Under a launcher the ranks start
  together, so without a warm cache each builds the same artifact and the
  first to finish is kept -- correct, but paid for N times. `ppy build --warm .`
  before the launch builds every kernel once, and no rank builds anything.
- `PPY_IMPORT=python` runs the same files as plain Python, for the comparison
  below. `PPY_QUIET=1` silences the per-rank `[ppy]` notes.

## Measured

CPU, 2 ranks, each on 20 000 rows x 16 columns; 100 steps of a 32-unit MLP.

| per rank | `PPY_IMPORT=python` | `import ppy` |
|---|---:|---:|
| preprocessing (`standardize` + `bucketize`) | 109.9 ms | **1.6 ms** |
| 100 training steps (`forward_loss` region) | 399 ms | 250-600 ms |

The checksums, bucket counts, and loss trajectories are identical on both
paths and under every launcher. The preprocessing is the point: the loops
lower to native code writing into memory the caller owns. The region
removes four Python round trips per step and is worth about 20% on small
CPU tensors (`21_training_torch` measures that in isolation); here the
step time is dominated by the tensor work itself and the run-to-run noise
of a 2-rank CPU launch is wider than the gain. On an accelerator the region
changes nothing measurable.

A cold first launch pays for the region: compiling it against the installed
PyTorch takes tens of seconds, once, into `.ppy-cache/torch/`. `ppy build
--warm .` moves that cost to before the launch.

## Run it

```bash
ppy build --warm .                                              # once, before the launch
python train.py                                                 # one process
torchrun --standalone --nproc_per_node=2 train.py               # two ranks
accelerate launch --multi_gpu --num_processes 2 train.py        # the same two ranks, via accelerate
PPY_IMPORT=python python train.py                               # the Python comparison

ppy check .                                                     # kernels and trainer, strict
python features.ppy && ppy run features.ppy                     # the kernels' own checks
python model.ppy    && ppy run model.ppy
```

<!-- outputs:start -->
## What it prints

**`ppy build --warm .`**

*(prints nothing; exits 0)*

**`python train.py`**

```text
# rank 0/1 device=cpu native=True region=True loader=GeneratedLoader
rank 0: prep       2.1 ms   checksum=-21015.470416 outside=4103
rank 0: train    281.4 ms   loss 1.0412 -> 1.0107
```

**`torchrun --standalone --nproc_per_node=2 train.py`**

```text
# rank 0/2 device=cpu native=True region=True loader=GeneratedLoader
rank 0: prep       2.1 ms   checksum=-21015.470416 outside=4103
rank 0: train    598.9 ms   loss 1.0412 -> 1.0123
# rank 1/2 device=cpu native=True region=True loader=GeneratedLoader
rank 1: prep       2.0 ms   checksum=-20883.104755 outside=4005
rank 1: train    602.6 ms   loss 1.0486 -> 1.0239
```

**`accelerate launch --multi_gpu --num_processes 2 train.py`**

*not run here: `accelerate` is not installed*

**`PPY_IMPORT=python python train.py`**

```text
# rank 0/1 device=cpu native=False region=False loader=PPySourceLoader
rank 0: prep     110.5 ms   checksum=-21015.470416 outside=4103
rank 0: train    188.5 ms   loss 1.0412 -> 1.0107
```

**`ppy check .`**

*(prints nothing; exits 0)*

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
