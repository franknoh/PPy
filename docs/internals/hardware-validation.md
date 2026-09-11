# Hardware validation

The hosted CI runs on CPUs. What the accelerator stack does on a real GPU
-- whether JAX sees the card or quietly answers `CpuDevice(id=0)`, whether
`ppy.cuda` launches, whether `ppy.xla` compiles for the GPU, whether a
sharded trainer spans two physical cards -- is checked on rented machines
by `scripts/cloud/runpod_matrix.py`, on demand, never on a push.

```bash
python scripts/cloud/runpod_matrix.py --cuda        # one NVIDIA GPU
python scripts/cloud/runpod_matrix.py --multigpu    # two or more NVIDIA GPUs in one Pod
python scripts/cloud/runpod_matrix.py --rocm        # one AMD Instinct GPU
python scripts/cloud/runpod_matrix.py --all
```

The script needs an authenticated `runpodctl` (`runpodctl doctor`) and a
public key in `~/.ssh`; it never reads or prints the API key. For each
environment it asks what is in stock, prefers a common card (an RTX 4090,
an A6000, an A40, an L40S) over an expensive one, makes a Pod named
`ppy-test-<env>-<id>` from a vendor image with a four-hour termination
set on it, waits for SSH, and runs `scripts/cloud/remote_test.sh` there
against the exact commit named (`--sha`, the checkout's HEAD by default,
which the Pod clones, so it must be pushed). Every step of the remote
script is logged and its status recorded; nothing is skipped silently.
The results come back under `cloud-results/<stamp>/<env>/`, which git
ignores, and the Pod is deleted in a `finally` and confirmed gone. Only
Pods the run created are touched.

## What a run checks

- **The environment as installed.** The lock's JAX is synced, then the
  accelerator plugin for that same version is installed (`jax[cuda12]`,
  or AMD's ROCm plugin), and `scripts/cloud/accelerator_check.py` records
  the interpreter, JAX, jaxlib, every plugin package, the driver and
  toolkit the vendor tools report, `jax.default_backend()`, and every
  device -- then fails, with the report written, if every device is a CPU.
  An import that succeeded proves nothing; a CPU is not a pass for a GPU
  test.
- **A computation on the device.** A jitted matrix multiplication and a
  reduction, synchronized with `block_until_ready()`, against a NumPy
  reference; with two or more devices, an array sharded across them and
  a reduction across the shards.
- **PPy's own paths.** `examples/39_xla` under `ppy run` with the bridge
  compiling for the GPU; `examples/38_cuda` and `examples/44_tile` under
  `ppy run`, with `ppy explain` on the kernels so a launch that fell back
  to the reference is seen as one; the GPU, tile, and XLA test files; and
  the lowering-limit tests, so a function that stays in Python is the
  one the docs say stays.
- **Multi-GPU.** `examples/45_multi_gpu_jax` under `ppy run` and `python`
  -- the batch sharded over a mesh of the cards, the gradient summed
  across them, the loss falling, the result held to a single-device run
  -- and `scripts/cloud/multiprocess_smoke.py`, one JAX process per card
  with a coordinator on the loopback address, asserting
  `jax.process_count()`, `jax.process_index()`, the device lists, and one
  collective across the processes.
- **A few benchmarks**, for a sanity check of accelerator performance:
  the JAX and PyTorch comparisons, the CUDA and tile examples against
  CuPy and Triton. These are validation numbers from a rented machine;
  the tables in the READMEs come from the dedicated bench runner and are
  never overwritten by them.

## Last run

*No run is recorded yet.*
