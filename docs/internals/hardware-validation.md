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

Commit `ea6a5a5` (the `semantics-0.3` branch, 2026-09-11), from
`cloud-results/` of that run; the numbers are that machine's and are not
the READMEs' tables.

| Environment | Hardware | Count | JAX backend | PPy backend | JAX compute | Training |
|---|---|---:|---|---|---|---|
| Local | Intel Core Ultra 9 386H (CPU) | 1 | cpu | LLVM | PASS (two virtual CPU devices, `tests/test_multi_device_jax.py`) | PASS on virtual devices, not a hardware pass |
| CUDA single | NVIDIA GeForce RTX 4090 (RunPod) | 1 | gpu (`cuda:0`) | CUDA: kernels compiled for the device, `ppy.xla` on `cuda:0` | PASS | N/A |
| CUDA multi | @@MULTI_GPU@@ | 2 | @@MULTI_BACKEND@@ | CUDA | @@MULTI_COMPUTE@@ | @@MULTI_TRAINING@@ |
| ROCm single | AMD Instinct MI300X (RunPod, EU-RO-1) | 1 | -- | HIP is `ppy emit hip` only: there is no HIP launch runtime | NOT RUN | NOT RUN |
| ROCm multi | -- | -- | -- | -- | NOT RUN | NOT RUN |

**CUDA single**: Python 3.13.8; jax 0.11.1, jaxlib 0.11.1, jax-cuda12-plugin
0.11.1, jax-cuda12-pjrt 0.11.1, nvidia-nccl-cu12 2.31.2 (CUDA 12.9 pip
libraries); driver 570.195.03, CUDA 12.8 runtime in the image, no `nvcc`;
`jax.default_backend()` = `gpu`, `jax.devices()` = `[cuda:0]`,
`jax.local_devices()` = `[cuda:0]`. A jitted 1024×512×256 float32 matmul at
full precision against NumPy: relative error 5.9e-7, reduction exact. PPy:
`examples/38_cuda` and `examples/44_tile` under `ppy run` print `kernels
compiled for a device here: True`, and `ppy explain` shows the host
function boxed for its `GpuLaunch` effect, as the guide says it is;
`examples/39_xla` prints `# devices: ['cuda:0']`, the bridge having compiled
for the GPU; 27 tests in `test_gpu_frontend`, `test_ir_gpu`, `test_tile`,
`test_xla` pass, as do the lowering-limit tests. Benchmarks on that machine:
PPy `cuda.launch` saxpy 0.40 ms against CuPy 0.44 ms, block max 1.03
against 1.02, saxpy with copies 38 against 62; the JAX comparison ran
JAX on the GPU (`blend` 0.7 ms) against PPy's fused loop on the Pod's CPU
(205 ms), which measures the machines, not the compilers. The Triton
counterpart's wheel found no driver in that image (`0 active drivers`), so
its row is its own failure and the PPy tile row stands alone.

**ROCm**: `runpodctl gpu list` carried no AMD type; `runpodctl datacenter
list` named an MI300X in EU-RO-1 with no stock status, and asking for it
outright answered `There are no longer any instances available with the
requested specifications`. Nothing AMD ran. Independently of capacity,
`ppy.hip` is source only -- `ppy emit hip` writes HIP C++, and the guide
says the module has no launch runtime -- so an MI300X run would validate
JAX on ROCm and the emitted source, not a PPy HIP launch.

The harness's own findings, which are not PPy's: the first two-GPU Pod
hung in NCCL's peer-to-peer transport (both cards at 100%, the collective
never returning) and completed with `NCCL_P2P_DISABLE=1`, which is why a
cross-device step now has fifteen minutes and a recorded retry without
peer-to-peer; and a float32 matmul on an Ampere-class card is TF32 unless
the precision is asked for, which is why the check asks.
