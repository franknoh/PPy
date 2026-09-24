# Hardware validation

The hosted CI runs on CPUs. What the accelerator stack does on a real GPU is
checked on rented machines by `scripts/cloud/runpod_matrix.py`, on demand,
never on a push. The questions it answers:

- whether JAX sees the card or quietly answers `CpuDevice(id=0)`
- whether `ppy.cuda` launches
- whether `ppy.xla` compiles for the GPU
- whether a sharded trainer spans two physical cards

```bash
python scripts/cloud/runpod_matrix.py --cuda        # one NVIDIA GPU
python scripts/cloud/runpod_matrix.py --multigpu    # two or more NVIDIA GPUs in one Pod
python scripts/cloud/runpod_matrix.py --rocm        # one AMD Instinct GPU
python scripts/cloud/runpod_matrix.py --all
```

## How a run works

The script needs an authenticated `runpodctl` (`runpodctl doctor`) and a
public key in `~/.ssh`. It never reads or prints the API key.

For each environment it:

1. asks what is in stock, and prefers a common card (an RTX 4090, an A6000,
   an A40, an L40S) over an expensive one
2. makes a Pod named `ppy-test-<env>-<id>` from a vendor image, with a
   four-hour termination set on it
3. waits for SSH
4. runs `scripts/cloud/remote_test.sh` there against the exact commit named
   (`--sha`, the checkout's HEAD by default). The Pod clones it, so it must
   be pushed.

Every step of the remote script is logged and its status recorded; nothing
is skipped silently. The results come back under
`cloud-results/<stamp>/<env>/`, which git ignores. The Pod is deleted in a
`finally` and confirmed gone. Only Pods the run created are touched.

## What a run checks

- **The environment as installed.** The lock's JAX is synced, then the
  accelerator plugin for that same version is installed (`jax[cuda12]`, or
  AMD's ROCm plugin). `scripts/cloud/accelerator_check.py` records the
  interpreter, JAX, jaxlib, every plugin package, the driver and toolkit the
  vendor tools report, `jax.default_backend()`, and every device. It then
  fails, with the report written, if every device is a CPU. An import that
  succeeded proves nothing, and a CPU does not pass a GPU test.
- **A computation on the device.** A jitted matrix multiplication and a
  reduction, synchronized with `block_until_ready()`, against a NumPy
  reference. With two or more devices, an array is sharded across them and
  reduced across the shards.
- **PPy's own paths.**
    - `examples/39_xla` under `ppy run`, with the bridge compiling for the
      GPU.
    - `examples/38_cuda` and `examples/44_tile` under `ppy run`, with
      `ppy explain` on the kernels so a launch that fell back to the
      reference is seen as one.
    - The GPU, tile, and XLA test files.
    - The lowering-limit tests, so a function that stays in Python is the one
      the docs say stays.
- **Multi-GPU.**
    - `examples/45_multi_gpu_jax` under `ppy run` and `python`: the batch
      sharded over a mesh of the cards, the gradient summed across them, the
      loss falling, the result held to a single-device run.
    - `scripts/cloud/multiprocess_smoke.py`: one JAX process per card with a
      coordinator on the loopback address, asserting `jax.process_count()`,
      `jax.process_index()`, the device lists, and one collective across the
      processes.
- **A few benchmarks**, for a sanity check of accelerator performance: the
  JAX and PyTorch comparisons, and the CUDA and tile examples against CuPy
  and Triton. These are validation numbers from a rented machine. The tables
  in the READMEs come from the dedicated bench runner and are never
  overwritten by them.

## Last run

The `fixes-0.3` branch, 2026-09-11, all three environments at commit
`e280fe5`, from `cloud-results/` of those runs. The numbers are those
machines' and are not the READMEs' tables.

| Environment | Hardware | Count | JAX backend | PPy backend | JAX compute | Training |
|---|---|---:|---|---|---|---|
| Local | Intel Core Ultra 9 386H (CPU) | 1 | cpu | LLVM | PASS (two virtual CPU devices, `tests/test_multi_device_jax.py`) | PASS on virtual devices, not a hardware pass |
| CUDA single | NVIDIA GeForce RTX 4090 (RunPod) | 1 | gpu (`cuda:0`) | CUDA: kernels compiled for the device, `ppy.xla` on `cuda:0` | PASS | N/A |
| CUDA multi | 2 × NVIDIA GeForce RTX 4090 (RunPod) | 2 | gpu (`cuda:0`, `cuda:1`) | CUDA | PASS: sharded across both, reduced across both | PASS under `ppy run` and `python`; one process per GPU PASS |
| ROCm single | AMD Instinct MI300X (RunPod, EU-RO-1) | 1 | gpu (`rocm:0`) | `ppy.xla` on `rocm:0`; HIP is `ppy emit hip` only, the source compiled by the image's `hipcc` for gfx942, never launched by PPy | PASS | N/A |
| ROCm multi | -- | -- | -- | -- | NOT RUN | NOT RUN |

### CUDA single

**Stack.** Python 3.13.8; jax 0.11.1, jaxlib 0.11.1, jax-cuda12-plugin
0.11.1, jax-cuda12-pjrt 0.11.1, nvidia-nccl-cu12 2.31.2 (CUDA 12.9 pip
libraries); driver 580.159.04, CUDA 13.0 image, no `nvcc`.
`jax.default_backend()` = `gpu`, `jax.devices()` = `[cuda:0]`.

**JAX compute.** A jitted 1024×512×256 float32 matmul at full precision
against NumPy: relative error 5.9e-7, reduction exact.

**PPy.**

- `examples/38_cuda` and `examples/44_tile` under `ppy run` print
  `kernels compiled for a device here: True`, and `ppy explain` shows the
  host function boxed for its `GpuLaunch` effect, as the guide says it is.
- `examples/39_xla` prints `# devices: ['cuda:0']`, the bridge having
  compiled for the GPU.
- 32 tests in `test_gpu_frontend`, `test_ir_gpu`, `test_tile`, `test_xla`
  pass, as do the lowering-limit tests.

**Benchmarks on that machine.**

| benchmark | PPy `cuda.launch` | CuPy |
|---|---|---|
| saxpy | 0.40 ms | 0.44 ms |
| block max | 1.03 | 1.02 |
| saxpy with copies | 42 | 59 |

The JAX comparison ran JAX on the GPU against PPy's fused loop on the Pod's
CPU, which measures the machines and says nothing about the compilers. The
Triton counterpart's wheel found no driver in that image
(`0 active drivers`), so its row is its own failure and the PPy tile row
stands alone.

### CUDA multi

**Stack.** The same stack (Python 3.13.8, jax 0.11.1 with the CUDA 12
plugin, NCCL 2.31.2) on driver 580.95.05 with a CUDA 13.0 image.
`jax.devices()` = `[cuda:0, cuda:1]`, `jax.local_devices()` the same, both
`NVIDIA GeForce RTX 4090`.

**JAX compute.** The check sharded an array over both cards and reduced
across them (exact).

**`examples/45_multi_gpu_jax`.** The batch was sharded over the two cards,
and a hundred steps were dispatched without a host synchronization between
them. The `train` time is the steps and does not include a wait per step.

| | `ppy run` | `python` |
|---|---|---|
| `train` time | 573 ms | 570 ms |
| largest parameter difference | 7e-8 (the standardization native, 1.1 ms) | 5e-7 (71 ms) |
| result | `PASS` | `PASS` |

Loss went from 1.0414 to 1.0108 on two devices, and the same on one.

**`scripts/cloud/multiprocess_smoke.py`.** Two processes, one per card,
`jax.process_count()` 2, each seeing one local and two global devices, the
collective sum exact in both.

**PPy.** 32 GPU, tile, and XLA tests pass. PPy `cuda.launch` saxpy 0.39 ms
against CuPy 0.44, saxpy with copies 34 against 52.

#### Hosts that failed

Two other two-GPU hosts were tried in an earlier run and are the reason the
harness records what it does.

- On a two-A6000 host, NCCL's peer-to-peer transport hung (both cards at
  100%, the collective never returning) and completed with
  `NCCL_P2P_DISABLE=1`.
- On a two-A40 host every check passed, but the sharded trainer's loss came
  out wrong and different on each run (0.52, then 0.64, against 1.04 on one
  device). A collective answered wrongly instead of hanging.

The same program on the two-4090 host, and on two virtual CPU devices,
agrees with the single-device run to 7e-8. The faults are in those hosts'
NCCL, and not in PPy's or JAX's arithmetic: `sharding_probe.py` compares
every stage with NumPy and found nothing on the host that works. A run is
therefore judged on its own transcript, never on a step's exit status alone,
and the trainer exits non-zero on `FAIL`.

### ROCm single

**Stack.** An MI300X (gfx942) in EU-RO-1, refused earlier that day and
granted in the evening, in AMD's `rocm/jax:rocm10.0-jax0.11.0-py3.12` image:
Python 3.12.3, jax 0.11.0, jaxlib 0.11.0, jax-rocm10-plugin and
jax-rocm10-pjrt 0.11.0+rocm10.0.0, ROCm 10.0.0 as the pip SDK, driver
6.10.5. `jax.default_backend()` = `gpu`, `jax.devices()` = `[rocm:0]`.

**JAX compute.** The matmul and the reduction report a relative error of 0.0
against NumPy.

**Install.** PPy was installed beside that stack with the image's `jax` and
`jaxlib` as constraints (without the `jax` group's flax, which wants a newer
jax than the image has). The run checks that every `jax*` distribution is the
one that was there before the install; it is.

**PPy.**

- `examples/39_xla` prints `# devices: ['rocm:0']`, the bridge having
  compiled for the AMD card through the same PJRT path as on CUDA.
- `examples/38_cuda` and `examples/44_tile` run their CPU path there
  (`kernels compiled for a device here: False`), since `ppy.cuda` and
  `ppy.tile` launch CUDA only.
- 31 GPU, tile, and XLA tests pass with one skipped, as do the
  lowering-limit tests.
- `ppy emit hip` on `examples/38_cuda/saxpy.ppy` writes 79 lines of HIP C++
  with the `__global__` kernel and the `<<<>>>` launch. The image's `hipcc`
  compiles it for gfx942 (`HIP_DEVICE_LIB_PATH` pointed at the SDK's device
  library).

That is the extent of ROCm in PPy: `ppy.hip` is source only, and this run
does not make it a launch runtime.

### ROCm multi

Not run; one MI300X was in stock.

## What the runs found

### Bugs in PPy

The PJRT bridge behind `ppy.xla` had two bugs, both fixed:

- It compiled for every device of the platform. On the two-GPU hosts its
  executables expected one argument shard per device and refused the single
  buffers the bridge places
  (`Expected args to execute_sharded_on_local_devices to have 2 shards, got: [1, 1]`).
  It compiles for one device now, and a test over two virtual CPU devices
  holds it there.
- It defaulted to the `cpu` platform whatever the machine had. It follows
  JAX's default backend now, and refuses to call an initialization failure or
  an installed-but-idle accelerator plugin "cpu". That is how
  `examples/39_xla` came to print `cuda:0` and `rocm:0`.

### Changes to the harness

- A float32 matmul on an Ampere-class card is TF32 unless full precision is
  asked for, so the check asks.
- A cross-device step gets fifteen minutes and a recorded retry without
  peer-to-peer.
- The one-process-per-GPU step counts its transcripts, since a loop that ran
  nothing had once exited zero.
- A Pod that is still listed after the run fails the run.
- AMD's image runs no `sshd`, so the harness starts one for the injected key
  (RunPod's own pattern for a plain image) and hands the container's
  environment to the SSH session.
