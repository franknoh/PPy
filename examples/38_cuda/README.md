# GPU kernels

A saxpy and a block reduction with shared memory and a warp shuffle, written
once in `ppy.cuda`: the reference launch under CPython, PTX through the CUDA
driver under `ppy run` where a device is present, and CUDA or HIP source
from `ppy emit`.

## Provenance

Hand-written. `saxpy.ppy` is written directly; there is no `.py` source and
no conversion step involved.

## What it shows

- `@cuda.kernel` marks a function of scalars and pointers that runs once
  per thread of a launch, `@cuda.device` one a kernel calls; `thread_id`,
  `block_id`, `block_dim`, `global_id` say where a thread is.
- `cuda.shared[float, 64]()` is memory a block shares, `syncthreads` waits
  for the block, and `shfl_xor` trades a scalar across the warp -- the
  reference launch runs a block's threads together, each a Python thread that
  knows its position, so the barrier and the shuffle are real.
- `cuda.launch(kernel, grid, block, *args)` runs the kernel and waits; a
  `native` pointer's whole array goes to the device and, when the pointer is
  mutable, comes back, so a launch means what the reference launch means.
- Inside device code `int` arithmetic wraps and nothing guards, as on the
  device; the CPU backends leave device code alone.
- `cuda.compiled(saxpy)` says whether PTX ran; without a driver, a device,
  or the NVPTX backend the reference runs and `W2008` says why. The line
  that prints it starts with `# `.

## Run it

```bash
python  saxpy.ppy
ppy run saxpy.ppy                 # PTX through the driver where there is a device
ppy emit cuda saxpy.ppy           # the kernels, the device function, and the launch as CUDA C++
ppy emit hip saxpy.ppy            # the same as HIP
ppy emit ptx saxpy.ppy            # what the driver receives (PPY_CUDA_ARCH picks the target)
ppy inspect saxpy.ppy --stage gpu # the device code alone, as IR
```
