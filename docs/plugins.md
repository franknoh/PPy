# Plugins

Each supported library is a plugin (spec 19–23): the compiler learns that
library's types and effects, takes a faster path where it can prove
equivalence, and falls back to the ordinary Python call everywhere else. A
guard that fails at runtime is a fallback, never a different answer.

Every plugin has a `fingerprint()` — its own version plus the library build it
found (`v1:torch=2.11+cu128:cxx11abi=1:cuda=12.8:...`) — and every cache key
includes the fingerprints of the plugins whose library the module actually
imports. An artifact built against one build of a library is never reused
against another, and a module that imports none of them pays for none of them.
`ppy doctor` prints what each plugin detected.

## NumPy

- Array expressions are typed with dtype and shape refinements; `tolist()`
  and friends follow the declared dtype.
- Elementwise expressions fuse into one strided loop with no temporaries,
  compiled through LLVM; contiguity and shape are guarded at runtime.
- `dot`, `matmul`, `inner`, `vdot`, `tensordot` route to the linear-algebra
  path.
- Reduction order is preserved bit-for-bit unless the function is
  `@ppy.fastmath`.

## PyTorch

- A function whose body is entirely curated tensor operations (55 ops,
  `plugins/torch_plugin.CURATED_OPS`) compiles into one C++ region calling
  ATen directly — one Python round trip per call instead of one per operator.
- The region still calls through the dispatcher, so autograd, device
  selection, and backend keys behave identically; a tensor subclass or
  `__torch_function__` override fails the guard and the Python body runs.
- Building the region needs a C++ compiler and `ninja`; `toolchain_ready()`
  reports what is missing. CUDA is used when available.
- Worth ~20% on small CPU tensors; nothing on an accelerator, where kernel
  launch latency dominates. Measured honestly in `examples/21_training_torch`.

## JAX

- A `@jax.jit` function (or one marked `@ppy.jax`) whose inputs carry
  `ppy.Shape` and `ppy.DType` exports to StableHLO at build time via
  `jax.export`, so the trace is not repeated at startup; shapes may be
  symbolic, so one artifact serves every batch size.
- Export executes project code at build time, so it is governed by
  `[tool.ppy] build-execution` and **off by default** (`"deny"`).
- At runtime the artifact executes through PJRT; a mismatch falls back to the
  ordinary jitted call.

## Pydantic

- Models are typed from their fields; the constructor signature and the
  validated output shape are kept distinct.
- Field constraints become integer-range refinements the checker propagates —
  both spellings: `Annotated[int, Field(ge=0, le=100)]` and
  `count: int = Field(ge=0, le=100)` (`ge`/`gt`/`le`/`lt`, `conint`).
- Schema-building code execution is policy-gated like JAX export.

## Uvicorn

- `uvicorn.run(app)` with a statically resolvable application skips the
  per-worker re-import by module string.
- The reloader is told to watch `*.ppy` alongside `*.py`.

## Writing against the interface

A plugin implements the `Plugin` protocol in `plugins/base.py`: type results
for calls and attributes (`CallResult`), effect declarations, an optional
build-time stage, and `fingerprint()`. Registration is explicit in the
registry; per-plugin options live under `[tool.ppy.plugins.<name>]`, and
`enabled = false` turns one off.
