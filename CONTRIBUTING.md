# Contributing

## Setting up

```bash
uv sync            # the compiler core, LLVM, NumPy, pydantic, and the linters
./scripts/check.sh # the one gate; CI runs exactly this
```

`uv sync` installs the `dev` group. That is what a contributor to the
compiler needs and more than a user needs: a user installs `ppy` and gets the
compiler, the runtime, and nothing else.

The plugin runtimes are separate groups, so you install only what you intend
to test:

```bash
uv sync --group torch     # PyTorch, CPU wheels
uv sync --group jax       # JAX and Flax, CPU wheels
uv sync --group uvicorn   # FastAPI and Uvicorn
uv sync --group all       # everything
```

Torch and JAX resolve from the CPU index, which works on every platform.
On a CUDA machine, override it for your checkout rather than for the
repository:

```bash
uv sync --group torch --index pytorch-cuda=https://download.pytorch.org/whl/cu128
```

## The gate

`./scripts/check.sh` is the single source of truth for "clean": ruff, ruff
format, pylint, the test suite, the conversion check, the three-path example
run, and the example lint. CI runs that script and nothing else, so a local
pass and a CI pass are the same claim. Run it before every commit.

A plugin's own claim is `./scripts/plugin_check.sh <torch|jax|uvicorn>`,
which fails if the plugin's tests all skip — a skipped test proves nothing.

## What a change has to keep true

- **The three paths agree.** Plain CPython, the optimized Python backend, and
  the native backend produce the same answer for the same program.
  `examples/run_all.py` checks every example on all three.
- **A guard that fails falls back.** Native code that cannot keep a promise
  runs the Python body; it never answers differently.
- **The cache is disposable.** Corrupting or deleting any part of it may cost
  a rebuild and must never cost the answer. See
  [docs/compatibility.md](docs/compatibility.md).
- **`ppy_runtime` never imports `ppy_compiler`.** A built artifact keeps
  working with the compiler uninstalled, and a test enforces it.
- **A generated `.ppy` is exactly what `ppy convert` writes.** Never hand-edit
  one; `examples/verify_conversions.py` regenerates and diffs them.

## Examples

An example is either hand-written or generated, and its README says which.
A generated one keeps its `.py` source beside it and is regenerated with
`ppy convert <name>.py` (some folders add `--promote-buffers`; see
`examples/verify_conversions.py`).

`python scripts/refresh.py` reports anything that has drifted — an example
that no longer checks, a conversion that no longer matches, a measurement
that has moved — and `--write` brings the first two back in line and records
the third.

## Measurements

Numbers in the documentation come from `examples/15_algorithms/bench.py` and
are recorded with the machine they were taken on. They are not a per-change
gate: a scheduled workflow re-measures and reports drift beyond a tolerance,
because absolute wall times differ between runners. If you change one, say
what machine it was measured on.
