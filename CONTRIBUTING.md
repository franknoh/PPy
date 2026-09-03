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
that has moved, a README table that is behind the record — and `--write`
brings all of them back in line. `--quick` skips the benchmark; `--record
FILE` also writes the raw numbers of that run, drift or not.

## The compiler as its own test corpus

`python scripts/dogfood.py` migrates `src/ppy_compiler`, `src/ppy_runtime`,
and `src/ppy` with `ppy migrate --dry-run` and holds three lines: no
traceback, no `<unknown>` in any message, and no more errors than
`scripts/dogfood.json` records for each. The count only comes down --
`--write` records a lower one -- so a change that makes the converter worse
at real code fails CI, and one that makes it better is asked to say so.
It is not part of `check.sh`, because a minute of migration on every local
run is too much; CI runs it as its own job on every push.

## Measurements

Numbers in the documentation come from `examples/15_algorithms/bench.py`,
are recorded in `measurements.json` with the machine they were taken on, and
the README tables are rendered from that file rather than typed. They are
not a per-change gate: a scheduled workflow re-measures and reports drift
beyond a tolerance, because absolute wall times differ between runners.

What counts as drift needs both signals: the milliseconds moved beyond the
tolerance *and* the ratio to the C reference moved with them. A busy machine
slows every path at once, so the ratio holds where the times do not; and the
reference is a few milliseconds on the smaller problems, so a wobble there
moves every ratio at once. Either on its own reports the machine. `ppy run`
is exempt from the ratio entirely — it is mostly the compiler, and there is
no ratio to take against a C program that compiled beforehand — so its
movement is reported and never fatal.

A machine that cannot build every path fails the run and records nothing. A
record with a column missing would replace a whole one, and the gap would
read as a result rather than as a machine without `gcc` or without a shared
libpython; `bench.py` says up front which paths it had to skip and why, and
`--record` still writes what it measured so a failed scheduled run keeps its
evidence. That file may not be `measurements.json` itself: the baseline is
written only once the run is judged worth keeping.

## Releasing

The distribution is **`ppy-lang`**; the packages it installs are `ppy`,
`ppy_compiler`, and `ppy_runtime`.

`COMPILER_VERSION` in `src/ppy_compiler/version.py` is the version. The
packaging metadata reads it (`[tool.hatch.version]`), so `pyproject.toml`
does not repeat it. `ppy.__version__` is a second literal, deliberately: the
runtime package does not import the compiler, and giving it one just to
learn a string would be a dependency in the wrong direction. A test holds
the two together, along with the installed distribution's metadata, because
the compiler keys its caches on that string and a stale copy would serve
artifacts from a version that is not running.

To cut a release:

1. Move `COMPILER_VERSION` and `ppy.__version__` together, and write the
   release into `CHANGELOG.md`.
2. `./scripts/check.sh`, then `uv build` and
   `uv run --with twine twine check dist/*`.
3. Tag it `vX.Y.Z` — matching the declared version, which the workflow
   verifies — and push the tag. `.github/workflows/release.yml` runs the
   gate, builds, installs the built wheel into a clean environment and runs
   it, and publishes.

A workflow pins actions by **ref**, and a ref is not a release. Some
publishers cut releases past the last moving major tag they maintain, so
`gh api repos/OWNER/REPO/releases/latest` can name a version that
`uses:` cannot resolve. Check the ref itself before changing one:

```bash
gh api repos/astral-sh/setup-uv/git/ref/tags/v10.0.1 --jq .ref
```

The workflow publishes through PyPI's trusted publishing, so there is no API
token in the repository. It needs, once: a pending publisher on PyPI for
`ppy-lang` naming this repository, the workflow `release.yml`, and the
environment `pypi`; the same on TestPyPI with the environment `testpypi`;
and both environments created under the repository's settings.
`workflow_dispatch` on the workflow publishes to TestPyPI by default, which
is the way to rehearse one.
