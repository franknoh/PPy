# Contributing

## Setting up

```bash
uv sync            # the compiler core, LLVM, NumPy, pydantic, the linters, the docs tools
./scripts/check.sh # the one gate; CI runs exactly this
```

`uv sync` installs the `dev` group. That is what a contributor to the
compiler needs and more than a user needs: a user installs `ppy-lang` and gets
the compiler, the runtime, and nothing else.

The plugin runtimes are separate groups, so you install only what you intend
to test:

```bash
uv sync --group torch     # PyTorch, CPU wheels
uv sync --group jax       # JAX and Flax, CPU wheels
uv sync --group uvicorn   # FastAPI and Uvicorn
uv sync --group scipy     # SciPy
uv sync --group pandas    # pandas
uv sync --group pyarrow   # PyArrow
uv sync --group all       # everything
```

Torch and JAX resolve from the CPU index, which works on every platform.
On a CUDA machine, override it for your checkout rather than for the
repository:

```bash
uv sync --group torch --index pytorch-cuda=https://download.pytorch.org/whl/cu128
```

## Branches

Two long-lived branches, and short-lived ones for work:

| branch | holds | declares | publishes |
|---|---|---|---|
| `main` | the current stable release, and fixes to it | `X.Y.Z` | the docs site's `latest` follows it; a tag `vX.Y.Z` publishes to PyPI |
| `dev` | everything in progress toward the next release | `X.Y.0aN` | the docs site's `dev` follows it; a tag `vX.Y.0aN` publishes an alpha to PyPI |
| anything else | one change | — | nothing; deleted when merged |

Work branches off `dev` and merges back into it by pull request; name it for
the change (`columnar-join`, `fix-e1306-protocol-bound`), not for yourself.
`main` receives `dev` when a release is cut, or a fix that has to ship
before the next one, and nothing else. `main` is protected: it takes pull
requests whose checks passed, never a direct push, never a force-push, and it
cannot be deleted. When a branch has merged, delete it — the repository keeps
`main`, `dev`, `gh-pages` (the built site, written by the docs workflow), and
whatever is being worked on right now.

## Commits

One line, and it says what changed as a sentence someone else could read
without the diff:

```
the documentation as a site: MkDocs from docs/, an example gallery and API pages generated at build time
call an nn.Module through forward, know its members, and say when the boundary is ctypes
pin libcst below 1.8 where a glibc-2.17 wheel exists, and say which libc a machine has
```

Lower case, no trailing period, no prefix tags (`feat:`, `fix:`), no body,
no trailers. The reasoning goes in the pull request, the code comments, and
the changelog, each of which outlives a commit message in `git log`. A
branch may carry several commits while it is open; each is still one line.

Every commit that changes behavior gets a line under the unreleased section
of `CHANGELOG.md` in the same pull request, in the order the work landed,
written for a user: what they can do now, or what stopped happening to them.

## Pull requests

A pull request is one change, into `dev` (or into `main` for a release or a
fix that cannot wait). Before opening it:

```bash
./scripts/check.sh                          # the gate, green
python scripts/refresh.py --quick           # examples and tables in step
python examples/record_outputs.py <folder>  # if an example's output moved
```

The description says what changed and why in a few paragraphs, in the same
register as the commit lines; a reader six months from now should be able to
reconstruct the decision from it. Link the issue it closes with `Closes #N`.
CI runs the gate on Python 3.12, 3.13, and 3.14, the plugin suites (each
fails if all its tests skipped), the dogfood migration, and a built
artifact; all of them have to pass. A pull request is merged with a merge
commit whose subject is the pull request's title, and its branch is deleted.

Review is for the invariants below and for whether the change says what it
does — in its diagnostics, its docs, and its tests. Style is the gate's
business, not the reviewer's.

## Issues

Open one for a bug, a program the compiler answers differently on one of
the three paths, a diagnostic that is wrong or unhelpful, or a feature with
a program that should work and does not. Include:

- the `.ppy` (or `.py`) that shows it, as small as it can be made;
- the command, and what it printed against what you expected — the three
  paths' outputs when they disagree;
- `ppy doctor` and `ppy --version`.

A disagreement between `python f.ppy`, `ppy f.ppy`, and `ppy run f.ppy` is
always a bug, whatever the program does. A performance report needs the
numbers and the machine (`examples/15_algorithms/bench.py --json` records
both). Questions about whether something is meant to work are issues too;
the answer becomes documentation.

## The gate

`./scripts/check.sh` is the single source of truth for "clean": ruff, ruff
format, pylint, the test suite, the conversion check, the three-path example
run, the example lint, the documentation's code fences, and a strict build
of the documentation site. CI runs that script and nothing else, so a local
pass and a CI pass are the same claim. Run it before every commit.

A plugin's own claim is `./scripts/plugin_check.sh <torch|jax|uvicorn|scipy|pandas|pyarrow>`,
which fails if the plugin's tests all skip — a skipped test proves nothing.

## What a change has to keep true

- **The three paths agree.** Plain CPython, the optimized Python backend, and
  the native backend produce the same answer for the same program.
  `examples/run_all.py` checks every example on all three.
- **A guard that fails falls back.** Native code that cannot keep a promise
  runs the Python body; it never answers differently.
- **The cache is disposable.** Corrupting or deleting any part of it may cost
  a rebuild and must never cost the answer. See
  [docs/reference/compatibility.md](docs/reference/compatibility.md).
- **`ppy_runtime` never imports `ppy_compiler`.** A built artifact keeps
  working with the compiler uninstalled, and a test enforces it.
- **A generated `.ppy` is exactly what `ppy convert` writes.** Never hand-edit
  one; `examples/verify_conversions.py` regenerates and diffs them.
- **Every diagnostic code is documented once.** `docs/reference/diagnostics.md`
  and the registry are one claim, and a test holds them together.

## Examples

An example is either hand-written or generated, and its README says which.
A generated one keeps its `.py` source beside it and is regenerated with
`ppy convert <name>.py` (some folders add `--promote-buffers`; see
`examples/verify_conversions.py`).

Every README ends with `## Run it`, the commands, and `## What it prints`,
their output. The second is written by `python examples/record_outputs.py`
(one folder: `record_outputs.py 40_`), which runs each command in the
folder and puts what it printed back into the README, so a command and its
answer sit side by side and a change to either is a change to the file.
Run it with the plugin groups installed (`uv sync --group all`) so the torch,
JAX, and pandas examples record rather than fail; a tool the repository does
not install (`accelerate`) is shown as not run.

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
The exception was 0.2.0, which re-recorded the ceilings upward once: the
corpus roughly doubled, and the count is a property of the corpus as much
as of the converter. Ceilings are recorded on Python 3.13, the version the
job runs, because counts can differ between versions.
It is not part of `check.sh`, because a minute of migration on every local
run is too much; CI runs it as its own job on every push.

## Measurements

Numbers in the documentation come from `examples/15_algorithms/bench.py`,
are recorded in `measurements.json` with the machine they were taken on, and
the README tables are rendered from that file rather than typed. The C
reference is built with gcc and with clang, and both columns are shown;
the drift check takes its ratio against gcc. They are not a per-change
gate: a scheduled workflow re-measures and reports drift beyond a
tolerance, because absolute wall times differ between runners.

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

Measure from a checkout on a native filesystem. A launcher built from a
project on a mounted Windows drive bakes that path into `sys.path`, and
every import-bound number roughly quadruples; the numbers in the tree were
taken from a worktree under `/tmp`, one session, nothing else running.

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

Versions follow the release model above: `dev` declares the *next* release
with an alpha suffix (`0.3.0a1`, then `a2` as alphas ship), `main` declares
a stable `X.Y.Z`. A build from `dev` never carries a stable number it has
not earned, and `pip` and `uv` never pick an alpha unless asked
(`--prerelease allow`, or an exact pin).

**An alpha**, from `dev`:

1. Write what shipped into the unreleased section of `CHANGELOG.md` if it is
   not there yet; leave the heading `— unreleased`.
2. `./scripts/check.sh`, then tag the commit `vX.Y.0aN` — matching the
   declared version, which the workflow verifies — and push the tag.
3. Move `COMPILER_VERSION` and `ppy.__version__` to `aN+1` in the next pull
   request.

**A stable release**, onto `main`:

1. On `dev`, move `COMPILER_VERSION` and `ppy.__version__` to `X.Y.0` and
   date the changelog section (`## X.Y.0 — YYYY-MM-DD`), in a pull request.
2. Merge `dev` into `main` by pull request.
3. `./scripts/check.sh`, then `uv build` and
   `uv run --with twine twine check dist/*`.
4. Tag `main` `vX.Y.0` and push the tag. `.github/workflows/release.yml`
   runs the gate, builds, installs the built wheel into a clean environment
   and runs it, and publishes to PyPI; the docs workflow deploys the release
   as `X.Y` and moves `latest` onto it.
5. Back on `dev`, move the version to `X.(Y+1).0a1` and open the next
   changelog section.

`workflow_dispatch` on the release workflow publishes to TestPyPI by
default, which is the way to rehearse either kind. The workflow publishes
through PyPI's trusted publishing, so there is no API token in the
repository; it needs, once, a pending publisher on PyPI for `ppy-lang`
naming this repository, the workflow `release.yml`, and the environment
`pypi`, the same on TestPyPI with the environment `testpypi`, and both
environments under the repository's settings.

A workflow pins actions by **ref**, and a ref is not a release. Some
publishers cut releases past the last moving major tag they maintain, so
`gh api repos/OWNER/REPO/releases/latest` can name a version that
`uses:` cannot resolve. Check the ref itself before changing one:

```bash
gh api repos/astral-sh/setup-uv/git/ref/tags/v10.0.1 --jq .ref
```
