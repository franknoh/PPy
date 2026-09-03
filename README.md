# PPY — Pretty Python

[![PyPI](https://img.shields.io/pypi/v/ppy-lang?logo=pypi&logoColor=white)](https://pypi.org/project/ppy-lang/)
[![Python](https://img.shields.io/pypi/pyversions/ppy-lang?logo=python&logoColor=white)](https://pypi.org/project/ppy-lang/)
[![CI](https://github.com/franknoh/PPy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/franknoh/PPy/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A statically analyzable language in Python's syntax. A `.ppy` file *is* valid
Python: it runs under plain CPython with no compiler involved. The compiler
adds static checking, an optimized Python backend, and an LLVM native
backend — and all three must produce the same answer.

PPY is source-compatible with Python's syntax and ecosystem, not with every
dynamic Python behavior: `exec`/`eval`, monkey-patching, dynamic namespace
mutation, and unrestricted runtime reflection are deliberately restricted —
or isolated behind an explicit `ppy.dynamic` boundary — in exchange for
analysis, optimization, and native compilation that can be trusted. Running
existing Python is a migration feature (`ppy migrate`), not the definition of
the language.

## Install

Add PPY to your project with [uv](https://docs.astral.sh/uv/) (Python 3.12+):

```bash
uv add "ppy-lang[llvm]"
```

or with pip: `pip install "ppy-lang[llvm]"`. The distribution is `ppy-lang`
and what it installs is `ppy`, so your code writes `import ppy`.

Releases are alphas — the language and the diagnostics are in use and
tested, and neither is promised to stay put — so pin an exact version:
`uv add "ppy-lang[llvm]==0.1.0a1"`. For the development tip instead:

```bash
uv add "ppy-lang[llvm] @ git+https://github.com/franknoh/PPy.git"
```

The base package is the compiler and the runtime; extras enable the rest, so
you install only what you use:

| extra | enables |
|---|---|
| `llvm` | the native backend (llvmlite) |
| `numpy` / `pydantic` / `uvicorn` | the matching plugin |
| `torch` / `jax` | the matching plugin, CPU builds by default — point at a CUDA index in your own project if you want one |

Everything degrades cleanly: a missing library only disables its plugin, and
`uv run ppy doctor` reports what was found. For the fastest native call
boundary, have the CPython headers installed (`python3-dev`); without them
PPY says so once and uses the slower boundary.

## One file, four ways

```python
# collatz.ppy
import ppy


@ppy.pure
@ppy.opt(3)
def longest(limit: int) -> int:
    best: int = 0
    for start in range(1, limit):
        n: int = start
        steps: int = 0
        while n != 1:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            steps += 1
        best = max(best, steps)
    return best


print(longest(ppy.input[int]()))
```

`ppy.input[T]()` reads the next value the way `T` says to read it, straight
into memory rather than through a Python object per field. Piping the limit
in, `echo 300000 |` before each of these:

```bash
uv run python    collatz.ppy           # 1  plain CPython, no compiler   1170.2 ±  9.1 ms
uv run ppy       collatz.ppy           # 2  optimized Python backend     1167.9 ± 13.1 ms
uv run ppy run   collatz.ppy           # 3  LLVM, JIT every run            45.4 ±  1.3 ms  (+ ~1200 ms JIT)
uv run ppy build collatz.ppy -o dist   # 4  LLVM, built once (~800 ms)...
./dist/collatz                         #    ...then the native binary      33.6 ±  1.1 ms
                                       #    ...--host-cpu, not portable    31.8 ±  0.9 ms

gcc -O3 collatz.c && ./a.out           # reference: the same loop in C     45.0 ±  0.6 ms  (+ ~110 ms gcc)
```

Numbers are the kernel's wall time on one machine — mean ± standard
deviation over ten runs, each a fresh process; the parenthesized figure is
what that row spends turning source into machine code, and how often.
Ways 3 and 4 are one compilation path, ahead of time or not — the binary is
`ppy run` in a compiled coat, machine code taken from the library built next
to it. They differ in one default: `ppy run` keeps Python-integer semantics
(overflow is guarded and falls back to arbitrary precision — that is the
45.4 ms, level with C with the guards in), while `ppy build` produces a
wrap-semantics artifact like every native compiler — that is the 33.6 ms,
past C. `run --unsafe` and `build --safe` flip either one; bounds
checks stay in both. The built binary is compiled software: it starts an
embedded interpreter and imports `ppy_runtime` — about 35 ms before the
program begins — and keeps working with the compiler uninstalled.
`ppy build --standalone` removes even that, for a program whose reachable
graph is entirely native. (Python's floor semantics help too:
`n // 2` lowers to one arithmetic shift exactly, where C's truncating
division needs a sign fixup.)

The same kernel through the neighbors, same machine and methodology:

| compiler | kernel | integer semantics |
|---|---:|---|
| **PPY** `ppy build --host-cpu` | **31.8 ± 0.9 ms** | 64-bit, wraps on overflow (this machine's instruction set) |
| **PPY** `ppy build` binary | **33.6 ± 1.1 ms** | 64-bit, wraps on overflow (`--safe` to keep Python ints) |
| Codon `-release` | 35.9 ± 1.9 ms | 64-bit, wraps on overflow |
| Numba `@njit` | 36.5 ± 1.0 ms | 64-bit, wraps on overflow |
| C (`gcc -O3`) | 45.0 ± 0.6 ms | 64-bit, wraps on overflow |
| **PPY** `ppy run` | **45.4 ± 1.3 ms** | **Python ints: guarded, falls back to arbitrary precision** |
| PyPy 3.11 | 56.9 ± 2.5 ms | Python ints |
| mypyc | 78.4 ± 1.4 ms | Python ints |
| Cython (`cdef long long`) | 81.4 ± 3.0 ms | 64-bit, wraps on overflow |
| Nuitka | 776.9 ± 11.6 ms | Python ints, no type specialization |
| CPython 3.14 | 1170.2 ± 9.1 ms | Python ints |

`--host-cpu` is the opt-in that compiles for the machine doing the build
instead of the portable baseline — 33.6 ms to 31.8 ms here, and about 20% on
a matmul kernel where the vectorizer has something to work with. It is off
by default because an artifact is meant to be shipped and host code faults
on an older CPU; JIT code under `ppy run` always targets the host, which is
free because it never leaves the machine. Giving C the same option changes
nothing on this kernel (`gcc -O3 -march=native`: 45.5 ± 1.2 ms), so the row
above it is not winning on a flag C was denied.

Same compiler, `run` and `build`: the 12 ms between them is the price of
Python's integers, and it is a per-command default rather than a language decision —
wrap semantics where a native artifact is expected, full Python semantics
where a Python program is. Ports are the straightforward one for each tool:
`@njit`, a `cdef long long` `.pyx`, an annotated module for mypyc, `codon
build -release`, `nuitka --module`; Numba 0.67, Cython 3.3, mypy 2.3.1,
Nuitka on CPython 3.13, Codon 0.19.6, PyPy 3.11.15 (warm).

## Turning ordinary Python into it

```bash
uv run ppy convert src/ --in-place    # strict: the output must be valid strict PPY
uv run ppy migrate src/ --in-place    # permissive: rewrite toward it, report the rest
```

`convert` is the compiler-facing command: it refuses to produce anything
`ppy check` would reject, and says why. `migrate` is for normal existing
Python — it first rewrites dynamic-but-static patterns (`setattr` with a
constant name, `globals()["X"] = ...`, constant `importlib.import_module`),
then staticizes, and classifies whatever remains (`--report migration.json`,
`--diff`).

Given untyped Python:

```python
import math

LIMIT = 3.0


def clamp(value):
    return min(value, LIMIT)


def spread(samples):
    total = 0.0
    for sample in samples:
        total += clamp(sample)
    return math.sqrt(total / len(samples))


print(spread([1.0, 2.0, 9.0]), spread((4.0, 5.0)))
```

`ppy convert` writes:

```python
import math
from collections.abc import Sequence
from typing import Final

import ppy

LIMIT: Final[float] = 3.0


@ppy.pure
def clamp(value: float) -> float:
    return min(value, LIMIT)


@ppy.pure
def spread(samples: Sequence[float]) -> float:
    total: float = 0.0
    for sample in samples:
        total += clamp(sample)
    return math.sqrt(total / len(samples))


print(spread([1.0, 2.0, 9.0]), spread((4.0, 5.0)))
```

Types come from the whole call graph, not one file. `samples` is `Sequence`
rather than `list` because the body only reads it and one call site passes a
tuple; `LIMIT` is `Final` because nothing rebinds it; `@ppy.pure` is attached
only where the checker proved it. Nothing is renamed and no function is split —
those are design decisions, not mechanical ones.

## Works with

Each library is a plugin: the compiler learns that library's types and effects,
takes a faster path where it can prove one is equivalent, and falls back to the
ordinary Python call everywhere else. A guard that fails is a fallback, never a
different answer.

**NumPy** — elementwise expressions fuse into a single loop with no
temporaries; `dot`, `matmul`, `inner`, `vdot`, and `tensordot` route to the
linear-algebra path. Contiguity and shape are guarded at runtime, not assumed.
Reduction order is preserved unless `@ppy.fastmath` permits reassociation, so a
sum stays bit-identical to NumPy's.

**PyTorch** — a function whose body is entirely curated tensor operations (55 of
them) compiles into one C++ region calling ATen directly, removing a Python
round trip per operator. Every call still goes through the dispatcher, so
autograd, device selection, and backend keys are unchanged; a tensor subclass or
a `__torch_function__` override trips the guard and the Python body runs. CUDA
is used when it is there.

**JAX / Flax** — a `@jax.jit` function whose inputs carry `ppy.Shape` and
`ppy.DType` can be exported to StableHLO at build time, so the trace is not
repeated at startup; shapes may be symbolic, so one artifact serves every
batch size (export runs project code and is off until the project opts in).
The same plugin models Flax and optax — layers, activations, `Module.init`/
`apply` through the external MRO, optimizers — so a Flax training loop
checks under strict mode as-is.

**Pydantic** — models are typed, constructor and output shapes are kept
distinct, and field constraints become refinements the checker can use.

**Uvicorn / FastAPI** — the ASGI application is resolved statically instead of
re-imported by module string per worker, and the reloader is told to watch
`.ppy`. FastAPI rides the same plugin: its surface is modeled so strict mode
checks a FastAPI service as-is, while route handlers keep the exact
signatures FastAPI reads at import.

The plugin's exact version is part of every cache key, so an artifact built
against one build of a library is never reused against another. `ppy doctor`
prints what it found.

## Docs

- [docs/guide.md](docs/guide.md) — overview, measurements, the import hook
- [docs/language.md](docs/language.md) — the subset, directives, markers
- [docs/conversion.md](docs/conversion.md) — how `ppy convert` and `ppy migrate` infer what they write
- [docs/migrating.md](docs/migrating.md) — migrating a real project: profile, carve the kernels, leave the rest
- [docs/architecture.md](docs/architecture.md) — pipeline, cache, threads
- [docs/plugins.md](docs/plugins.md) — how each library integration works
- [docs/config.md](docs/config.md) — every `[tool.ppy]` key
- [docs/diagnostics.md](docs/diagnostics.md) — every diagnostic code
- [docs/cli.md](docs/cli.md) — every command and option
- [docs/compatibility.md](docs/compatibility.md) — what is stable, what moves, and what the cache and artifact ABI promise
- [examples/README.md](examples/README.md) — 30 folders, 39 runnable programs
- [CONTRIBUTING.md](CONTRIBUTING.md) — setting up, the gate, and what a change has to keep true
- [the implementation spec](ppy-compiler-implementation-spec-v1.md) — the normative baseline the source cites by section number (`spec 11.2`, `spec 16.4`, …)

## Development

```bash
git clone https://github.com/franknoh/PPy.git
cd PPy
uv sync              # the compiler core, LLVM, NumPy, pydantic, and the linters
./scripts/check.sh   # the one gate; CI runs exactly this
```

Plugin runtimes are separate groups (`uv sync --group torch|jax|uvicorn|all`),
and everything else a contributor needs — the gate, the invariants a change
has to keep, how examples and measurements are kept current — is in
[CONTRIBUTING.md](CONTRIBUTING.md).
