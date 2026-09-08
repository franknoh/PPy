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
uv run python    collatz.ppy           # 1  plain CPython, no compiler   1047.8 ±  3.4 ms
uv run ppy       collatz.ppy           # 2  optimized Python backend     1055.2 ± 17.0 ms
uv run ppy run   collatz.ppy           # 3  LLVM, built on the first run   39.3 ±  0.4 ms  (+ ~640 ms, once)
uv run ppy build collatz.ppy -o dist   # 4  LLVM, built once (~640 ms)...
./dist/collatz                         #    ...then the native binary      29.5 ±  0.4 ms
                                       #    ...--host-cpu, not portable    27.5 ±  0.5 ms

gcc -O3 collatz.c && ./a.out           # reference: the same loop in C     40.4 ±  0.6 ms  (+ ~90 ms gcc)
```

Numbers are the kernel's wall time on one machine — mean ± standard
deviation over ten runs, each a fresh process; the parenthesized figure is
what that row spends turning source into machine code, and how often.
Ways 3 and 4 are one compilation path, ahead of time or not — the binary is
`ppy run` in a compiled coat, machine code taken from the library built next
to it, and the first `ppy run` is that build into the cache, every later
one the launcher alone. They differ in one default: `ppy run` keeps
Python-integer semantics (overflow is guarded and falls back to arbitrary
precision — that is the 39.3 ms, level with C with the guards in), while
`ppy build` produces a wrap-semantics artifact like every native compiler —
that is the 29.5 ms, past C. `run --unsafe` and `build --safe` flip either
one; bounds checks stay in both. The built binary is compiled software: it starts an
embedded interpreter and imports `ppy_runtime` — about 35 ms before the
program begins — and keeps working with the compiler uninstalled.
`ppy build --standalone` removes even that, for a program whose reachable
graph is entirely native. (Python's floor semantics help too:
`n // 2` lowers to one arithmetic shift exactly, where C's truncating
division needs a sign fixup.)

The same kernel through the neighbors, same machine and methodology:

| compiler | kernel | integer semantics |
|---|---:|---|
| **PPY** `ppy build --host-cpu` | **27.5 ± 0.5 ms** | 64-bit, wraps on overflow (this machine's instruction set) |
| **PPY** `ppy build` binary | **29.5 ± 0.4 ms** | 64-bit, wraps on overflow (`--safe` to keep Python ints) |
| Numba `@njit` | 31.4 ± 0.7 ms | 64-bit, wraps on overflow |
| **PPY** `ppy run` | **39.3 ± 0.4 ms** | **Python ints: guarded, falls back to arbitrary precision** |
| C (`gcc -O3`) | 40.4 ± 0.6 ms | 64-bit, wraps on overflow |
| Codon `-release` | 44.5 ± 0.9 ms | 64-bit, wraps on overflow |
| PyPy 3.11 | 52.5 ± 1.7 ms | Python ints |
| Cython (`cdef long long`) | 61.7 ± 0.9 ms | 64-bit, wraps on overflow |
| mypyc | 69.5 ± 1.2 ms | Python ints |
| Nuitka | 700.9 ± 4.0 ms | Python ints, no type specialization |
| CPython 3.14 | 1047.8 ± 3.4 ms | Python ints |

`--host-cpu` is the opt-in that compiles for the machine doing the build
instead of the portable baseline — 29.5 ms to 27.5 ms here, and about a
third on a matmul kernel where the vectorizer has something to work with. It is off
by default because an artifact is meant to be shipped and host code faults
on an older CPU; JIT code under `ppy run` always targets the host, which is
free because it never leaves the machine. Giving C the same option changes
nothing on this kernel (`gcc -O3 -march=native`: 40.2 ± 1.0 ms), so the rows
above it are not winning on a flag C was denied.

Same compiler, `run` and `build`: the 10 ms between them is the price of
Python's integers, and it is a per-command default rather than a language decision —
wrap semantics where a native artifact is expected, full Python semantics
where a Python program is. Ports are the straightforward one for each tool:
`@njit`, a `cdef long long` `.pyx`, an annotated module for mypyc, `codon
build -release`, `nuitka --module`; Numba 0.67 (warm), Cython 3.3, mypy
2.3.1, Nuitka 4.2 on CPython 3.13, Codon 0.19.6, PyPy 3.11.15 (warm), gcc
13.3, CPython 3.14.5.

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

## The compiler platform

Between the analysis and every backend sits one typed canonical IR
([docs/ir.md](docs/ir.md)): SSA values, blocks, an explicit control-flow
graph, and operations named in dialects — `core`, `math`, `simd`, `cpu`,
`atomic`, `concurrency`, `parallel`, `tensor`, `linalg`, `fft`, `sparse`,
`columnar`, `arrow`, `gpu`, `async`, `prof` — with a verifier, a printer and
parser (`.ppyir` is public text), a pass manager, and rewrite patterns.
Every backend reads that IR and nothing else: LLVM (JIT and object code, for
the host or a `--target`), C11 and C++17 source, CUDA and HIP source, NVVM
IR and PTX for a device, StableHLO for XLA. What a program can ask for
grew with it — `ppy.simd`, `ppy.cpu`, `ppy.atomic`, `ppy.concurrent`,
`ppy.cuda` and `ppy.hip` kernels, `ppy.xla.jit`, `ppy.aio` coroutines with
a native runtime, `ppy.autodiff` — and every one of them is inert under
plain CPython. A package builds as one program: modules link at the IR,
whole-program optimization inlines across them, and one object comes out.
For the developer there are sanitizers (`--sanitize`), the stage debugger
(`ppy inspect --stage`), the optimization report (`--report-opt`), and
profile-guided optimization (`ppy run --profile`, `ppy build --pgo`).
Plugins register dialects, passes, patterns, and lowerings through the same
APIs the builtin ones use.

## Docs

- [docs/guide.md](docs/guide.md) — overview, measurements, the import hook
- [docs/language.md](docs/language.md) — the subset, directives, markers
- [docs/conversion.md](docs/conversion.md) — how `ppy convert` and `ppy migrate` infer what they write
- [docs/migrating.md](docs/migrating.md) — migrating a real project: profile, carve the kernels, leave the rest
- [docs/architecture.md](docs/architecture.md) — pipeline, cache, threads
- [docs/ir.md](docs/ir.md) — the canonical IR: dialects, passes, `.ppyir`, the linker, sanitizers, profiles
- [docs/plugins.md](docs/plugins.md) — how each library integration works
- [docs/config.md](docs/config.md) — every `[tool.ppy]` key
- [docs/diagnostics.md](docs/diagnostics.md) — every diagnostic code
- [docs/cli.md](docs/cli.md) — every command and option
- [docs/compatibility.md](docs/compatibility.md) — what is stable, what moves, and what the cache and artifact ABI promise
- [docs/solver.md](docs/solver.md) — where an SMT solver would fit: proving overflow guards away, validating the optimizer
- [examples/README.md](examples/README.md) — 42 folders, 52 runnable programs
- [CONTRIBUTING.md](CONTRIBUTING.md) — setting up, the gate, and what a change has to keep true

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
