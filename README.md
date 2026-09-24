# PPy

[![PyPI](https://img.shields.io/pypi/v/ppy-lang?logo=pypi&logoColor=white)](https://pypi.org/project/ppy-lang/)
[![Python](https://img.shields.io/pypi/pyversions/ppy-lang?logo=python&logoColor=white)](https://pypi.org/project/ppy-lang/)
[![CI](https://github.com/franknoh/PPy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/franknoh/PPy/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-ppy.franknoh.dev-0d7560)](https://ppy.franknoh.dev/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

PPy is a compiler for the statically analyzable part of Python. A `.ppy`
file is valid Python: it runs under plain CPython with no compiler
installed, and that run is the reference answer. The compiler reads the same
file, proves what it can about it, lowers it through a canonical IR, and
hands the result to a backend: native code through LLVM, C or C++ source,
CUDA, PTX, StableHLO, or a backend from another package.

Code the compiler cannot prove stays Python. `ppy check` says where a
program leaves the supported subset, and at runtime a failed guard falls
back to the Python body. An overflowing multiply returns to CPython and
finishes in arbitrary precision. A library call PPy has no equivalence
argument for runs as the library's own call.

Documentation: **[ppy.franknoh.dev](https://ppy.franknoh.dev/)**

## Highlights

- Plain CPython, the optimized Python backend, and the LLVM backend must
  print the same output. `examples/run_all.py` checks this for every example.
- 7 code generation targets off one IR: LLVM (JIT and objects, host or
  cross), C11, C++17, CUDA, HIP, NVVM/PTX, and StableHLO, plus any backend
  a package installs.
- 8 library plugins: NumPy, PyTorch, JAX/Flax, pydantic, FastAPI/Uvicorn,
  SciPy, pandas, and PyArrow.
- A native call with two `int` arguments costs 47 ns, against 28 ns for a
  plain Python call. It is 65 ns with a borrowed buffer, and 86 ns when a
  guard fails and the Python body runs. No Python frames are on the native
  path.
- `ppy convert` and `ppy migrate` bring existing Python over one module at a
  time.
- 1,736 tests on Python 3.12, 3.13, and 3.14, covering 74% of the compiler's
  40k statements, plus differential fuzzing of generated programs. 78
  diagnostic codes, each documented once.

## Installation

```bash
uv add "ppy-lang[llvm]"        # or: pip install "ppy-lang[llvm]"
uv run ppy doctor              # reports what it found
```

Pin an exact version before 1.0 (`ppy-lang[llvm]==0.3.0`). A minor release
may change the language, and the [changelog](CHANGELOG.md) says what
changed.

Extras enable the rest: `solver`, `numpy`, `torch`, `jax`, `pydantic`,
`uvicorn`, `scipy`, `pandas`, `pyarrow`, `bind`. A missing library only
disables its plugin. The details are in
[Installing](https://ppy.franknoh.dev/latest/installing/).

## Example

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

Run it with `echo 300000 |` in front of each command:

```bash
uv run python    collatz.ppy           # 1  plain CPython, no compiler   1068.8 ±  5.5 ms
uv run ppy       collatz.ppy           # 2  optimized Python backend     1074.3 ± 10.3 ms
uv run ppy run   collatz.ppy           # 3  LLVM, built on the first run   41.7 ±  1.8 ms  (+ ~640 ms, once)
uv run ppy build --unsafe collatz.ppy -o dist   # 4  LLVM, built once (~640 ms)...
./dist/collatz                         #    ...then the native binary      30.8 ±  0.4 ms
                                       #    ...--host-cpu, not portable    28.8 ±  0.5 ms

gcc   -O3 collatz.c && ./a.out         # the same loop in C                41.9 ±  0.9 ms
clang -O3 collatz.c && ./a.out         #                                   32.8 ±  0.5 ms
```

The same source can go to other places:

```bash
uv run ppy check   collatz.ppy                  # the static checker alone
uv run ppy build   collatz.ppy --backend python # the optimized Python backend's modules
uv run ppy emit    ir  collatz.ppy              # the canonical IR (.ppyir)
uv run ppy emit    c   collatz.ppy              # one C11 translation unit
uv run ppy build   collatz.ppy --backend toy    # a backend some package installed
```

The first two commands in the timing block need no compiler toolchain. The
rest share the frontend and the IR and differ only in the backend.

## How it works

```text
Python / .ppy source
      ↓   frontend: the CPython grammar, the module graph
static analysis: types, effects, refinements, contracts
      ↓   what cannot be proven stays Python, and is said so
canonical IR (typed SSA, 19 dialects, a public .ppyir on disk)
      ↓
shared optimization + the plugins' semantic lowerings
      ↓   the backend boundary
backend-specific passes, then backend validation
      ↓
LLVM · C11 · C++17 · CUDA · HIP · NVVM/PTX · StableHLO · your backend
```

Every backend consumes the same canonical IR after the same shared passes.
That is why a backend can be a Python package with an entry point instead of
a fork of this repository.

## Performance

The Collatz kernel above, compiled by PPy and by other tools. One machine,
ten fresh processes each, kernel wall time as mean ± standard deviation.

| compiler | kernel | integer semantics |
|---|---:|---|
| **PPy** `ppy build --unsafe --host-cpu` | **28.8 ± 0.5 ms** | 64-bit, wraps on overflow (this machine's instruction set) |
| C (`clang -O3 -march=native`) | 30.8 ± 0.7 ms | 64-bit, wraps on overflow |
| **PPy** `ppy build --unsafe` | **30.8 ± 0.4 ms** | 64-bit, wraps on overflow |
| Numba `@njit` | 32.5 ± 0.9 ms | 64-bit, wraps on overflow |
| C (`clang -O3`) | 32.8 ± 0.5 ms | 64-bit, wraps on overflow |
| **PPy** `ppy run` | **41.7 ± 1.8 ms** | **Python ints: guarded, falls back to arbitrary precision** |
| C (`gcc -O3`) | 41.9 ± 0.9 ms | 64-bit, wraps on overflow |
| Codon `-release` | 42.1 ± 12.1 ms | 64-bit, wraps on overflow |
| PyPy 3.11 | 52.8 ± 0.5 ms | Python ints |
| Cython (`cdef long long`) | 61.6 ± 0.8 ms | 64-bit, wraps on overflow |
| mypyc | 71.6 ± 1.4 ms | Python ints |
| Nuitka | 714.7 ± 12.8 ms | Python ints, no type specialization |
| CPython 3.14 | 1068.8 ± 5.5 ms | Python ints |

Notes on the table:

- `ppy run` and `ppy build` are one compiler: the binary is `ppy run` ahead
  of time, and the first `ppy run` is the build into the cache. Both keep
  **Python's integers** (overflow is guarded and falls back to arbitrary
  precision) and land near gcc.
- `--unsafe` drops the overflow guards for 64-bit wrap semantics like C's,
  which lands near clang. Bounds checks stay either way. The rows with wrap
  semantics were measured with it.
- `--host-cpu` compiles for the machine doing the build instead of the
  portable baseline. It is off by default because an artifact is meant to be
  shipped. JIT code under `ppy run` always targets the host, which costs
  nothing because it never leaves the machine.
- `gcc -O3 -march=native` changes nothing here (42.1 ± 1.3 ms). Codon's
  spread is its own: ten runs between 35 and 60 ms.
- Each port is the straightforward one for its tool: `@njit`, a
  `cdef long long` `.pyx`, an annotated module for mypyc,
  `codon build -release`, `nuitka --module`. Versions: Numba 0.67 (warm),
  Cython 3.3, mypy 2.3.1, Nuitka 4.2 on CPython 3.13, Codon 0.19.6, PyPy
  3.11.15 (warm), gcc 13.3, clang 22.1, CPython 3.14.5.

Eight more kernels and six whole programs against both C compilers are in
[`examples/15_algorithms`](examples/15_algorithms/README.md).

## Why Python syntax

There is no new language to learn. `.ppy` uses Python's grammar, semantics,
modules, and imports, so your editor, debugger, formatter, and test runner
work on it unchanged. `python program.ppy` runs it with no compiler
installed, and the other paths are held to that run. A disagreement is a
compiler bug.

Types are ordinary annotations. The static subset is documented, and `ppy
check` reports where a program leaves it, with a diagnostic code that has
one meaning and one page of documentation.

## Libraries

The eight plugins tell the compiler what NumPy, PyTorch, JAX/Flax, pydantic,
FastAPI/Uvicorn, SciPy, pandas, and PyArrow mean: their types, their
effects, and how an operation lowers. They describe this as a
backend-neutral spec, not as backend code.

A plugin only adds a faster path where the compiler can argue the result is
the same. Everywhere else the library's own call runs at its usual speed. A
PyTorch training loop is still `.backward()` and the optimizer. No plugin
compiles a library's whole surface, and none changes what a library returns.

## Migrating existing code

```bash
uv run ppy convert src/ --in-place    # strict: the output must be valid strict PPy
uv run ppy migrate src/ --in-place    # permissive: rewrite toward it, report the rest
```

`convert` refuses to produce anything `ppy check` would reject, and says
why. `migrate` is for ordinary existing Python. It first rewrites dynamic
patterns that are static in practice (`setattr` with a constant name,
`globals()["X"] = ...`, a constant `importlib.import_module`), then adds
types, and classifies whatever remains (`--report migration.json`,
`--diff`).

A module that converts cleanly gets the compiler. A module that does not
keeps running as Python next to it.

Given this untyped Python:

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

Types come from the whole call graph, not one file. `samples` is a
`Sequence` because the body only reads it and one call site passes a tuple.
`LIMIT` is `Final` because nothing rebinds it. `@ppy.pure` is attached only
where the checker proved it. Nothing is renamed and no function is split.

## Native features

`ppy.native` memory and FFI, `ppy.simd`, `ppy.cpu`, `ppy.atomic`,
`ppy.concurrent`, `ppy.parallel.range`, `ppy.grad`, `ppy.aio` coroutines,
`ppy.cuda` kernels, and `ppy.xla.jit` all have a Python reference
implementation, so a program that uses them is still a Python program.

`ppy.cuda` launches PTX through the CUDA driver where there is one.
`ppy.hip` is source emission only: `ppy emit hip` writes HIP C++ for
`hipcc`, with no launch runtime of its own.

## Extending the compiler

**The canonical IR** is typed SSA with 19 dialects: `core`, `math`, `simd`,
`cpu`, `atomic`, `concurrency`, `parallel`, `layout`, `tensor`, `linalg`,
`fft`, `sparse`, `special`, `columnar`, `arrow`, `gpu`, `async`, `regex`,
`prof`. It has a text form and an on-disk form (`.ppyir`) that records its
schema and dialect versions, so other tools can read it, and `ppy build
foo.ppyir` builds one without the Python that produced it.

**A library plugin** registers types, effects, lowering specs, dialects,
patterns, and passes for the libraries it claims.

**A backend** consumes the canonical IR after the shared passes. Backends
are found through the `ppy.backends` entry-point group, so a backend can
live in its own package with its own release schedule:

```toml
[project.entry-points."ppy.backends"]
toy = "ppy_toy:create_backend"
```

```python
class ToyBackend(Backend):
    name = "toy"
    api_version = 1

    def emit_formats(self):
        return (EmitFormat("toy", ".toy"),)

    def register_passes(self, passes):
        passes.add(CountOps)

    def validate(self, module, context):
        """Refuse what this target cannot express, naming what and why."""

    def emit(self, module, format, context):
        """Canonical IR in; text, or bytes for a binary format, out."""
        return render(module)
```

```bash
ppy emit toy program.ppy             # the backend's own format
ppy build program.ppy --backend toy  # the backend's own artifacts
ppy doctor                           # every backend, its toolchain, its fingerprint
```

A backend declares the interface version it implements, adds passes at the
backend stage, and refuses IR it cannot express instead of emitting it
wrongly. It reads its own `[tool.ppy.backends.<name>]` configuration. Its
distribution, version, and fingerprint are part of the cache key, so
upgrading it invalidates what the old version built.

The backend interface is **experimental** in 0.3. It is documented, tested,
and versioned separately from the compiler, and it may still change. See
[Backends](https://ppy.franknoh.dev/latest/internals/backends/),
[the Backend API](https://ppy.franknoh.dev/latest/api/backends/),
[Plugins](https://ppy.franknoh.dev/latest/internals/plugins/), and
[The IR](https://ppy.franknoh.dev/latest/internals/ir/).

## Documentation

[ppy.franknoh.dev](https://ppy.franknoh.dev/) has the tutorial, the guide,
every example with its output, the CLI, the API, and the internals. It is
built from [`docs/`](docs/) on every push: `latest` is the release and `dev`
is the development branch.

## Development

```bash
git clone https://github.com/franknoh/PPy.git
cd PPy
uv sync              # the compiler core, LLVM, NumPy, pydantic, and the linters
./scripts/check.sh   # the one gate; CI runs exactly this
```

Plugin runtimes are separate dependency groups
(`uv sync --group torch|jax|uvicorn|all`). The rest of what a contributor
needs, including the checks, the invariants a change has to keep, and how
examples and measurements stay current, is in
[CONTRIBUTING.md](CONTRIBUTING.md).
