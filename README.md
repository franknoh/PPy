# PPY — Pretty Python

[![PyPI](https://img.shields.io/pypi/v/ppy-lang?logo=pypi&logoColor=white)](https://pypi.org/project/ppy-lang/)
[![Python](https://img.shields.io/pypi/pyversions/ppy-lang?logo=python&logoColor=white)](https://pypi.org/project/ppy-lang/)
[![CI](https://github.com/franknoh/PPy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/franknoh/PPy/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-ppy.franknoh.dev-indigo)](https://ppy.franknoh.dev/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Python as the frontend. A compiler underneath.**

PPY is a general-purpose compiler whose source language is Python. A `.ppy`
file *is* valid Python: it runs under plain CPython with no compiler
involved, and that run is the reference answer. The compiler reads the same
file, proves what it can about it, lowers that through a canonical IR, and
hands the result to a backend — native code through LLVM, C or C++ source,
CUDA, PTX, StableHLO, or a backend shipped in someone else's package.

What it does not prove, it does not compile. Python is a dynamic language
and PPY does not pretend otherwise: the compiler works on a supported static
subset, says exactly where a program leaves it, and falls back to the Python
body rather than guessing. An overflowing multiply returns to CPython and
finishes in arbitrary precision. A library call PPY has no equivalence
argument for stays the library's own call.

Keep Python. Change the compiler.

```bash
uv add "ppy-lang[llvm]"        # or: pip install "ppy-lang[llvm]"
```

Pin an exact version before 1.0 (`ppy-lang[llvm]==0.3.0`): a minor release
may move the language, and the [changelog](CHANGELOG.md) says what moved.
Extras enable the rest — `solver`, `numpy`, `torch`, `jax`, `pydantic`,
`uvicorn`, `scipy`, `pandas`, `pyarrow`, `bind` — and a missing library only
disables its plugin. `uv run ppy doctor` reports what it found; the details
are in [installing](https://ppy.franknoh.dev/latest/installing/).

## The road through the compiler

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
That is what makes the last item on the list possible: a backend is a Python
package with an entry point, not a fork of this repository.

## One file, several paths

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

`echo 300000 |` before each of these:

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

One source, and the compiler is told where to take it:

```bash
uv run ppy check   collatz.ppy                  # the static checker alone
uv run ppy build   collatz.ppy --backend python # the optimized Python backend's modules
uv run ppy emit    ir  collatz.ppy              # the canonical IR (.ppyir)
uv run ppy emit    c   collatz.ppy              # one C11 translation unit
uv run ppy build   collatz.ppy --backend toy    # a backend some package installed
```

Numbers 1 and 2 need no compiler toolchain at all; the rest are the same
frontend and the same IR under a different backend.

## Why PPY

### Python-native development

There is no PPY-like language to learn. `.ppy` is Python's grammar, Python's
semantics, Python's modules and imports. Your editor, your debugger, your
formatter, and your test runner work on it because it is Python. `python
program.ppy` runs it with no compiler installed, and that path is the
reference the other paths are held to: `examples/run_all.py` runs every
example on plain CPython, the Python backend, and the LLVM backend and diffs
the output. A disagreement is a bug in the compiler, not a dialect.

Types are ordinary annotations. The static subset is documented, and `ppy
check` tells you where a program leaves it, with a diagnostic code that has
one meaning and one page of documentation.

### The Python ecosystem

PPY keeps the ecosystem rather than replacing it. Eight library plugins —
NumPy, PyTorch, JAX/Flax, pydantic, FastAPI/Uvicorn, SciPy, pandas, PyArrow
— teach the compiler what those libraries mean: their types, their effects,
and how an operation lowers, as a backend-neutral spec rather than as
backend code.

The rule is equivalence, not enthusiasm. A plugin gives a faster path only
where the compiler can argue the result is the same; everywhere else the
library's own call runs, exactly as fast as it always was. A PyTorch
training loop is still `.backward()` and the optimizer. Nothing in a plugin
makes a library's whole surface compiled, and nothing silently changes what
a library returns.

### Existing code, migrated a piece at a time

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

A module that converts cleanly gets the compiler; a module that does not
keeps running as Python beside it. There is no all-or-nothing switch.

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

Types come from the whole call graph, not one file: `samples` is `Sequence`
because the body only reads it and one call site passes a tuple, `LIMIT` is
`Final` because nothing rebinds it, and `@ppy.pure` is attached only where the
checker proved it. Nothing is renamed and no function is split.

## Performance

Compilation is the point of a compiler, so here is the evidence: the same
Collatz kernel, through PPY and through the neighbours.

Kernel wall time on one machine, ten fresh processes each, mean ± standard
deviation. Ways 3 and 4 are one compiler: the binary is `ppy run` in a
compiled coat, and the first `ppy run` is the build into the cache. They
mean the same program: both keep **Python's integers** — overflow is
guarded and falls back to arbitrary precision — and land on gcc. `--unsafe`,
on either, drops the guards for 64-bit wrap semantics like C's, which lands
on clang; bounds checks stay in both. The wrap-semantics rows below were
measured with it.

The same kernel through the neighbours, same machine and methodology:

| compiler | kernel | integer semantics |
|---|---:|---|
| **PPY** `ppy build --unsafe --host-cpu` | **28.8 ± 0.5 ms** | 64-bit, wraps on overflow (this machine's instruction set) |
| C (`clang -O3 -march=native`) | 30.8 ± 0.7 ms | 64-bit, wraps on overflow |
| **PPY** `ppy build --unsafe` | **30.8 ± 0.4 ms** | 64-bit, wraps on overflow |
| Numba `@njit` | 32.5 ± 0.9 ms | 64-bit, wraps on overflow |
| C (`clang -O3`) | 32.8 ± 0.5 ms | 64-bit, wraps on overflow |
| **PPY** `ppy run` | **41.7 ± 1.8 ms** | **Python ints: guarded, falls back to arbitrary precision** |
| C (`gcc -O3`) | 41.9 ± 0.9 ms | 64-bit, wraps on overflow |
| Codon `-release` | 42.1 ± 12.1 ms | 64-bit, wraps on overflow |
| PyPy 3.11 | 52.8 ± 0.5 ms | Python ints |
| Cython (`cdef long long`) | 61.6 ± 0.8 ms | 64-bit, wraps on overflow |
| mypyc | 71.6 ± 1.4 ms | Python ints |
| Nuitka | 714.7 ± 12.8 ms | Python ints, no type specialization |
| CPython 3.14 | 1068.8 ± 5.5 ms | Python ints |

`--host-cpu` compiles for the machine doing the build instead of the
portable baseline; it is off by default because an artifact is meant to be
shipped. JIT code under `ppy run` always targets the host, which is free
because it never leaves the machine. `gcc -O3 -march=native` changes
nothing here (42.1 ± 1.3 ms), and Codon's spread is its own: ten runs
between 35 and 60 ms. Ports are the straightforward one for each
tool: `@njit`, a `cdef long long` `.pyx`, an annotated module for mypyc,
`codon build -release`, `nuitka --module`; Numba 0.67 (warm), Cython 3.3,
mypy 2.3.1, Nuitka 4.2 on CPython 3.13, Codon 0.19.6, PyPy 3.11.15 (warm),
gcc 13.3, clang 22.1, CPython 3.14.5. Eight more kernels and six whole
programs against both C compilers are in
[`examples/15_algorithms`](examples/15_algorithms/README.md).

## Extending the compiler

PPY is built to be extended from the outside, and the 0.3 line is where that
became a public surface rather than an internal one.

**The canonical IR** is typed SSA with 19 dialects — `core`, `math`, `simd`,
`cpu`, `atomic`, `concurrency`, `parallel`, `layout`, `tensor`, `linalg`,
`fft`, `sparse`, `special`, `columnar`, `arrow`, `gpu`, `async`, `regex`,
`prof`. It has a text form and an on-disk form (`.ppyir`) that carries its
schema and dialect versions, so a tool can read it, and `ppy build
foo.ppyir` builds one without the Python that produced it.

**A library plugin** registers types, effects, lowering specs, dialects,
patterns, and passes for the libraries it claims.

**A backend** consumes the canonical IR after the shared passes and makes
something of it. Backends are found through the `ppy.backends` entry-point
group, so one lives in its own package, on its own release schedule:

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

A backend declares the interface version it implements, hangs passes at the
backend stage, refuses IR it cannot take rather than emitting it wrongly,
reads its own `[tool.ppy.backends.<name>]` configuration, and has its
distribution, version, and fingerprint folded into the cache key, so
upgrading it invalidates what the old one built. The backend interface is
**experimental** in 0.3: it is documented, tested, and versioned
independently of the compiler, and it may still change.

The full surface is in [Backends](https://ppy.franknoh.dev/latest/internals/backends/),
[the Backend API](https://ppy.franknoh.dev/latest/api/backends/),
[Plugins](https://ppy.franknoh.dev/latest/internals/plugins/), and
[The IR](https://ppy.franknoh.dev/latest/internals/ir/).

## By the numbers

| | |
|---|---|
| **3** execution paths that must agree — plain CPython, the Python backend, LLVM — checked on every example by `examples/run_all.py` | **1,502** tests on Python 3.12, 3.13, and 3.14 covering **74%** of the compiler's 40k statements, plus differential fuzzing of generated programs |
| **45** example folders, **55** programs, every `.ppy` next to a `.py` regenerated by `verify_conversions.py` to prove it is what `ppy convert` wrote | **78** diagnostic codes, each documented once, each with a stable meaning |
| **19** IR dialects — `core`, `math`, `simd`, `cpu`, `atomic`, `concurrency`, `parallel`, `layout`, `tensor`, `linalg`, `fft`, `sparse`, `special`, `columnar`, `arrow`, `gpu`, `async`, `regex`, `prof` | **7** code-generation targets off one IR: LLVM (JIT and objects, host or cross), C11, C++17, CUDA, HIP, NVVM/PTX, StableHLO — and any backend a package installs |
| **8** library plugins — NumPy, PyTorch, JAX/Flax, pydantic, FastAPI/Uvicorn, SciPy, pandas, PyArrow — each a faster path where equivalence is proven and the library's own call everywhere else | **0** Python frames on a native call: **47 ns** for a two-`int` call against **28 ns** for a plain Python call, **65 ns** with a borrowed buffer, **86 ns** when a guard fails and the Python body runs |

The whole native surface — `ppy.native` memory and FFI, `ppy.simd`,
`ppy.cpu`, `ppy.atomic`, `ppy.concurrent`, `ppy.parallel.range`,
`ppy.grad`, `ppy.aio` coroutines, `ppy.cuda` kernels, `ppy.xla.jit` — has a
Python reference implementation, so a program using it is still a Python
program. `ppy.cuda` launches PTX through the CUDA driver where there is one;
`ppy.hip` is source emission — `ppy emit hip` writes HIP C++ for `hipcc` —
and has no launch runtime of its own. What is not native is untouched: a
PyTorch training loop is `.backward()` and the optimizer, and stays exactly
as fast.

## Docs

**[ppy.franknoh.dev](https://ppy.franknoh.dev/)** — getting started, the
guide, every example with what it prints, the CLI, the API, and the internals.
Built from [`docs/`](docs/) on every push; `latest` is the release, `dev` the
tip.

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
