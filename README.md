# PPY — Pretty Python

[![PyPI](https://img.shields.io/pypi/v/ppy-lang?logo=pypi&logoColor=white)](https://pypi.org/project/ppy-lang/)
[![Python](https://img.shields.io/pypi/pyversions/ppy-lang?logo=python&logoColor=white)](https://pypi.org/project/ppy-lang/)
[![CI](https://github.com/franknoh/PPy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/franknoh/PPy/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-ppy.franknoh.dev-indigo)](https://ppy.franknoh.dev/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Python's syntax, compiled.** A `.ppy` file *is* valid Python and runs under
plain CPython with no compiler involved. The compiler adds a strict static
checker, an optimized Python backend, and an LLVM native backend — and the
three must print the same answer, or it is a bug.

```bash
uv add "ppy-lang[llvm]"        # or: pip install "ppy-lang[llvm]"
```

Pin an exact version before 1.0 (`ppy-lang[llvm]==0.2.0`): a minor release
may move the language, and the [changelog](CHANGELOG.md) says what moved.
Extras enable the rest — `solver`, `numpy`, `torch`, `jax`, `pydantic`,
`uvicorn`, `scipy`, `pandas`, `pyarrow`, `bind` — and a missing library only
disables its plugin. `uv run ppy doctor` reports what it found; the details
are in [installing](https://ppy.franknoh.dev/latest/installing/).

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

`echo 300000 |` before each of these:

```bash
uv run python    collatz.ppy           # 1  plain CPython, no compiler   1068.8 ±  5.5 ms
uv run ppy       collatz.ppy           # 2  optimized Python backend     1074.3 ± 10.3 ms
uv run ppy run   collatz.ppy           # 3  LLVM, built on the first run   41.7 ±  1.8 ms  (+ ~640 ms, once)
uv run ppy build collatz.ppy -o dist   # 4  LLVM, built once (~640 ms)...
./dist/collatz                         #    ...then the native binary      30.8 ±  0.4 ms
                                       #    ...--host-cpu, not portable    28.8 ±  0.5 ms

gcc   -O3 collatz.c && ./a.out         # the same loop in C                41.9 ±  0.9 ms
clang -O3 collatz.c && ./a.out         #                                   32.8 ±  0.5 ms
```

Kernel wall time on one machine, ten fresh processes each, mean ± standard
deviation. Ways 3 and 4 are one compiler: the binary is `ppy run` in a
compiled coat, and the first `ppy run` is the build into the cache. They
differ in one default. `ppy run` keeps **Python's integers** — overflow is
guarded and falls back to arbitrary precision — and lands on gcc; `ppy build`
is a wrap-semantics artifact like every native compiler and lands on clang.
`run --unsafe` and `build --safe` flip either; bounds checks stay in both.

The same kernel through the neighbours, same machine and methodology:

| compiler | kernel | integer semantics |
|---|---:|---|
| **PPY** `ppy build --host-cpu` | **28.8 ± 0.5 ms** | 64-bit, wraps on overflow (this machine's instruction set) |
| C (`clang -O3 -march=native`) | 30.8 ± 0.7 ms | 64-bit, wraps on overflow |
| **PPY** `ppy build` | **30.8 ± 0.4 ms** | 64-bit, wraps on overflow (`--safe` to keep Python ints) |
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

Types come from the whole call graph, not one file: `samples` is `Sequence`
because the body only reads it and one call site passes a tuple, `LIMIT` is
`Final` because nothing rebinds it, and `@ppy.pure` is attached only where the
checker proved it. Nothing is renamed and no function is split.

## By the numbers

| | |
|---|---|
| **3** execution paths that must agree — plain CPython, the Python backend, LLVM — checked on every example by `examples/run_all.py` | **1,110** tests on Python 3.12, 3.13, and 3.14 covering **74%** of the compiler's 36k statements, plus differential fuzzing of generated programs |
| **42** example folders, **52** programs, every `.ppy` next to a `.py` regenerated by `verify_conversions.py` to prove it is what `ppy convert` wrote | **76** diagnostic codes, each documented once, each with a stable meaning |
| **18** IR dialects — `core`, `math`, `simd`, `cpu`, `atomic`, `concurrency`, `parallel`, `tensor`, `linalg`, `fft`, `sparse`, `columnar`, `arrow`, `gpu`, `async`, `prof`, … | **7** backends off one IR: LLVM (JIT and objects, host or cross), C11, C++17, CUDA, HIP, NVVM/PTX, StableHLO |
| **8** library plugins — NumPy, PyTorch, JAX/Flax, pydantic, FastAPI/Uvicorn, SciPy, pandas, PyArrow — each a faster path where equivalence is proven and the library's own call everywhere else | **0** Python frames on a native call: **47 ns** for a two-`int` call against **28 ns** for a plain Python call, **65 ns** with a borrowed buffer, **86 ns** when a guard fails and the Python body runs |

The whole native surface — `ppy.native` memory and FFI, `ppy.simd`,
`ppy.cpu`, `ppy.atomic`, `ppy.concurrent`, `ppy.parallel.range`,
`ppy.grad`, `ppy.aio` coroutines, `ppy.cuda`/`ppy.hip` kernels,
`ppy.xla.jit` — has a Python reference implementation, so a program using it
is still a Python program. What is not native is untouched: a PyTorch
training loop is `.backward()` and the optimizer, and stays exactly as fast.

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
