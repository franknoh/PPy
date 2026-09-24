---
hide:
  - navigation
  - toc
---

<div class="ppy-home" markdown>

# PPy

<p class="ppy-lead">A compiler for the statically analyzable part of Python.
It reads ordinary Python files and builds them into native code, C, CUDA,
or StableHLO.</p>

A `.ppy` file is valid Python. You can run it with `python` and no compiler
installed, and that run is the reference answer. The compiler reads the same
file, proves what it can about the types and effects, lowers it through a
canonical IR, and hands it to a backend.

```python
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

```bash
echo 300000 | python    collatz.ppy      # plain CPython                 1068.8 ms
echo 300000 | ppy run   collatz.ppy      # LLVM, Python's integers         41.7 ms
echo 300000 | ppy build collatz.ppy -o dist && ./dist/collatz   # native    30.8 ms
```

The same loop in C takes 41.9 ms with `gcc -O3` and 32.8 ms with `clang -O3`.
These are ten fresh processes each on one machine. The
[performance page](reference/performance.md) has the full table, including
Numba, Codon, PyPy, Cython, mypyc, and Nuitka.

## Install

```bash
uv add "ppy-lang[llvm]"      # or: pip install "ppy-lang[llvm]"
uv run ppy doctor            # reports the compiler, LLVM, the C toolchain, plugin runtimes
```

Then follow [Getting started](getting-started.md), which goes from this file
to a native build in about ten minutes.

## Highlights

- **Three paths, one answer.** `python f.ppy` runs plain CPython,
  `ppy f.ppy` runs the optimized Python backend, and `ppy run f.ppy` runs
  LLVM native code. All three must print the same output. A difference is a
  compiler bug, and every example is checked for it.
- **Falls back instead of guessing.** Code outside the supported subset is
  reported with its location. A guard that fails at runtime, such as an
  overflowing multiply, returns to the Python body and finishes there.
- **Ahead-of-time builds.** `ppy build` produces a launcher and a library
  that keep working after the compiler is uninstalled. `ppy build
  --standalone` produces a native executable with no CPython inside.
- **Seven backends off one IR:** LLVM, C11, C++17, CUDA, HIP, NVVM/PTX, and
  StableHLO, across 19 IR dialects. Other packages can
  [register their own backend](internals/backends.md).
- **Collections without pointers.** [`Vec`, `Deque`, `Heap`, `LinkedList`,
  `HashMap`, `HashSet`, `TreeMap`, and `TreeSet`](guide/collections.md)
  compile to native code and give the same answers as their Python
  reference classes.
- **Eight library plugins:** NumPy, PyTorch, JAX/Flax, pydantic,
  FastAPI/Uvicorn, SciPy, pandas, and PyArrow.
- **A cheap call boundary.** A native call with two `int` arguments costs
  47 ns, against 28 ns for a plain Python call, with no Python frames on the
  native path.
- **Tested.** @@TEST_FUNCTIONS@@ test functions on Python 3.12, 3.13, and
  3.14, with 74% statement coverage. @@DIAGNOSTIC_CODES@@ diagnostic codes,
  each documented in one place.

## Documentation

| | |
|---|---|
| [Getting started](getting-started.md) | Install, run a file three ways, check it, build it. |
| [Guide](guide/index.md) | The language, one topic per page: the subset, directives, native memory, SIMD, threads, parallel loops, derivatives, coroutines, GPU kernels, XLA, generics, regular expressions. |
| [Examples](howto/index.md) | @@EXAMPLE_FOLDERS@@ folders and @@EXAMPLE_PROGRAMS@@ programs with their commands and output. @@COMPARED_FOLDERS@@ of them are measured against Numba, Cython, NumPy, numexpr, JAX, PyTorch, CuPy, Triton, Taichi, Mojo, Codon, Rust, C, pandas, polars, asyncio, and uvloop; the results are collected on the [comparisons page](howto/comparisons.md). |
| [Reference](reference/index.md) | The command line, diagnostics, configuration, performance, compatibility, and the Python API. |
| [Internals](internals/index.md) | The pipeline, the IR and its dialects, the cache, the solver. |

## Limitations

PPy accepts Python's syntax and works with its ecosystem, but it does not
support every dynamic behavior. `exec` and `eval`, monkey-patching, dynamic
namespace mutation, and unrestricted runtime reflection are restricted, or
isolated behind an explicit `ppy.dynamic` boundary. That is the trade for
analysis and native code you can rely on.

To bring an existing Python project over, use `ppy migrate`
([how a real project went](internals/migrating.md)).

</div>
