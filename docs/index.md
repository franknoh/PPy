# PPY

**Python's syntax, compiled.** A `.ppy` file *is* valid Python and runs under
plain CPython with no compiler involved. The compiler adds a strict static
checker, an optimized Python backend, and an LLVM native backend — and the
three must print the same answer, or it is a bug.

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

The same loop in C: `gcc -O3` 41.9 ms, `clang -O3` 32.8 ms. One machine, ten
fresh processes each; the full table with Numba, Codon, PyPy, Cython, mypyc,
and Nuitka is on the [performance page](reference/performance.md).

<div class="grid cards" markdown>

-   **[Getting started](getting-started.md)**

    ---

    From install to a native kernel in ten minutes.

-   **[Guide](guide/index.md)**

    ---

    The subset, directives, native memory, SIMD, threads, parallel loops,
    derivatives, coroutines, GPU kernels, XLA, generics, regular expressions.

-   **[Examples](howto/index.md)**

    ---

    @@EXAMPLE_FOLDERS@@ folders, @@EXAMPLE_PROGRAMS@@ programs — the code, the commands, and what they print,
    the same on all three paths; seven of them measured against Numba, Cython, NumPy, JAX,
    PyTorch, Taichi, Mojo, Codon, Rust, and C, collected on one [comparisons page](howto/comparisons.md).

-   **[CLI](cli.md)**

    ---

    `ppy check`, `run`, `build`, `emit`, `inspect`, `explain`, the
    sanitizers, and profile-guided optimization.

-   **[API](api/index.md)**

    ---

    The `ppy` package, the runtime, the plugin interface, the canonical IR.

-   **[Internals](internals/index.md)**

    ---

    The pipeline, the IR and its dialects, the cache, the solver.

</div>

## By the numbers

| | |
|---|---|
| **3** execution paths that must agree, checked on every example | **1,110** tests on Python 3.12, 3.13, and 3.14, **74%** statement coverage |
| **42** example folders, **52** programs, every conversion regenerated to prove it | **76** diagnostic codes, each documented once |
| **18** IR dialects, **7** backends off one IR: LLVM, C11, C++17, CUDA, HIP, NVVM/PTX, StableHLO | **8** library plugins: NumPy, PyTorch, JAX/Flax, pydantic, FastAPI/Uvicorn, SciPy, pandas, PyArrow |
| **47 ns** for a native two-`int` call, against **28 ns** for a plain Python call | **0** Python frames on the native call path |

## The three paths

| | |
|---|---|
| `python f.ppy` | plain CPython, no compiler |
| `ppy f.ppy` | the optimized Python backend |
| `ppy run f.ppy` | LLVM native; the first run builds into the cache, every later one is the launcher alone |

`ppy build` is the third path ahead of time — a launcher and a library that
keep working with the compiler uninstalled — and `ppy build --standalone` is
a native executable with no CPython inside. A guard that fails at runtime
falls back to the Python body; it never answers differently.

## What it is, and is not

PPY is source-compatible with Python's syntax and ecosystem, not with every
dynamic Python behavior: `exec`/`eval`, monkey-patching, dynamic namespace
mutation, and unrestricted runtime reflection are deliberately restricted —
or isolated behind an explicit `ppy.dynamic` boundary — in exchange for
analysis, optimization, and native compilation that can be trusted. Running
existing Python is a migration feature (`ppy migrate`), not the definition of
the language.
