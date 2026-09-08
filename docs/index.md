# PPY

**A statically analyzable language in Python's syntax.** A `.ppy` file *is*
valid Python: it runs under plain CPython with no compiler involved. The
compiler adds static checking, an optimized Python backend, and an LLVM
native backend — and all three must produce the same answer.

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
echo 300000 | python    collatz.ppy   # plain CPython                 1047.8 ms
echo 300000 | ppy run   collatz.ppy   # LLVM, built on the first run    39.3 ms
echo 300000 | ppy build collatz.ppy -o dist && ./dist/collatz   #        29.5 ms
```

The same loop in C, `gcc -O3`, takes 40.4 ms. The numbers, the machine, and
the neighbouring compilers are on the [performance page](reference/performance.md).

<div class="grid cards" markdown>

-   **[Getting started](getting-started.md)**

    ---

    From install to a native kernel in ten minutes.

-   **[Guide](guide/index.md)**

    ---

    The subset, directives, native memory, SIMD, threads, parallel loops,
    derivatives, coroutines, GPU kernels, XLA, generics.

-   **[Examples](howto/index.md)**

    ---

    42 folders, 52 programs, every one printing the same output on all
    three paths.

-   **[CLI](cli.md)**

    ---

    `ppy check`, `run`, `build`, `emit`, `inspect`, `explain`, the
    sanitizers, and profile-guided optimization.

-   **[API](api/index.md)**

    ---

    The `ppy` package, the runtime, the plugin interface, the canonical IR.

-   **[Internals](internals/index.md)**

    ---

    The pipeline, the IR and its dialects, the cache, the solver — in the
    order someone adding a backend would read them.

</div>

## What it is, and is not

PPY is source-compatible with Python's syntax and ecosystem, not with every
dynamic Python behavior: `exec`/`eval`, monkey-patching, dynamic namespace
mutation, and unrestricted runtime reflection are deliberately restricted —
or isolated behind an explicit `ppy.dynamic` boundary — in exchange for
analysis, optimization, and native compilation that can be trusted. Running
existing Python is a migration feature (`ppy migrate`), not the definition of
the language.

## The three paths

| | |
|---|---|
| `python f.ppy` | plain CPython, no compiler |
| `ppy f.ppy` | the optimized Python backend |
| `ppy run f.ppy` | LLVM native; the first run builds into the cache, every later one is the launcher alone |

Any observable difference between the three is a compiler bug. The test
suite and `examples/run_all.py` compare all three on every example. `ppy
build` is the third path ahead of time, and `ppy build --standalone` is a
native executable with no CPython inside.

## 0.2.0: the compiler as a platform

Between the analysis and every backend sits one typed canonical IR
([The IR](internals/ir.md)): SSA values, blocks, an explicit control-flow
graph, operations named in dialects, a verifier, a printer and parser
(`.ppyir` is public text), a pass manager, and rewrite patterns. The LLVM,
C/C++, CUDA/HIP source, NVVM/PTX, and StableHLO backends all read that IR
and nothing else. A program writes to the same IR through the `ppy`
namespaces — `ppy.native`, `ppy.simd`, `ppy.cpu`, `ppy.atomic`,
`ppy.concurrent`, `ppy.parallel.range`, `ppy.grad`, `ppy.aio`, `ppy.cuda`
and `ppy.hip`, `ppy.xla.jit` — each with a reference implementation that
is inert under plain CPython.
