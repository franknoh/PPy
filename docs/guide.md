# Guide

The map. Details live one link away:

- [language.md](language.md) — the subset, directives, markers, the three paths
- [conversion.md](conversion.md) — how `ppy convert` and `ppy migrate` infer what they write
- [architecture.md](architecture.md) — pipeline, module map, cache, threads
- [plugins.md](plugins.md) — NumPy, PyTorch, JAX, Pydantic, Uvicorn
- [config.md](config.md) — every `[tool.ppy]` key
- [diagnostics.md](diagnostics.md) — every code
- [cli.md](cli.md) — every command and option

## The three paths

One file, three ways. Any disagreement is a compiler bug, and the suite plus
`examples/run_all.py` check for it.

| | |
|---|---|
| `python f.ppy` | plain CPython, no compiler |
| `ppy f.ppy` | optimized Python backend |
| `ppy run f.ppy` | LLVM native |

## Measured

WSL2, RTX 5080, torch 2.11+cu128, jax 0.11.1. Answers identical on all three
paths in every row.

| algorithm | plain | ppy run | C (`gcc -O3`) |
|---|---:|---:|---:|
| sieve 2e6 | 191.9 ms | 9.7 ms | 14.6 ms |
| collatz 3e5 | 1204.3 ms | 42.8 ms | 42.9 ms |
| knapsack 400×2e4 | 476.7 ms | 5.4 ms | 2.5 ms |
| edit distance 2000² | 531.1 ms | 3.5 ms | 3.8 ms |
| Floyd–Warshall 220 | 539.9 ms | 3.5 ms | 5.3 ms |
| matmul 220 | 527.4 ms | 3.7 ms | 2.2 ms |
| union-find 5e5 | 186.2 ms | 3.2 ms | 3.9 ms |
| fermat 6e4 | 25.9 ms | 2.8 ms | 1.8 ms |

The C column is `examples/15_algorithms/algorithms.c`: the same kernels
hand-written, printing the same answers, compiled at the same optimization
level the kernels declare (`@ppy.opt(3)`). Guard hoisting
(`[tool.ppy.llvm] safeguards`) moves the Python-int overflow and bounds
guards of multiplied index chains into one check ahead of the loop, so the
body keeps no side exits and LLVM can strength-reduce and vectorize; the
kernels where gcc still leads carry guards on data values no range analysis
can prove away.

The Python backend is not faster and is not meant to be: it optimizes the AST,
and the interpreter overhead is unchanged.

| model training | plain | ppy run |
|---|---:|---:|
| torch, preprocessing | 70.9 ms | 0.9 ms |
| torch, training loop | 507 ms | 532 ms |
| jax, preprocessing | 69.6 ms | 0.9 ms |
| jax, training loop | 26.5 ms | 26.3 ms |

The large factor is the Python *around* the model, not the tensor math.
The training loops are dominated by `.backward()` and the optimizer, which
stay in the framework either way — the native path neither helps nor is it
supposed to.

Caches earn their keep inside a run, not around it: on the collatz module,
lowering to IR costs 777 ms cold and 12 ms from the content-addressed store.
What no cache can remove is per-process — importing the compiler (~850 ms)
and MCJIT machine-code emission (~190 ms) — which is exactly what the
launcher `ppy build` emits pays once instead of every run. `PPY_CACHE_DIR`
moves the store itself off a slow filesystem.

## `.ppy` under ordinary CPython

Importing `ppy` installs a `sys.meta_path` finder, so a plain `.py` file can
import `.ppy` modules with no compiler and no build step:

```python
import ppy        # installs the hook
import geometry   # loads geometry.ppy as ordinary Python source
```

Without `import ppy` the module is invisible — the hook is never implicit. If
`foo.py` and `foo.ppy` both exist the `.ppy` wins and a warning is raised;
`ppy check` rejects the ambiguity outright, so use `--in-place` to migrate a
project.

## Examples

30 example folders (33 runnable programs) under `examples/`, each with a
README. Where a
folder holds both `<name>.py` and `<name>.ppy`, the `.ppy` is exactly what
`ppy convert` writes — `ppy migrate` for the folder that says so — and
`verify_conversions.py` regenerates it to prove it.

```bash
python examples/run_all.py             # every example x 3 paths, compared
python examples/verify_conversions.py  # every .ppy regenerated from its .py
python examples/lint_all.py            # pylint over every example
```

Both runners skip an example whose optional library (torch, jax, uvicorn) is
not installed, and say so — a skip is never counted as a pass.

## Differential fuzzing

`tests/test_differential_fuzz.py` generates deterministic small programs —
arithmetic, branches, bounded loops, lists, aliases, mutation, exceptions —
and requires the three paths to agree byte for byte. Fixed seeds (including
every seed that ever regressed) run in CI; sweep wider locally with:

```bash
PPY_FUZZ_SEEDS=0:500 pytest tests/test_differential_fuzz.py
```

Its first three hundred seeds caught three real optimizer bugs.

## Installing

Using PPY from another project (`uv add "ppy[llvm] @ git+..."`) and setting
up this repo for development (`uv sync` and its groups) are both in the
[README](../README.md#install).
