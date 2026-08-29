# Guide

The map. Details live one link away:

- [language.md](language.md) — the subset, directives, markers, the three paths
- [conversion.md](conversion.md) — how `ppy convert` infers what it writes
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

| algorithm | plain | ppy run |
|---|---:|---:|
| sieve 2e6 | 187.7 ms | 8.6 ms |
| collatz 3e5 | 1203.6 ms | 88.4 ms |
| knapsack 400×2e4 | 469.0 ms | 5.3 ms |
| edit distance 2000² | 423.3 ms | 2.9 ms |
| Floyd–Warshall 220 | 492.2 ms | 6.1 ms |
| matmul 220 | 527.7 ms | 7.2 ms |
| union-find 5e5 | 167.9 ms | 3.2 ms |

The Python backend is not faster and is not meant to be: it optimizes the AST,
and the interpreter overhead is unchanged.

| model training | plain | ppy run |
|---|---:|---:|
| torch, preprocessing | 66.2 ms | 0.9 ms |
| torch, 100 steps | 1244.9 ms | 715.3 ms |
| jax, preprocessing | 62.8 ms | 0.9 ms |
| jax, 100 steps | 24.9 ms | 24.2 ms |

The large factor is the Python *around* the model, not the tensor math.
Training time is dominated by `.backward()` and the optimizer, which stay in
Python.

Incremental builds, 30-example tree: cold 2415 ms, no change 371 ms (LLVM
never loads), one file changed 1785 ms.

## `.ppy` under ordinary CPython

Importing `ppy` installs a `sys.meta_path` finder, so a plain `.py` file can
import `.ppy` modules with no compiler and no build step:

```python
import ppy        # installs the hook
import geometry   # loads geometry.ppy as ordinary Python source
```

Without `import ppy` the module is invisible — the hook is never implicit. If
`foo.py` and `foo.ppy` both exist the `.ppy` wins and a warning is raised;
`ppy check` rejects the ambiguity outright, so use `convert --in-place` to
migrate a project.

## Examples

29 example folders (31 runnable programs) under `examples/`, each with a
README. Where a
folder holds both `<name>.py` and `<name>.ppy`, the `.ppy` is exactly what
`ppy convert` writes and `verify_conversions.py` regenerates it to prove it.

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
