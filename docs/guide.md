# Guide

Commands and options are in [cli.md](cli.md).

## The three paths

One file, three ways. Any disagreement is a compiler bug, and `ppy test`
checks for it.

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

The large factor is the Python *around* the model, not the tensor math. A
PyTorch ATen region removes one Python round trip per operator, worth about 20%
on small CPU tensors and nothing on an accelerator, where kernel launch latency
dominates. Training time is dominated by `.backward()` and the optimizer, which
stay in Python.

## Conversion

`ppy convert` infers parameter and return types across the whole call graph,
instance field types from `__init__`, container element types from what is
appended, and a parameter's type from the arithmetic it takes part in when no
call site exists. It attaches `@ppy.pure` where the checker proved it, moves a
class above the function that annotates against it so a quoted forward
reference becomes an ordinary one, formats what it writes, and introduces no
pylint finding the source did not have.

With `--promote-buffers` it declares a read-only numeric list parameter as
`Buffer[T]` and rewrites the values feeding it into `array.array`, which is what
lets a loop lower to native code. When it cannot, it says why:

```
remark[R3003]: `raw` could be a borrowed buffer, but `raw` is sliced, which
               copies; indexing it element by element instead would let the
               memory be borrowed
```

It does not rename, split a function that does several jobs, or restructure an
algorithm. Those are design decisions.

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

27 worked examples under `examples/`, one folder each with a README. Where a
folder holds both `<name>.py` and `<name>.ppy`, the `.ppy` is exactly what
`ppy convert` writes and `verify_conversions.py` regenerates it to prove it.

```bash
python examples/run_all.py             # every example x 3 paths, compared
python examples/verify_conversions.py  # every .ppy regenerated from its .py
python examples/lint_all.py            # pylint over every example
```

## Dependency groups

```bash
uv sync                  # dev: pytest, pylint, llvmlite, numpy, pydantic
uv sync --group torch    # + PyTorch (cu128 index)
uv sync --group jax      # + JAX (cuda12)
uv sync --group all      # everything
```
