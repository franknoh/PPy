# Guide

The map. Details live one link away:

- [language.md](language.md) — the subset, directives, markers, the three paths
- [conversion.md](conversion.md) — how `ppy convert` and `ppy migrate` infer what they write
- [architecture.md](architecture.md) — pipeline, module map, cache, threads
- [plugins.md](plugins.md) — NumPy, PyTorch, JAX, Pydantic, Uvicorn
- [config.md](config.md) — every `[tool.ppy]` key
- [diagnostics.md](diagnostics.md) — every code
- [cli.md](cli.md) — every command and option
- [compatibility.md](compatibility.md) — what is stable, what moves, what the cache and artifact ABI promise
- [the implementation spec](../ppy-compiler-implementation-spec-v1.md) — the normative baseline; comments in the source cite it by section

## The three paths

One file, three ways. Any disagreement is a compiler bug, and the suite plus
`examples/run_all.py` check for it.

| | |
|---|---|
| `python f.ppy` | plain CPython, no compiler |
| `ppy f.ppy` | optimized Python backend |
| `ppy run f.ppy` | LLVM native |

## Measured

Numbers live where the thing they measure lives, so there is one copy of
each and it is regenerated rather than retyped:

| what | where |
|---|---|
| the collatz kernel against nine other compilers | [README](../README.md#one-file-four-ways) |
| six competitive-programming problems, end to end, against C | [examples/15_algorithms](../examples/15_algorithms/README.md) |
| the Python/native call boundary, per call | `examples/bench_boundary.py` |
| startup: cold build, warm build, launcher, prebuilt, JIT | `examples/bench_startup.py` |
| what input costs, three ways | [examples/15_algorithms/15f_input](../examples/15_algorithms/15f_input/README.md) |

`examples/15_algorithms/measurements.json` holds the recorded run with the
machine it was taken on, and `scripts/refresh.py` re-measures and reports
what has drifted past a tolerance. A scheduled workflow does the same weekly;
absolute wall times differ between runners, so they are not a per-change gate.

Two shapes recur in all of it. The native path wins where the work is a loop
over machine-word data — that is what lowering is for. It neither helps nor
is meant to help where the time is inside a library: a torch or jax training
loop is `.backward()` and the optimizer, and stays exactly as fast.

The Python backend is not faster and is not meant to be: it optimizes the AST,
and the interpreter overhead is unchanged.

Caches earn their keep inside a run, not around it: on the collatz module,
lowering to IR costs 777 ms cold and 12 ms from the content-addressed store.
What no cache can remove from `ppy run` is per-process — importing the
compiler and MCJIT machine-code emission — and that is what `ppy build` pays
once. `PPY_CACHE_DIR` moves the store itself off a slow filesystem, which is
worth doing: a checkout on a mounted Windows drive answers `stat` several
times slower, and a launch is mostly imports.

## `.ppy` under ordinary CPython

Importing `ppy` installs a `sys.meta_path` finder, so a plain `.py` file can
import `.ppy` modules with no compiler and no build step:

```python
import ppy  # installs the hook
import geometry  # loads geometry.ppy as ordinary Python source
```

Without `import ppy` the module is invisible — the hook is never implicit. If
`foo.py` and `foo.ppy` both exist the `.ppy` wins and a warning is raised;
`ppy check` rejects the ambiguity outright, so use `--in-place` to migrate a
project.

## Examples

30 example folders under `examples/`, 39 runnable programs, each folder with
a README. Where a folder holds both `<name>.py` and `<name>.ppy`, the `.ppy`
is exactly what `ppy convert` writes — `ppy migrate` for the folder that says
so — and `verify_conversions.py` regenerates it to prove it.

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

Using PPY from another project (`uv add "ppy-lang[llvm]"`) and setting
up this repo for development (`uv sync` and its groups) are both in the
[README](../README.md#install).
