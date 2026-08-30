# PPY — Pretty Python

A statically analyzable language in Python's syntax. A `.ppy` file *is* valid
Python: it runs under plain CPython with no compiler involved. The compiler
adds static checking, an optimized Python backend, and an LLVM native
backend — and all three must produce the same answer.

PPY is source-compatible with Python's syntax and ecosystem, not with every
dynamic Python behavior: `exec`/`eval`, monkey-patching, dynamic namespace
mutation, and unrestricted runtime reflection are deliberately restricted —
or isolated behind an explicit `ppy.dynamic` boundary — in exchange for
analysis, optimization, and native compilation that can be trusted. Running
existing Python is a migration feature (`ppy migrate`), not the definition of
the language.

## Install

Add PPY to your project with [uv](https://docs.astral.sh/uv/) (Python 3.12+):

```bash
uv add "ppy[llvm] @ git+https://github.com/franknoh/PPy.git"
```

or with pip: `pip install "ppy[llvm] @ git+https://github.com/franknoh/PPy.git"`.

The base package is the compiler and the runtime; extras enable the rest, so
you install only what you use:

| extra | enables |
|---|---|
| `llvm` | the native backend (llvmlite) |
| `numpy` / `pydantic` / `uvicorn` | the matching plugin |
| `torch` / `jax` | the matching plugin, resolved from PyPI — pick your own CUDA index in your project if you need one |

Everything degrades cleanly: a missing library only disables its plugin, and
`uv run ppy doctor` reports what was found. For the fastest native call
boundary, have the CPython headers installed (`python3-dev`); without them
PPY says so once and uses the slower boundary.

## One file, three ways

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


print(longest(300000))
```

```bash
uv run python  collatz.ppy    # plain CPython, no compiler   1170 ms
uv run ppy     collatz.ppy    # optimized Python backend     1181 ms
uv run ppy run collatz.ppy    # LLVM native                    45 ms  (JIT ~1200 ms, every run)
gcc -O3 collatz.c && ./a.out  # the same loop in C             44 ms  (gcc ~110 ms, once)
```

And the fourth way — build the native artifacts once, then run the binary:

```bash
uv run ppy build collatz.ppy -o dist   # ~800 ms, once: object, libppy_*.so, manifest, launcher
./dist/collatz                         # native binary                 45 ms
```

Run numbers are the kernel's wall time on one machine, best of five; the
parenthesized figure is what that row spends turning source into machine
code, and how often. The C row is the same algorithm hand-written with
64-bit integers — the target that keeps the native backend honest: within
measurement noise of it today, overflow guards included, from source that is
still plain Python. (Python's floor semantics help: `n // 2` lowers to one
arithmetic shift exactly, where C's truncating division needs a sign fixup.)
`./dist/collatz` is `ppy run` in a compiled coat: the launcher enters the
same pipeline and the same guarded bindings, and only takes its machine code
from the library built next to it — delete the library and it refuses,
rather than quietly interpreting.

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

Types come from the whole call graph, not one file. `samples` is `Sequence`
rather than `list` because the body only reads it and one call site passes a
tuple; `LIMIT` is `Final` because nothing rebinds it; `@ppy.pure` is attached
only where the checker proved it. Nothing is renamed and no function is split —
those are design decisions, not mechanical ones.

## Works with

Each library is a plugin: the compiler learns that library's types and effects,
takes a faster path where it can prove one is equivalent, and falls back to the
ordinary Python call everywhere else. A guard that fails is a fallback, never a
different answer.

**NumPy** — elementwise expressions fuse into a single loop with no
temporaries; `dot`, `matmul`, `inner`, `vdot`, and `tensordot` route to the
linear-algebra path. Contiguity and shape are guarded at runtime, not assumed.
Reduction order is preserved unless `@ppy.fastmath` permits reassociation, so a
sum stays bit-identical to NumPy's.

**PyTorch** — a function whose body is entirely curated tensor operations (55 of
them) compiles into one C++ region calling ATen directly, removing a Python
round trip per operator. Every call still goes through the dispatcher, so
autograd, device selection, and backend keys are unchanged; a tensor subclass or
a `__torch_function__` override trips the guard and the Python body runs. CUDA
is used when it is there.

**JAX / Flax** — a `@jax.jit` function whose inputs carry `ppy.Shape` and
`ppy.DType` can be exported to StableHLO at build time, so the trace is not
repeated at startup; shapes may be symbolic, so one artifact serves every
batch size (export runs project code and is off until the project opts in).
The same plugin models Flax and optax — layers, activations, `Module.init`/
`apply` through the external MRO, optimizers — so a Flax training loop
checks under strict mode as-is.

**Pydantic** — models are typed, constructor and output shapes are kept
distinct, and field constraints become refinements the checker can use.

**Uvicorn / FastAPI** — the ASGI application is resolved statically instead of
re-imported by module string per worker, and the reloader is told to watch
`.ppy`. FastAPI rides the same plugin: its surface is modeled so strict mode
checks a FastAPI service as-is, while route handlers keep the exact
signatures FastAPI reads at import.

The plugin's exact version is part of every cache key, so an artifact built
against one build of a library is never reused against another. `ppy doctor`
prints what it found.

## Docs

- [docs/guide.md](docs/guide.md) — overview, measurements, the import hook
- [docs/language.md](docs/language.md) — the subset, directives, markers
- [docs/conversion.md](docs/conversion.md) — how `ppy convert` and `ppy migrate` infer what they write
- [docs/architecture.md](docs/architecture.md) — pipeline, cache, threads
- [docs/plugins.md](docs/plugins.md) — how each library integration works
- [docs/config.md](docs/config.md) — every `[tool.ppy]` key
- [docs/diagnostics.md](docs/diagnostics.md) — every diagnostic code
- [docs/cli.md](docs/cli.md) — every command and option
- [examples/README.md](examples/README.md) — 30 worked examples

## Development

```bash
git clone https://github.com/franknoh/PPy.git
cd PPy
uv sync                  # dev group: llvmlite, numpy, pydantic, pytest, linters
uv sync --group torch    # + PyTorch (CUDA 12.8 index)
uv sync --group jax      # + JAX (cuda12)
uv sync --group all      # everything
uv run ppy doctor
```

```bash
uv run pytest -q
uv run python examples/run_all.py             # every example x 3 paths, compared
uv run python examples/verify_conversions.py  # every .ppy regenerated from its .py
uv run python examples/lint_all.py            # pylint over every example
```
