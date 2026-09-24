# Compatibility

PPy is at version 0.3. Some parts of it are more settled than others.
This page says which parts you can build on, which parts will change, and
what happens when the two sides of a boundary disagree.

## Stability levels

| level | meaning |
|---|---|
| stable | changes only with a deprecation period and a changelog entry |
| settling | the shape is right and the details may still change; breaking changes are listed in the changelog |
| experimental | may change or be withdrawn in any release |

The levels of each part:

| surface | level | notes |
|---|---|---|
| the language subset: statements, expressions, the type system | settling | new constructs are added; accepted code is not un-accepted without a note |
| `ppy.pure`, `ppy.opt`, `ppy.native`, `ppy.jit`, `ppy.dynamic`, `ppy.check` | settling | the directives a program is written around |
| `ppy.input`, `ppy.scan`, `ppy.buffer`, `ppy.read_ints`, `ppy.read_token` | experimental | `input` reads lines and `scan` tokens since 0.3; the spelling may still change |
| `ppy.check`, `ppy.assume` | experimental | `check` validates all the way down since 0.3; `assume` is the unchecked crossing |
| `ppy.native` memory and `ppy.ffi`; `ppy.simd`, `ppy.cpu`, `ppy.atomic`, `ppy.concurrent`, `ppy.parallel.range`; `ppy.grad` | settling | added in 0.2; each has a reference implementation under CPython, and a lowering that agrees with it |
| `ppy.aio`, `ppy.cuda`, `ppy.hip`, `ppy.xla` | experimental | added in 0.2; the runtimes behind them (epoll, the CUDA driver, PJRT) are the newest code in the tree |
| the canonical IR (`ppy_compiler.ir`) and `.ppyir` | settling | the text carries a schema and dialect versions a reader refuses rather than guesses at; public from 0.2.0 at schema 1 |
| `ppy emit`, `ppy inspect --stage`, `--report-opt`, `--sanitize`, `--profile`/`--pgo` | experimental | developer tools; the text they print is for people and may be reworded |
| `ppy check` / `ppy run` / `ppy build` and their flags | settling | flags are added; removals get a deprecation release |
| `ppy convert` / `ppy migrate` output | settling | the output is regenerated from source, so a change shows up as a diff, not a break |
| diagnostic codes (`E1xxx`, `W2xxx`, `R3xxx`) | settling | a code keeps its meaning; new codes are added freely |
| the backend interface in `backend/base.py` (`BACKEND_API_VERSION` 1) | experimental | added in 0.3. A package registers a backend through `ppy.backends`, declares `api_version = 1` as a literal of its own, receives the canonical IR after the shared passes, hangs passes at the `backend` stage alone, and emits (per module or per program) or builds. The version number is the interface's own, bumped when a method's meaning changes |
| the plugin interface in `plugins/base.py` | settling | the second version. Types, effects, lowerings, and the IR hooks (`register_dialects`, `register_passes`, `register_patterns`, `register_lowerings`) the builtin plugins use themselves |
| the cache format | internal | see below; never read it yourself |
| the built-artifact ABI | versioned | see below |

## The cache is disposable

The build cache holds optimization state. It is not a source of truth: each
artifact in it is content-addressed and can be recomputed from the source it
was derived from.

- Deleting any part of it costs a rebuild and nothing else.
- A damaged SQLite index is quarantined next to itself as
  `index.sqlite.corrupt-<timestamp>`, rebuilt empty, and reported once as
  `warning[W2101]`. Compilation continues with cache misses.
- If even a fresh index cannot be written (a read-only directory, a full
  disk), the store keeps working in memory. Every lookup is a miss, nothing
  persists, and the answer is the same.

The cache schema version is internal. It changes without notice, and a
mismatch is handled by rebuilding rather than by migrating.

## The built-artifact ABI

`ppy build` writes `ppy-bindings.json` with an `abi_version`. The runtime
that launches an artifact refuses a version it does not speak, with the
remedy in the message:

```
error[E1801]: <path> speaks ABI 2; this runtime speaks 1 -- rebuild the
artifact with `ppy build`
```

The manifest also records the Python version it was built for. The
launcher refuses a different one for the same reason: the wrappers are
compiled against one interpreter's ABI. An artifact and the `ppy_runtime`
that launches it are expected to come from the same release.

## Platforms and the C library

The `ppy-lang` wheel is pure Python (`py3-none-any`). Nothing in it was
compiled on a build machine. Everything native is compiled where it runs, by
the C compiler on `PATH`: the `ppy._io` scanner, the Python-ABI wrappers,
the native objects, a standalone executable. Those bind to the running
machine's C library and Python.

So the rule is to build where you run. Don't copy a built artifact to an
older machine: a library linked against glibc 2.35 does not load on glibc
2.27. `ppy doctor` prints the libc it found.

The oldest supported glibc is set by the dependencies' wheels, not by PPy:

| package | Linux x86_64 wheels | glibc |
|---|---|---|
| `llvmlite` 0.49 | manylinux2014 | 2.17 |
| `libcst` 1.7 | manylinux2014 | 2.17 |
| `libcst` 1.8 and later | manylinux_2_28 only | 2.28 |
| `z3-solver` 4.13 to 4.15 | manylinux2014 | 2.17 |
| `z3-solver` 5.x | manylinux_2_27 | 2.27 |
| `numpy` up to 2.2 (Python 3.12) | manylinux2014 | 2.17 |
| `numpy` 2.3 and later | manylinux_2_28 | 2.28 |

`libcst` is the one dependency that requires more than glibc 2.27. From 1.8
it ships `manylinux_2_28` wheels only. An installer that cannot use them
falls back to building the Rust sources, which fails without a Rust
toolchain. For that reason `ppy-lang` pins `libcst<1.8` on Python 3.13 and
earlier, until a wheel for older machines returns. Python 3.14 needs
`libcst` 1.8 and runs on machines new enough for it.

With that pin, Ubuntu 18.04 (glibc 2.27) works:

- it installs `ppy-lang[llvm,solver]`
- `uv` chooses a `numpy` that has a wheel for it
- a `uv`-managed interpreter (`uv python install 3.12`) runs there and
  brings its headers, so the fast Python boundary is available without a
  system `python3-dev`

## Python versions

3.12, 3.13, and 3.14 are tested on every change. A release supports the
versions its CI matrix runs. Dropping one is a changelog entry.

## What a change to PPy may not do

- Make a program that checked clean produce a different answer on any of the
  three paths. A native path that cannot keep a promise falls back to Python
  rather than answering differently.
- Turn a cache or artifact problem into a failure to compile correct source.
