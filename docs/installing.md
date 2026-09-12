# Installing

## The package

```bash
uv add "ppy-lang[llvm]"          # add to a project
pip install "ppy-lang[llvm]"     # or with pip
```

The distribution is `ppy-lang`; what it installs is three packages, `ppy`
(the runtime and the directives), `ppy_compiler`, and `ppy_runtime`.
Before 1.0 a minor release may move the language and the diagnostics — the
[changelog](changelog.md) says what moved — so pin an exact version
(`ppy-lang[llvm]==0.3.0`). For the development tip instead:

```bash
uv add "ppy-lang[llvm] @ git+https://github.com/franknoh/PPy.git"
```

Python 3.12, 3.13, and 3.14 are tested on every change.

## Extras

The base package is the compiler and the runtime; extras enable the rest,
so you install only what you use.

| extra | enables |
|---|---|
| `llvm` | the native backend (llvmlite) |
| `solver` | the Z3 solver that proves overflow guards away |
| `numpy` / `pydantic` / `uvicorn` | the matching plugin |
| `torch` / `jax` | the matching plugin, CPU builds by default — point at a CUDA index in your own project if you want one |
| `scipy` / `pandas` / `pyarrow` | the matching plugin |
| `bind` | `ppy bind header`: libclang for reading C headers |

A missing library only disables its plugin, and `uv run ppy doctor` reports
what was found and what was not.

## The toolchain

Everything native is compiled where it runs, by the C compiler on `PATH`:
the small C scanner that reads input, the Python-ABI wrappers, the native
objects, a standalone executable. So:

- **A C compiler** (`cc` or `gcc`) is what makes the native path complete.
  Without one the boundary falls back to the slower ctypes trampoline, and
  `W2004` says so once.
- **The CPython headers** (`python3-dev`) are what the fastest
  `METH_FASTCALL` boundary is built against. An interpreter installed by `uv
  python install 3.13` brings its headers with it.
- **GPU**: running a `ppy.cuda` kernel on a device needs the CUDA driver,
  a device, and libdevice (the CUDA toolkit, `CUDA_HOME` or `PPY_LIBDEVICE`).
  Without them the reference launch runs and `W2008` says why. `ppy emit
  cuda`/`hip` write source only and need no device.
- **XLA**: the PJRT bridge that runs an `@xla.jit` function on a device
  needs JAX installed (`uv sync --group jax`). The compiler that writes the
  StableHLO does not.
- **Coroutines**: the native runtime behind `ppy.aio` is built once into the
  cache on Linux (epoll) with a C compiler. Elsewhere asyncio gives the same
  answer.

## Platforms and the C library

The `ppy-lang` wheel is pure Python (`py3-none-any`); nothing in it was
compiled on a build machine. Everything native binds to the running
machine's C library and Python, which is why a built artifact is not
something to copy to an older machine — a library linked against glibc 2.35
does not load on glibc 2.27 — and why "build where you run" is the rule.
`ppy doctor` prints the libc it found.

The floor is set by the dependencies' wheels, not by PPY:

| package | Linux x86_64 wheels | glibc |
|---|---|---|
| `llvmlite` 0.49 | manylinux2014 | 2.17 |
| `libcst` 1.7 | manylinux2014 | 2.17 |
| `libcst` 1.8 and later | manylinux_2_28 only | 2.28 |
| `z3-solver` 4.13 to 4.15 | manylinux2014 | 2.17 |
| `z3-solver` 5.x | manylinux_2_27 | 2.27 |
| `numpy` up to 2.2 (Python 3.12) | manylinux2014 | 2.17 |
| `numpy` 2.3 and later | manylinux_2_28 | 2.28 |

`libcst` is the one dependency that moved past glibc 2.27: from 1.8 it ships
`manylinux_2_28` wheels only, and an installer that cannot use them falls
back to building the Rust sources, which fails without a Rust toolchain.
`ppy-lang` therefore pins `libcst<1.8` on Python 3.13 and earlier; Python
3.14 needs `libcst` 1.8 and runs on machines new enough for it. With that,
Ubuntu 18.04 (glibc 2.27) installs `ppy-lang[llvm,solver]`, `uv` chooses a
`numpy` that has a wheel for it, and a `uv`-managed interpreter brings its
headers, so the fast boundary is available without a system `python3-dev`.

## Developing in the repository

```bash
git clone https://github.com/franknoh/PPy.git
cd PPy
uv sync              # the compiler, LLVM, NumPy, pydantic, the linters, the docs tools
./scripts/check.sh   # the one gate; CI runs exactly this
```

Plugin runtimes are separate groups, so you install only what you intend to
test: `uv sync --group torch`, `--group jax`, `--group uvicorn`, `--group
scipy`, `--group pandas`, `--group pyarrow`, `--group all`.
[Contributing](contributing.md) has the rest.
