# Getting started

From install to the first native kernel. Every command below runs in the
project directory.

## 1. Install

```bash
uv add "ppy-lang[llvm]"
```

Without `uv`, `pip install "ppy-lang[llvm]"`. The distribution is `ppy-lang`
and what it installs is `ppy`, `ppy_compiler`, and `ppy_runtime`, so your
code writes `import ppy`. The native backend needs the `llvm` extra
(llvmlite), and the fastest call boundary is built when the CPython headers
(`python3-dev`) are present. [Installing](installing.md) has the details.

```bash
uv run ppy doctor   # what was found: the compiler, LLVM, the C toolchain, plugin runtimes
```

## 2. The first file

A `.ppy` file is valid Python. What the compiler adds is carried by the
decorators and annotations of the `ppy` package, all of which are inert
under plain CPython.

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


print(longest(ppy.input[int]()))
```

`@ppy.pure` is a contract that the function has no observable effects, and
the checker verifies it. `@ppy.opt(3)` is this function's optimization
level. `ppy.input[int]()` reads one line of standard input as an integer, straight
into memory rather than through a Python object per field.

## 3. Run it three ways

```bash
echo 300000 | uv run python  collatz.ppy   # 1. plain CPython
echo 300000 | uv run ppy     collatz.ppy   # 2. the optimized Python backend
echo 300000 | uv run ppy run collatz.ppy   # 3. LLVM native
```

The three must print the same answer; a difference is a compiler bug. The
first run of the third builds into the cache (about 0.6 s); every later run
is the launcher alone, serving the built artifact. Editing any source under
the project changes the artifact's name and builds a fresh one.

## 4. Check and explain

```bash
uv run ppy check collatz.ppy               # static checking only; strict is the default
uv run ppy explain collatz.ppy:longest     # why it went native, or what blocked it
```

An implicit `Any` is an error (`E1201`), and so is a decorator the checker
cannot vouch for (`E1204`). Every code is in
[Diagnostics](reference/diagnostics.md).

## 5. Build

```bash
uv run ppy build collatz.ppy -o dist
./dist/collatz < input.txt
```

`ppy build` writes the objects, `libppy_<project>.so`, a manifest, and a
launcher executable. The launcher starts an embedded interpreter, imports
`ppy_runtime`, and runs; it keeps working with the compiler uninstalled.
`run` and `build` mean the same program: both keep Python-integer
semantics -- overflow is guarded and falls back to arbitrary precision --
and `--unsafe`, on either, drops the guards for 64-bit wrap semantics like
C's.

```bash
uv run ppy build --standalone collatz.ppy -o native   # an executable with no CPython inside
```

## 6. In a real program

In an existing Python project, do not move the whole thing: carve out the
kernels. Put the loops where the time goes into a `.ppy` module and add one
line, `import ppy`; a plain `.py` imports that module, and where the
compiler is installed it is served natively
([Interop](howto/24_interop.md), [Migrating a real
project](internals/migrating.md)).

```bash
uv run ppy convert kernel.py     # untyped Python to strict PPY, inferred from the call sites
uv run ppy migrate legacy.py     # the permissive form: dynamic features go behind boundaries
```

## Next

- [Guide](guide/index.md): from the subset and the directives to native memory, SIMD, threads, parallel loops, derivatives, coroutines, GPU kernels, XLA, and generics.
- [Examples](howto/index.md): @@EXAMPLE_FOLDERS@@ folders, one page each.
- [CLI](cli.md): every command and option — `ppy emit`, `inspect --stage`, `--report-opt`, `--sanitize`, `--profile`/`--pgo`.
