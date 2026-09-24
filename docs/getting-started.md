# Getting started

This tutorial takes you from installing PPy to running your first native
kernel. Run every command below in your project directory.

## 1. Install PPy

```bash
uv add "ppy-lang[llvm]"
```

If you don't use `uv`, run `pip install "ppy-lang[llvm]"` instead.

The distribution is called `ppy-lang`. It installs three packages: `ppy`,
`ppy_compiler`, and `ppy_runtime`. Your code writes `import ppy`.

The native backend needs the `llvm` extra (llvmlite). The fastest call
boundary is built when the CPython headers (`python3-dev`) are present.
[Installing](installing.md) has the details.

Check what PPy found on your machine:

```bash
uv run ppy doctor   # what was found: the compiler, LLVM, the C toolchain, plugin runtimes
```

## 2. Write your first file

A `.ppy` file is valid Python. The compiler reads the decorators and
annotations of the `ppy` package. Under plain CPython they do nothing.

Save this as `collatz.ppy`:

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

What the `ppy` parts do:

- `@ppy.pure` is a contract that the function has no observable effects.
  The checker verifies it.
- `@ppy.opt(3)` sets this function's optimization level.
- `ppy.input[int]()` reads one line of standard input as an integer. The
  value goes straight into memory, without a Python object per field.

## 3. Run it three ways

```bash
echo 300000 | uv run python  collatz.ppy   # 1. plain CPython
echo 300000 | uv run ppy     collatz.ppy   # 2. the optimized Python backend
echo 300000 | uv run ppy run collatz.ppy   # 3. LLVM native
```

All three must print the same answer. If they differ, that is a compiler
bug.

The first run of the third command builds into the cache (about 0.6 s).
Later runs only start the launcher, which serves the built artifact. If you
edit any source under the project, the artifact's name changes and a fresh
one is built.

## 4. Check and explain

```bash
uv run ppy check collatz.ppy               # static checking only; strict is the default
uv run ppy explain collatz.ppy:longest     # why it went native, or what blocked it
```

`ppy check` runs the checker without running your code. An implicit `Any`
is an error (`E1201`). So is a decorator the checker cannot vouch for
(`E1204`). [Diagnostics](reference/diagnostics.md) lists every code.

`ppy explain` tells you why a function went native, or what blocked it.

## 5. Build an executable

```bash
uv run ppy build collatz.ppy -o dist
./dist/collatz < input.txt
```

`ppy build` writes the objects, `libppy_<project>.so`, a manifest, and a
launcher executable. The launcher starts an embedded interpreter, imports
`ppy_runtime`, and runs your program. It keeps working after you uninstall
the compiler.

`run` and `build` produce the same program. Both keep Python-integer
semantics: overflow is guarded and falls back to arbitrary precision. Pass
`--unsafe` to either one to drop the guards and get 64-bit wrap semantics
like C's.

To get an executable with no CPython inside:

```bash
uv run ppy build --standalone collatz.ppy -o native   # an executable with no CPython inside
```

## 6. Use it in an existing project

You don't need to move a whole Python project to PPy. Carve out the
kernels instead:

1. Put the loops where the time goes into a `.ppy` module.
2. Add one line to it: `import ppy`.
3. Import that module from your plain `.py` code. Where the compiler is
   installed, the module is served natively.

[Interop](howto/24_interop.md) and [Migrating a real
project](internals/migrating.md) cover this in depth.

Two commands help you turn existing Python into PPy:

```bash
uv run ppy convert kernel.py     # untyped Python to strict PPy, inferred from the call sites
uv run ppy migrate legacy.py     # the permissive form: dynamic features go behind boundaries
```

## Next steps

- [Guide](guide/index.md): the subset and the directives, then native memory, SIMD, threads, parallel loops, derivatives, coroutines, GPU kernels, XLA, and generics.
- [Examples](howto/index.md): @@EXAMPLE_FOLDERS@@ folders, one page each.
- [CLI](cli.md): all commands and options, including `ppy emit`, `inspect --stage`, `--report-opt`, `--sanitize`, and `--profile`/`--pgo`.
