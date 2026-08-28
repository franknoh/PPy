# CLI reference

What any of it is for is in [guide.md](guide.md).

## Global options

Before the subcommand.

| option | effect |
|---|---|
| `--version` | print the version and exit |
| `-q`, `--quiet` | only errors |
| `--color {auto,always,never}` | ANSI colour in diagnostics |
| `-O`, `--opt-level {0,1,2,3}` | override `[tool.ppy] opt-level` |
| `--no-strict` | downgrade strict-mode errors where a sound fallback exists |

`-O` overrides the project default, not a per-function `@ppy.opt(n)`, which is
a contract on that function.

## Running a file

```bash
ppy FILE.ppy [-- ARGS...]        # optimized Python backend
ppy run FILE.ppy [-- ARGS...]    # compile through LLVM, then run
```

Everything after `--` reaches the program as `sys.argv[1:]`.

## `ppy convert` — `.py` to `.ppy`

```bash
ppy convert PATH [--in-place] [--force] [--dry-run] [--promote-buffers]
```

`PATH` is a file or a directory; a directory is analyzed as one call graph, so
a function's types can come from call sites in other files.

| option | effect |
|---|---|
| `--dry-run` | print, write nothing |
| `--force` | overwrite an existing `.ppy` |
| `--in-place` | write the `.ppy` and remove the `.py` it came from |
| `--promote-buffers` | declare read-only numeric list parameters as `Buffer[T]` and rewrite the values feeding them into `array.array` |

Use `--in-place` to migrate a project. Without it both `foo.py` and `foo.ppy`
are left on disk, which a project may not contain — the module would be
ambiguous — so the converter warns and `ppy check` then refuses.

In a converted module `import ppy` is placed before any sibling import, because
that import is what installs the loader those modules need.

## `ppy check` — static validation

```bash
ppy check [PATH] [--remarks]
```

Types, effects, purity and native contracts, dynamic-feature policy. Exits
non-zero on any error. `--remarks` also prints which functions lowered natively
and which stayed boxed, with the reason.

## `ppy build` — compile without running

```bash
ppy build TARGET [--backend {llvm,python}] [-o DIR]
```

`--backend llvm` (default) writes objects, `libppy_<project>.so`,
`ppy-bindings.json`, and a launcher. `-o` puts them somewhere other than the
cache. With the JAX plugin enabled and permitted, staged functions are exported
here too.

## `ppy explain` — why it compiled that way

```bash
ppy explain LOCATION
```

`LOCATION` is a `FILE:LINE`, a function name or qualname, or a diagnostic code.
For a function it reports the semantic type, effects, purity, the backend
decision, the representation chosen for every parameter, and each library
call's lowering with its guards.

## `ppy inspect` — generated artifacts

```bash
ppy inspect TARGET [--backend {python,llvm}] [--ir]
```

The optimized Python by default, including plugin rewrites, so it is what to
read when a result differs from plain CPython. `--ir` prints what the native
path compiles: LLVM IR, then the C for the CPython-ABI wrappers, then the C++
for any ATen region.

## `ppy test` — differential conformance

```bash
ppy test [PATH]
```

Runs each program on all three paths and compares stdout, stderr, and exit
status.

## `ppy fmt` — formatting

```bash
ppy fmt [PATH] [--check]
```

Delegates to an installed `ruff` or `black`, otherwise normalizes through the
concrete syntax tree. `--check` writes nothing and exits non-zero if a file
would change. `ppy convert` already formats its output, with the built-in
normalizer so the result is identical on every machine.

## `ppy cache`

```bash
ppy cache status
ppy cache clean
ppy cache gc [--max-age-days N] [--max-bytes N]
```

A cache key covers the source digest, compiler version, optimization level,
directives, dependency hashes, and the fingerprints of the plugins the module
actually imports.

## `ppy clean`

Removes the whole cache directory. `ppy cache clean` empties it but leaves the
directory; neither touches an output directory named with `build -o`.

## `ppy doctor`

```bash
ppy doctor [--verbose]
```

Versions, project root, cache location, effective configuration, whether the
LLVM backend and native toolchain are usable, and each plugin's fingerprint.
Run it first when something compiles on one machine and not another.

## `ppy lsp`

```bash
ppy lsp [--root DIR]
```

LSP over stdio.

## Configuration

CLI options override `[tool.ppy]` in `pyproject.toml` for one invocation.

```toml
[tool.ppy]
python = ">=3.12,<3.15"
strict = true
opt-level = 2
cache-dir = ".ppy-cache"
dynamic-boundaries = "explicit"   # "explicit" | "deny"
build-execution = "deny"          # "deny" | "allow"
source-roots = ["src", "."]

[tool.ppy.python-backend]
enabled = true

[tool.ppy.llvm]
enabled = true
target = "native"
jit = true
lto = "thin"
cpython-api = "version-specific"

[tool.ppy.parallel]
enabled = true
threads = "auto"

[tool.ppy.inference]
interprocedural = true
write-local-annotations = true
implicit-any = "error"

[tool.ppy.diagnostics]
optimization-remarks = false

[tool.ppy.plugins.torch]
enabled = true
cpp-regions = true

[tool.ppy.plugins.jax]
enabled = true
allow-build-export = true
```

Two settings gate things that run project code or weaken analysis, so they
default to the safe value: `build-execution = "deny"` keeps build-time
execution (which JAX export needs) off until the project opts in, and
`dynamic-boundaries = "explicit"` requires a `ppy.dynamic` boundary for dynamic
features, with `"deny"` forbidding them outright.

## Exit codes

| code | meaning |
|---|---|
| `0` | success |
| `1` | ran and reported a problem: a failed check, a differential mismatch, files `fmt --check` would reformat |
| `2` | could not run: a missing path, an unreadable file, an unavailable backend |
