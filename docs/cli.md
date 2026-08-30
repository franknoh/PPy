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

Everything after `--` reaches the program as `sys.argv[1:]`. `ppy run`
keeps Python-integer semantics by default; `--unsafe` drops the overflow
guards on data arithmetic (64-bit wrap, bounds checks stay), and
`--safeguards {hoisted,inline,off}` names the guard mode outright.

## `ppy convert` — strict `.py` to `.ppy`

```bash
ppy convert PATH [--in-place] [--force] [--dry-run] [--promote-buffers]
```

Strict staticization: the input is expected to already be reasonably static,
and the output must be valid strict PPY. After planning the annotations, the
converter re-analyzes its own output in strict mode; whatever `ppy check`
would reject tomorrow — a dynamic feature without its `ppy.dynamic` boundary,
a parameter no annotation reaches, an unvouched decorator holding one back —
`ppy convert` refuses to produce today, with the checker's own explanation of
why and a pointer to `ppy migrate`.

`PATH` is a file or a directory; a directory is analyzed as one call graph, so
a function's types can come from call sites in other files.

| option | effect |
|---|---|
| `--dry-run` | print, write nothing |
| `--force` | overwrite an existing `.ppy` |
| `--in-place` | write the `.ppy` and remove the `.py` it came from |
| `--format` | hand the result to the project's formatter afterwards |
| `--promote-buffers` | declare read-only numeric list parameters as `Buffer[T]` and rewrite the values feeding them into `array.array` |
| `--hoist-classes {safe,aggressive,off}` | which classes may move above their uses; `safe` (default) moves only provably inert definitions |

There is deliberately no `--no-strict` here: a convert that can be asked not
to be strict is two pipelines wearing one name. The permissive pipeline is
`ppy migrate`.

Conversion is atomic: an error anywhere means no file is written anywhere, so
`--in-place` can never leave a tree half `.py` and half `.ppy`.

Without `--in-place` both `foo.py` and `foo.ppy` are left on disk, which a
project may not contain — the module would be ambiguous — so the converter
warns and `ppy check` then refuses.

In a converted module `import ppy` is placed before any sibling import, because
that import is what installs the loader those modules need.

## `ppy migrate` — permissive Python to PPY

```bash
ppy migrate PATH [--in-place] [--force] [--dry-run] [--diff] [--report FILE]
```

The migration tool for normal existing Python. It shares every flag and every
guarantee of `ppy convert` — deterministic output, atomic failure, one call
graph per directory — but it writes work-in-progress code on purpose: dynamic
features convert faithfully with an advisory (`E1504`) instead of an error,
functions whose annotations could not be written stay untouched, and `ppy
check` is the command that later insists on the remaining boundaries. The
natural workflow is

```bash
ppy migrate project/ --in-place   # rewrite what can be rewritten
ppy check project/                # see what manual migration remains
```

and iterating on the check findings until the project is strict PPY.

Before staticizing, migration runs its rewrite passes
(`ppy_compiler/migration/`), each of which proves its rewrite equivalent
before making it:

| pass | rewrite |
|---|---|
| `literal-attributes` | `setattr(o, "name", v)` → `o.name = v`; two-argument `getattr` → `o.name`; `delattr` → `del o.name` — constant, identifier-shaped names only, and only when the builtin still means the builtin |
| `static-imports` | `m = importlib.import_module("pkg.mod")` → `import pkg.mod as m`, under any spelling the lexical bindings resolve to importlib's importer (`il.import_module`, a `from importlib import import_module as imp` alias); an import that fed only rewritten calls is removed with them |
| `module-namespace-writes` | `globals()["NAME"] = value` in the module body → `NAME = value` (function scope differs, and stays) |

Afterwards the strict checker runs over the result once more — not to fail
the migration, but to classify what remains:

| classification | meaning |
|---|---|
| `AUTOFIXED` | a pass rewrote the site; nothing left to do |
| `REQUIRES_REWRITE` | valid Python the analysis cannot yet hold still; needs a manual rewrite |
| `DYNAMIC_BOUNDARY` | needs an explicit `ppy.dynamic` boundary to stay dynamic |
| `UNSUPPORTED` | `eval`/`exec`-class constructs no rewrite recovers |
| `OPTIMIZATION_OPPORTUNITY` | already valid, and one change away from a faster lowering |

After writing, migration re-analyzes its own final output in strict mode --
the same pass `ppy convert` gates on -- and classifies what the strict
language still rejects. A strict failure is not a migration failure: the
files land either way, and the report carries the verdict as `strict_ready`
and `strict_errors`, so

```bash
ppy migrate project/ --report migration.json
```

is by itself an accurate account of how far the migration got.

`--report FILE` writes the full accounting as JSON; the summary block prints
either way. `--diff` prints a unified diff of what migration would write and
writes nothing, which is the right first command on a project you have not
migrated before.

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
`ppy-bindings.json`, and a launcher. A build is a wrap-semantics artifact by
default — data arithmetic overflows at 64 bits like every native compiler's
output, while bounds checks stay; `--safe` keeps Python's integers
bit-for-bit instead, and the launcher always runs with exactly the mode it
was built with. It is a native executable that embeds the
interpreter and is `ppy run` in a compiled coat: it enters the same CLI, the
same pipeline, and the same guarded bindings, and only takes its machine code
from the library built next to it instead of a JIT (`ppy run --prebuilt
MANIFEST` is the spelled-out form). A manifest that names a library which has
gone missing is an error, never a silent fall back to interpretation; a
program with nothing native simply binds nothing, like `ppy run` would.
`-o` puts the artifacts somewhere other than the cache. With the JAX plugin
enabled and permitted, staged functions are exported here too.

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

## `ppy test`

```bash
ppy test [PATH] [--backend {differential,pytest}] [-- ARGS...]
```

`differential` (default) runs each program on all three paths and compares
stdout, stderr, and exit status.

`pytest` runs an ordinary test suite with the `.ppy` import hook already
installed and the project's source roots registered, so tests can import the
modules under test. Arguments after `--` reach pytest.

```bash
ppy test --backend pytest tests -- -k buffers -q
```

## `ppy lint`

```bash
ppy lint [PATH] [--backend {auto,pyright,pylint,ruff,mypy}] [--no-strict]
```

External tools key off the `.py` extension, so the sources are mirrored into a
staging tree, the tool runs there, and the paths in its output are mapped back
to the `.ppy` files you have. `auto` picks the first installed backend. A type checker runs in its strict
mode — `pyright` gets `typeCheckingMode = "strict"` — while a linter runs the
project's own rule selection, because "every rule there is" is not the same
kind of setting. `--all-rules` turns `ruff` up to `--select ALL` when that is
what you want; `--no-strict` turns a type checker down.

```bash
ppy lint --backend pyright src
```

The staging tree mirrors the project: every tool config at the root
(`pyproject.toml`, `pyrightconfig.json`, `.pylintrc`, `ruff.toml`,
`mypy.ini`, ...) and every plain `.py` module is copied in alongside the
staged sources, so imports resolve and the project's own configuration —
`extraPaths`, per-rule overrides, execution environments — keeps applying.

## `ppy fmt` — formatting

```bash
ppy fmt [PATH] [--check]
```

The built-in pass runs first and settles what an external formatter has no
opinion about: import grouping that keeps `ppy` ahead of a sibling module, and
a signature wrapped after annotation. An installed `ruff` or `black` then
applies the project's own style on top. `--check` writes nothing and exits
non-zero if a file would change. An installed formatter that fails — bad
config, crash, timeout — is an error (`E1802`), not a silent fallback.

`ppy convert` uses the built-in normalizer only, so a converted file is
byte-identical on every machine. Pass `--format`, or set
`[tool.ppy.convert] format = true`, to apply the project's style on top.

## `ppy cache`

```bash
ppy cache status
ppy cache clean
ppy cache gc [--max-age-days N] [--max-bytes N]
```

A cache key covers the source digest, compiler version, optimization level,
directives, dependency hashes, and the fingerprints of the plugins the module
actually imports. Nothing keys off modification time.

The native build is incremental per module. A rebuild with no source change
recompiles nothing and never initializes LLVM; a rebuild after editing one
module recompiles that module alone, and relinks only because its object
changed. Editing a module invalidates the modules that depend on it, because a
dependent's key includes the public summaries it compiled against.

| | |
|---|---|
| `lowered` | what lowering decided: the IR and each function's native ABI |
| `llvm` | the optimized IR |
| `native` | the object file, and the linked library keyed by its inputs |
| `python` | the generated Python |
| `jit` | guarded specializations |

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
Every key, with defaults, is in [config.md](config.md).

## Exit codes

| code | meaning |
|---|---|
| `0` | success |
| `1` | ran and reported a problem: a failed check, a differential mismatch, files `fmt --check` would reformat |
| `2` | could not run: a missing path, an unreadable file, an unavailable backend |
