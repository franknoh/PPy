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
`--prover {off,z3}` asks the solver to prove overflow guards away where the
analysis allows it, overriding `[tool.ppy.llvm] prover`; see `docs/solver.md`.

The first `ppy run` of a program is a build into the cache followed by the
launcher; the second is the launcher alone. Before importing the compiler,
`ppy run` names the artifact by everything that could change it — the
compiler's version and build, the interpreter, the configuration and flags,
every source under the project root, the installed packages — and when a
directory by that name holds a manifest, `ppy_runtime` runs it the way a
built launcher would: no analysis, no LLVM, nothing the compiler imports.
An edit anywhere in the project is a different name and a fresh build,
whose per-module caches make it cheap. Three kinds of program stay on the
in-process path every time, because the launcher cannot serve them: one
that specializes at runtime (`@ppy.jit`, `@ppy.specialize`), one with a
fused NumPy kernel, and one that imports JAX; each leaves a `needs-jit`
note in its directory so the next run knows without analyzing. A program
that imports torch is cached like any other: its ATen regions are compiled
into the artifact and the launcher loads them from there.

## `ppy convert` — strict `.py` to `.ppy`

```bash
ppy convert PATH [--in-place] [--force] [--dry-run] [--format]
                 [--promote-buffers] [--hoist-classes {safe,aggressive,off}]
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

and iterating on the check findings until the project is strict PPY. On a
real codebase, start with the kernels rather than the repository —
[migrating.md](migrating.md) says how to pick them and how to read what
comes back.

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
ppy build TARGET [--safe] [--host-cpu] [--standalone]
                 [--target TRIPLE] [--python-extension] [--library]
                 [--backend {llvm,python}] [-o DIR]
ppy build --warm TARGET
ppy build foo.ppyir                  # from the IR alone; see `ppy emit`
```

`--backend llvm` (default) writes objects, `libppy_<project>.so`,
`ppy-bindings.json`, a launcher, and -- when a function is
`@ppy.native.export`ed -- a C header declaring the public symbols. A build is a wrap-semantics artifact by
default — data arithmetic overflows at 64 bits like every native compiler's
output, while bounds checks stay; `--safe` keeps Python's integers
bit-for-bit instead, and the launcher always runs with exactly the mode it
was built with. It is a native executable that embeds the
interpreter and is `ppy run` in a compiled coat: it enters the same CLI, the
same pipeline, and the same guarded bindings, and only takes its machine code
from the library built next to it instead of a JIT (`ppy run --prebuilt
MANIFEST` is the spelled-out form, and takes the same runtime-only fast
path — no analysis, manifest validation only). A manifest that names a library which has
gone missing is an error, never a silent fall back to interpretation; a
program with nothing native simply binds nothing, like `ppy run` would.
`--host-cpu` compiles the object code for the machine doing the build
rather than the portable baseline: faster where the code vectorizes (~20%
on a matmul kernel), and the artifact then requires a CPU with the same
instruction set, so it is off by default. `-o` puts the artifacts somewhere
other than the cache. With the JAX plugin
enabled and permitted, staged functions are exported here too.

The artifact is complete: the manifest carries the full native ABI and a
program section, `generated/` holds the optimized Python the build wrote,
and the compiled `METH_FASTCALL` boundary wrapper ships alongside the
library — a launched artifact crosses the same fast boundary a JIT run
does, not a ctypes one. The launcher loads `ppy_runtime`, binds, and executes — it does not
discover a project, parse, analyze, or touch LLVM, and it keeps working with
`ppy_compiler` uninstalled. What it does pay is starting the embedded
interpreter and importing the runtime — about 35 ms before the program
begins, against ~2 s for a cold `ppy run` that compiles first (a warm
`ppy run` takes this same launcher path, from the cache).
`examples/bench_startup.py` measures the categories separately, and
`--standalone` below removes that 35 ms too.

### `--target`, `--python-extension`, `--library`

```bash
ppy build foo.ppy --target aarch64-linux-gnu     # objects, library, header for it
ppy build foo.ppy --python-extension -o dist     # dist/foo.so: `import foo`
ppy build lib.ppy --library -o dist              # dist/lib, dist/include, a .pc
```

`--target TRIPLE` (or `[tool.ppy.llvm] target`) compiles for another
machine: the objects carry that triple and data layout, the library is
linked with a toolchain for it -- `<triple>-gcc` on the path, or clang
with `--target` -- and the header is the same. The parts only the running
interpreter can build, the CPython boundary wrapper and the launcher, are
left out with a note; the manifest names its target and a runtime on a
different machine refuses it rather than loading it. Everything the
compiler knows about a machine sits in one `TargetInfo` (triple, CPU and
features, pointer width, endianness, ABI, OS, object format, data
layout); there is no `sys.platform` to trip over elsewhere. `ppy doctor`
prints the host's.

`--python-extension` writes one importable module -- `foo.so`, `foo.pyd`
on Windows -- holding the module's native code, the generated
`METH_FASTCALL` boundary, and the module's own optimized Python. `import
foo` runs that Python, so every class, constant, and helper the module
defines exists, and each native-eligible function is bound to its
compiled code as it is defined, keeping its Python definition as the
fallback a refused guard runs. Nothing is bound by name at runtime and no
manifest is read. The module still imports `ppy` for its markers, like
the source did; it is built against the interpreter that builds it.

`--library` lays the exports out for a C consumer: `lib/` with the shared
library, `include/` with the header, `lib/pkgconfig/<name>.pc`, and the
manifest describing the ABI. A module with no `@ppy.native.export` has
nothing to package and says so (`E1805`).

### `--warm`

```bash
ppy build --warm train/kernels/        # every .ppy under it
ppy build --warm train/reward.ppy
```

Builds ahead of time exactly what `ppy run FILE` and `import ppy` build on
their first use -- the artifact in the project cache, under the key an
import of that module will look for -- and stops. It takes no flags that
would change the artifact (`--safe`, `--host-cpu`, `--prover`, `-o`, and
the rest are refused): an import takes none either, so the project
configuration is the only thing that names the build, and what a flag built
nothing would find. Its place is the step before a launch that starts many
processes at once. Without it, every rank of a `torchrun` imports the
kernel, finds no build, and builds one; the builds are identical and the
first to finish is the one kept, so the result is right, but each rank
paid for it. With it, every rank finds the build. A module that does not
check clean is an error here (exit 1) rather than a note on each rank's
stderr, and a module that needs the in-process JIT is reported and skipped.
The key covers every source under the project root, so run it after the
last edit, not before.

### `--standalone`

```bash
ppy build --standalone app.ppy
```

Links a fully native executable with **no CPython inside** — `ldd` shows
libc and nothing else, and startup is C startup (~a few ms). It asks less of
the machine than the hybrid build does: a C compiler is enough, because
there are no CPython headers to include and no libpython to embed. The reachable
graph from `main` must be entirely native: functions the hybrid path could
lower, plus `print` of integers, booleans, and string literals through C
shims (floats wait until native formatting can reproduce Python's
shortest-round-trip repr exactly), plus the memory the program makes for
itself — `ppy.buffer[int](n)` is a zeroed native allocation here and
`ppy.input[Buffer[int]](n)` is that allocation with the input read straight
into it, because there is no `array.array` to build. Anything else is `E1803` with the path
that reaches it, never a workaround. There is no Python to fall back to, so
in `--safe` mode a failed guard aborts with a message instead of retrying
in Python; `ppy.input[int]()` reads through the C support rather than the
runtime's reader, and answers 0 at end of input because there is no
exception to raise; the default wrap-semantics build has almost no guards left to
fail.

Five of the six problems in `examples/15_algorithms` build this way once
their buffers come from `ppy.buffer` rather than `array.array`, and four of
those beat their C reference. Substring search is the one that cannot: its
text arrives as a token, and `ppy.read_token` has no standalone lowering yet.
`examples/15_algorithms/standalone/` holds the five, timed against every
other path by the benchmark beside them.

## `ppy emit` — a compiler stage as text

```bash
ppy emit ir foo.ppy                  # the canonical IR, to stdout
ppy emit ir foo.ppy -o foo.ppyir     # ... to a file
ppy emit ir src/ -o build/ir/        # one .ppyir per module
ppy emit llvm-ir foo.ppy             # what the LLVM backend makes of it
ppy emit c foo.ppy                   # what the C backend makes of it: one C11 unit
ppy emit cpp foo.ppy                 # ... as C++17, exports behind extern "C"
ppy emit cuda foo.ppy                # the kernels, device functions, and launches as CUDA C++
ppy emit hip foo.ppy                 # ... as HIP C++
ppy emit c --header-only foo.ppy     # every function static inline in a header
ppy emit c --standalone prog.ppy     # the whole program from main(), shims and all
ppy emit header foo.ppy              # the C declarations of the exports
ppy emit stablehlo foo.ppy           # the @ppy.xla.jit functions as StableHLO for XLA
```

One rule for every kind: a single file with no `-o` prints to standard
output, `-o FILE` writes that file, and a directory target writes one file
per module into the directory `-o` names (and refuses to guess without
it). `ir` is the canonical IR after the shared passes (`docs/ir.md`);
`llvm-ir` is the optimized LLVM IR. The output is deterministic for one
input and configuration.

`c` and `cpp` are the C backend's reading of the same IR: a translation
unit per module in the internal ABI the runtime binds (atoms in, result
slots out, a status back), every `@native.export` behind its public C
signature, the overflow helpers and runtime shims the unit actually uses,
and nothing else -- so it compiles on its own with any C11 or C++17
compiler, and answers exactly what the LLVM road answers, fallbacks
included. `--header-only` makes every function `static inline` under an
include guard, for a header a program includes from any number of
translation units; a feature that needs state the process owns (reading
standard input) is refused there with `E1804` and its name.
`--standalone` takes a program the way `ppy build --standalone` does --
`main` and everything it reaches, all of it native -- and ends the unit in
a C `main`, so the text is a whole program. `header` is the declarations
of a module's exports, the same text `ppy build` writes beside a library.

`.ppyir` is the IR's on-disk form, experimental in 0.2.0, and `ppy build
foo.ppyir` builds one without the Python that produced it: the file
carries its schema and dialect versions, every function's ABI, and its
source locations, so the build is the passes, the LLVM backend, an object,
a library, and a manifest whose entries the runtime binds. A file from
another schema or a dialect this compiler lacks is refused with the reason.

## `ppy bind` — bindings for foreign code

```bash
ppy bind header foo.h                     # the bindings module, to stdout
ppy bind header foo.h -o foo.ppy          # ... to a file
ppy bind header foo.h --library foo -I include/
```

Clang reads the header -- the real parser, through libclang
(`ppy-lang[bind]`), never a regular expression -- and the importer walks
the declarations the header itself makes. A function becomes an
`@ffi.bind` stub with typed parameters (`int` is `ppy.i32`, `long` the
target's width, `double` `float`, `const T *` `native.const_ptr[T]`,
`void *` a byte pointer); a typedef of a scalar an alias; an enum its
constants and an `int` alias; a struct of scalars a dataclass; a `#define`
of one number a typed constant. What has no PPY spelling yet -- a
variadic function, a function pointer, an array or a struct passed by
value, an opaque struct, a macro that is not one number -- is left out
and listed by name at the end of the module. The module type-checks under
`ppy check` and calls the library on every path, ctypes under CPython and
directly in native code. A header Clang cannot read is refused with its
line (`E1806`).

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
ppy lint [PATH] [--backend {auto,pyright,pylint,ruff,mypy}]
         [--all-rules] [--no-strict]
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
actually imports. Nothing keys off modification time, with one exception: the
record of the whole-project scan that `convert` and `migrate` make for `Final`
and for annotation materialization is keyed by the compiler fingerprint and
the project root, and each file's entry in it is trusted while the file's size
and modification time still match. Hashing every file of the project is what
that record exists to avoid.

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
| `scan` | the whole-project scan's record: per file, the writes it makes on other modules and the annotation readers it holds |

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
