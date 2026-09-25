# CLI reference

This page lists every `ppy` command and its options. The
[guide](guide/index.md) explains what each feature is for.

## Global options

These go before the subcommand.

| option | effect |
|---|---|
| `--version` | print the version and exit |
| `-q`, `--quiet` | only errors |
| `--color {auto,always,never}` | ANSI colour in diagnostics |
| `-O`, `--opt-level {0,1,2,3}` | override `[tool.ppy] opt-level` |
| `--no-strict` | report strict-mode errors that have a sound fallback as `W2010` warnings |

`-O` overrides the project default. It does not override a per-function
`@ppy.opt(n)`, which is a contract on that function.

## `ppy` and `ppy run`

Run a file.

```bash
ppy FILE.ppy [-- ARGS...]        # optimized Python backend
ppy run FILE.ppy [-- ARGS...]    # compile through LLVM, then run
```

Everything after `--` reaches the program as `sys.argv[1:]`.

`ppy run` keeps Python-integer semantics by default, as `ppy build` does.
Its options:

| option | effect |
|---|---|
| `--unsafe` | drop the overflow guards on data arithmetic (64-bit wrap); bounds checks stay |
| `--safeguards {hoisted,inline,off}` | name the guard mode outright |
| `--prover {off,z3}` | ask the solver to prove overflow guards away where the analysis allows it; overrides `[tool.ppy.llvm] prover`. See [Where a solver fits](internals/solver.md) |
| `--prebuilt MANIFEST` | run a built artifact (see [The launcher](#the-launcher)) |
| `--sanitize KINDS` | see [Sanitizers](#sanitizers-sanitize) |
| `--profile`, `--profile-out FILE`, `--pgo FILE` | see [Profile-guided optimization](#profile-guided-optimization-profile-pgo) |

### How the run cache works

The first `ppy run` of a program builds into the cache and then starts the
launcher. The second run is the launcher alone.

Before importing the compiler, `ppy run` names the artifact by everything
that could change it:

- the compiler's version and build
- the interpreter
- the configuration and flags
- every source under the project root
- the installed packages

When a directory by that name holds a manifest, `ppy_runtime` runs it the
way a built launcher would: no analysis, no LLVM, nothing the compiler
imports. An edit anywhere in the project gives a different name and a
fresh build. The per-module caches make that build cheap.

Three kinds of program stay on the in-process path every time, because the
launcher cannot serve them:

- one that specializes at runtime (`@ppy.jit`, `@ppy.specialize`)
- one with a fused NumPy kernel
- one that imports JAX

Each of these leaves a `needs-jit` note in its directory, so the next run
knows without analyzing. A program that imports torch is cached like any
other: its ATen regions are compiled into the artifact and the launcher
loads them from there.

## `ppy convert`

Convert strict `.py` to `.ppy`.

```bash
ppy convert PATH [--in-place] [--force] [--dry-run] [--format]
                 [--promote-buffers] [--hoist-classes {safe,aggressive,off}]
```

| option | effect |
|---|---|
| `--dry-run` | print, write nothing |
| `--force` | overwrite an existing `.ppy` |
| `--in-place` | write the `.ppy` and remove the `.py` it came from |
| `--format` | hand the result to the project's formatter afterwards |
| `--promote-buffers` | declare read-only numeric list parameters as `Buffer[T]` and rewrite the values feeding them into `array.array` |
| `--hoist-classes {safe,aggressive,off}` | which classes may move above their uses; `safe` (default) moves only provably inert definitions |

`PATH` is a file or a directory. A directory is analyzed as one call graph,
so a function's types can come from call sites in other files.

### Strictness

This is strict staticization. The input is expected to already be
reasonably static, and the output must be valid strict PPy.

After planning the annotations, the converter re-analyzes its own output in
strict mode. It refuses to produce anything that `ppy check` would reject
later, for example:

- a dynamic feature without its `ppy.dynamic` boundary
- a parameter no annotation reaches
- an unvouched decorator holding one back

The refusal comes with the checker's own explanation of why and a pointer
to `ppy migrate`.

`ppy convert` has no `--no-strict` option on purpose. A convert that can be
asked not to be strict would be two pipelines under one name. The
permissive pipeline is `ppy migrate`.

### Output files

Conversion is atomic. An error anywhere means no file is written anywhere,
so `--in-place` can never leave a tree half `.py` and half `.ppy`.

Without `--in-place`, both `foo.py` and `foo.ppy` stay on disk. A project
may not contain both, because the module would be ambiguous, so the
converter warns and `ppy check` then refuses.

In a converted module, `import ppy` is placed before any sibling import,
because that import installs the loader those modules need.

## `ppy migrate`

Migrate permissive Python to PPy.

```bash
ppy migrate PATH [--in-place] [--force] [--dry-run] [--diff] [--report FILE]
```

`ppy migrate` is the migration tool for ordinary existing Python. It shares
the flags and guarantees of `ppy convert`: deterministic output, atomic
failure, one call graph per directory. Options it adds:

| option | effect |
|---|---|
| `--diff` | print a unified diff of what migration would write, and write nothing. The right first command on a project you have not migrated before |
| `--report FILE` | write the full accounting as JSON. The summary block prints either way |

Unlike `convert`, it writes work-in-progress code on purpose:

- dynamic features convert faithfully with an advisory (`E1504`) instead of
  an error
- functions whose annotations could not be written stay untouched
- `ppy check` is the command that later insists on the remaining boundaries

The usual workflow:

```bash
ppy migrate project/ --in-place   # rewrite what can be rewritten
ppy check project/                # see what manual migration remains
```

Then iterate on the check findings until the project is strict PPy. On a
real codebase, start with the kernels rather than the whole repository.
[Migrating a real project](internals/migrating.md) says how to pick them
and how to read what comes back.

### Rewrite passes

Before staticizing, migration runs its rewrite passes
(`ppy_compiler/migration/`). Each pass proves its rewrite equivalent before
making it.

| pass | rewrite |
|---|---|
| `literal-attributes` | `setattr(o, "name", v)` → `o.name = v`; two-argument `getattr` → `o.name`; `delattr` → `del o.name`. Constant, identifier-shaped names only, and only when the builtin still means the builtin |
| `static-imports` | `m = importlib.import_module("pkg.mod")` → `import pkg.mod as m`, under any spelling the lexical bindings resolve to importlib's importer (`il.import_module`, a `from importlib import import_module as imp` alias); an import that fed only rewritten calls is removed with them |
| `module-namespace-writes` | `globals()["NAME"] = value` in the module body → `NAME = value` (function scope differs, and stays) |

### Classification of what remains

Afterwards the strict checker runs over the result once more. It does not
fail the migration. It classifies what remains:

| classification | meaning |
|---|---|
| `AUTOFIXED` | a pass rewrote the site; nothing left to do |
| `REQUIRES_REWRITE` | valid Python the analysis cannot yet hold still; needs a manual rewrite |
| `DYNAMIC_BOUNDARY` | needs an explicit `ppy.dynamic` boundary to stay dynamic |
| `UNSUPPORTED` | `eval`/`exec`-class constructs no rewrite recovers |
| `OPTIMIZATION_OPPORTUNITY` | already valid, and one change away from a faster lowering |

After writing, migration re-analyzes its own final output in strict mode
(the same pass `ppy convert` gates on) and classifies what the strict
language still rejects. A strict failure is not a migration failure. The
files land either way, and the report carries the verdict as `strict_ready`
and `strict_errors`. So this command alone gives an accurate account of how
far the migration got:

```bash
ppy migrate project/ --report migration.json
```

## `ppy check`

Static validation.

```bash
ppy check [PATH] [--remarks]
```

Checks types, effects, purity and native contracts, and the
dynamic-feature policy. Exits non-zero on any error.

| option | effect |
|---|---|
| `--remarks` | also print which functions lowered natively and which stayed boxed, with the reason |

## `ppy build`

Compile without running.

```bash
ppy build TARGET [--unsafe] [--host-cpu] [--standalone]
                 [--target TRIPLE] [--python-extension] [--library]
                 [--backend NAME] [-o DIR]
                 [--sanitize KINDS] [--pgo FILE]
                 [--report-opt] [--report-opt-json FILE]
ppy build --warm TARGET
ppy build foo.ppyir                  # from the IR alone; see `ppy emit`
```

| option | effect |
|---|---|
| `--backend NAME` | `llvm` (the default), `python`, or the name of an installed backend ([Backends](internals/backends.md)); `ppy doctor` lists them. See [`--backend NAME`](#-backend-name) |
| `-o DIR` | put the artifacts somewhere other than the cache |
| `--unsafe` | the same flag as on `run`: data arithmetic wraps at 64 bits, bounds checks stay |
| `--host-cpu` | compile for the build machine instead of the portable baseline |
| `--standalone` | a fully native executable with no CPython inside. See [`--standalone`](#-standalone) |
| `--target TRIPLE` | compile for another machine |
| `--python-extension` | write one importable module |
| `--library` | lay the exports out for a C consumer |
| `--sanitize KINDS` | see [Sanitizers](#sanitizers-sanitize) |
| `--pgo FILE` | see [Profile-guided optimization](#profile-guided-optimization-profile-pgo) |
| `--report-opt`, `--report-opt-json FILE` | see [Optimization report](#optimization-report-report-opt) |
| `--warm` | build ahead of time what `ppy run` and `import ppy` build on first use. See [`--warm`](#-warm) |

### What a build writes

`--backend llvm` writes:

- objects
- `libppy_<project>.so`
- `ppy-bindings.json`
- a launcher
- a C header declaring the public symbols, when a function is
  `@ppy.native.export`ed

A build keeps Python's integers bit-for-bit, the same as `ppy run`:
overflow is guarded and falls back to arbitrary precision. `--unsafe` is
the same flag it is on `run`. Data arithmetic wraps at 64 bits like other
native compilers' output, while bounds checks stay. The launcher always
runs with the mode it was built with.

`--host-cpu` compiles the object code for the machine doing the build
rather than the portable baseline. It is faster where the code vectorizes
(about a third on a matmul kernel). The artifact then requires a CPU with
the same instruction set, so the flag is off by default.

With the JAX plugin enabled and permitted, staged functions are exported
here too.

The artifact is complete:

- the manifest carries the full native ABI and a program section
- `generated/` holds the optimized Python the build wrote
- the compiled `METH_FASTCALL` boundary wrapper ships alongside the
  library, so a launched artifact crosses the same fast boundary a JIT run
  does, not a ctypes one

### The launcher

The launcher is a native executable that embeds the interpreter. It is
`ppy run` in a compiled coat: it enters the same CLI, the same pipeline,
and the same guarded bindings. The only difference is that it takes its
machine code from the library built next to it instead of from a JIT.

`ppy run --prebuilt MANIFEST` is the spelled-out form. It takes the same
runtime-only fast path: no analysis, manifest validation only.

The launcher loads `ppy_runtime`, binds, and executes. It does not discover
a project, parse, analyze, or touch LLVM, and it keeps working with
`ppy_compiler` uninstalled.

- A manifest that names a library which has gone missing is an error. It
  never silently falls back to interpretation.
- A program with nothing native binds nothing, as `ppy run` would.

What the launcher does pay for is starting the embedded interpreter and
importing the runtime: about 35 ms before the program begins. A cold
`ppy run` that compiles first takes about 0.7 s. A warm `ppy run` takes
this same launcher path, from the cache. `examples/bench_startup.py`
measures the categories separately, and `--standalone` (below) removes the
35 ms too.

### `--backend NAME`

```bash
ppy build foo.ppy --backend toy -o out/
```

For an installed backend, the build goes through these steps:

1. The backend is loaded with its `[tool.ppy.backends.NAME]` table and
   asked for its toolchain.
2. The modules go through the shared passes, then the backend's own passes
   and validation.
3. Its `build` writes into `-o` (or `<cache>/backends/NAME`).
4. What it wrote is listed, with any notes the backend adds.

A `.ppyir` target builds through the chosen backend too. The file is
canonical IR, so the backend's passes, validation, and build run over it
without a frontend above them.

`-O` reaches every backend, and `-o` names the directory a backend writes
its artifacts into. The LLVM road's options belong to the LLVM road. Every
backend except `llvm` (`python` and installed ones alike) refuses them with
`E1002` rather than accepting and ignoring them:

- `--unsafe`, `--sanitize`, `--pgo`, `--prover`
- `--host-cpu`, `--standalone`, `--python-extension`, `--library`
- `--report-opt`, `--report-opt-json`
- `--target`. What an installed backend builds for is
  `[tool.ppy.backends.NAME] target`, which reaches it through the backend
  context and the artifact's identity, where a compiler triple would not.

`--warm` builds the artifact that `ppy run` and `import ppy` take, which is
the LLVM backend's, so it is refused for any other backend.

The diagnostics are the same as `ppy emit`'s:

| code | cause |
|---|---|
| `E1903` | a backend that cannot be used |
| `E1801` | a missing toolchain |
| `E1802` | IR the backend refuses; also a builtin backend that only emits (`--backend c`) |
| `E1904` | a pass of the backend's that broke the IR |

`--backend python` builds the same optimized Python that `ppy FILE.ppy`
runs and publishes it to the project cache, where `import ppy` and
`ppy run` read it. It takes `-O` and the analysis flags. It refuses:

- the LLVM road's options above
- `--warm` (that artifact is the LLVM backend's)
- a `.ppyir` target (it reads PPy source, not canonical IR)
- `-o` (its modules go to the cache, which `[tool.ppy] cache-dir` and
  `PPY_CACHE_DIR` move)

Whichever backend you name, no other backend's road runs. Each option is
either understood by the chosen backend or refused by name.

### `--target`, `--python-extension`, `--library`

```bash
ppy build foo.ppy --target aarch64-linux-gnu     # objects, library, header for it
ppy build foo.ppy --python-extension -o dist     # dist/foo.so: `import foo`
ppy build lib.ppy --library -o dist              # dist/lib, dist/include, a .pc
```

**`--target TRIPLE`** (or `[tool.ppy.llvm] target`) compiles for another
machine. The objects carry that triple and data layout. The library is
linked with a toolchain for it: `<triple>-gcc` on the path, or clang with
`--target`. The header is the same.

The parts only the running interpreter can build, the CPython boundary
wrapper and the launcher, are left out with a note. The manifest names its
target, and a runtime on a different machine refuses it rather than
loading it.

Everything the compiler knows about a machine sits in one `TargetInfo`:
triple, CPU and features, pointer width, endianness, ABI, OS, object
format, data layout. There is no `sys.platform` to trip over elsewhere.
`ppy doctor` prints the host's.

**`--python-extension`** writes one importable module (`foo.so`, or
`foo.pyd` on Windows). It holds the module's native code, the generated
`METH_FASTCALL` boundary, and the module's own optimized Python.

`import foo` runs that Python, so each class, constant, and helper the
module defines exists. Each native-eligible function is bound to its
compiled code as it is defined, and keeps its Python definition as the
fallback a refused guard runs. Nothing is bound by name at runtime and no
manifest is read. The module still imports `ppy` for its markers, like the
source did. It is built against the interpreter that builds it.

**`--library`** lays the exports out for a C consumer:

- `lib/` with the shared library
- `include/` with the header
- `lib/pkgconfig/<name>.pc`
- the manifest describing the ABI

A module with no `@ppy.native.export` has nothing to package and says so
(`E1805`).

### `--warm`

```bash
ppy build --warm train/kernels/        # every .ppy under it
ppy build --warm train/reward.ppy
```

`--warm` builds ahead of time what `ppy run FILE` and `import ppy` build on
their first use, and stops. The artifact goes into the project cache, under
the key an import of that module will look for.

It takes no flags that would change the artifact (`--unsafe`, `--host-cpu`,
`--prover`, `-o`, and the rest are refused). An import takes none either,
so the project configuration is the only thing that names the build. What a
flag built, nothing would find.

Use it before a launch that starts many processes at once. Without it,
each rank of a `torchrun` imports the kernel, finds no build, and builds
one. The builds are identical and the first to finish is kept, so the
result is right, but each rank paid for it. With `--warm`, each rank finds
the build.

- A module that does not check clean is an error here (exit 1), rather than
  a note on each rank's stderr.
- A module that needs the in-process JIT is reported and skipped.

The key covers every source under the project root, so run it after the
last edit, not before.

### `--standalone`

```bash
ppy build --standalone app.ppy
```

`--standalone` links a fully native executable with no CPython inside.
`ldd` shows libc and nothing else, and startup is C startup (~a few ms).

It asks less of the machine than the hybrid build does. A C compiler is
enough, because there are no CPython headers to include and no libpython to
embed.

The reachable graph from `main` must be entirely native. That means:

- functions the hybrid path could lower
- `print` of integers, booleans, and string literals through C shims.
  Floats wait until native formatting can reproduce Python's
  shortest-round-trip repr exactly
- the memory the program makes for itself. `ppy.buffer[int](n)` is a zeroed
  native allocation here, and `ppy.scan[Buffer[int]](n)` is that allocation
  with the input read straight into it, because there is no `array.array`
  to build

Anything else is `E1803` with the path that reaches it. There is no
workaround.

There is no Python to fall back to. A failed guard prints the line CPython's
traceback would end with (`IndexError: list index out of range`) on standard
error and exits with status 1, as CPython does, instead of retrying in
Python. A standalone build
keeps the guards, like other builds, and never claims Python's integers it
cannot provide. `--unsafe` asks for wrap semantics outright, with almost no
guards left to fail.

`ppy.input` and `ppy.scan` of numbers, tuples of numbers, and number
buffers are the scanner itself, the same C the runtime's reader compiles.
Where Python would raise (the end of the input, a token that is not a
number, a value outside its declared width), the binary names the same
exception on standard error and stops. [Reading input](guide/input.md#native-code-and-standalone-binaries)
lists what reads natively.

Five of the six problems in `examples/15_algorithms` build this way once
their buffers come from `ppy.buffer` rather than `array.array`, and four of
those beat their C reference. Substring search is the one that cannot: its
text arrives as a token, and `ppy.read_token` has no standalone lowering
yet. `examples/15_algorithms/standalone/` holds the five, timed against the
other paths by the benchmark beside them.

## `ppy emit`

Print a compiler stage as text.

```bash
ppy emit ir foo.ppy                  # the canonical IR, to stdout
ppy emit ir foo.ppy -o foo.ppyir     # ... to a file
ppy emit ir src/ -o build/ir/        # one .ppyir per module
ppy emit linked-ir app.ppy           # the whole program: every module linked and optimized as one
ppy emit llvm-ir foo.ppy             # what the LLVM backend makes of it
ppy emit c foo.ppy                   # what the C backend makes of it: one C11 unit
ppy emit cpp foo.ppy                 # ... as C++17, exports behind extern "C"
ppy emit cuda foo.ppy                # the kernels, device functions, and launches as CUDA C++
ppy emit hip foo.ppy                 # ... as HIP C++
ppy emit nvvm-ir foo.ppy             # the kernels as LLVM IR for NVPTX
ppy emit ptx foo.ppy                 # ... as PTX (PPY_CUDA_ARCH names the architecture, sm_70 by default)
ppy emit c --header-only foo.ppy     # every function static inline in a header
ppy emit c --standalone prog.ppy     # the whole program from main(), shims and all
ppy emit c --standalone --unsafe --format app.ppy -o app.c
ppy emit cpp --standalone --unsafe --format app.ppy -o app.cpp
ppy emit c --standalone --unsafe --int-width 32 --format app.ppy -o app.c
ppy emit c --format foo.ppy          # ... laid out by clang-format (the project's .clang-format, or LLVM style)
ppy emit header foo.ppy              # the C declarations of the exports
ppy emit stablehlo foo.ppy           # the @ppy.xla.jit functions as StableHLO for XLA
ppy emit toy foo.ppy                 # a format an installed backend registers; ppy doctor lists them
```

| option | effect |
|---|---|
| `-o FILE` / `-o DIR` | write to a file, or one file per module into a directory |
| `--header-only` | (`c`, `cpp`) every function `static inline` under an include guard |
| `--standalone` | (`c`, `cpp`) the whole program from `main`, ending in a C `main` |
| `--format` | (`c`, `cpp`, `cuda`, `hip`, `header`) run the text through `clang-format` |
| `--unsafe` | (`c`, `cpp`) the build command's safeguard configuration. See [Unsafe standalone source](#unsafe-standalone-source) |
| `--int-width {32,64}` | integer width for unsafe standalone C/C++ source. See [`--int-width 32`](#-int-width-32) |

`--header-only`, `--standalone`, and `--format` belong to the builtin
kinds.

### Where the output goes

One rule applies to every kind:

- a single file with no `-o` prints to standard output
- `-o FILE` writes that file
- a directory target writes one file per module into the directory `-o`
  names, and refuses to guess without it

`ir` is the canonical IR after the shared passes ([The IR](internals/ir.md)).
`llvm-ir` is the optimized LLVM IR. The output is deterministic for one
input and configuration.

### Formats from installed backends

Any other kind is a format an installed backend registers
([Backends](internals/backends.md)). The backend is loaded and its
toolchain checked. The modules run through the shared passes and the
backend's own, are validated by it, and are written under the same rule:
text, or bytes for a format the backend declares binary.

A format is written per module (the default) or per program:

- A per-module format writes one artifact for each module. One module goes
  to standard output or `-o FILE`. A target that resolves to several
  modules needs `-o DIR`, since two artifacts are not one file and are
  never concatenated into one.
- A per-program format is one artifact for all the modules together.

Errors:

| code | cause |
|---|---|
| `E1903` | a format no backend emits, a format two backends claim, or a backend that cannot be loaded; with the reason and the formats there are |
| `E1801` | a backend whose toolchain is missing |
| `E1802` | IR the backend refuses, with what it cannot take and where |
| `E1904` | a backend pass that breaks the IR |

### C and C++

`c` and `cpp` are the C backend's reading of the same IR. Each module
becomes a translation unit in the internal ABI the runtime binds (atoms in,
result slots out, a status back). The unit contains:

- each `@native.export` behind its public C signature
- the overflow helpers and runtime shims the unit uses, and nothing else

So it compiles on its own with any C11 or C++17 compiler, and answers what
the LLVM road answers, fallbacks included.

`--header-only` makes every function `static inline` under an include
guard, for a header a program includes from any number of translation
units. A feature that needs state the process owns (reading standard input)
is refused there with `E1804` and its name.

`--standalone` takes a program the way `ppy build --standalone` does
(`main` and everything it reaches, all of it native) and ends the unit in a
C `main`, so the text is a whole program.

`header` is the declarations of a module's exports, the same text
`ppy build` writes beside a library.

The C is written to be read:

- loops are `while`, branches are `if`/`else`
- a local keeps its Python name
- a value read once is written where it is read

`--format` (for `c`, `cpp`, `cuda`, `hip`, and `header`) runs the text
through `clang-format`. It uses the project's `.clang-format` where there
is one, and LLVM style at four spaces and a hundred columns otherwise. A
missing `clang-format` is `E1802`.

### Unsafe standalone source

`--unsafe` applies only to `emit c` and `emit cpp`, using the build
command's safeguard configuration. With `--standalone`, it emits readable
source with:

- direct scalar/void returns
- one global `main`
- source function names
- C++17 module namespaces (nested for packages). C qualifies names only
  when they collide.

Project imports are linked after checking that every imported module has no
executable initialization and every reachable function is native.

**Arithmetic.** Unsafe standalone source selects native C/C++ integer
arithmetic in the IR, before presentation lowering.

- Addition, subtraction and multiplication use ordinary operators. Signed
  overflow is outside the portable input domain. This differs from the
  guaranteed 64-bit wrap of `build/run --unsafe` and non-standalone unsafe
  emission.
- Division retains Python floor rounding. When integer bounds prove that
  truncation gives the same answer, division and remainder use plain `/`
  and `%`. The proof follows acyclic branches and local assignments,
  discarding bounds when arithmetic could overflow. Loops retain the
  general correction.
- Expressions keep their required arithmetic width without redundant
  casts.

**Layout.** Functions appear before their callers where possible, with
prototypes for recursion and runtime callbacks. Local declarations move to
their first write when every use stays in that scope.

**Output.** Adjacent canonical print operations fuse within a block into
`printf`, with byte-exact literals, Python `True`/`False`, and `end`.
Explicit `print(..., flush=True)` emits `fflush(stdout)` (in C++,
`std::fflush(stdout)`). Omitting `flush` or passing `flush=False` does not
emit a flush.

**Input.** Integer `ppy.input` and `ppy.scan` use `scanf`: native
whitespace/token parsing, not Python's line parsing or underscore syntax.
Unsafe standalone source assumes every integer read succeeds and fits its
machine type, and emits a plain `scanf(...);` without checking the return
value.

!!! warning "Unchecked input"
    Malformed input, premature EOF, and out-of-range values are outside
    this mode's input contract. A failed read can leave the destination
    uninitialized.

Safe standalone emission and builds retain the existing scanner and
guards. Programs sharing a buffered scanner retain that scanner for all
reads.

**Headers.** The C++ spelling uses `<cstdint>`, `<cinttypes>`, `<cstdio>`
and `std::` stdio. Headers and helpers are included only when needed.
Aggregate-returning and runtime-managed functions retain their internal ABI
when required.

### `--int-width 32`

`--int-width 32` selects 32-bit integers for unsafe standalone C/C++
source.

- Signed integers are spelled `int`, unsigned integers `unsigned int`, and
  stdio uses `%d`/`%u`.
- The generated unit asserts that the target's `int` has the required
  32-bit range.
- This changes the IR types **before optimization**, including function
  parameters, return values, local slots, and arithmetic. It is not a
  spelling alias for 64-bit integers.
- All signed input and intermediate results must fit
  `[-2147483648, 2147483647]`. Out-of-range integer constants are rejected,
  and signed overflow follows native C/C++ semantics.
- Python floor rounding is still preserved for negative operands.

The 32-bit model currently accepts scalar programs with integer/bool
output, literal strings, and scalar integer input. Buffers, aggregates,
foreign functions, and runtime-managed operations have fixed ABIs and are
rejected with a diagnostic directing you to `--int-width 64`. Fixed-width
`i64`/`u64` values in the selected scalar program are narrowed too.

The default remains 64 bits, and `--int-width 64` selects it explicitly.
The flag requires `--standalone` and safeguards off (`--unsafe` or project
configuration). It does not change `run`, `build`, or ordinary module
emission.

### `.ppyir` files

`.ppyir` is the IR's on-disk form, public from 0.2.0 at schema 1.
`ppy build foo.ppyir` builds one without the Python that produced it. The
file carries its schema and dialect versions, each function's ABI, and its
source locations. The build is the passes, the LLVM backend, an object, a
library, and a manifest whose entries the runtime binds. A file from
another schema or a dialect this compiler lacks is refused with the reason.

A package builds as one program:

- a call from one module into another's native function is a declaration
  the linker answers with the definition
- the linked program is optimized as a whole: what Python never binds is
  internalized, small callees are inlined across the seam, dead private
  code goes
- one object comes out

`ppy emit linked-ir` shows that program.

## `ppy bind`

Generate bindings for foreign code.

```bash
ppy bind header foo.h                     # the bindings module, to stdout
ppy bind header foo.h -o foo.ppy          # ... to a file
ppy bind header foo.h --library foo -I include/
```

Clang reads the header. This is the real parser, through libclang
(`ppy-lang[bind]`), not a regular expression. The importer walks the
declarations the header itself makes:

| C declaration | becomes |
|---|---|
| a function | an `@ffi.bind` stub with typed parameters |
| a typedef of a scalar | an alias |
| an enum | its constants and an `int` alias |
| a struct of scalars | a dataclass |
| a `#define` of one number | a typed constant |

Parameter types map as follows: `int` is `ppy.i32`, `long` the target's
width, `double` `float`, `const T *` `native.const_ptr[T]`, `void *` a
byte pointer.

What has no PPy spelling yet is left out and listed by name at the end of
the module:

- a variadic function
- a function pointer
- an array or a struct passed by value
- an opaque struct
- a macro that is not one number

The module type-checks under `ppy check` and calls the library on every
path: ctypes under CPython, and directly in native code. A header Clang
cannot read is refused with its line (`E1806`).

## `ppy explain`

Explain why a function compiled the way it did.

```bash
ppy explain LOCATION
```

`LOCATION` is a `FILE:LINE`, a function name or qualname, or a diagnostic
code. For a function it reports:

- the semantic type, effects, and purity
- the backend decision
- the representation chosen for each parameter
- each library call's lowering, with its guards

## `ppy inspect`

Print generated artifacts.

```bash
ppy inspect TARGET [--backend {python,llvm}] [--ir]
ppy inspect TARGET --stage {analysis,ir,canonical,optimized,tensor,columnar,gpu,stablehlo,llvm}
```

By default it prints the optimized Python, including plugin rewrites. Read
it when a result differs from plain CPython.

| option | effect |
|---|---|
| `--ir` | print what the native path compiles: LLVM IR, then the C for the CPython-ABI wrappers, then the C++ for any ATen region |
| `--stage STAGE` | print the program as one stage of the compiler holds it (below) |

| stage | contents |
|---|---|
| `analysis` | what the checker knows of each function (type, effects, whether it is native, bound, a kernel, a coroutine, marked for XLA) |
| `ir` | the frontend's module before any pass |
| `canonical` | after canonicalization |
| `tensor` | after fusion and before the tensor dialect lowers to loops |
| `columnar` | the same point as `tensor`, for the modules holding columnar operations |
| `optimized` | what a backend receives |
| `gpu` | the device code alone |
| `stablehlo`, `llvm` | what those backends write |

## `ppy test`

```bash
ppy test [PATH] [--backend {differential,pytest}] [-- ARGS...]
```

| backend | effect |
|---|---|
| `differential` (default) | run each program on all three paths and compare stdout, stderr, and exit status |
| `pytest` | run an ordinary test suite with the `.ppy` import hook already installed and the project's source roots registered, so tests can import the modules under test. Arguments after `--` reach pytest |

```bash
ppy test --backend pytest tests -- -k buffers -q
```

## `ppy lint`

```bash
ppy lint [PATH] [--backend {auto,pyright,pylint,ruff,mypy}]
         [--all-rules] [--no-strict]
```

| option | effect |
|---|---|
| `--backend` | the tool to run; `auto` picks the first installed backend |
| `--all-rules` | turn `ruff` up to `--select ALL` |
| `--no-strict` | turn a type checker down |

External tools key off the `.py` extension. So `ppy lint` mirrors the
sources into a staging tree, runs the tool there, and maps the paths in its
output back to your `.ppy` files.

A type checker runs in its strict mode (`pyright` gets
`typeCheckingMode = "strict"`). A linter runs the project's own rule
selection, because "every rule there is" is a different kind of setting.
Use `--all-rules` when that is what you want.

```bash
ppy lint --backend pyright src
```

The staging tree mirrors the project. Each tool config at the root
(`pyproject.toml`, `pyrightconfig.json`, `.pylintrc`, `ruff.toml`,
`mypy.ini`, ...) and each plain `.py` module is copied in alongside the
staged sources. Imports resolve, and the project's own configuration
(`extraPaths`, per-rule overrides, execution environments) keeps applying.

## `ppy fmt`

Format source.

```bash
ppy fmt [PATH] [--check]
```

| option | effect |
|---|---|
| `--check` | write nothing and exit non-zero if a file would change |

The built-in pass runs first. It settles what an external formatter has no
opinion about: import grouping that keeps `ppy` ahead of a sibling module,
and a signature wrapped after annotation. An installed `ruff` or `black`
then applies the project's own style on top.

An installed formatter that fails (bad config, crash, timeout) is an error
(`E1802`), not a silent fallback.

`ppy convert` uses the built-in normalizer only, so a converted file is
byte-identical on every machine. Pass `--format`, or set
`[tool.ppy.convert] format = true`, to apply the project's style on top.

## `ppy cache`

```bash
ppy cache status
ppy cache clean
ppy cache gc [--max-age-days N] [--max-bytes N]
```

### Cache keys

A cache key covers:

- the source digest
- the compiler version
- the optimization level
- directives
- dependency hashes
- the fingerprints of the plugins the module imports

Nothing keys off modification time, with one exception. `convert` and
`migrate` make a record of the whole-project scan for `Final` and for
annotation materialization. That record is keyed by the compiler
fingerprint and the project root, and each file's entry in it is trusted
while the file's size and modification time still match. Avoiding hashing
every file of the project is why that record exists.

### Incremental builds

The native build is incremental per module.

- A rebuild with no source change recompiles nothing and never initializes
  LLVM.
- A rebuild after editing one module recompiles that module alone, and
  relinks only because its object changed.
- Editing a module invalidates the modules that depend on it, because a
  dependent's key includes the public summaries it compiled against.

### Cache contents

| | |
|---|---|
| `lowered` | what lowering decided: the IR and each function's native ABI |
| `llvm` | the optimized IR |
| `native` | the object file, and the linked library keyed by its inputs |
| `python` | the generated Python |
| `jit` | guarded specializations |
| `scan` | the whole-project scan's record: per file, the writes it makes on other modules and the annotation readers it holds |

## `ppy clean`

Removes the whole cache directory. `ppy cache clean` empties it but leaves
the directory. Neither touches an output directory named with `build -o`.

## `ppy doctor`

```bash
ppy doctor [--verbose]
```

Run it first when something compiles on one machine and not another. It
prints:

- versions, project root, cache location, and the effective configuration
- whether the LLVM backend and native toolchain are usable
- each plugin's fingerprint
- each backend, builtin and installed, with its toolchain status, the
  formats it emits, and its fingerprint

An installed backend that cannot be loaded is printed as `unusable` with
the reason. A name that two distributions register is reported.

## `ppy lsp`

```bash
ppy lsp [--root DIR]
```

Runs a language server (LSP) over stdio.

## Shared build and run options

### Sanitizers: `--sanitize`

```bash
ppy run --sanitize bounds,overflow foo.ppy
ppy build --sanitize pointer,alignment .
```

A sanitizer instruments the IR with checks the program did not ask for.

| kind | checks |
|---|---|
| `bounds` | every buffer index, whether or not a proof or a hoist removed the frontend's guard |
| `overflow` | every wrapping or proven `int` operation |
| `pointer` | that a pointer read or written through is not null |
| `alignment` | that a pointer is aligned for what it points at |

A failed check is not a fallback. The function returns a sanitizer status
and the boundary raises `ppy_runtime.binding.SanitizerFailure` naming the
kind and the function. A standalone program exits as it does for a guard.

`[tool.ppy.llvm] sanitize = ["bounds"]` configures the same.

`lifetime` and `alias` are refused with the reason: stack lifetime is held
by the verifier, and aliasing has no runtime check yet.

### Optimization report: `--report-opt`

`ppy build --report-opt` prints, per module:

- which functions became native and are bound to Python
- which stay in Python, and why
- how many guards a proof removed
- each remark the passes left, filed under a stable category
- what the build staged for XLA or a device

The categories:

- `function inlined`
- `tensor ops fused`
- `columnar ops fused`
- `parallel loop emitted`
- `GPU kernel emitted`
- `StableHLO region emitted`
- `bounds guard removed`
- `overflow guard proven unnecessary`
- `allocation stack-promoted`
- `generic specialization emitted`
- `sanitizer checks inserted`
- `dead code removed`

`--report-opt-json FILE` writes the same as JSON, keyed the same way, so a
tool can count by category across versions. A build guided by a profile
lists the profile first: the file, its runs, and each measured function's
calls, hotness, and argument kinds.

### Profile-guided optimization: `--profile`, `--pgo`

```bash
ppy run --profile foo.ppy            # runs, then writes foo.ppyprof
ppy run --profile --profile-out p.ppyprof foo.ppy -- args
ppy build --pgo foo.ppyprof foo.ppy
ppy run --pgo foo.ppyprof foo.ppy
```

**Recording.** A profiling run is a JIT run with counters. Each native
function counts its blocks and the taken edge of each conditional branch.
The boundary records what kinds of value each native function was called
with: an `int`, an `ndarray[float64;4x3]`, a
`DataFrame[a:int64,b:float64;1000 rows]`, a `list[400]`.

When the program ends, the counters are read back through the engine and
the profile is written as JSON. Per function it holds:

- its calls
- each block's count
- each branch's (taken, not taken)
- the loop trip counts these imply
- the argument kinds
- which generic it is an instance of

A profile already at the path is merged, so several runs (or several
inputs) add up. `runs` says how many.

**Using the profile.** A build with `--pgo` (or
`[tool.ppy.llvm] pgo = "foo.ppyprof"`, relative to the project root) reads
the counts back at the same point of the pipeline. A function whose graph
still matches what was measured is annotated:

- `hot`: at least a twentieth of the most-called function's calls, and at
  least two
- `cold`: never called
- or neither

Each conditional branch carries its weights, and each loop's back edge its
average trip count.

What the annotations change:

- The inliner inlines a hot callee at four times the usual budget, leaves a
  cold one alone, and leaves alone a call the profile never reached.
- LLVM receives the entry counts and branch weights as `!prof` metadata,
  which drive its block placement, its estimated trip counts, and its own
  unrolling and inlining heuristics.
- `hot`/`cold` become function attributes.

A function that changed since the profile was recorded is named by `W2009`
and built as if there were no profile. A missing or foreign profile is
refused (`E1002`).

A profile changes what is fast, not what is computed: the program's output
under `--pgo` is the program's output. The profile's content is part of
every cache key and of the warm run directory's name, so a new profile
means a new build.

## Configuration

CLI options override `[tool.ppy]` in `pyproject.toml` for one invocation.
[Configuration](reference/config.md) lists every key, with defaults.

## Exit codes

| code | meaning |
|---|---|
| `0` | success |
| `1` | ran and reported a problem: a failed check, a differential mismatch, files `fmt --check` would reformat |
| `2` | could not run: a missing path, an unreadable file, an unavailable backend |
