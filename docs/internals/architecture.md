# Architecture

This page is the map of the compiler: the stages a `.ppy` source passes
through, the package that owns each stage, and the rules that hold across
them. The diagram below is the pipeline from parsing to the backends.

```
.ppy sources
     │  frontend/          parse (CPython grammar), module graph, .py/.ppy shadowing
     ▼
symbol tables            analysis/symbols       declarations, imports, fields, directives
     ▼
type & effect analysis   analysis/checker       flow typing, refinements, purity fixpoint
     ▼
     ├── driver/convert + analysis/inference    call-site fixpoint → ConversionPlan
     ├── driver/rewrite                          the plan applied to source, through a CST
     ├── migration/                              rewrite passes + report behind `ppy migrate`
     ├── opt/                                   AST passes for the Python backend
     ├── lowering/                              typed AST → canonical IR (ir/)
     │      driver/ir_pipeline                  the shared passes (ir/transforms); ir/linker joins modules
     │      ├── backend/llvm/                   IR → LLVM IR → wrapper → link/JIT
     │      ├── backend/c/                      IR → C11, C++17, CUDA, HIP source
     │      ├── backend/nvvm/                   IR → NVVM IR → PTX, launched by ppy_runtime.cuda
     │      ├── backend/stablehlo/              IR → StableHLO, run by ppy_runtime.xla
     │      └── an installed backend            IR → whatever it makes (ppy.backends entry point)
     └── lsp/, driver/explain                   the same analysis, served interactively
```

Analysis produces data. Only `driver/rewrite.py` touches source text, and only
through the `ConversionPlan` that `driver/convert.py` filled in. Nothing in
`analysis/` knows the output is text, and nothing in the backends re-derives
what the checker already proved.

The backends, the compiler's own and an installed one alike, read the
canonical IR after the shared passes and nothing above it
([Backends](backends.md)).

## Modules

Each package and what it holds:

| package | contents |
|---|---|
| `ppy` (runtime) | the import hook and the inert directives/markers, and the modules a program writes against (`native`, `ffi`, `simd`, `cpu`, `atomic`, `concurrent`, `autodiff`, `cuda`, `hip`, `xla`, `aio`), each a Python implementation that answers the same as the compiled one. This is all a plain CPython run ever loads. |
| `ppy_runtime` | everything a *built artifact* needs at launch: the native ABI as data (`abi`), the guarded binding trampolines (`binding`), generated-module identity and execution (`generated`, `execute`), the binder protocol (`dispatch`), and the manifest-driven launch path (`manifest`, `launch`), and the runtimes native code calls into: the collections runtime (`collections.c`, `strings.c`, `exceptions.c`, with `stdlib.c` and `random.c` for the standard library and `pyio.c` for held output and calls into Python, compiled once per machine into the user cache), `effects` (the Python side of held output and barriers), `aio` (the epoll loop, one C file compiled once), `cuda` (the driver API through ctypes, launching PTX), `xla` (the PJRT bridge), `exported` and `regions` (staged artifacts and compiled torch regions). The hard rule: this package never imports `ppy_compiler`. Uninstalling the compiler must not break a built application, and a test keeps that true by poisoning the compiler and running a launcher. |
| `frontend/` | source loading, the module graph, ambiguity detection (`E1003`). |
| `migration/` | the `ppy migrate` layer over the shared conversion engine: deterministic rewrite passes (`pipeline`, `dynamic`, `globals`) that prove each rewrite equivalent before making it, and the classified report (`report`) that says what remains. |
| `analysis/` | `results` (what analysis produced: the types every other package reads), `symbols` (declarations), `checker` (types, refinements, effects), `binding` (one shared call-argument binder), `lexical` (point-sensitive name resolution: what a name means at each statement, shared by decorator identity, reflection, and the write index), `aliasing` (flow-sensitive local alias analysis: mutation and escape resolve through what a name may refer to rather than its spelling), `inference` (staged evidence/generalization fixpoint with a convergence guard), `decorators` (what each known decorator does, that unknown means opaque, and the shared `class_construction` facts behind both strict class checking and safe hoisting), `global_writes` (scope-aware project-wide write index behind `Final`), `reflection` (who reads annotations at runtime, blocking their materialization), `codec` (exact-inverse serialization of analysis facts for the cache), `render` (types back to annotation source). |
| `lowering/` | the frontend of the native road: `ast_to_ir` turns a typed, effect-checked function into canonical IR (guards spelled, overflow and rounding on the operation, cross-module calls as declarations), `abi` the native signatures. |
| `ir/` | the typed canonical IR every backend lowers: `model` (modules, functions, blocks, SSA values with use lists), `types`, `dialect` (the registry and `OpSpec`), `dialects/` (core, math, simd, cpu, atomic, concurrency, parallel, layout, tensor, linalg, fft, special, sparse, columnar, arrow, gpu, async, prof), `verify`, `printer`/`parser`/`codec` (`.ppyir`), `pattern` (rewrites to a fixed point), `passes` (the pass manager with analyses and stages), `transforms/` (canonicalize, simplify-cfg, dce, promote-slots, tensor fusion and lowering, columnar lowering, parallel lowering, async lowering, autodiff, sanitize, profile, whole-program), `linker` (modules into one program). See [The IR](ir.md). |
| `opt/` | AST-level passes: constant folding, inlining, LICM, loop transforms; used by the Python backend and as pre-lowering cleanup. |
| `backend/python/` | runs optimized AST under CPython with the loader installed. |
| `target` | `TargetInfo`: the triple, CPU and features, pointer width, endianness, ABI, OS, object format, and data layout of the machine a build is for; the host is one target among others, and nothing else consults `sys.platform`. |
| `bind/` | `ppy bind header`: a C header read through libclang, written as `ppy.ffi` bindings. |
| `backend/c/` | the source backends: `emit` reads the canonical IR and writes one C11 or C++17 translation unit (or a header-only form), and with `gpu` the CUDA or HIP spelling of the same IR, kernels and launches included; `runtime` holds the C shims a standalone program links (the LLVM standalone build compiles the same table). |
| `backend/nvvm/` | the device backend: IR → NVVM IR → PTX through LLVM's NVPTX target with libdevice linked in; `ppy_runtime.cuda` launches it. |
| `backend/stablehlo/` | IR → StableHLO for `@ppy.xla.jit` functions; `ppy_runtime.xla` compiles and runs it through PJRT. |
| `backend/` | `base` (the interface a backend implements: formats, passes, validation, emit, build, toolchain, fingerprint), `registry` (the builtin backends and the installed ones, found through the `ppy.backends` entry-point group and loaded when asked for), `builtin` (the compiler's own backends described through the same interface). See [Backends](backends.md). |
| `backend/llvm/` | `ir_pipeline` (the LLVM road over the shared passes, which live in `driver/ir_pipeline`) and `from_ir` (canonical IR → LLVM IR, dialect by dialect; `lowering` keeps the native ABI and eligibility rules, `lowering_cache` what a build reuses), `wrapper` (generated CPython-ABI entry points, `METH_FASTCALL`, GIL release), `fusion` (NumPy elementwise loops), `specialize`/`jit` (guarded runtime specialization), `parallel` (the worker pool), `link` (objects → shared library, for the host or a `--target`), `extension`/`packaging` (`--python-extension`, `--library`). |
| `plugins/` | numpy, torch, jax, pydantic, uvicorn. See [Plugins](plugins.md). |
| `cache/` | the content-addressed store (SQLite) and key construction. |
| `driver/` | CLI, pipeline orchestration, `ir_pipeline` (the canonical IR of a project after the shared passes, which is the one road every backend takes, and the backend boundary after it), `convert` (what to write) and `rewrite` (writing it) either side of `plan`, fmt, lint, test, explain. |
| `lsp/` | the language server, on the same analysis. |

## The three-path invariant

Plain CPython, the Python backend, and the LLVM backend must produce the same
answer. A guard that fails at runtime falls back to the Python body instead of
answering differently.

The invariant is checked by tests. `examples/run_all.py` runs all
@@EXAMPLE_PROGRAMS@@ example programs on all three paths and diffs the output,
and the test suite does the same per feature.

## Cache and incremental builds

The store is content-addressed: a key is a blake2b digest over the source, the
compiler version and schema, the opt level, the active directives, the
dependency digests, and the fingerprints of every plugin whose library the
module imports. Nothing is invalidated by time: a key either describes the
artifact or misses.

The LLVM path caches per stage, so a rebuild does only what changed:

| stage | keyed by | on a no-change rebuild |
|---|---|---|
| lowering (canonical IR, LLVM IR, ABI decisions, remarks) | module source + deps + opt level + sanitizers + profile | reused; LLVM never loads |
| the program's object | every module's lowering key | reused; nothing links |
| linked library | the set of object keys | reused |

Editing one file re-lowers that file, links the program again, and touches nothing else;
a change in a module's interface invalidates its dependents through the
dependency digests.

### Damage and recovery

The store holds optimization state and nothing else: every artifact in it can be
recomputed from the source it came from.

- A damaged SQLite index is moved aside as `index.sqlite.corrupt-<timestamp>`,
  rebuilt empty, and reported once as `W2101`. Compilation continues with
  cache misses.
- Where even a fresh index cannot be written, the store works in memory and
  every lookup is a miss.
- Recording an artifact spans two tables and runs in one transaction, so a
  reader never sees a row whose dependencies have not landed.

[Compatibility](../reference/compatibility.md) states the contract.

## Runtime specialization

`@ppy.jit` compiles a version specialized to the argument classes actually
seen, guarded on exact class identity. A guard miss runs the Python body and
may compile another specialization. All-scalar `@dataclass` value classes are
flattened to scalar SSA values at the ABI, with reads guarded on exact class.

## Code generation

JIT-compiled code targets the host CPU. Its name and feature set are handed
to LLVM, so AVX2/FMA and friends are on where the machine has them. This
measured a third faster on a 384×384 matmul kernel; memory-bound kernels are
unchanged. It costs nothing, because JIT code never leaves the machine that
made it.

Emitted objects, built artifacts, and standalone binaries stay on the
portable baseline instead. They may run on another machine, like a C
compiler's output without `-march=native`.

`ppy build --host-cpu` (`[tool.ppy.llvm] host-cpu`) trades that away on
purpose: the artifact gets this machine's instruction set and faults on an
older one. The choice is part of the cache key, along with the host CPU's
name and features, so a baseline object is never handed to a host build or
carried between machines.

## Reading input

`ppy/_io.py` is a runtime-only reader: a small C scanner over file
descriptor 0, compiled on first use into the user cache and bound through
ctypes. It is exposed as:

- `ppy.input[T]()` for lines
- `ppy.scan[T]()` for tokens
- the lower-level `ppy.read_ints` / `ppy.read_token`

`ppy.buffer[T](n)` beside it is the allocation both a CPython run and a
standalone binary understand.

The reader lives in the `ppy` package, outside the compiler, because it is
useful on every path, plain CPython included. It degrades to a pure-Python
implementation where no C compiler exists. The checker types `ppy.input[T]()`
and `ppy.scan[T]()` from their subscript, the way it types `ppy.check[T]` and
`ppy.assume[T]`.

## The boundary

Python callers cross into native code through a generated `METH_FASTCALL`
wrapper that parses, guards, calls, and boxes in C. The wrapper also holds the
Python implementation, so a refused guard is a C-to-Python call instead of a
`NotImplemented` bounced through a Python frame. No Python code stands on the
call path (`@ppy.jit` keeps a thin Python watcher only while it is still
learning which argument shapes repeat).

The same wrapper carries what used to need Python frames:

- Lists, dicts, sets, and objects of the project's classes cross in C
  (`backend/llvm/crossing.c`), from type tables emitted for each signature.
  An object reached twice becomes one handle.
  - A call that writes through no parameter and takes no objects reads its
    containers in place: its lists, inner lists included, are laid out in
    an arena the call owns (`px_chunk`, kept between calls) and marked
    immortal for the call; a string is the Python string's own UTF-8,
    borrowed; a dict or set is filled without hashing and indexed at its
    first lookup. A row or string handed back is the caller's object. When
    the call answers, `px_settle` counts the call's references, and
    anything still held past the call (a string a cached callee kept) is
    copied into memory of its own.
  - A call that writes copies each argument into a runtime handle. Every
    entry records whether a written parameter brought it in, and only those
    are copied back into the caller's objects. Of a list of numbers only
    the elements whose words changed are set again; an element that did not
    change keeps its Python object.
  - Objects of a class that may stay resident (`CrossingClass.resident`,
    decided in `lowering/collections.py`; the wrapper checks the type again
    in `px_plain_type`) keep their native records between calls from the
    second crossing on. One world per process (`px_world`), shared by every
    wrapper module through a capsule (`collection_boundary._share_world`),
    maps each object, handle, and `__dict__` to its record. The world holds
    a weak reference to each object, not its dict, and drops the records of
    dead objects in a purge once half its entries are dead. Handles it holds
    are taken off their thread's heap list (`ppy_coll_adopt`), so neither
    the sweep after a failed call nor the cycle collector touches them.
  - Python writes are seen through `PyDict_Watch` (3.12+) on each resident
    object's `__dict__`: a change marks the object stale, and the next
    resident call reads its fields again (`px_refresh`). The boundary's own
    writes back set an expected dict first, so they mark nothing. Each
    resident class is watched with `PyType_Watch`; after a change its fields
    are looked up again, and a data descriptor named like a field ends
    residency for the process.
  - Native writes are found by the lowering: after each field store
    (`_field_store`) it tests a header word that marks a resident record,
    and the first store puts the record on a runtime list
    (`ppy_coll_touch`). When the call answers, `px_drain` sets the changed
    fields of every record on the list on its Python object; an object
    native code made and linked in is created in Python then (through
    `object`'s `tp_new` where that is the class's only built-in base) and
    admitted. A call that falls back or raises leaves what it wrote stale,
    so the next call reads it from Python. The world is used only with the
    GIL held, and a call made while it is busy (a finalizer that runs during
    a crossing) copies. A free-threaded build has no world, and neither does
    a process with `PPY_RESIDENT=0`.
  - A number that may be `None` crosses as the number's atom and a byte
    (`NativeParam.kind = "optional"`, `NativeSignature.optional` for a
    result); in a field or a container element it is two words, the number
    then a flag (`PX_OPTIONAL`). A `str | None` is a string handle, null for
    `None`, and as a text parameter a length of -1.
- A function that draws from `random` saves the generator's state in the
  wrapper with one `memcpy` and puts it back only where Python runs the call
  again (`binding.attach_random` hands the wrapper the state's address once
  per module).
- A function that prints, reads input, or calls into Python enters and
  leaves its call, commits held output, and drops it on a fallback in C
  (`wrapper_effects.c`); it calls back into Python only for a raise after a
  barrier or a write that failed.
- A function that reads settled module globals keeps a Python frame that
  reads them, which then calls the C entry (`binding._bind_globals`). The
  same frame reads a nested function's shared variables from the
  `__closure__` of the function object it was called through, and packs
  the positions a `*args` parameter takes into one list.
- A call with keywords, or with defaults left out, is bound in C: the
  wrapper matches `kwnames` against the Python function's parameter names
  (read from its code object when the wrapper is bound) and fills what is
  left out from its `__defaults__` and `__kwdefaults__`. A call that does
  not bind that way goes to the Python function, which raises CPython's
  `TypeError`; a guard that refuses the bound arguments hands Python the
  call as it was spelled.

A two-int call costs 35 ns through the wrapper, against 33 ns for
CPython's own call. [What a call costs](../guide/native-lowering.md#what-a-call-costs)
has the measured table for each shape, from `examples/bench_boundary.py`.

Built artifacts ship the compiled wrapper and bind through it at launch. The
ctypes trampoline remains only as the fallback where no C toolchain exists
(`W2004` says so once).

## Threads

Generated wrappers release the GIL around a native call whose body loops
or calls another function (`Py_BEGIN_ALLOW_THREADS`), so `@ppy.native`
functions scale on threads:
measured 1.95× on two threads against 0.98× for the same code on plain
CPython (`examples/28_threads`). A short straight-line body keeps the GIL,
since dropping and retaking it costs about 20 ns, and so does a function
that prints, reads, or calls into Python, and a call whose objects are
resident (the world of records is not locked; the GIL guards it).

`@ppy.parallel` loops run on a process-wide worker pool sized by
`[tool.ppy.parallel] threads`.
