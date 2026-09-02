# Architecture

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
     ├── backend/llvm/                          lowering → LLVM IR → wrapper → link/JIT
     └── lsp/, driver/explain                   the same analysis, served interactively
```

Analysis produces data; only `driver/rewrite.py` touches source text, and only
through the `ConversionPlan` that `driver/convert.py` filled in. Nothing in
`analysis/` knows the output is text, and nothing in the backends re-derives
what the checker already proved.

## Modules

| package | contents |
|---|---|
| `ppy` (runtime) | the import hook and the inert directives/markers. This is all a plain CPython run ever loads. |
| `ppy_runtime` | everything a *built artifact* needs at launch: the native ABI as data (`abi`), the guarded binding trampolines (`binding`), generated-module identity and execution (`generated`, `execute`), the binder protocol (`dispatch`), and the manifest-driven launch path (`manifest`, `launch`). The hard rule: this package never imports `ppy_compiler` — uninstalling the compiler must not break a built application, and a test keeps that true by poisoning the compiler and running a launcher. |
| `frontend/` | source loading, the module graph, ambiguity detection (`E1003`). |
| `migration/` | the `ppy migrate` layer over the shared conversion engine: deterministic rewrite passes (`pipeline`, `dynamic`, `globals`) that prove each rewrite equivalent before making it, and the classified report (`report`) that says what remains. |
| `analysis/` | `results` (what analysis produced — the types every other package reads), `symbols` (declarations), `checker` (types, refinements, effects), `binding` (one shared call-argument binder), `lexical` (point-sensitive name resolution: what a name means at each statement, shared by decorator identity, reflection, and the write index), `aliasing` (flow-sensitive local alias analysis: mutation and escape resolve through what a name may refer to, not its spelling), `inference` (staged evidence/generalization fixpoint with a convergence guard), `decorators` (what each known decorator does, that unknown means opaque, and the shared `class_construction` facts behind both strict class checking and safe hoisting), `global_writes` (scope-aware project-wide write index behind `Final`), `reflection` (who reads annotations at runtime, blocking their materialization), `codec` (exact-inverse serialization of analysis facts for the cache), `render` (types back to annotation source). |
| `opt/` | AST-level passes: constant folding, inlining, LICM, loop transforms; used by the Python backend and as pre-lowering cleanup. |
| `backend/python/` | runs optimized AST under CPython with the loader installed. |
| `backend/llvm/` | `lowering` (AST → LLVM IR), `wrapper` (generated CPython-ABI entry points, `METH_FASTCALL`, GIL release), `fusion` (NumPy elementwise loops), `specialize`/`jit` (guarded runtime specialization), `parallel` (the worker pool), `link` (objects → shared library). |
| `plugins/` | numpy, torch, jax, pydantic, uvicorn — see [plugins.md](plugins.md). |
| `cache/` | the content-addressed store (SQLite) and key construction. |
| `driver/` | CLI, pipeline orchestration, `convert` (what to write) and `rewrite` (writing it) either side of `plan`, fmt, lint, test, explain. |
| `lsp/` | the language server, on the same analysis. |

## The three-path invariant

Plain CPython, the Python backend, and the LLVM backend must produce the same
answer; a guard that fails at runtime falls back to the Python body rather than
ever answering differently. The invariant is enforced, not assumed:
`examples/run_all.py` runs all 39 example programs on all three paths and
diffs the output, and the test suite does the same per feature.

## Cache and incremental builds

The store is content-addressed: a key is a blake2b digest over the source, the
compiler version and schema, the opt level, the active directives, the
dependency digests, and the fingerprints of every plugin whose library the
module imports. Nothing is ever invalidated by time — a key either describes
the artifact or misses.

The LLVM path caches per stage, so a rebuild does only what changed:

| stage | keyed by | on a no-change rebuild |
|---|---|---|
| lowering (IR + ABI decisions) | module source + deps + opt level | reused; LLVM never loads |
| object code | the lowering key | reused |
| linked library | the set of object keys | reused |

Editing one file re-lowers that file, relinks, and touches nothing else;
a change in a module's interface invalidates its dependents through the
dependency digests.

The store is optimization state and nothing else: every artifact in it can be
recomputed from the source it came from. A damaged SQLite index is moved aside
as `index.sqlite.corrupt-<timestamp>`, rebuilt empty, and reported once as
`W2101`; compilation continues with cache misses. Where even a fresh index
cannot be written the store works in memory, every lookup a miss. Recording an
artifact spans two tables and runs in one transaction, so a reader never sees
a row whose dependencies have not landed. [compatibility.md](compatibility.md)
states the contract.

## Runtime specialization

`@ppy.jit` compiles a version specialized to the argument classes actually
seen, guarded on exact class identity; a guard miss runs the Python body and
may compile another specialization. All-scalar `@dataclass` value classes are
flattened to scalar SSA values at the ABI, with reads guarded on exact class.

## Code generation

JIT-compiled code targets the exact host CPU — its name and feature set are
handed to LLVM, so AVX2/FMA and friends are on where the machine has them
(measured ~20% on a matmul kernel; memory-bound kernels are unchanged).
That is free because JIT code never leaves the machine that made it.

Emitted objects, built artifacts, and standalone binaries stay on the
portable baseline instead: they may run on another machine, exactly like a
C compiler's output without `-march=native`. `ppy build --host-cpu`
(`[tool.ppy.llvm] host-cpu`) trades that away deliberately — the artifact
gets this machine's instruction set and faults on an older one. The choice
is part of the cache key, along with the host CPU's name and features, so a
baseline object is never handed to a host build or carried between
machines.

## Reading input

`ppy/_io.py` is a runtime-only reader: a small C scanner over file
descriptor 0, compiled on first use into the user cache and bound through
ctypes, exposed as `ppy.input[T]()` and the lower-level `ppy.read_ints` /
`ppy.read_token`; `ppy.buffer[T](n)` beside it is the allocation both a
CPython run and a standalone binary understand. It lives in the `ppy` package rather than the compiler
because it is useful on every path, plain CPython included, and it degrades
to a pure-Python implementation where no C compiler exists. The checker
types `ppy.input[T]()` from its subscript, the way it types `ppy.check[T]`.

## The boundary

Python callers cross into native code through a generated `METH_FASTCALL`
wrapper that parses, guards, calls, and boxes in C — and holds the Python
implementation itself, so a refused guard is a C-to-Python call, not a
`NotImplemented` bounced through a Python frame. No Python code stands on
the call path at all (`@ppy.jit` keeps a thin Python watcher only while it
is still learning which argument shapes repeat). Measured against a plain
Python call at 29 ns: 51 ns forced-native two-int call, 70 ns borrowed
buffer, 80 ns guard failure into the fallback
(`examples/bench_boundary.py`). Built artifacts ship the compiled wrapper
and bind through it at launch; the ctypes trampoline remains only as the
fallback where no C toolchain exists (`W2004` says so once).

## Threads

Generated wrappers release the GIL around native calls
(`Py_BEGIN_ALLOW_THREADS`), so `@ppy.native` functions scale on threads:
measured 1.94× on two threads against 1.02× for the same code on plain
CPython. `@ppy.parallel` loops run on a process-wide worker pool sized by
`[tool.ppy.parallel] threads`.
