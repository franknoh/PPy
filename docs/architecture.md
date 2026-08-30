# Architecture

```
.ppy sources
     │  frontend/          parse (CPython grammar), module graph, .py/.ppy shadowing
     ▼
symbol tables            analysis/symbols       declarations, imports, fields, directives
     ▼
type & effect analysis   analysis/checker       flow typing, refinements, purity fixpoint
     ▼
     ├── driver/convert + analysis/inference    call-site fixpoint → ConversionPlan → CST rewrite
     ├── migration/                              rewrite passes + report behind `ppy migrate`
     ├── opt/                                   AST passes for the Python backend
     ├── backend/llvm/                          lowering → LLVM IR → wrapper → link/JIT
     └── lsp/, driver/explain                   the same analysis, served interactively
```

Analysis produces data; only `driver/convert.py` touches source text, and only
through a `ConversionPlan`. Nothing in `analysis/` knows the output is text,
and nothing in the backends re-derives what the checker already proved.

## Modules

| package | contents |
|---|---|
| `ppy` (runtime) | the import hook and the inert directives/markers. This is all a plain CPython run ever loads. |
| `ppy_runtime` | everything a *built artifact* needs at launch: the native ABI as data (`abi`), the guarded binding trampolines (`binding`), generated-module identity and execution (`generated`, `execute`), the binder protocol (`dispatch`), and the manifest-driven launch path (`manifest`, `launch`). The hard rule: this package never imports `ppy_compiler` — uninstalling the compiler must not break a built application, and a test keeps that true by poisoning the compiler and running a launcher. |
| `frontend/` | source loading, the module graph, ambiguity detection (`E1003`). |
| `migration/` | the `ppy migrate` layer over the shared conversion engine: deterministic rewrite passes (`pipeline`, `dynamic`, `globals`) that prove each rewrite equivalent before making it, and the classified report (`report`) that says what remains. |
| `analysis/` | `symbols` (declarations), `checker` (types, refinements, effects), `binding` (one shared call-argument binder), `lexical` (point-sensitive name resolution: what a name means at each statement, shared by decorator identity, reflection, and the write index), `aliasing` (flow-sensitive local alias analysis: mutation and escape resolve through what a name may refer to, not its spelling), `inference` (staged evidence/generalization fixpoint with a convergence guard), `decorators` (what each known decorator does, that unknown means opaque, and the shared `class_construction` facts behind both strict class checking and safe hoisting), `global_writes` (scope-aware project-wide write index behind `Final`), `reflection` (who reads annotations at runtime, blocking their materialization), `codec` (exact-inverse serialization of analysis facts for the cache), `render` (types back to annotation source). |
| `opt/` | AST-level passes: constant folding, inlining, LICM, loop transforms; used by the Python backend and as pre-lowering cleanup. |
| `backend/python/` | runs optimized AST under CPython with the loader installed. |
| `backend/llvm/` | `lowering` (AST → LLVM IR), `wrapper` (generated CPython-ABI entry points, `METH_FASTCALL`, GIL release), `fusion` (NumPy elementwise loops), `specialize`/`jit` (guarded runtime specialization), `parallel` (the worker pool), `link` (objects → shared library). |
| `plugins/` | numpy, torch, jax, pydantic, uvicorn — see [plugins.md](plugins.md). |
| `cache/` | the content-addressed store (SQLite) and key construction. |
| `driver/` | CLI, pipeline orchestration, convert, fmt, lint, test, explain. |
| `lsp/` | the language server, on the same analysis. |

## The three-path invariant

Plain CPython, the Python backend, and the LLVM backend must produce the same
answer; a guard that fails at runtime falls back to the Python body rather than
ever answering differently. The invariant is enforced, not assumed:
`examples/run_all.py` runs all 30 examples on all three paths and diffs the
output, and the test suite does the same per feature.

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
dependency digests. Measured cold/no-change/one-file: 2415 / 371 / 1785 ms on
the 30-example tree.

## Runtime specialization

`@ppy.jit` compiles a version specialized to the argument classes actually
seen, guarded on exact class identity; a guard miss runs the Python body and
may compile another specialization. All-scalar `@dataclass` value classes are
flattened to scalar SSA values at the ABI, with reads guarded on exact class.

## Code generation

JIT-compiled code targets the exact host CPU — its name and feature set are
handed to LLVM, so AVX2/FMA and friends are on where the machine has them
(measured 14% on a matmul kernel; memory-bound kernels are unchanged).
Emitted objects, built artifacts, and standalone binaries stay on the
portable baseline: they may run on another machine, exactly like a C
compiler's output without `-march=native`.

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
