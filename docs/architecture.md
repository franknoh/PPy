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
| `frontend/` | source loading, the module graph, ambiguity detection (`E1003`). |
| `analysis/` | `symbols` (declarations), `checker` (types, refinements, effects), `binding` (one shared call-argument binder), `lexical` (point-sensitive name resolution: what a name means at each statement, shared by decorator identity, reflection, and the write index), `aliasing` (flow-sensitive local alias analysis: mutation and escape resolve through what a name may refer to, not its spelling), `inference` (staged evidence/generalization fixpoint with a convergence guard), `decorators` (what each known decorator does, and that unknown means opaque), `global_writes` (scope-aware project-wide write index behind `Final`), `reflection` (who reads annotations at runtime, blocking their materialization), `codec` (exact-inverse serialization of analysis facts for the cache), `render` (types back to annotation source). |
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

## Threads

Generated wrappers release the GIL around native calls
(`Py_BEGIN_ALLOW_THREADS`), so `@ppy.native` functions scale on threads:
measured 1.94× on two threads against 1.02× for the same code on plain
CPython. `@ppy.parallel` loops run on a process-wide worker pool sized by
`[tool.ppy.parallel] threads`.
