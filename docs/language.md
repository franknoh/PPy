# Language reference

A `.ppy` file is valid Python 3.12+. Everything PPY adds is carried by
annotations and decorators from the `ppy` package, all of which are inert at
runtime: under plain CPython the decorators return the function unchanged and
the markers are ordinary `typing.Annotated` aliases. What the compiler adds is
enforcement and speed, never behavior.

## The subset

The compiler analyzes the whole project as one call graph. Inside it:

- Every parameter and return type must be declared or inferable — an implicit
  `Any` is an error (`E1201`), not a silence.
- Attributes are resolved on statically known types; `__init__` declares the
  instance fields.
- `eval`, `exec`, `from x import *`, computed imports, monkey-patching,
  frame manipulation, computed base classes, and unvouched metaclasses are
  rejected (`E15xx`) unless isolated behind `ppy.dynamic`.
- Class construction must be declarative: a class body that executes
  statements, a body value constructing a project descriptor whose
  `__set_name__` runs at creation, or a base whose `__init_subclass__` does
  real work are all `E1507`. The strict checker and the safe hoister judge
  this from the same shared facts (`class_construction` in
  `analysis/decorators.py`), so they cannot disagree about what a class body
  runs.
- A decorator must have vouched semantics — the built-in table, a plugin, or
  `ppy.*` — because a decorator may replace the decorated object: believing
  the `def` while the runtime holds whatever the decorator returned would be
  unsound. An unvouched decorator is `E1204` unless the definition is marked
  `@ppy.dynamic`. `@partial(vouched, ...)` counts as the vouched decorator it
  binds.
- Everything else — classes, generators, closures, `match`, comprehensions,
  decorators the compiler knows, the stdlib it models — is ordinary Python.

Strict mode is the default. `--no-strict` downgrades only the errors that have
a sound fallback.

## Compatibility policy

Three different claims, deliberately held to three different standards:

- **Syntax compatibility — very high.** A `.ppy` file is valid Python; the
  tooling, editors, and formatters that read Python read PPY.
- **Library compatibility — high, through plugins and boundaries.** NumPy,
  PyTorch, JAX/Flax, pydantic, FastAPI and the modeled stdlib work as-is;
  everything else works behind an explicit `ppy.dynamic` boundary.
- **Semantic compatibility — intentionally incomplete.** PPY does not aim to
  preserve arbitrary dynamic Python behavior. `exec`/`eval`, monkey-patching,
  dynamic namespace mutation, computed class construction, and unrestricted
  runtime reflection are restricted in exchange for reliable analysis,
  optimization, and native compilation.

Running existing Python is a migration feature (`ppy migrate`), not the
definition of the language: a valid Python program is not necessarily a valid
PPY program.

## Directives

All directives work bare (`@ppy.pure`) and called (`@ppy.pure()`), and are
contracts the compiler verifies, not hints it trusts.

| directive | meaning |
|---|---|
| `@ppy.pure` | no observable effects: no I/O, no global or nonlocal writes, no mutation of arguments. Local allocation and mutation of locally created values are fine (spec 11.2). Violations are `E1601`/`E1602`. |
| `@ppy.opt(n)` | per-function optimization level 0–3, overriding the project default. |
| `@ppy.native` | lower to LLVM. `require=True` makes any fallback to the Python body an error (`E1702`). |
| `@ppy.parallel` | parallelize the eligible loop. `require=True` makes failure an error (`E1701`). |
| `@ppy.jit` | specialize at runtime on the argument classes actually seen. |
| `@ppy.specialize` | ahead-of-time specialization on declared value classes. |
| `@ppy.inline` / `@ppy.noinline` | force or forbid inlining into callers. |
| `@ppy.fastmath` | permit floating-point reassociation in this function; without it, reduction order is preserved bit-for-bit. |
| `@ppy.jax` | stage this function for build-time StableHLO export (a plain `@jax.jit` decorator marks it too). |
| `@ppy.dynamic` / `with ppy.dynamic():` | an explicit boundary inside which dynamic features are allowed; every value it produces is `Dynamic`, and stays `Dynamic` through attribute hops and arithmetic until a `ppy.check[T]` clears it. |
| `@ppy.reflective` | the function's annotations are runtime-visible state, exactly as written: `ppy convert`/`ppy migrate` will never add to or rely on rewriting them. |

A typo in a directive name is `E1205`, with a suggestion.

## Unknown, Any, and Dynamic

Three different absences of a type, held apart on purpose:

- **Unknown** is internal compiler state — inference has not resolved the
  value. It must not survive strict compilation: it is reported (`E1201`,
  `E1304`), never silently widened.
- **`typing.Any`** is the permissive legacy spelling. It absorbs anything and
  the compiler polices nothing about it — use it for interop annotations you
  already trust.
- **`ppy.Dynamic`** is the policed boundary. Any value may become `Dynamic`;
  a `Dynamic` value fits only `Dynamic`, `Any`, or `object`. Crossing into
  typed code — a typed return, parameter, field, or declared variable — is
  `E1508` until it passes through `ppy.check[T](value)`, which validates at
  runtime (raising `TypeError`) and hands back a typed value. `ppy.check` is
  the inverse of `typing.cast`: it checks and asserts nothing, where `cast`
  asserts and checks nothing. Validation is shallow — `list[int]` is checked
  to be a `list`, not walked — because the check runs on the boundary.

## Markers

Ordinary `Annotated` aliases from `ppy`:

| marker | meaning |
|---|---|
| `i8 i16 i32 i64 u8 u16 u32 u64` | fixed-width integer contract. A value provably outside the range is `E1401`; a check the contract mode forbids is `E1402`. |
| `f16 f32 f64` | floating-point width. |
| `Buffer[T]` | a borrowed writable buffer (`memoryview` over `array.array`) — zero-copy in and out of native code. |
| `Array[T]`, `Vector[T]` | contiguous numeric containers with a known element type. |
| `Range(lo, hi)` | an integer refinement the checker propagates. |
| `Dynamic` | an explicit Python-dynamic boundary value. Entering is free; leaving is not: `Dynamic -> Dynamic` flows freely, but `Dynamic -> int` is `E1508` until a `ppy.check[T]` validates it. `Any` at runtime. |
| `Length(n)`, `Shape(...)`, `DType("f32")`, `Contiguous`, `NoAlias` | container refinements; `Shape`/`DType` are what makes a `@ppy.jax` function exportable. |

## Effects and purity

Every function gets an inferred effect set (I/O, global write, argument
mutation, allocation, randomness, unknown external call). `@ppy.pure` asserts
the set is empty of the forbidden ones, and the checker proves it
interprocedurally — a pure function calling something with unknown effects is
`E1602`, and a callee that mutates the caller's argument is charged to the
caller.

## The three execution paths

| | |
|---|---|
| `python f.ppy` | plain CPython. `import ppy` installs a `sys.meta_path` finder so `.py` files can import `.ppy` modules; nothing else happens. |
| `ppy f.ppy` | the optimized Python backend: AST-level optimization (folding, inlining, LICM, loop transforms) executed by CPython. |
| `ppy run f.ppy` | eligible functions compile through LLVM; everything else runs the Python body. |

Any observable difference between the three is a compiler bug. The test suite
and `examples/run_all.py` compare all three on every example. `ppy build`
produces the third path ahead of time: the launcher it emits re-enters the
same pipeline and the same guarded bindings, taking machine code from the
library built next to it instead of a JIT.

## Native lowering

A function lowers when its types are scalars, `Buffer[T]`, homogeneous
`list[int]`/`list[float]`, `Sequence` of those, or all-scalar `@dataclass`
value classes (flattened into scalar arguments), and its body stays inside the
modeled subset. The generated wrapper releases the GIL around the native
call, so `@ppy.native` functions scale across threads. `ppy explain FILE.ppy:name` reports the decision and, when the
answer is no, the first blocking construct.
