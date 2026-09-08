# The subset

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
  PyTorch, JAX/Flax, pydantic, FastAPI, SciPy, pandas, PyArrow and the
  modeled stdlib work as-is; everything else works behind an explicit
  `ppy.dynamic` boundary.
- **Semantic compatibility — intentionally incomplete.** PPY does not aim to
  preserve arbitrary dynamic Python behavior. `exec`/`eval`, monkey-patching,
  dynamic namespace mutation, computed class construction, and unrestricted
  runtime reflection are restricted in exchange for reliable analysis,
  optimization, and native compilation.

Running existing Python is a migration feature (`ppy migrate`), not the
definition of the language: a valid Python program is not necessarily a valid
PPY program.

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

Examples: [Dynamic boundaries](../howto/16_dynamic.md),
[Narrowing](../howto/10_narrowing.md), [Classes](../howto/04_classes.md).
