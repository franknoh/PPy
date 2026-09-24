# The subset

This page covers what the compiler accepts, how far PPy stays compatible with
Python, and how it treats a value whose type is missing.

## What the compiler accepts

The compiler analyzes the whole project as one call graph. Inside it:

- Every parameter and return type must be declared or inferable. An implicit
  `Any` is an error (`E1201`). It is not silently accepted.
- Attributes are resolved on statically known types. `__init__` declares the
  instance fields.
- `eval`, `exec`, `from x import *`, computed imports, monkey-patching,
  frame manipulation, computed base classes, and unvouched metaclasses are
  rejected (`E15xx`) unless isolated behind `ppy.dynamic`.
- Class construction must be declarative. These are all `E1507`:
    - a class body that executes statements
    - a body value constructing a project descriptor whose `__set_name__`
      runs at creation
    - a base whose `__init_subclass__` does real work

    The strict checker and the safe hoister judge this from the same shared
    facts (`class_construction` in `analysis/decorators.py`), so they cannot
    disagree about what a class body runs.
- A decorator must have vouched semantics: the built-in table, a plugin, or
  `ppy.*`. A decorator may replace the decorated object, so believing the
  `def` while the runtime holds whatever the decorator returned would be
  unsound. An unvouched decorator is `E1204` unless the definition is marked
  `@ppy.dynamic`. `@partial(vouched, ...)` counts as the vouched decorator it
  binds.
- Everything else is ordinary Python: classes, generators, closures,
  `match`, comprehensions, decorators the compiler knows, and the stdlib it
  models.

Strict mode is the default. `--no-strict` downgrades only the errors that have
a sound fallback.

## Compatibility policy

PPy makes three separate compatibility claims and holds each to a different
standard:

| claim | level | what it means |
|---|---|---|
| Syntax compatibility | very high | A `.ppy` file is valid Python. The tooling, editors, and formatters that read Python read PPy. |
| Library compatibility | high, through plugins and boundaries | NumPy, PyTorch, JAX/Flax, pydantic, FastAPI, SciPy, pandas, PyArrow and the modeled stdlib work as-is. Everything else works behind an explicit `ppy.dynamic` boundary. |
| Semantic compatibility | intentionally incomplete | PPy does not aim to preserve arbitrary dynamic Python behavior. `exec`/`eval`, monkey-patching, dynamic namespace mutation, computed class construction, and unrestricted runtime reflection are restricted in exchange for reliable analysis, optimization, and native compilation. |

Running existing Python is a migration feature (`ppy migrate`). It is not the
definition of the language: a valid Python program is not necessarily a valid
PPy program.

## Unknown, Any, and Dynamic

PPy keeps three kinds of missing type apart on purpose.

### Unknown

Unknown is internal compiler state: inference has not resolved the value. It
must not survive strict compilation. It is reported (`E1201`, `E1304`) and is
not silently widened.

### `typing.Any`

`typing.Any` is the permissive legacy spelling. It absorbs anything, and the
compiler polices nothing about it. Use it for interop annotations you already
trust.

### `ppy.Dynamic`

`ppy.Dynamic` is the policed boundary. Any value may become `Dynamic`, but a
`Dynamic` value fits only `Dynamic`, `Any`, or `object`. Crossing into typed
code (a typed return, parameter, field, or declared variable) is `E1508`
until the value passes through `ppy.check[T](value)` or
`ppy.assume[T](value)`.

## `ppy.check` and `ppy.assume`

`ppy.check[T](value)` validates each runtime-checkable part of `T`,
recursively. It raises `TypeError` where the value falls short, and hands
back a value typed as `T`. It checks:

- a `list[int]` element by element
- a dataclass field by field
- an `i8`'s range
- an `Array[int, 3]`'s length
- a `Buffer[float]`'s format
- a `Range`, `Length`, `Shape`, `DType`, or `Contiguous` refinement against
  the value's own metadata

A `T` that PPy cannot validate soundly at runtime is rejected, never checked
in part. That includes:

- a callable
- an iterator
- a protocol, `@runtime_checkable` or not. `isinstance` against one answers
  whether the attributes are there and nothing of what they take or answer,
  so a class whose `f(self)` returns a string satisfies a protocol declaring
  `f(self, x: int) -> int`.
- a contract between caller and callee such as `Owned[T]`, `Borrowed[T]`,
  `Mut[T]`, or `NoAlias`, which no single value can bear witness to

`ppy.assume[T](value)` performs no runtime validation. It is an explicit
unchecked assertion the programmer takes responsibility for, and it should
look like one.

Both differ from `typing.cast` in behavior:

- `cast` asserts and checks nothing.
- `check` checks and asserts nothing.
- `assume` asserts and says so.

Examples: [Dynamic boundaries](../howto/16_dynamic.md),
[Narrowing](../howto/10_narrowing.md), [Classes](../howto/04_classes.md).
