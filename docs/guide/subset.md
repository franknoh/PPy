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
a sound fallback: an unannotated parameter (`E1201`), an unknown attribute
(`E1202`), an unvouched decorator (`E1204`), iterating something that is not
iterable (`E1302`), and a call with no known signature (`E1306`). Each is
still reported, as a `W2010` warning that names the code it replaces, and the
code involved runs on CPython. Type mismatches (`E1301`) and the rest stay
errors, with one exception: a value that may be `None` where one that is not
is needed.

A program often passes a `Node | None` where a `Node` is declared, or reads
`.value` from something that may be `None`, because it knows the `None`
never arrives on that path. Strict mode refuses this (`E1301`, `E1206`).
`--no-strict` reports it as `W2011` and runs the program. If the value is
`None` after all, the program raises where CPython raises: native code checks
every field read and method call through such a value and raises CPython's
`AttributeError`.

Without strict mode, a module-private function (one whose name starts with
`_`) that has unannotated parameters takes their types from its call sites,
when every use of it is a direct call in its own module. `_scale(xs, k)`
called only as `_scale([1, 2, 3], 2)` has `xs: list[int]` and `k: int`, and
compiles like an annotated function. A call from elsewhere with other types,
through reflection or `doctest`, runs the function's Python body.

## Accepted forms of ordinary Python

These are valid Python that the checker accepts and types:

- `typing.Self` in a class's signatures, fields, and method bodies.
- Old-style type variables: `T = TypeVar("T")`, with `bound=` or
  constraints, and `class Stack(Generic[T])`. See [Generics](generics.md).
- `namedtuple("P", "x y")`, `NamedTuple("P", [("x", int)])`, and
  `class P(NamedTuple)`. An instance reads as the tuple it is (`p[0]`,
  `x, y = p`, `for v in p`), and has `_replace`, `_asdict`, and `_fields`.
  The fields of `namedtuple` have no type.
- `Cls.attr = value` and `setattr(Cls, "attr", value)` for an attribute the
  class body sets. It is a class variable that changes: functions that read
  or write it run on CPython. `setattr(obj, "name", value)` with a constant
  name is checked as `obj.name = value`.
- `Cls.method(obj, arg)`, a method called through its class with the
  receiver written out.
- `__import__("doctest")` with a constant name, as `import doctest`.
- `isinstance(x, (list, tuple))` narrows `x` to what its declared type
  shares with the classes: a `list[int]` stays a `list[int]`.
- A container passed where a wider element type is declared. `list[int]`
  goes where `Sequence[float]` is declared, and where `list[int | float]` is
  declared if the callee only reads the list. A function that appends a
  float to it is still refused (`E1301`). Calls like these run on CPython,
  since native code would convert the ints.
- A list written in place with narrower elements than declared:
  `m: list[list[float]] = [[0] * n for _ in range(n)]`.
- A generator expression where a `Generator[T, None, None]` is declared.

The standard library's `Queue`, `LifoQueue`, `PriorityQueue`, `deque`,
`Counter`, `OrderedDict`, and `defaultdict` have their methods typed by
their type arguments (`Queue[int].get()` is an `int`). `datetime` and `date`
with `timedelta`, `Decimal` and `Fraction` with themselves and with numbers,
and `Counter` with `Counter` under `+ - & |` have their operators typed, and
their fields and common methods too. Code that uses them runs on CPython.

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
