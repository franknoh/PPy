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

Strict mode is the default. `--no-strict` on the command line, or
`strict = false` in `[tool.ppy]`, turns it off for a run or for the
project. Without strict mode the compiler downgrades only the errors that have
a sound fallback: an unannotated parameter (`E1201`), an unknown attribute
(`E1202`), an unvouched decorator (`E1204`), iterating something that is not
iterable (`E1302`), and a call with no known signature (`E1306`). Code the
analysis cannot read but CPython runs is downgraded too:

- an annotation it cannot resolve, such as a type from a module that is not
  there or one it does not model (`E1101`, or `E1301` for the annotation's
  form). The value is treated as unknown.
- a name nothing defines (`E1101`). CPython raises `NameError` if the line
  runs.
- a star import (`E1103`). It may rebind any name, so no function in that
  module is compiled.
- `eval`, `exec`, `globals()`, `locals()`, `vars()`, `__import__` of a
  computed name, `getattr` or `setattr` with a computed name, and a class
  built by a metaclass or on a computed base (`E1501` to `E1504`, `E1506`,
  `E1507`).
- an operator on an instance of a class whose base the analysis cannot see
  (`E1302`).
- a value stored into a list or dict element whose type does not match the
  element type (`E1301`), such as a list stored where the rows hold strings.

Each is still reported, as a `W2010` warning that names the code it replaces,
and the function involved runs on CPython, which runs it or raises its own
error. So a script whose sibling import fails ends with CPython's
`ImportError`, not with a compile error. Other type mismatches (`E1301`)
and the rest stay errors, with one exception: a value that may be `None`
where one that is not is needed.

A program often passes a `Node | None` where a `Node` is declared, or reads
`.value` from something that may be `None`, because it knows the `None`
never arrives on that path. Strict mode refuses this (`E1301`, `E1206`).
`--no-strict` reports it as `W2011` and runs the program. If the value is
`None` after all, the program raises where CPython raises: native code checks
every field read and method call through such a value and raises CPython's
`AttributeError`.

### Types from call sites

Without strict mode, a function or method with unannotated parameters takes
their types from the calls the project makes to it. `count_divisors(n)`
called as `count_divisors(28)` and `count_divisors(36)` has `n: int`, and
compiles like an annotated function:

```python
def count_divisors(n):
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 2
        i += 1
    return count


print(count_divisors(28), count_divisors(36))
```

Every call in every file of the project counts, and they must agree. An
`int` on one call and a `float` on another make a `float`, as a declared
`float` takes an `int`: where the function's result would show that it was
given an `int` (`x * 3` is `12` for `4`, not `12.0`), the native entry
refuses the `int` and the Python body runs. Other mixes leave the
parameter unknown: a function is not compiled once per type its calls
pass. More evidence:

- A value whose type the program states counts as a typed argument:
  `int(input())`, `float(...)`, `input().split()`,
  `list(map(int, input().split()))`, and `args.k` from an
  `argparse.ArgumentParser` with `add_argument("--k", type=int, default=3)`.
  An option that may be `None` (no default, not required), a parser
  reconfigured with `set_defaults` or groups, or one passed elsewhere does
  not count.
- A default value is a call that passes it: `def f(xs, lo=0)` with
  `f(ys, 2)` has `lo: int`. So is a default that is a module constant
  written as a literal (`tol=EPSILON`), or `int(...)`, `float(...)`,
  `str(...)`, `bool(...)`, or `len(...)` (`seed=int(time())`).
- A method takes the calls made to it through any class of its family:
  `shape.area(2)` with `shape: Shape` may run `Square.area`, so both take an
  `int`.
- An operator on an instance is a call of its method: `a + b` with `a: Vec`
  calls `Vec.__add__` (or `b`'s `__radd__`), and so do `a < b`, `a[k]`,
  `a[k] = v`, `del a[k]`, and `x in a` for theirs. A comparison method
  (`__eq__`, `__lt__`, ...) is only typed as taking its own class, because
  the runtime also calls it with two of the class's objects to sort a list
  or look one up.
- A parameter declared with an open element type (`xs: list`,
  `dict[str, Any]`) takes the element type its calls agree on, as
  `list[int]`; with no agreement it keeps the declaration.
- A parameter no call in the project types takes the type its doctests
  pass as literals: `>>> digit_sum(1234)` gives `n: int`, and
  `>>> s = Stack()` then `>>> s.push(3)` gives `value: int`. Arithmetic on
  literals (`2 << 31`), an empty display beside filled ones
  (`{1: [2], 2: []}` is a `dict[int, list[int]]`), a loop over a literal or
  a `range` (`for v in [3, 4]: f(v)`), and operators on a name the doctest
  bound (`>>> heap["B"]`) count too. An example that expects an exception
  is not counted. The cases of `@pytest.mark.parametrize`, as literals or a
  module constant written as one, count the same way.
- A parameter nothing calls with a type, with no default, takes the type
  its body's use admits when that is one builtin: `range(n)` makes `n` an
  `int`, and a parameter whose only attribute uses are string methods
  (`s.split()`, `s.upper()`) is a `str`. A test against `None`, another
  attribute, a subscript of an `int`, a new binding, or a nested function
  leaves it unknown.
- A result takes the type of the body's `return` statements, a recursive
  function's too (`fib(n - 1) + fib(n - 2)`).

A decorator is looked through when it keeps the function's parameters:
`@staticmethod`, `@classmethod`, `@functools.cache`, `@functools.lru_cache`,
`@abc.abstractmethod`, a pytest mark, and a project decorator whose wrapper
is made with `functools.wraps(fn)` and only calls `fn(*args, **kwargs)`.
Looking through a decorator gives the parameters their types and nothing
else. Under `--no-strict` a function with a decorator nobody vouches for
stays in Python, and every call by its name, from Python or from native
code, goes to the object the decorator returned, so a wrapper that prints,
doubles the result, or caches runs just as it does on CPython. Such a call
counts as one with unknown effects: it is never inlined, folded, or moved
out of a loop.

Nothing is inferred for a function the program uses as a value (`key=f`,
`map(f, xs)`, `g = obj.method`), one with another decorator, one called with
`*args` or `**kwargs`, a nested function, or a dunder method other than
`__init__` and the operator methods above. An inferred type that makes the
checker report an error (`x + y` on a path the program never takes, with
`y` now an `int`) is taken back, and the function runs on CPython as before.

An inferred type is guarded like a declared one. A Python caller (a doctest,
another program that imports the module, a call through `getattr`) goes
through the function's native entry, which checks the exact type of each
argument and runs the function's Python body when one does not match. So
`count_divisors(True)` from Python prints CPython's answer. `ppy explain`
says which types were inferred and from where:

```text
inferred (not annotated):
  n: int, from 2 calls (prog.py:11)
  (the Python boundary checks these at each call, and runs the Python body otherwise)
  return: int, from the body's return statements
```

Strict mode does not infer: an unannotated parameter is still `E1201`.

[An unannotated module](../howto/52_unannotated.md) is a worked example:
seven functions with no annotations, each typed from its calls and
defaults, and the boundary running the Python body for an argument that
does not fit. [Options and doctests](../howto/56_options_and_doctests.md)
types a script from its `argparse` options, operators on a class, an
`lru_cache` function, and a constant default.
[Linked structures](../howto/55_linked_structures.md) shows the fields of
unannotated classes typed from the stores into them
([Classes](classes.md#fields-without-annotations)).

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
  float to it is still refused (`E1301`). Native code converts the ints
  only where the callee cannot show the difference, and otherwise runs the
  call on CPython (see
  [A `float` given an `int`](native-lowering.md#a-float-given-an-int)).
- A list written in place with narrower elements than declared:
  `m: list[list[float]] = [[0] * n for _ in range(n)]`.
- A generator expression where a `Generator[T, None, None]` is declared.

The standard library's `Queue`, `LifoQueue`, `PriorityQueue`, `deque`,
`Counter`, `OrderedDict`, and `defaultdict` have their methods typed by
their type arguments (`Queue[int].get()` is an `int`). `datetime` and `date`
with `timedelta`, `Decimal` and `Fraction` with themselves and with numbers,
and `Counter` with `Counter` under `+ - & |` have their operators typed, and
their fields and common methods too. `deque`, `Counter`, `OrderedDict`, and
`defaultdict` also compile to native code
([The standard library](stdlib.md)); code that uses the queues,
`datetime`, `Decimal`, or `Fraction` runs on CPython.

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
