# Exceptions and generators

`raise`, `try`, `assert`, and generators lower to native code. A function
that catches an exception, or consumes a generator, goes native under
`ppy run`, in a standalone binary, and in emitted C and C++, and gives
CPython's answer on every path.

```python
from collections.abc import Iterator

from ppy import Vec


class BadRow(ValueError):
    pass


def rows(text: str) -> Iterator[int]:
    for part in text.split(","):
        if not part.strip().isdigit():
            raise BadRow(f"not a number: {part!r}")
        yield int(part)


def total(text: str, weights: Vec[int]) -> int:
    got = 0
    try:
        for i, row in enumerate(Vec[int](rows(text))):
            got += row * weights[i]
    except BadRow as error:
        return -len(str(error))
    except IndexError:
        return -1
    return got
```

`weights[i]` past the end is an `IndexError` the handler takes natively.
Nothing falls back to Python.

## Exceptions

### What lowers

| | |
|---|---|
| `raise T`, `raise T(message)` | a builtin exception, or a class deriving from one; the message is a string, an `int`, or a `bool` |
| `raise T(...) from cause`, `raise ... from None` | the cause is a name or a new exception, as CPython chains it |
| `raise`, `raise e` | re-raising the exception a handler holds |
| `try` with `except T`, `except (A, B)`, `except T as e`, `except`, `else`, `finally` | a handler matches by class, a base catches its subclasses |
| `assert test`, `assert test, message` | `AssertionError` |
| `str(e)`, `f"{e}"`, `print(e)`, `isinstance(e, T)`, `e.field` | in a handler |

A project exception class may have fields, methods, and an `__init__` of its
own. Its `str()` is what `super().__init__(...)` was given, or, where
`__init__` does not call it, the arguments the class was called with, as
CPython keeps them:

```python
class ParseError(ValueError):
    def __init__(self, line: int, text: str) -> None:
        super().__init__(f"line {line}: {text}")
        self.line: int = line
        self.text: str = text


def parse(n: int) -> int:
    total = 0
    for i in range(n):
        try:
            if i % 7 == 3:
                raise ParseError(i, "bad token")
            total += i
        except ParseError as e:
            total += e.line * 1000 + len(str(e))
    return total
```

`str()` of a `KeyError` is `repr()` of its key, as CPython has it.
`finally` runs when the body, a handler, or `else` finishes, raises,
`return`s, `break`s, or `continue`s.

### Checks are exceptions

In a module that raises or catches, the checks native code makes where
CPython would raise are exceptions too:

- an index out of range
- a missing key
- a division by zero
- `int()` of text that is not a number
- `None` where an object is wanted

So `except IndexError:` takes the exception native code raised. Each
exception is CPython's class with CPython's message. A check whose text
leaves out what CPython adds, such as `int("x")`'s literal, gives the class
natively, and `str(e)` of it falls back to Python.

A check that stands for what only native code cannot do is not an
exception: an integer past 64 bits, or a NaN compared in a collection.
Under `ppy run` it falls back and Python answers, since CPython would not
have raised there. A check the optimizer would hoist out of a loop is not
hoisted in such a module: caught, a check that failed before the loop ran
would give a different answer.

### Across calls

An exception crosses native calls. The function it leaves lets go of what
it held and returns a raised status, and the caller catches it (a `try`
around the call) or returns the same status on up. An exception nothing
native catches ends the call:

- Under `ppy run` Python runs the call again and raises it itself; the
  runtime frees what the native call made.
- A standalone binary, and emitted C and C++, print the line CPython's
  traceback ends with (`Broken: at the bottom of 3`) and exit with status 1.

### What stays in Python

A function using any of these runs as Python:

- `e.args`, and any other use of `e` than the ones above
- a cause that is neither a name nor a new exception (`raise X from f()`)
- a `finally` that returns, breaks, or continues out of itself

## Generators

A generator function, and a generator expression, is lowered into the
code that consumes it. Its body runs in its own scope, and at each `yield`
the consumer's step runs with the value, then the body carries on after
the `yield`. Nothing is allocated for the generator itself, so a loop over
one costs what the loop inside it costs.

### What consumes one

| | |
|---|---|
| `for x in gen(...)`, with `break`, `continue`, and `else` | `break` closes the generator |
| `sum`, `min`, `max`, `sorted`, `any`, `all` | over a generator or a generator expression |
| `next(gen(...))`, `next(gen(...), default)` | the first value, or `StopIteration` |
| `it = gen(...)`, then `next(it)`, `next(it, default)`, and `for x in it` | stepped from one place to the next, as below |
| `Vec[T](gen(...))` and the other collections | built from what it yields |

In the generator, `yield value` and `yield from` another generator or a
collection lower, and `return` ends it. It yields numbers, tuples of
numbers, strings, collections, or objects.

A generator held in a name and stepped more than once runs in step with the
function holding it. The code between two steps runs as the generator hands
over its next value, and a `for` over it may `break` and leave the rest for
the next step:

```python
def header_then_rows(n: int) -> int:
    it = squares(n)
    first = next(it)
    second = next(it, -1)
    total = first * 100 + second
    for row in it:
        total += row
    return total * 1000 + next(it, 77)
```

The steps are `x = next(it)` and `for` statements of the function itself,
not inside a loop or a branch, and the code between two of them does not
`return`, `break`, or `continue`.

### What stays in Python

A generator that is returned, passed to another function, or stepped from
inside a loop or a branch keeps the function that does so in Python, and so
does:

- a generator method, a generic generator, or an async one
- a generator with a `try`, a nested function, or a `with`
- `x = yield`, whose value `send` gives
- a generator that yields from itself

Examples: [Errors and generators](../howto/50_errors_and_generators.md).
