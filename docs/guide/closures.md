# Functions as values

Nested functions, `lambda`, and functions passed, returned, and stored as
values lower to native code. A function that makes or calls one goes native
under `ppy run`, in a standalone binary, and in emitted C and C++, and gives
CPython's answer on every path.

```python
from collections.abc import Callable


def make_scale(k: int) -> Callable[[int], int]:
    return lambda v: v * k


def compose(f: Callable[[int], int], g: Callable[[int], int]) -> Callable[[int], int]:
    return lambda v: f(g(v))


def total(n: int) -> int:
    count = 0

    def bump(by: int) -> None:
        nonlocal count
        count += by

    twice = compose(make_scale(2), lambda v: v + 1)
    for i in range(n):
        bump(twice(i))
    values = [i * 37 % 11 for i in range(n)]
    best = max(values, key=lambda v: (v % 4, v))
    return count + sum(map(lambda v: v * v, range(n))) + best
```

## What lowers

| | |
|---|---|
| `def` inside a function | reads the enclosing function's variables, and writes them with `nonlocal` |
| `lambda` | typed by where it goes: a `Callable` parameter, a declared local or field, a sort key, `map` or `filter` |
| a function of the module used as a value | `apply(double, xs)`, `ops = [add, sub]` |
| `Callable[[A, B], R]` | a parameter, a result, a local, a list or dict element, an object's field |
| `f(x)` where `f` is a value | a local, a parameter, `ops[i](x)`, `self.f(x)`, `make()(x)` |
| `xs.sort(key=f)`, `sorted(xs, key=f)` | with `reverse=`; a stable sort |
| `min(xs, key=f)`, `max(xs, key=f)` | the first element whose key no later one beats |
| `map(f, xs)`, `filter(f, xs)`, `filter(None, xs)` | where a `for` loop, `sum`, `min`, `max`, `sorted`, or a comprehension consumes it |

A key or a `map` function is a lambda, a function's name, or any expression
giving a function value. `len` is a key too.

## How a closure works

A function value is a closure: a handle, reference counted like a list,
holding the address of the function's native entry and a cell for each
variable it shares with the function that made it. A shared variable lives
in its cell for its whole life, and both functions read and write it there.

So a closure sees a variable as it is when the closure runs, not as it was
when the closure was made, which is CPython's late binding:

```python
def late(n: int) -> int:
    def get() -> int:
        return x

    x = n
    first = get()
    x = n * 2
    return first + get()  # n + 2 * n
```

A loop variable is one variable too: after `for i in range(3):
fs.append(lambda: i)`, every closure in `fs` gives 2, in CPython and natively.

A call through a value is an indirect call of the entry, with the same
arguments and result a direct call has. A closure that raises raises through
the call, and one that falls back takes its caller with it. A closure that
refers to itself, such as a nested recursive function, is a cycle the
collector frees.

## A nested function in a Python function

A function that stays in Python can still define nested functions that run
natively. Where a nested function shares no variable with the functions
around it, every function object its `def` makes behaves the same, so it
gets a native entry of its own. Python binds that entry where the `def`
runs, once per process, and calls it like any other native function.

```python
def solve(rows: list[tuple[int, int]], **options: int) -> int:
    def fib(n: int) -> int:
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)

    return sum(fib(a % 20) for a, _ in rows)
```

`solve` stays in Python, since a native function takes no `**options`, but
`fib` is native. A nested function may call itself by its own name. One that
reads or writes a variable of the function around it, uses its own name as
a value, or has a decorator runs as Python with that function, and `ppy
explain` names the shared variables.

A call by a plain name inside a function follows Python's scopes: a nested
function the function around it defines is called before a module function
of the same name.

## The Python boundary

A function value never crosses to Python. A function that takes or returns
one runs as Python when Python calls it, and natively when native code calls
it. A function that only makes and calls closures inside itself crosses as
usual: Python calls it natively and gets back what it returns.

## What stays in Python

A function using any of these runs as Python:

- a lambda nothing types, such as `f = lambda v: v + 1` with no annotation
- a nested function or lambda with defaults, `*args`, keyword-only
  parameters, or a decorator
- calling a function value with keyword arguments
- a closure sharing a tuple, a buffer, or a value class
- `min` and `max` with a key over strings or objects; a key giving a string
- `map` over several iterables
- a function of another module used as a value, or a method bound to its
  object (`f = obj.method`)

`tests/test_closures.py` holds each form in the table to CPython on every path.
