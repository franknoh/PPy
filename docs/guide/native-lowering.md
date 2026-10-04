# Native lowering

This page covers when the compiler lowers a function to native code and what
a function may contain to lower.

## Eligibility and profitability

The compiler asks two different questions:

- `can_lower_native` decides whether correct native code exists.
- `should_lower_native` decides whether crossing the Python/native boundary
  pays for it.

A function with a loop, a buffer parameter, enough straight-line work, or an
explicit `@ppy.native`/`@ppy.jit`/`@ppy.specialize`/`@ppy.parallel` gets the
boundary. Native callers call its native symbol directly, boundary or not.

The generated wrapper's call costs about what a Python call does: 29 ns for
`def add(x: int, y: int) -> int: return x + y` called from Python, against
30 ns for CPython's own call of it. So straight-line work pays from two
operations (`0.5 * base * height`), and a one-operation helper stays on the
Python side (remarked as `R3004`). The rest costs more:

| what crosses | straight-line work it takes |
|---|---|
| numbers, tuples of numbers | 2 operations |
| a `str` parameter or result | 4 more each: a native string is made for it |
| a value class | 1 more per field |
| output held while the call runs (a `print`) | a loop: one line written through Python costs more than CPython's `print`, many cost much less |
| a module global the function reads | 6 more: a Python frame reads it for the wrapper |
| a call that draws from `random` | 16: the Python-level binding saves its state |

A straight-line function that calls itself, as `power(b, e - 1)` does,
stays off the boundary: how deep it goes is the argument's to decide, and
CPython raises `RecursionError` where native code has no limit to stop at.

A function that returns nothing gets the boundary where it loops over what
it is given, or fills a container the caller passed. A function that only
checks its arguments and raises stays a native caller's, and so does a
`main()` that takes nothing: it runs once, and what it calls goes native on
its own terms.

A container or an object is copied whole on each call, so the body has to
do work in proportion to it ([Lists, dicts, and sets](containers.md#between-functions)).

`ppy explain module.name` (or `FILE.ppy:LINE`) reports the decision and,
when the answer is no, the first blocking construct.

### What a call costs

`examples/bench_boundary.py` calls each shape below many times from a
Python loop and prints the time per call. The `ppy run` column is the
program as `ppy run` runs it; the CPython column is the same program under
`python`. The median of three runs, in nanoseconds:

| call | `ppy run` | CPython |
|---|---:|---:|
| `x + y` of two ints, kept in Python by the cost model | 29 | 30 |
| `x + y` of two ints, `@ppy.native` | 29 | 30 |
| the same, `y` passed by keyword | 38 | 34 |
| the same, `y` left to its default | 32 | 32 |
| a loop of 100 additions | 65 | 886 |
| `sum` of a borrowed buffer of 100 ints | 67 | 254 |
| a guard that fails, so the Python body runs | 82 | 36 |
| a `list[int]` of 100 written in place, `@ppy.native` | 1,496 | 2,538 |
| a `dict[int, int]` of 100 read, `@ppy.native` | 2,109 | 1,742 |
| a chain of 10 objects walked, `@ppy.native` | 831 | 141 |
| a function returning `None` that fills a list of 100 | 553 | 553 |

The `@ppy.native` `x + y` row is the wrapper alone: parsing the arguments,
the exact type checks, and boxing the result. A failed guard costs the
wrapper plus a Python call. The written list copies 100 ints in and back
and still wins, because the body does a multiplication and a remainder per
element; filling a list of 100 costs the same both ways, since copying it
back takes what the native loop saves. The dict and object rows are
the shapes the cost model keeps off the boundary: one addition per entry or
per object does not pay for copying it, and without `@ppy.native` Python
runs their Python bodies. Measured on Python 3.14 on an Intel Core Ultra 9
386H under WSL2.

## Byte-wide buffers

A buffer's element is a machine word unless it says otherwise.
`Buffer[ppy.i8]` and `Buffer[ppy.u8]` are one byte each, which is what text
and packed data want: four million characters cost four megabytes rather than
thirty-two.

The width is storage. Reading one hands out an `int`, and arithmetic on it is
integer arithmetic. Writing a value that does not fit in a byte falls back to
CPython rather than wrapping. An `array.array("b", ...)` is what such a buffer
is made of.

## Types that lower

A function lowers when each of its parameters, locals, and its result has
a type native code holds, and its body stays inside the modeled subset.
The types are:

- `int`, `float`, `bool`, `None`, the fixed-width markers, and tuples of
  these
- `str` ([Strings](strings.md))
- `list`, `dict`, and `set` of any of these, nested too
  ([Lists, dicts, and sets](containers.md)), and `Sequence` of them; a list
  of numbers a function only reads is lent as a buffer
- `Buffer[T]` and the `ppy` collections (`Vec`, `HashMap`, ...)
- value classes, copied as their fields, and object classes, held by
  handle ([Classes](classes.md))
- `deque`, `Counter`, `defaultdict`, `OrderedDict`, and `random.Random`
  ([The standard library](stdlib.md))
- generators and iterators of one of these types
  ([Exceptions and generators](exceptions-and-generators.md)), and
  function values ([Functions as values](closures.md))

A parameter has to be annotated, or under `--no-strict` inferred from the
calls the project makes ([Types from call sites](subset.md#types-from-call-sites)).
`list[Any]`, a bare `list`, and NumPy arrays have no native form, and a
function that takes one runs as Python. Under `--no-strict`, a parameter
declared as a bare `list` takes the element type its calls agree on.

## What the body may contain

The subset includes what a loop is normally made of:

- `break` and `continue` (a `continue` in a `for` still advances the counter)
- statement-level calls whose result is discarded
- buffers handed on to another native function
- the bitwise operators, including `~`
- `raise`, `try` with its handlers, `else`, and `finally`, and `assert`
- a generator consumed where it is made: by `for`, `next`, `sum`, `min`,
  `max`, a comprehension, or a collection built from it; and one that is
  returned, passed on, stepped in a loop, or made by `__iter__`, which gets a
  frame of its own

[Exceptions and generators](exceptions-and-generators.md) has the details.

### Expressions

- `isinstance(x, T)` and `isinstance(x, (A, B))` with builtin classes
  (`int`, `float`, `bool`, `str`, `list`, `dict`, `set`, `tuple`,
  `type(None)`) fold to a constant where the checker's type of `x` decides
  the answer. An `int` may be a `bool` and a `float` may be an `int`, so
  `isinstance(n, bool)` of an `n: int` stays in Python. Object classes are
  tested by the class tag the instance carries.
- A chained comparison, `0 <= i < n`, is its comparisons joined by `and`,
  each operand evaluated once and the ones after a false comparison not at
  all.
- `x in (a, b)`, `x in {a, b}`, and `x in [a, b]` are `x == a or x == b`.
  `x in range(start, stop, step)` is a bounds check and a remainder, with no
  range made. `not in` is the negation. A NaN on the left falls back,
  since CPython matches it by identity.
- `a ** b` and `pow(a, b)` of ints multiply by squaring. A product past 64
  bits falls back, and so does a negative exponent, whose result is a float.
  `pow(a, b, m)` reduces as it goes, and `pow(a, b, 0)` raises CPython's
  `ValueError`.
- `a = b = value` evaluates the value once and binds each name.
  `r, c = (x, y) if flag else (y, x)` makes and unpacks only the chosen side.

### Loops

A `for` walks:

- `range` with any step, including one known only at run time; a zero step
  raises `ValueError`
- `reversed(range(...))`, from its last value
- a string's characters, a tuple display (`for dx, dy in ((0, 1), (1, 0))`),
  and a tuple local
- a list parameter lent as a buffer

Each of these also works under `enumerate` (with `start=`), `zip`, and, for
ranges, buffers, and tuples, `reversed`, alone or mixed with the
collections. A `for` over any of them, or over a collection, may have an
`else`, which runs when the loop ends without `break`. Identity between two
values the checker types `bool` (`flag is True`) is equality, since there is
one `True` and one `False`.

### A `float` given an `int`

CPython lets an `int` stand where a `float` is declared and keeps it an
`int`: `f(0)` of `def f(x: float) -> float: return x` is `0`. Native code
converts at the boundary, which is only right where the difference cannot
show. The compiler follows each `float` parameter (and each tuple, list, or
value class holding floats) through the body. Where the value, or something
computed from it the way an `int` stays an `int`, is returned, printed or
formatted, stored where the caller sees it, or asked its class, the boundary
takes only a `float` for it and an `int` keeps the call in Python. A native
caller that passes an `int` to such a parameter stays in Python too, and so
does one that builds a value class with an `int` for a `float` field. Every
other `float` parameter takes an `int` up to 2**53, converted exactly.

A module constant written as an expression, such as `MOD = 10**9 + 7` or
`LIMIT = 1 << 20`, folds into the code rather than staying a global read.
So does a table of numbers bound once at module level, such as
`RATES = (0.1, 0.3, 0.6)`: a tuple of up to 16 ints, floats, or bools of
one type. Native code builds it once per call and reads `RATES[i]` with an
index known only at run time, after the bounds check CPython makes. Being a
constant, it reaches a standalone binary too, where a global read does not.

### Calls

A call between native functions may name its arguments and leave some to
their defaults. The compiler binds the call the way Python would: each
keyword goes to the parameter it names, including a keyword-only one, and
each parameter left out takes its default. The call then goes ahead in
order. This holds for functions, methods, and `__init__`.

```python
def scale(x: int, factor: int = 3, *, offset: int = 0) -> int:
    return x * factor + offset


def use(n: int) -> int:
    return scale(n) + scale(x=n, offset=1) + scale(n, factor=4)
```

Python evaluates arguments in the order they are written. Binding moves a
keyword argument to its parameter's place, so it may only move where the
order cannot be seen: at most one of the keyword arguments that change
places runs any code, and the rest are names, constants, or attributes.
Python evaluates a default once, when the `def` runs, so the compiler puts a
default into the call only where it is a constant: a number, a string,
`None`, or a tuple of those. A call that leaves out a parameter whose
default is anything else (`xs: list[int] = []`), a call with `*args` or
`**kwargs`, and a method call bound by keyword where a subclass overrides
the method stay in Python.

When Python calls a native function with keywords or with defaults left
out, the boundary binds the call by the Python function's own signature,
with its own default values, and calls the native entry in order. A call
that does not bind goes to the Python function, which raises CPython's
`TypeError`. The generated C wrapper does this binding itself: it matches
the call's keyword names against the parameter names and takes left-out
values from the function's `__defaults__` and `__kwdefaults__`, so such a
call costs about what a positional one does ([the table
above](#what-a-call-costs)). When a guard refuses the bound arguments,
Python gets the call as it was written.

A function whose writes all happen inside a callee it handed a buffer to
lowers too: the write lands in the caller's memory either way. The reverse
also lowers: filling memory you allocated and then passing it on.

A call to a function that did not lower goes through Python under `ppy run`:
the native caller boxes the arguments, calls the Python function with the
GIL held, and takes back a result of the type the checker gave the call.
Which results it may take, and why such a call is often a barrier, is in
[Calls into Python](native-effects.md#calls-into-python). A standalone
binary has no Python to call, so there the caller does not lower either,
and the build stops with `E1803`.

`int(x)` of a float truncates toward zero, as CPython does, when the result
is a 64-bit word. The machine's conversion has no answer for the rest (x86
gives -2**63, and C leaves it undefined), so each case is a guard:

| `x` | CPython | `ppy run` | a standalone binary, emitted C or C++ |
|---|---|---|---|
| NaN | `ValueError: cannot convert float NaN to integer` | falls back and raises it | prints it and exits 1 |
| an infinity | `OverflowError: cannot convert float infinity to integer` | falls back and raises it | prints it and exits 1 |
| past 2**63 | the exact integer | falls back and returns it | `OverflowError: the result does not fit in a 64-bit integer`, exit 1 |

The guards are range checks, so `--unsafe` keeps them. Nothing saturates to
the largest or smallest word: Python's integers have no largest.

## Finding what stays in Python

`ppy explain --summary` answers, for a file, a directory, or a project, how
much of the code goes native and what keeps the rest in Python. Here it is
over the `sorts` folder of TheAlgorithms/Python with `--no-strict`, trimmed
to the first two reasons:

```text
175 functions, 1389 statements
  native, called from Python              2 functions (  1%)       24 statements (  2%)
  native, called from native code         2 functions (  1%)       18 statements (  1%)
  Python                                171 functions ( 98%)     1347 statements ( 97%)
  (73 of the Python functions are generic: each native caller compiles its own instance)

what keeps functions in Python, by statements kept out (a function can count under more than one):
      143 statements     10 functions  writes to a parameter native code copies
      return the new value, or take a `Buffer`, a list, or a ppy collection
      see https://ppy.franknoh.dev/latest/guide/native/
      sorts/bead_sort.py:7 sorts.bead_sort.bead_sort
      sorts/circle_sort.py:50 sorts.circle_sort.circle_sort.<locals>.circle_sort_util
      sorts/dutch_national_flag_sort.py:33 sorts.dutch_national_flag_sort.dutch_national_flag_sort
       53 statements     10 functions  a parameter or result with no annotation the checker could infer
      annotate it, or run `ppy convert` to write the inferred annotations
      see https://ppy.franknoh.dev/latest/guide/subset/
      sorts/external_sort.py:13 sorts.external_sort.FileSplitter.__init__
      sorts/external_sort.py:26 sorts.external_sort.FileSplitter.split
      sorts/external_sort.py:48 sorts.external_sort.NWayMerge.select
  ... 60 more reasons, 70 functions (--limit to see more, --json for all)

native, but Python calls the Python body (why its boundary is not used):
      1 functions  copying its strings across costs what one pass over them saves
      1 functions  copying the collections in costs more than the body does with them
```

Read it from the top down:

- The first block counts every function once. "Called from native code"
  means the function compiled but Python calls its Python body, because the
  crossing costs more than the body saves or it passes objects by handle;
  the summary lists those reasons last.
- The reasons are ordered by statements kept out, so the first one is where
  a change moves the most code. A function with several effects counts
  under each.
- A generic function is not a blocker: it has no entry point of its own and
  is compiled for each native caller that names its types. Most of `sorts`
  is generic sorts that nothing calls natively.
- A nested function that shares no variable with the functions around it
  is counted on its own, since it has an entry of its own
  ([Functions as values](closures.md#a-nested-function-in-a-python-function)).
  One that shares a variable lowers with the function around it: it counts
  as native and called from native code when that function is native, and
  otherwise says that the function around it stays in Python, which is the
  reason to fix.
- Each reason says what to do and links the page that explains it. The
  first places it occurs are listed with their line.

`--json` gives every function with its tier and reason, for a script or a
dashboard. See [the command](../cli.md#ppy-explain). Without `--no-strict`
the summary analyzes as `ppy check` does, so an unannotated parameter is an
error and its function counts as Python; on code written without
annotations, pass `--no-strict` to see what `ppy run --no-strict` compiles.

## Threads

The generated wrapper releases the GIL around a native call that loops or
calls another function, so `@ppy.native` functions scale across threads. A
short straight-line body keeps it: dropping the GIL and taking it back costs
about 20 ns, what two operations cost. A function that prints, reads, or
calls into Python keeps the GIL too: its wrapper holds its output until the
call ends and writes it out then, and the call takes the GIL where it
reaches Python ([Effects in native code](native-effects.md)).

Reading input is its own guide: [Reading input](input.md).

Examples: [Algorithms](../howto/15_algorithms.md),
[Buffers and JIT](../howto/12_buffers_and_jit.md).
