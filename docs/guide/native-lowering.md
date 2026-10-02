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

The generated wrapper's call costs about what a Python call does: 31 ns for
`def add(x: int, y: int) -> int: return x + y` called from Python, against
33 ns for CPython's own call of it, measured with `examples/bench_boundary.py`.
So straight-line work pays from two operations (`0.5 * base * height`), and
a one-operation helper stays on the Python side (remarked as `R3004`). The
rest costs more:

| what crosses | straight-line work it takes |
|---|---|
| numbers, tuples of numbers | 2 operations |
| a `str` parameter or result | 4 more each: a native string is made for it |
| a value class | 1 more per field |
| output held while the call runs (a `print`) | a loop: one line written through Python costs more than CPython's `print`, many cost much less |
| a module global the function reads | 6 more: a Python frame reads it for the wrapper |
| a call that draws from `random` | 16: the Python-level binding saves its state |

A function that returns nothing gets the boundary where it loops, or fills
a container the caller passed. A function that only checks its arguments
and raises stays a native caller's.

A container or an object is copied whole on each call, so the body has to
do work in proportion to it ([Lists, dicts, and sets](containers.md#between-functions)).

`ppy explain FILE.ppy:name` reports the decision and, when the answer is no,
the first blocking construct.

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

A function lowers when its types are:

- scalars
- `Buffer[T]`
- homogeneous `list[int]`/`list[float]`, or `Sequence` of those
- all-scalar `@dataclass` value classes (flattened into scalar arguments)

and its body stays inside the modeled subset.

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

A function whose writes all happen inside a callee it handed a buffer to
lowers too: the write lands in the caller's memory either way. The reverse
also lowers: filling memory you allocated and then passing it on.

A call to a function that did not lower keeps its caller on the Python side,
because the call would otherwise name a symbol nothing defines.

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
over the `sorts` folder of TheAlgorithms/Python, trimmed:

```text
167 functions, 1389 statements
  native, called from Python              1 functions (  1%)        8 statements (  1%)
  native, called from native code         0 functions (  0%)        0 statements (  0%)
  Python                                166 functions ( 99%)     1381 statements ( 99%)
  (76 of the Python functions are generic: each native caller compiles its own instance)

what keeps functions in Python, by statements kept out (a function can count under more than one):
      161 statements     13 functions  calls code whose effects are unknown (a library, or a call the checker cannot type)
      annotate it, add a stub or plugin, or call it outside the hot function
      most often: `dict.fromkeys` (1), `file.readlines` (1), `file.write` (1), `heapq.heapify` (1)
      see https://ppy.franknoh.dev/latest/guide/effects/
      sorts/benchmark_sorts.py:56 sorts.benchmark_sorts.is_sorted
      155 statements      9 functions  writes to a parameter native code copies
      return the new value, or take a `Buffer`, a list, or a ppy collection
      see https://ppy.franknoh.dev/latest/guide/native/
      sorts/bead_sort.py:7 sorts.bead_sort.bead_sort
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
- A nested function lowers with the function around it. When that one
  stays in Python, the nested one says so, and the reason to fix is the
  outer function's.
- Each reason says what to do and links the page that explains it. The
  first places it occurs are listed with their line.

`--json` gives every function with its tier and reason, for a script or a
dashboard. See [the command](../cli.md#ppy-explain).

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
