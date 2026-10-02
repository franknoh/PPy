# Native coverage of a real corpus

This page records what `ppy explain --summary` found on a large body of
ordinary Python, and what running that code under `ppy run` against CPython
found. It is input for choosing what 0.6.0 lowers next.

## The corpus

[TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) (MIT), commit
`0a72d14` of 2026-09-28, shallow-cloned outside the repository: 1,603 `.py`
files in 46 top-level directories, 4,686 functions (101 of them nested), 36,969 statements.
It is a good test of code nobody wrote for PPy: mostly annotated, heavy on
lists, dicts, and doctests, with some NumPy, Matplotlib, and network code at
the edges.

The summary runs over the whole tree as one project, with `[tool.ppy]
strict = false`. It takes 32 seconds with the project's cache cold and 19
warm, and peaks under 500 MB. A module whose lowering raises is
reported under "could not be analyzed" and the rest of the report still
comes out. On the current tree no module fails.

## How much goes native

| tier | functions | statements |
|---|---:|---:|
| native, called from Python | 231 (5%) | 2,290 (6%) |
| native, called from native code | 421 (9%) | 1,699 (5%) |
| Python | 4,034 (86%) | 32,980 (89%) |

247 of the Python functions are generic. They have no entry point of their
own, and each native caller compiles its own instance, so the report does not
count them as blocked.

A nested function counts its own body. The function around it counts its
`def` as one statement, so no statement is counted twice. A nested function
has no entry point of its own either: it runs where the function around it
runs.

"Called from native code" means the function compiled, but a call from Python
runs its Python body. The reasons, by count:

| functions | why its boundary is not used |
|---:|---|
| 155 | the boundary crossing costs more than the body saves |
| 115 | returns nothing, which has no Python boundary |
| 95 | takes an object, which native callers pass by handle |
| 22 | returns an object, which native callers receive by handle |
| 20 | copying the collections in costs more than the body does with them |
| 9 | copying its strings across costs what one pass over them saves |
| 5 | a closure, lowered inside the function around it |

## What keeps the rest in Python

By statements kept out. A function can count under more than one reason, so
the rows add up to more than the Python total.

| statements | functions | reason |
|---:|---:|---|
| 8,213 | 840 | calls code whose effects are unknown |
| 3,755 | 355 | does I/O |
| 2,466 | 249 | writes to a parameter native code copies |
| 2,027 | 217 | reads a module global that can change |
| 1,800 | 140 | writes to an object native code does not own |
| 1,554 | 154 | calls `isinstance` |
| 1,388 | 138 | draws random numbers |
| 1,156 | 241 | a parameter or result with no annotation the checker could infer |
| 1,002 | 74 | writes through a name the compiler cannot follow |
| 744 | 91 | calls back into Python |
| 641 | 96 | a nested function whose enclosing function stays in Python |
| 539 | 72 | a generator that is returned, passed on, or stepped inside a loop or a branch |
| 535 | 55 | a `for` over something native code does not walk |
| 530 | 102 | a `numpy.ndarray` parameter |
| 449 | 49 | a chained comparison |
| 407 | 56 | a `list[Any]` parameter |
| 348 | 27 | `in` or `is` against a tuple, a range, or a string |
| 309 | 37 | `**` between two integers |
| 213 | 16 | an empty `[]` or `{}` whose element type is never told |

Closures lower now, and they are not a blocker of their own. Of the 101
nested functions, 96 stay in Python only because the function around them
does, for one of the other reasons in this table. Among the reasons that are
about closures themselves: four closures share a variable of a type native
code has no cell for (a `set[Any]`, a `defaultdict`, a list of lists of
`float | int`), three functions define a nested function with defaults or a
decorator, and one function value returns a tuple of four values. No function
in the corpus stays in Python for an untyped lambda or for a keyword argument
through a function value.

The unknown-effect calls spread thinly. The most frequent callees are
`pytest.raises` (40), `httpx2.get` (39), `np.random.default_rng` (23),
`plt.show` (23), `os.path.dirname` (18), and `random.sample` (17). No one
stub removes much of that row.

What this suggests for 0.6.0, in order of statements freed per unit of work:

1. **`isinstance`** (1,554 statements). Most calls test a parameter against a
   builtin type, as input validation, where the checker already knows the
   answer or the check is one type tag.
2. **Chained comparisons, `in` on a tuple or range, integer `**`, and `for`
   over a tuple** (about 1,600 statements together). Each is small in the
   lowering.
3. **Module globals that can change** (2,027). Most are never written after
   import. Reading them as constants once the module has finished loading
   would free most of the row.
4. **Parameter writes** (2,466) and **writes to unowned objects** (1,800).
   These need the copy-back that lists already have, extended to the other
   shapes.
5. I/O, random numbers, and callbacks are working as intended: those
   functions belong in Python.

## Running the corpus under `ppy run`

Separately, 400 of the corpus's scripts ran under CPython and under `ppy
run`, one at a time, each in its own directory, with empty stdin, a timeout,
and `PYTHONHASHSEED=0`. Both sides used the same interpreter, CPython 3.14. A
program counts as the same when the exit code, stdout, and the exception that
ended it all match. 13 of the 400 were skipped as nondeterministic under
CPython (two runs disagreed).

On the first pass 323 matched and 64 differed. After the fixes below, on the
current tree, 353 match and 34 differ. None of the 34 is `ppy run` printing a
different answer from a program both sides accept, except the one listed
under "not fixed". The differences, sorted:

**PPy bugs, fixed on this branch**, each with a regression under
`tests/fuzz_regressions/`:

- `ppy run` crashed with an LLVM verification error when a function returning
  a string called one that stayed in Python.
- `d.get(k)` without a default crashed lowering.
- The C entry point that stands in for a native function refused keyword
  arguments, and answered to the wrapper module's name with no docstring, so
  `doctest` silently skipped its tests. The Python-level binding had the same
  keyword problem and reported the runtime's module.
- Loop-invariant motion hoisted `row = []` out of a loop, so every pass
  appended to one list (project_euler/problem_049 printed a wrong answer).
- Common-subexpression elimination bound `a, b = [0] * n, [0] * n` to one
  list (sorts/pigeon_sort).
- A `# type: ignore` comment crashed the Python backend's source map.
- A native `sum` over a `list[float]` argument added without CPython's
  compensation, so `sum([0.1, 0.2, 0.3])` printed `0.6000000000000001`.
- The IR verifier took the join after an `if` whose branches both return as
  dominated by nothing, and refused a later read of a local, so a whole
  module failed to lower (backtracking/match_word_pattern).

**A PPy difference, fixed in 0.6.0:** a native function with a `float`
parameter accepted an `int` and converted it, so its result was a float where
CPython's would be an int. `cross_product((0, 0), (1, 1), (2, 2))`, declared
over `tuple[float, float]`, returned `0.0` natively and `0` in CPython
(maths/ear_clipping_polygon_triangulation's doctest showed it). The compiler
now follows each float parameter through the body, and the boundary refuses
an `int` only for the parameters whose int-ness would show; see
[A `float` given an `int`](../guide/native-lowering.md#a-float-given-an-int).

**Checker refusals of valid code, fixed on this branch:**

- A builtin such as `int` did not meet a Protocol bound (`T: Comparable`).
- A `list[Dog]` was refused where a `Sequence[Animal]` was expected.
- `return NotImplemented` in `__eq__` was a return type error.
- A bare `Callable` annotation was "not a type the project can analyze".
- `seen = {}` filled by `seen[k] = 1` in one branch refused `seen[k] += 1` in
  the other, and `[] + names` was refused.
- A `return None` after a `while 1:` that leaves only by `return` was checked
  as reachable.

**Checker refusals of valid code, fixed in 0.6.0:** `typing.Self`,
`queue.Queue` and the other generic collections of the standard library,
old-style `TypeVar` with bounds and constraints and `Generic[T]`, a
`Generator` annotation given a generator expression, `Counter & Counter`,
`datetime + timedelta`, `Decimal / int`, `setattr` on a class and on an
object, a class attribute assigned through its class, a method called
through its class, `__import__("doctest")`, `namedtuple("P", "x y")`,
`list[int]` passed where the callee only reads a `list[int | float]` or a
`Sequence[float]` (the call runs on CPython), and the recursive generic
`RandomizedHeapNode[T] | None`, which is now inferred through the union.

**Code that may pass `None`:** several programs pass `Node | None` where
`Node` is declared, or read `.value` from something that may be `None`.
CPython runs them because the `None` never arrives on that input. Strict
mode still refuses them. `--no-strict` reports them as `W2011` and runs
them, and native code raises CPython's `AttributeError` where the `None`
would arrive.

With these, the same 400 scripts give 377 matching, 10 differing, and 13
skipped. Of the 10, seven are harness artifacts (below), one is a program
that ends in a `NameError` CPython reaches and PPy refuses first
(matrix/validate_sudoku_board), one is the `float` parameter difference
above, and one was a PPy bug the newly accepted programs exposed: constant
folding turned `(-2) ** c` into `-2 ** c` in the Python backend
(conversions/negative_binary_base_to_int), now fixed, with a regression.

**Harness artifacts:** copied to a directory of their own, programs that
import a sibling (`from .stack import Stack`, `from data_structures...`) fail
on both sides, and programs that read `input()` end in `EOFError` on both
sides. They count as different only because `ppy run` stops earlier, at the
check, with its own error.

## After the standard library went native

With `random`, `math`, `heapq`, `bisect`, `itertools`, and `string` lowered
(see [the guide](../guide/stdlib.md)), the same summary over the same tree:

| tier | functions | statements |
|---|---:|---:|
| native, called from Python | 254 (5%) | 2,497 (7%) |
| native, called from native code | 463 (10%) | 1,870 (5%) |
| Python | 3,969 (85%) | 32,602 (88%) |

"Draws random numbers" is gone from the reasons: 138 functions and 1,388
statements. Calls with unknown effects went from 840 functions (8,213
statements) to 753 (7,416). Some of the freed functions now stop at the
next reason in their body, which is why a few rows grew: a `for` over a
tuple (55 to 62 functions), integer `**` (37 to 42), a `list[Any]`
parameter (56 to 59).

The 400-script comparison ran again with `random` seeded before each
program on both sides (a `sitecustomize` that calls `random.seed`), so
programs that draw are compared too. 357 match, 33 differ, and 10 were
skipped as nondeterministic or slow under CPython. Of the programs that
import `random` and ran, all but two match. All 33 differences also differ
on the tree before this change, with the same seed: none is new.

## With all of 0.6.0

The same summary over the same tree, with everything 0.6.0 lowers: effects
in native code, module globals and objects across the boundary, the new
expressions and loops, and the standard library.

| tier | functions | statements |
|---|---:|---:|
| native, called from Python | 523 (11%) | 5,312 (14%) |
| native, called from native code | 713 (15%) | 3,161 (9%) |
| Python | 3,450 (74%) | 28,496 (77%) |

Against 0.5.0, the functions Python calls natively went from 231 to 523, and
all compiled functions from 652 to 1,236.

Several reasons left the table: calls with unknown effects (840 functions)
and I/O (355), which native code now holds as effects; `isinstance` (154);
random numbers (138); calls back into Python (91); generators that are
returned or passed on (72); `for` over a tuple (55); and chained comparisons
(49). Reading a module global that can change fell from 217 functions to 61.

Some freed functions stop at the next reason in their body, so the top of
the table is now:

| statements | functions | reason |
|---:|---:|---|
| 3,008 | 428 | a parameter or result with no annotation the checker could infer |
| 2,059 | 202 | writes to a parameter native code copies |
| 1,803 | 140 | writes to an object native code does not own |
| 905 | 143 | a `numpy.ndarray` parameter |
| 731 | 79 | a `list[Any]` parameter |
| 630 | 94 | a nested function whose enclosing function stays in Python |
| 615 | 61 | reads a module global that can change |
| 540 | 41 | writes through a name the compiler cannot follow |
| 396 | 36 | calls a Python function whose result native code cannot take back |
| 388 | 41 | keyword arguments have no native ABI |

Of the compiled functions Python does not call natively, 329 are too small
for the boundary to pay off and 208 do less with their collections than
copying them in costs.

The 400-script comparison, run again on the same programs: 387 match and 13
are skipped as nondeterministic or slow under CPython. None differs, down
from 34 when the 0.6.0 work began. The last eight were programs `ppy run`
refused at compile time under `strict = false`: an import of a sibling
module that CPython also fails to find, a `MutableSequence[T]` parameter, a
generator expression assigned to a bare `Generator`, and `globals()` passed
to `timeit`. Under `--no-strict` these are now warnings, and that code runs
on CPython.
