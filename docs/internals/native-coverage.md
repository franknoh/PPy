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

**A PPy difference, not fixed:** a native function with a `float` parameter
accepts an `int` and converts it, so its result is a float where CPython's
would be an int. `cross_product((0, 0), (1, 1), (2, 2))`, declared over
`tuple[float, float]`, returns `0.0` natively and `0` in CPython
(maths/ear_clipping_polygon_triangulation's doctest shows it now that doctest
sees native functions). Refusing the int at the boundary is a one-line change
but sends every such call to Python; converting only where the result cannot
tell is the better fix and needs the lowering to know it. It is left for
0.6.0.

**Checker refusals of valid code, fixed on this branch:**

- A builtin such as `int` did not meet a Protocol bound (`T: Comparable`).
- A `list[Dog]` was refused where a `Sequence[Animal]` was expected.
- `return NotImplemented` in `__eq__` was a return type error.
- A bare `Callable` annotation was "not a type the project can analyze".
- `seen = {}` filled by `seen[k] = 1` in one branch refused `seen[k] += 1` in
  the other, and `[] + names` was refused.
- A `return None` after a `while 1:` that leaves only by `return` was checked
  as reachable.

**Checker refusals of valid code, not fixed:**

- `typing.Self`, `queue.Queue`, and old-style `TypeVar` (27 corpus files use
  it) are not supported annotations.
- A `Generator` annotation given a generator expression, which the checker
  types as `Iterator`.
- `Counter & Counter`, `datetime + timedelta`, and `Decimal / int` have no
  model in the standard-library stubs.
- `list[int]` passed where `list[int | float]` or `Sequence[float]` is
  expected. The first is correct under invariance, and the second is kept
  refused on purpose: accepted, the native code would compute in floats and
  print `9.0` where CPython prints `9`, the difference above. CPython runs
  both, and the corpus does it often.
- Old-style type variables (`T = TypeVar("T")`) used in annotations, and
  `setattr` on a class.
- A recursive generic `RandomizedHeapNode[T] | None` compared unequal to
  itself.

**Checker refusals of code that is wrong:** several programs pass
`Node | None` where `Node` is declared, or read `.value` from something that
may be `None`. CPython runs them because the `None` never arrives on that
input. PPy refuses them in either mode. Whether non-strict mode should warn
instead is a policy question for 0.6.0.

**Harness artifacts:** copied to a directory of their own, programs that
import a sibling (`from .stack import Stack`, `from data_structures...`) fail
on both sides, and programs that read `input()` end in `EOFError` on both
sides. They count as different only because `ppy run` stops earlier, at the
check, with its own error.
