# Native coverage of a real corpus

This page records what `ppy explain --summary` found on a large body of
ordinary Python, and what running that code under `ppy run` against CPython
found. It is input for choosing what 0.6.0 lowers next.

## The corpus

[TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) (MIT), commit
`0a72d14` of 2026-09-28, shallow-cloned outside the repository: 1,603 `.py`
files in 46 folders, 4,585 functions, 36,972 statements. It is a good test of
code nobody wrote for PPy: mostly annotated, heavy on lists, dicts, and
doctests, with some NumPy, Matplotlib, and network code at the edges.

The summary runs over the whole tree as one project, with `[tool.ppy]
strict = false`. Cold, it takes 45 seconds, because every module is analyzed
and lowered once. Warm, from the analysis and lowering cache, it takes 21
seconds. Both peak under 500 MB. A module whose lowering raises is reported under "could not be
analyzed" and the rest of the report still comes out.

## How much goes native

| tier | functions | statements |
|---|---:|---:|
| native, called from Python | 224 (5%) | 2,233 (6%) |
| native, called from native code | 410 (9%) | 1,620 (4%) |
| Python | 3,951 (86%) | 33,119 (90%) |

247 of the Python functions are generic. They have no entry point of their
own, and each native caller compiles its own instance, so the report does not
count them as blocked.

"Called from native code" means the function compiled, but a call from Python
runs its Python body. The reasons, by count:

| functions | why its boundary is not used |
|---:|---|
| 155 | the boundary crossing costs more than the body saves |
| 114 | returns nothing, which has no Python boundary |
| 90 | takes an object, which native callers pass by handle |
| 22 | returns an object, which native callers receive by handle |
| 20 | copying the collections in costs more than the body does with them |
| 9 | copying its strings across costs what one pass over them saves |

## What keeps the rest in Python

By statements kept out. A function can count under more than one reason, so
the rows add up to more than the Python total.

| statements | functions | reason |
|---:|---:|---|
| 9,310 | 902 | calls code whose effects are unknown |
| 3,865 | 355 | does I/O |
| 2,453 | 247 | writes to a parameter native code copies |
| 2,069 | 216 | reads a module global that can change |
| 1,777 | 135 | writes to an object native code does not own |
| 1,554 | 154 | calls `isinstance` |
| 1,395 | 138 | draws random numbers |
| 1,158 | 240 | a parameter or result with no annotation the checker could infer |
| 1,013 | 74 | writes through a name the compiler cannot follow |
| 765 | 92 | calls back into Python |
| 560 | 72 | a generator that is stored, returned, or stepped by hand |
| 530 | 102 | a `numpy.ndarray` parameter |
| 520 | 54 | a `for` over something native code does not walk |
| 443 | 48 | a chained comparison |
| 399 | 55 | a `list[Any]` parameter |
| 313 | 26 | `in` or `is` against a tuple, a range, or a string |
| 235 | 33 | `**` between two integers |
| 187 | 14 | an empty `[]` or `{}` whose element type is never told |

The unknown-effect calls spread thinly. The most frequent callees are
`pytest.raises` (40), `httpx2.get` (39), `np.random.default_rng` (23),
`plt.show` (23), `os.path.dirname` (18), and `random.sample` (17). No one
stub removes much of that row.

What this suggests for 0.6.0, in order of statements freed per unit of work:

1. **`isinstance`** (1,554 statements). Most calls test a parameter against a
   builtin type, as input validation, where the checker already knows the
   answer or the check is one type tag.
2. **Chained comparisons, `in` on a tuple or range, integer `**`, and `for`
   over a tuple** (about 1,500 statements together). Each is small in the
   lowering.
3. **Module globals that can change** (2,069). Most are never written after
   import. Reading them as constants once the module has finished loading
   would free most of the row.
4. **Parameter writes** (2,453) and **writes to unowned objects** (1,777).
   These need the copy-back that lists already have, extended to the other
   shapes.
5. I/O, random numbers, and callbacks are working as intended: those
   functions belong in Python.

## Running the corpus under `ppy run`

Separately, 400 programs from the corpus ran under CPython and under `ppy
run`, one at a time, each in its own directory, with empty stdin, a timeout,
and `PYTHONHASHSEED=0`. Both sides used the same interpreter. A program counts
as the same when the exit code, stdout, and the exception that ended it all
match. 13 of the 400 were skipped as nondeterministic under CPython (two
runs disagreed).

323 matched and 64 differed on the first pass. The differences, sorted:

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

**Checker refusals of valid code, fixed on this branch:**

- A builtin such as `int` did not meet a Protocol bound (`T: Comparable`).
- A `list[Dog]` was refused where a `Sequence[Animal]` was expected.
- `return NotImplemented` in `__eq__` was a return type error.
- A bare `Callable` annotation was "not a type the project can analyze".
- `seen = {}` filled by `seen[k] = 1` in one branch refused `seen[k] += 1` in
  the other, and `[] + names` was refused.

**Checker refusals of valid code, not fixed:**

- `typing.Self`, `queue.Queue`, and old-style `TypeVar` (27 corpus files use
  it) are not supported annotations.
- A `Generator` annotation given a generator expression, which the checker
  types as `Iterator`.
- `Counter & Counter`, `datetime + timedelta`, and `Decimal / int` have no
  model in the standard-library stubs.
- `list[int]` passed where `list[int | float]` is expected. This is correct
  under invariance, but CPython runs it, and the corpus does it often.
- A recursive generic `RandomizedHeapNode[T] | None` compared unequal to
  itself.

**Checker refusals of code that is wrong:** several programs pass
`Node | None` where `Node` is declared, or read `.value` from something that
may be `None`. CPython runs them because the `None` never arrives on that
input. PPy refuses them in either mode. Whether non-strict mode should warn
instead is a policy question for 0.6.0.

**Harness artifacts, not differences:** copied to a directory of their own,
programs that import a sibling (`from .stack import Stack`, `from
data_structures...`) fail on both sides, and programs that read `input()`
end in `EOFError` on both sides. Some programs use Python 2 syntax that
CPython 3 rejects. All of these fail the same way under `ppy run`.
