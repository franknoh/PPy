# Options and doctests

A command-line script written without annotations: `argparse` options,
doctests on the functions, a `functools.lru_cache` recursion, and a small
vector class used through `+`, `*`, and `<`. The folder's `pyproject.toml`
turns strict mode off. `ppy run` types the functions from the parser's
`type=int` options, from the operators applied to the class, and from the
calls and doctests, and compiles them.

## Run it

```bash
python  puzzles.ppy
ppy run puzzles.ppy
ppy run puzzles.ppy --limit 1000 --terms 50 --steps 1000 --seed 3
ppy explain puzzles.walk
ppy explain --summary puzzles.ppy
```

<!-- outputs:start -->
<!-- outputs:end -->

## The program

- `Vector(x, y)` has `__add__`, `__mul__` by a number, `__lt__` that
  compares squared lengths, and `norm()`, the squared length.
- `sieve(limit)` marks composites in a list of bools and returns the
  primes up to `limit`.
- `partitions(total, largest)` counts the ways to write `total` as a sum
  of parts no larger than `largest`, under `@functools.lru_cache`.
- `walk(steps, seed)` takes `steps` steps of a random walk on the grid
  (from a linear congruential generator), each one a `Vector` added to the
  position, and returns the squared distance of the farthest point it
  reached, comparing points with `<`.
- `mean_gap(primes, tolerance=EPSILON)` is the mean gap between
  consecutive primes.
- `main()` reads `--limit`, `--terms`, `--steps`, and `--seed` with
  `argparse` and prints each result. The lines that start with `#` are
  timings, which differ between runs.

## Where the types come from

No parameter is annotated. In strict mode `ppy check puzzles.ppy` reports
27 errors: twelve `E1201`, one per parameter, and the reads and calls that
follow from them. With `strict = false`:

- **Options.** `parser.add_argument("--limit", type=int, default=5_000_000)`
  makes `args.limit` an `int`: the parser gives it a value of that type
  whether or not the option is on the command line. `sieve(args.limit)`
  is then a call with an `int`, and so are the calls of `partitions` and
  `walk`. An option with no default that is not required could be `None`
  and would not count.
- **Operators.** `position + step * (1 + state % 2)` in `walk` is a call of
  `Vector.__mul__` with an `int` and of `Vector.__add__` with a `Vector`,
  and `farthest < position` is a call of `Vector.__lt__`. A comparison
  method is only typed as taking its own class, since sorting and lookups
  call it with two of the class's objects.
- **Decorators.** `lru_cache` keeps the function's parameters, so
  `partitions` is typed from its calls as an undecorated function would
  be. A project decorator counts the same way only when its wrapper is
  made with `functools.wraps(fn)` and does nothing but call
  `fn(*args, **kwargs)`.
- **Defaults.** `tolerance=EPSILON` is a module constant written as a
  literal, so it counts as a call passing a `float`.
- **Doctests.** Here every function is also called from `main`, so the
  doctests only agree with the calls. Without `main`, they would type some
  of the same parameters: `sieve(20)` gives `limit` an `int`, and `a * 3`
  on the doctest's `a = Vector(1, 2)` gives `__mul__`'s `factor` one. The
  `Vector(3, 4)` and `Vector(2, 2)` operands of `+` and `<` in the doctest
  are not counted, so `other` would stay unknown.

`main` itself stays in Python: the parser is a Python object native code
has no value for. Everything it calls is native, and `ppy explain
--summary` counts `partitions` as native but called through its Python
body by Python, because it calls itself without a loop and CPython's
recursion limit stays in force.

## The boundary still checks

`walk` is native and checks its two arguments when Python calls it. The
third command passes small options: the values are still `int`s, so
nothing changes but the sizes. From `--terms 406` the count of
`partitions` passes 64 bits. Python calls `partitions` through its Python
body already, so the sum is CPython's own and every path prints the same
number.

## Timing

Wall time for the whole program, the mean of five runs, measured from a
checkout under `/tmp` on one machine (Python 3.14, an Intel Core Ultra 9
386H under WSL2):

| | seconds |
|---|---:|
| `python puzzles.ppy` | 1.14 |
| `ppy run puzzles.ppy`, after the first run built the cache | 0.42 |

Each part, as the program prints it, the mean of five runs in seconds:

| part | CPython | `ppy run` |
|---|---:|---:|
| `sieve(5_000_000)` | 0.242 | 0.128 |
| `mean_gap` | 0.010 | 0.001 |
| `partitions(400, 400)` | 0.035 | 0.030 |
| `walk(2_000_000, 7)` | 0.818 | 0.207 |

There is no standalone build: `main` stays in Python.

## Where the code comes from

`puzzles.ppy` is hand-written.

Read on: [Types from call sites](../../docs/guide/subset.md#types-from-call-sites)
and [Classes](../../docs/guide/classes.md#operators).
