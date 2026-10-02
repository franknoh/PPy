# An unannotated module

A module of number-theory helpers written the way script collections such
as TheAlgorithms/Python write them: no annotations, a doctest on each
function, and the work under `if __name__ == "__main__":`. The folder's
`pyproject.toml` turns strict mode off, and `ppy run` compiles every
function from the types its calls, its default values, and its doctests
pass.

## Run it

```bash
python  numbers.ppy
ppy run numbers.ppy
ppy explain numbers.digit_sum
ppy explain --summary numbers.ppy
```

<!-- outputs:start -->
## What it prints

**`python  numbers.ppy`**, **`ppy run numbers.ppy`**

```text
primes below 2,000,000: 148933
longest Collatz chain below 1,000,000: (837799, 524)
Harshad numbers below 1,000,000: 95427 the last 999990
mean digit sum: 27.0
is_prime(2**31 - 1): True
digit_sum(10**30 - 1): 270
```

**`ppy explain numbers.digit_sum`**

<details markdown="1">
<summary>22 lines</summary>

```text
function: digit_sum
qualname: numbers.digit_sum
semantic type: (int, int) -> int
effects: MayRaise[ZeroDivisionError]
purity: inferred pure
optimization: O2
python backend: optimized
llvm backend: native
python boundary: a generated CPython-ABI wrapper
jit: not requested
parallel: rejected
reason: `number` carries a dependency across iterations
reductions: +
representation:
  number: int -> i64 (guarded)
  base: int -> i64 (guarded)
  return: int -> PyLong*
inferred (not annotated):
  number: int, from 3 calls (numbers.ppy:90, numbers.ppy:72, numbers.ppy:88)
  base: int, from 1 call (numbers.ppy:72) and the default value
  (the Python boundary checks these at each call, and runs the Python body otherwise)
  return: int, from the body's return statements
```

</details>

**`ppy explain --summary numbers.ppy`**

```text
7 functions, 36 statements
  native, called from Python              7 functions (100%)       36 statements (100%)
  native, called from native code         0 functions (  0%)        0 statements (  0%)
  Python                                  0 functions (  0%)        0 statements (  0%)
```

<!-- outputs:end -->

## The program

- `is_prime(number)` tries odd divisors up to the square root.
- `digit_sum(number, base=10)` adds the digits of `number` in `base`.
- `collatz_steps(start)` counts the steps of the Collatz sequence from
  `start` down to 1, and `longest_collatz(limit)` finds the start below
  `limit` with the longest one. It returns a tuple `(start, steps)`.
- `count_primes(limit)` counts with `sum(1 for n in range(limit) if
  is_prime(n))`.
- `harshad_numbers(limit, base=10)` lists the numbers divisible by their
  digit sum.
- `mean(values)` divides a sum by a length.
- The main block counts the primes below 2,000,000, finds the longest
  Collatz chain below 1,000,000, lists the Harshad numbers below
  1,000,000, and takes the mean digit sum of the numbers below 1,000,000.
  Its last two lines call `is_prime(2**31 - 1)` and
  `digit_sum(10**30 - 1)`.

## Where the types come from

No parameter is annotated. In strict mode (`strict = true`, the default)
`ppy check numbers.ppy` reports nine `E1201` errors, one per parameter,
and `ppy run` stops. With `strict = false` in `[tool.ppy]` the compiler
looks for the types elsewhere:

- **Calls.** Every call in the project that the checker can type is
  evidence. `count_primes(2_000_000)` in the main block makes `limit` an
  `int`, and `is_prime(n)` inside `count_primes`, with `n` from a `range`,
  makes `number` an `int`. Where calls disagree, `int` and `float` join to
  `float`; any other mix leaves the parameter unknown, and the function
  runs on CPython.
- **Defaults.** `base=10` counts as a call passing `10`, so `base` is an
  `int` even in `harshad_numbers`, which no caller passes a base to.
- **Doctests.** A parameter that no call types takes what its module's
  doctests pass as literals. Here every function is also called from the
  main block, so the doctests only agree with the calls.
- **Results** come from the `return` statements, through recursion too:
  `longest_collatz` returns `tuple[int, int]`.

`ppy explain` lists each inferred type with the calls it came from. For
`digit_sum` (below) that is three calls for `number`, and one call plus the
default for `base`. `ppy explain --summary` counts the seven functions as
native and called from Python.

## The boundary still checks

An inferred type is a guess from the calls the project makes, and another
caller may pass something else. Every native entry checks the exact type of
each argument when Python calls it, and runs the function's Python body
when one does not match, as it does for a declared type. That is why
`digit_sum(10**30 - 1)` prints CPython's 270: the argument does not fit a
64-bit word, the entry's check fails, and the Python body adds the digits
in arbitrary precision. A doctest that passes a `float`, or a call through
`getattr`, takes the same path.

## Timing

Wall time for the whole program, the mean of five runs, measured from a
checkout under `/tmp` on one machine (Python 3.14, an Intel Core Ultra 9
386H under WSL2):

| | seconds |
|---|---:|
| `python numbers.ppy` | 7.22 |
| `ppy run numbers.ppy`, after the first run built the cache | 1.96 |

Timed one part at a time inside the program (one run each):

| part | CPython | `ppy run` |
|---|---:|---:|
| `count_primes(2_000_000)` | 2.53 | 0.13 |
| `longest_collatz(1_000_000)` | 4.19 | 0.17 |
| `harshad_numbers(1_000_000)` | 0.17 | 0.03 |
| `mean([float(digit_sum(n)) for n in range(1_000_000)])` | 0.21 | 1.55 |

The last line is slower under `ppy run`. Its comprehension runs in Python
and calls `digit_sum(n)` a million times, leaving `base` to its default.
A call from Python that leaves out a default, or names an argument, is
bound by the boundary in Python against the function's signature, which
costs about 1.5 µs; the same call with both arguments, `digit_sum(n, 10)`,
costs about 70 ns, against about 155 ns for CPython's own call. A loop
that calls a native function many times from Python should pass every
argument by position, or move into a function of its own.

The program has no `main()`, so there is no standalone build of it; a
standalone binary needs a native `main` to start from.

## Where the code comes from

`numbers.ppy` is hand-written.

Read on: [Types from call sites](../../docs/guide/subset.md#types-from-call-sites)
and [`ppy explain`](../../docs/cli.md#ppy-explain).
