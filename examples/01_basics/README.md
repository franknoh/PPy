# Basics

Fixed-width integer markers, a purity contract, and a per-function
optimization level. Run with `python` the file is ordinary Python; run with
`ppy run` the same three functions compile through LLVM, print the same
three numbers, and the checker has verified two contracts on the way.

## Fixed-width markers

```python
@ppy.pure
@ppy.opt(3)
def square(x: i64) -> i64:
    return x * x
```

A Python `int` is unbounded, so a native `x * x` on a plain `int` has to be
guarded: overflow falls back to CPython's arbitrary precision. `ppy.i64` is
a contract that the value fits 64 bits. The checker holds callers to it — a
value that provably leaves the range is `E1401` — and the native multiply
needs no guard. `@ppy.opt(3)` raises the optimization level for this one
function; the project default is 2.

## The purity contract

`@ppy.pure` says the function has no observable effect: no I/O, no global
writes, no mutation of its arguments. The checker proves it across calls —
`sum_of_squares` is pure because `square` is — and a violation is an error
(`E1601`), not a warning. `collatz_steps` keeps a plain `int`: its loop runs
natively with an overflow guard per operation, and a value that outgrows a
word takes the fallback and still answers correctly.

## Run it

```bash
python  basics.ppy
ppy     basics.ppy
ppy run basics.ppy
```

<!-- outputs:start -->
## What it prints

**`python  basics.ppy`**

```text
49 285 111
```

**`ppy     basics.ppy`**

```text
49 285 111
```

**`ppy run basics.ppy`**

```text
49 285 111
```

<!-- outputs:end -->

Read on: [Directives and markers](../../docs/guide/directives.md) ·
[Arbitrary precision](../02_arbitrary_precision/README.md)

`basics.ppy` is hand-written; there is no `.py` source and no conversion step.
