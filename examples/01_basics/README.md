# Basics

Three small functions show fixed-width integer markers, a purity contract,
and a per-function optimization level. Run it with `python` and it is
ordinary Python. Run it with `ppy run` and the same three functions compile
through LLVM, after the checker has verified two contracts.

## Run it

```bash
python  basics.ppy
ppy     basics.ppy
ppy run basics.ppy
```

## Fixed-width markers

```python
@ppy.pure
@ppy.opt(3)
def square(x: i64) -> i64:
    return x * x
```

A Python `int` is unbounded, so a native `x * x` on a plain `int` has to be
guarded: on overflow it falls back to CPython's arbitrary precision.

`ppy.i64` is a contract that the value fits 64 bits. The checker holds
callers to it (a value that provably leaves the range is `E1401`), so the
native multiply needs no guard.

`@ppy.opt(3)` raises the optimization level for this one function. The
project default is 2.

## The purity contract

`@ppy.pure` says the function has no observable effect: no I/O, no global
writes, no mutation of its arguments. The checker proves it across calls:
`sum_of_squares` is pure because `square` is. A violation is an error
(`E1601`), not a warning.

`collatz_steps` keeps a plain `int`. Its loop runs natively with an overflow
guard per operation, and a value that outgrows a word takes the fallback and
still gets the correct answer.

<!-- outputs:start -->
## What it prints

**`python  basics.ppy`**, **`ppy     basics.ppy`**, **`ppy run basics.ppy`**

```text
49 285 111
```

<!-- outputs:end -->

Read on: [Directives and markers](../../docs/guide/directives.md) ·
[Arbitrary precision](../02_arbitrary_precision/README.md)

`basics.ppy` is hand-written; there is no `.py` source and no conversion step.
