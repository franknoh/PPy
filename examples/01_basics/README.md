# Three annotations, one native loop

`basics.ppy` is a Python file. Run it with `python` and nothing PPY-specific
happens. Run it with `ppy run` and the same three functions compile through
LLVM, print the same three numbers, and the checker has verified two
contracts on the way: that `square` and `sum_of_squares` are pure, and that
their arguments fit a machine word.

## Say the width, skip the proof

```python
@ppy.pure
@ppy.opt(3)
def square(x: i64) -> i64:
    return x * x
```

A Python `int` is unbounded, so a native `x * x` on a plain `int` has to be
guarded: overflow falls back to CPython's arbitrary precision. `ppy.i64` is
a contract that the value fits 64 bits. The checker holds callers to it
(`E1401` when a value provably leaves the range) and the native code needs no
guard on the multiply. `@ppy.opt(3)` raises the optimization level for this
one function; the project default is 2.

## A contract, not a hint

`@ppy.pure` says the function has no observable effect: no I/O, no global
writes, no mutation of its arguments. The checker proves it across calls —
`sum_of_squares` is pure because `square` is — and a violation is an error
(`E1601`), not a warning. `collatz_steps` keeps a plain `int`: the loop runs
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

## Read on

- [Directives and markers](../../docs/guide/directives.md) — every directive and every marker, in one table each.
- [Arbitrary precision](../02_arbitrary_precision/README.md) — what happens when a plain `int` does outgrow a word.

`basics.ppy` is hand-written; there is no `.py` source and no conversion step.
