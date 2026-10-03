# Errors and generators

A ledger read from a generator, with the bad rows raised as exceptions and
caught, and primes from a generator reduced by generator expressions, all in
native code.

## Run it

```bash
python  ledger.ppy
ppy run ledger.ppy
ppy build --standalone ledger.ppy -o dist && ./dist/ledger
```

<!-- outputs:start -->
## What it prints

**`python  ledger.ppy`**, **`ppy run ledger.ppy`**, **`ppy build --standalone ledger.ppy -o dist && ./dist/ledger`**

```text
50 accounts, 10055 bad rows (longest message 23), richest 185856, overdrawn 30
78498 primes, widest gap 114, 8169 twin pairs, past the end -1
```

<!-- outputs:end -->

## The program

- `ledger(count)` is a generator. It yields 500,000 rows as text,
  `acct7,-120` and the like; now and then a row has no comma, or no amount.
- `amount_of(row)` splits a row with `partition` and raises `BadRow`, a
  subclass of `ValueError`, with a message saying what is missing.
- `settle` loops over `ledger(ROWS)`. Each row goes through `amount_of` in a
  `try`; the handler counts the bad row, keeps the longest `str(error)`, and
  `continue`s. The good rows add up per account in a `HashMap`. The richest
  balance and the number overdrawn come from `max` and `sum` over generator
  expressions.
- `primes(limit)` yields the primes below 1,000,000 by trial division, and
  `gaps` builds a `Vec` from it, finds the widest gap and the twin pairs
  with generator expressions, and reads one past the end in a `try` that
  catches the `IndexError`.

## How it runs natively

`ppy explain ledger.amount_of`, `ledger.settle`, and `ledger.gaps` each say
`llvm backend: native`.

- `raise BadRow(...)` makes an exception object holding the class, the
  message, and whether the message is CPython's. `amount_of` releases what
  it holds and returns a raised status.
- The call in `settle` sits in a `try`, so a raised status goes to the
  handlers, which compare the exception's class with `BadRow` and the
  classes deriving from it. The `continue` in the handler lets go of the
  exception first.
- `found[len(found)]` is a bounds check. In a module that raises or catches,
  a failed check raises CPython's `IndexError` natively, so the `except
  IndexError` takes it.
- `ledger` and `primes` are not objects. Each is lowered into the loop that
  consumes it: the generator's body runs, and at each `yield` the loop's
  body runs with the value. A generator expression is lowered the same way.

## Timing

Wall time for the whole program, the mean of five runs, measured from a
checkout under `/tmp` on one machine (Python 3.14, an Intel Core Ultra 9
386H under WSL2):

| | seconds |
|---|---:|
| `python ledger.ppy` | 2.14 |
| `ppy run ledger.ppy`, after the first run built the cache | 0.30 |
| `./dist/ledger`, the standalone binary | 0.31 |

`ppy run` also starts CPython, names the cached build from the project's
sources, and loads its native library, which the standalone binary does
not pay.

## Where the code comes from

`ledger.ppy` is hand-written.

Read on: [the exceptions and generators guide](../../docs/guide/exceptions-and-generators.md).
