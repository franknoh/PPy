# Exceptions

Exception behavior is part of the contract: `divide(1, 0)` raises
`ZeroDivisionError` and `at([10, 20, 30], 9)` raises `IndexError` at the
same point, with the same type, whichever path runs them.

## A guard where Python would raise

```python
@ppy.opt(3)
def at(values: list[int], index: int) -> int:
    return values[index]
```

Every place the program must be handed back to CPython — a zero divisor, an
index past the end, a shift past the word — is a `core.guard` in the IR.
Native code takes the guard's failure edge and the wrapper re-runs the
Python body, which raises exactly what Python raises. `safe_divide` checks
`b == 0` itself and never fails a guard; `divide` does not, and its
`ZeroDivisionError` comes from the fallback.

## The optimizer keeps the order

Optimization passes do not move or drop an operation that can raise unless
they can prove it cannot. Raising is an effect (`may_raise`) the passes read
like any other, so an `IndexError` on the third element still happens after
the print of the second.

## Run it

```bash
python  errors.ppy
ppy     errors.ppy
ppy run errors.ppy
```

<!-- outputs:start -->
## What it prints

**`python  errors.ppy`**, **`ppy     errors.ppy`**, **`ppy run errors.ppy`**

```text
3 0
3 -4
caught ZeroDivisionError
20
caught IndexError
```

<!-- outputs:end -->

Read on: [Effects and purity](../../docs/guide/effects.md) ·
[Numerics](../11_numerics/README.md)

`errors.ppy` is hand-written; there is no `.py` source and no conversion step.
