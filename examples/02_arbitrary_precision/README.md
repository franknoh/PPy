# Arbitrary precision

Integers stay Python integers: when a native multiply overflows a 64-bit
word, the call goes back to CPython and still returns the exact answer.
`cube(3)` runs as three native multiplies. `cube(10**7)` overflows a word on
the second one, and the guard hands the call back to CPython, which prints
the 21-digit answer.

## Run it

```bash
python  arbitrary_precision.ppy
ppy     arbitrary_precision.ppy
ppy run arbitrary_precision.ppy
```

## The overflow guard

```python
@ppy.pure
def cube(x: int) -> int:
    return x * x * x
```

Native code multiplies 64-bit words. Every `+`, `-`, and `*` on a plain
`int` lowers to LLVM's overflow-checking form. A set flag means the true
value no longer fits, so the function returns to its Python body with the
original arguments and computes there. The native path does not answer with
a wrapped number: it either has the right one or steps aside.

`ppy build` keeps the same guards, so an artifact answers as `ppy run`
does. `--unsafe`, on either command, drops them. The code then wraps at 64
bits like every native compiler's output. That is the 10 ms between
`ppy run` and `ppy build --unsafe` on the README's collatz kernel.

## Floor division and the sign of the remainder

`floor_and_mod(-7, 2)` is `-4 + 1`, and `floor_and_mod(7, -2)` is
`-4 + -1`: `//` rounds toward negative infinity and `%` takes the sign of
the divisor. C's truncating division would give `-3`. The IR carries
`rounding = "floor"` on the operation, and the LLVM backend emits the
sign-corrected sequence.

Division by zero is a guard as well: `divide(1, 0)` raises
`ZeroDivisionError` from the Python body.

<!-- outputs:start -->
## What it prints

**`python  arbitrary_precision.ppy`**, **`ppy     arbitrary_precision.ppy`**, **`ppy run arbitrary_precision.ppy`**

```text
27
1000000000000000000000
-3 -5
caught ZeroDivisionError
```

<!-- outputs:end -->

Read on: [Numerics](../11_numerics/README.md) ·
[The IR](../../docs/internals/ir.md)

`arbitrary_precision.ppy` is hand-written; there is no `.py` source and no
conversion step.
