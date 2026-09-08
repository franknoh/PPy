# A machine word until it is not

Native code multiplies 64-bit integers. Python multiplies integers of any
size. PPY keeps both promises at once: `cube(3)` runs as three native
multiplies, `cube(10**7)` overflows a word on the second one, and the guard
that catches it hands the call back to CPython, which prints the
21-digit answer Python always did.

## The fallback is a guard, not a different answer

```python
@ppy.pure
def cube(x: int) -> int:
    return x * x * x
```

Every `+`, `-`, and `*` on a plain `int` lowers to LLVM's overflow-checking
form. The flag set means the true value no longer fits, so the function
returns to its Python body with the original arguments and computes there.
The three paths print the same output because the native path never answers
with a wrapped number; it either has the right one or steps aside.

`ppy build` flips that default — an artifact wraps at 64 bits like every
native compiler's output, and `--safe` keeps the guards — which is why the
README's collatz kernel is 10 ms faster built than run.

## Floor division and the sign of the remainder

`floor_and_mod(-7, 2)` is `-4 + 1`, and `floor_and_mod(7, -2)` is `-4 + -1`:
Python's `//` rounds toward negative infinity and `%` takes the sign of the
divisor. The IR carries `rounding = "floor"` on the operation, and the LLVM
backend emits the sign-corrected sequence, so the native path agrees with
CPython where C's truncating division would not. Division by zero is a guard
too: `divide(1, 0)` raises `ZeroDivisionError` from the Python body on every
path.

## Run it

```bash
python  arbitrary_precision.ppy
ppy     arbitrary_precision.ppy
ppy run arbitrary_precision.ppy
```

<!-- outputs:start -->
## What it prints

**`python  arbitrary_precision.ppy`**

```text
27
1000000000000000000000
-3 -5
caught ZeroDivisionError
```

**`ppy     arbitrary_precision.ppy`**

```text
27
1000000000000000000000
-3 -5
caught ZeroDivisionError
```

**`ppy run arbitrary_precision.ppy`**

```text
27
1000000000000000000000
-3 -5
caught ZeroDivisionError
```

<!-- outputs:end -->

## Read on

- [Numerics](../11_numerics/README.md) — floor division, remainder sign, and overflow at the edges.
- [The IR](../../docs/internals/ir.md) — `overflow` and `rounding` as attributes of the operation.

`arbitrary_precision.ppy` is hand-written; there is no `.py` source and no
conversion step.
