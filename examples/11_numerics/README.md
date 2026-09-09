# Numerics

Where PPY refuses to differ from CPython: overflow, floor division, and the
sign of the remainder.

## Overflow falls back

```python
@ppy.pure
@ppy.opt(3)
def may_overflow(n: int) -> int:
    result: int = 1
    for i in range(1, n + 1):
        result *= i
    return result
```

Each `result *= i` is an overflow-checking multiply. `may_overflow(20)` runs
twenty of them natively. `may_overflow(30)` sets the flag on the
twenty-first, the function returns to its Python body, and CPython finishes
with arbitrary precision — the 33-digit number Python prints. Under `ppy
run` the guards are on by default; `ppy build` produces a wrap-semantics
artifact, and `--safe` puts them back.

## Floor, not truncation

```python
@ppy.pure
@ppy.opt(3)
def floor_semantics(a: int, b: int) -> int:
    return a // b
```

C rounds toward zero; Python rounds toward negative infinity and gives the
remainder the divisor's sign. The IR marks the operation `rounding =
"floor"`, and the LLVM backend emits the sign-corrected sequence — or a
single arithmetic shift when the divisor is a power of two, which is one
reason the collatz kernel keeps up with its C twin. `floor_semantics(-7, 2)`
is `-4` and `modulo_semantics(7, -2)` is `-1` on every path.

## Run it

```bash
python  numerics.ppy
ppy     numerics.ppy
ppy run numerics.ppy
```

<!-- outputs:start -->
## What it prints

**`python  numerics.ppy`**

```text
1000000
2432902008176640000
265252859812191058636308480000000
-4 3
1 -1
```

**`ppy     numerics.ppy`**

```text
1000000
2432902008176640000
265252859812191058636308480000000
-4 3
1 -1
```

**`ppy run numerics.ppy`**

```text
1000000
2432902008176640000
265252859812191058636308480000000
-4 3
1 -1
```

<!-- outputs:end -->

Read on: [Arbitrary precision](../02_arbitrary_precision/README.md) ·
[The IR](../../docs/internals/ir.md)

`numerics.ppy` is hand-written; there is no `.py` source and no conversion step.
