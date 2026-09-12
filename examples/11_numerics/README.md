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
run` and `ppy build` the guards are on by default; `--unsafe` on either
produces a wrap-semantics artifact.

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

## Compared with Numba, Codon, Mojo, C, and Rust

The same three functions, written for five other compilers, in
[`compare/`](compare/): [`semantics_numba.py`](compare/semantics_numba.py),
[`semantics_codon.py`](compare/semantics_codon.py), [`semantics.mojo`](compare/semantics.mojo),
[`semantics.c`](compare/semantics.c), and [`semantics.rs`](compare/semantics.rs).
There is nothing to time here; the question is what each prints for
`may_overflow(30)`, `floor_semantics(-7, 2)`, and `modulo_semantics(7, -2)`,
and the answers Python gives are `265252859812191058636308480000000`, `-4`,
and `-1`.

| | `may_overflow(30)` | `-7 // 2` | `7 % -2` | what the code says |
|---|---|---:|---:|---|
| **Python, PPY on every path** | 265252859812191058636308480000000 | -4 | -1 | `int` is an integer |
| Numba `@njit` | -8764578968847253504 | -4 | -1 | 64-bit wrap; Python's floor and sign |
| Mojo 1.0 `Int` | -8764578968847253504 | -4 | -1 | 64-bit wrap; Python's floor and sign |
| Codon | -8764578968847253504 | -3 | 1 | 64-bit wrap; C's truncation and sign |
| C `long long` (gcc, -O0 and -O3) | -8764578968847253504 | -3 | 1 | signed overflow is undefined; truncation |
| Rust `i64`, release | -8764578968847253504 | -3 | 1 | wraps; truncation |
| Rust `i64`, debug | *panics: attempt to multiply with overflow* | -3 | 1 | checked, then aborts |

Numba and Mojo keep Python's division and remainder and wrap the multiply
silently; Codon, C, and Rust in release keep C's rounding and wrap; Rust in
debug is the only other one that notices the overflow, and its answer is to
stop. PPY's is to notice and continue in Python: the same function, the
same source, the number Python prints. `--unsafe` on `ppy run` or `ppy build`
buys the wrap-semantics row, and says so.

Numba 0.67.0 on CPython 3.12.13, Codon 0.19.6, Mojo 1.0.0, gcc 13.3, rustc
1.95.0.

## Run it

```bash
python  numerics.ppy
ppy     numerics.ppy
ppy run numerics.ppy
```

<!-- outputs:start -->
## What it prints

**`python  numerics.ppy`**, **`ppy     numerics.ppy`**, **`ppy run numerics.ppy`**

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
