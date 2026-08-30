# Algorithms

Compute-heavy kernels from competitive programming.

## Provenance

Hand-written. `algorithms.ppy` is written directly; there is no `.py`
source and no conversion step involved. `algorithms.c` is the same eight
kernels hand-written in C — the reference the native path is measured
against, with the same workloads and the same printed answers.

## What it shows

- Eight problems: sieve, Collatz, knapsack, edit distance, Floyd-Warshall, matmul, union-find, Fermat.
- All three paths produce identical answers; the native path is 9x to 160x
  faster than plain CPython.
- Writes go through borrowed buffers, so the caller sees them.

## Against C

Same machine, same session, kernel wall time (`gcc -O3`, matching the
kernels' `@ppy.opt(3)`); every row prints the same answer in all four
columns:

| kernel | plain | ppy run | C |
|---|---:|---:|---:|
| sieve 2e6 | 191.9 ms | 9.7 ms | 14.6 ms |
| collatz 3e5 | 1204.3 ms | 42.8 ms | 42.9 ms |
| knapsack 400×2e4 | 476.7 ms | 5.4 ms | 2.5 ms |
| edit 2000×2000 | 531.1 ms | 3.5 ms | 3.8 ms |
| floyd 220 | 539.9 ms | 3.5 ms | 5.3 ms |
| matmul 220 | 527.4 ms | 3.7 ms | 2.2 ms |
| union-find 5e5 | 186.2 ms | 3.2 ms | 3.9 ms |
| fermat 6e4 | 25.9 ms | 2.8 ms | 1.8 ms |

Guard hoisting (`[tool.ppy.llvm] safeguards`, on by default) is what closed
most of the old gaps: a multiplied index like `i * n + k` proves its extreme
cases once in a guard block ahead of the loop, and the body runs plain `mul
nsw` with no side exits — which is what lets LLVM strength-reduce and
vectorize. PPY now wins sieve, edit distance, floyd, and union-find, draws
collatz, and gcc keeps matmul (1.7×), knapsack, and fermat, whose remaining
guards live on data values no range can prove.

## Run it

```bash
python  algorithms.ppy
ppy     algorithms.ppy
ppy run algorithms.ppy
gcc -O3 algorithms.c -o algorithms_c && ./algorithms_c
```
