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
| sieve 2e6 | 191.9 ms | 10.4 ms | 14.6 ms |
| collatz 3e5 | 1204.3 ms | 42.1 ms | 42.9 ms |
| knapsack 400×2e4 | 476.7 ms | 5.5 ms | 2.5 ms |
| edit 2000×2000 | 531.1 ms | 3.3 ms | 3.8 ms |
| floyd 220 | 539.9 ms | 6.2 ms | 5.3 ms |
| matmul 220 | 527.4 ms | 6.8 ms | 2.2 ms |
| union-find 5e5 | 186.2 ms | 3.8 ms | 3.9 ms |
| fermat 6e4 | 25.9 ms | 2.9 ms | 1.8 ms |

It goes both ways: PPY wins sieve and edit distance, draws collatz,
union-find, and floyd, and gcc wins matmul, knapsack, and fermat. Grafting
PPY's guard semantics into the C versions shows where those gaps live: the
bounds checks are essentially free, but Python-int overflow guards on the
index arithmetic (`i * n + k` behind `smul.with.overflow`) block strength
reduction and vectorization — C with the same guards runs matmul at 5.5 ms,
knapsack at 4.3 ms, fermat at 2.5 ms. The path to closing them is range
analysis proving the guards redundant, not faster arithmetic.

## Run it

```bash
python  algorithms.ppy
ppy     algorithms.ppy
ppy run algorithms.ppy
gcc -O3 algorithms.c -o algorithms_c && ./algorithms_c
```
