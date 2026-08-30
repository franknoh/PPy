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

Same machine, same session, kernel wall time (`gcc -O2`); every row prints
the same answer in all four columns:

| kernel | plain | ppy run | C |
|---|---:|---:|---:|
| sieve 2e6 | 191.9 ms | 10.4 ms | 17.7 ms |
| collatz 3e5 | 1204.3 ms | 42.1 ms | 44.4 ms |
| knapsack 400×2e4 | 476.7 ms | 5.5 ms | 2.5 ms |
| edit 2000×2000 | 531.1 ms | 3.3 ms | 2.0 ms |
| floyd 220 | 539.9 ms | 6.2 ms | 5.6 ms |
| matmul 220 | 527.4 ms | 6.8 ms | 2.3 ms |
| union-find 5e5 | 186.2 ms | 3.8 ms | 4.2 ms |
| fermat 6e4 | 25.9 ms | 2.9 ms | 1.8 ms |

It goes both ways: PPY wins where gcc's advantage does not apply (sieve,
union-find — and collatz is a draw with overflow guards included), and gcc
wins where auto-vectorization pays (matmul, edit distance) or the loop body
is one fused multiply (knapsack, fermat). Vectorization of the native
backend is where that remaining gap lives.

## Run it

```bash
python  algorithms.ppy
ppy     algorithms.ppy
ppy run algorithms.ppy
gcc -O2 algorithms.c -o algorithms_c && ./algorithms_c
```
