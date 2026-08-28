# Algorithms

Compute-heavy kernels from competitive programming.

## Provenance

Hand-written. `algorithms.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- Eight problems: sieve, Collatz, knapsack, edit distance, Floyd-Warshall, matmul, union-find, Fermat.
- All three paths produce identical answers; the native path is 10x to 150x faster.
- Writes go through borrowed buffers, so the caller sees them.

## Run it

```bash
python  algorithms.ppy
ppy     algorithms.ppy
ppy run algorithms.ppy
```
