# 15e — Longest increasing subsequence (Baekjoon 12015)

Input: `N`, then N integers. Output: the length of the longest strictly
increasing subsequence. One million values here.

## Provenance

Hand-written. The `.ppy` is written directly; there is no `.py` source and no
conversion step. The `.c` is the same solution hand-written in C, reading the
same input with `scanf`.

## What it shows

- The inner binary search is a `while` over indices with no allocation, so
  the whole kernel is one native loop nest.
- Every read is bounds-checked and the guards still hoist out of the loop.

## Numbers

Judge-style: the input is piped in and both halves are timed, mean ±
standard deviation over 5 runs. `examples/15_algorithms/bench.py` reproduces
the whole table.

| phase | plain | `ppy run` | `ppy build` | C (`scanf`) |
|---|---:|---:|---:|---:|
| read | 17.6 ± 0.5 ms | 17.9 ± 1.1 ms | 17.8 ± 0.7 ms | 34.0 ± 0.8 ms |
| solve | 505.5 ± 7.5 ms | 25.1 ± 0.7 ms | 24.2 ± 1.1 ms | 30.3 ± 1.2 ms |

## Run it

```bash
python  lis.ppy < input.txt
ppy run lis.ppy < input.txt
gcc -O3 lis.c -o lis_c && ./lis_c < input.txt
```
