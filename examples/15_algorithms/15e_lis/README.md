# 15e — Longest increasing subsequence

Patience sorting with a binary search, over two million values.

## Provenance

Hand-written. `lis.ppy` is written directly; there is no `.py` source and no
conversion step. `lis.c` is the same algorithm hand-written in C.

## What it shows

- The inner binary search is a `while` over indices with no allocation, so
  the whole kernel is one native loop nest.
- Every read is bounds-checked and the guards still hoist out of the loop.

## Numbers

Kernel wall time, mean ± standard deviation over 7 runs:

| path | 2e6 values |
|---|---:|
| plain CPython | 1048.9 ± 11.3 ms |
| `ppy run` | 57.4 ± 1.4 ms |
| `ppy build` binary | 53.7 ± 1.8 ms |
| C (`gcc -O3`) | 67.9 ± 1.8 ms |

## Run it

```bash
python  lis.ppy
ppy run lis.ppy
gcc -O3 lis.c -o lis_c && ./lis_c
```
