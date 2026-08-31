# 15e — Longest increasing subsequence ([BOJ 12015](https://www.acmicpc.net/problem/12015))

Input: `N`, then N integers. Output: the length of the longest strictly
increasing subsequence. One million values here.

## Provenance

Generated, not hand-written. `lis.ppy` is exactly what
`ppy convert lis.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run. `lis.c` is the same solution hand-written in C, reading the
same input with `scanf`.

## What it shows

- The inner binary search is a `while` over indices with no allocation, so
  the whole kernel is one native loop nest.
- The read loop in the source converts into one bulk `ppy.read_ints`.

## Numbers

Wall time of the whole process, measured from outside the way a judge does —
input, interpreter startup and all. Mean ± standard deviation over 5 runs;
`examples/15_algorithms/bench.py` reproduces it and
`scripts/refresh.py` says when these have drifted.

| path | wall |
|---|---:|
| plain | 764 ± 6 ms |
| ppy run | 1942 ± 65 ms |
| ppy build | 112 ± 5 ms |
| C scanf | 68 ± 1 ms |

`ppy run` compiles before it runs, which is most of its two seconds; it is
the development path, not the one to submit. `ppy build` produces a binary
that still starts an embedded CPython and imports the runtime: ~35 ms before
a line of the program runs, against C's ~1 ms.

## Run it

```bash
python  lis.ppy < input.txt
ppy run lis.ppy < input.txt
ppy build lis.ppy -o dist && ./dist/lis < input.txt
gcc -O3 lis.c -o lis_c && ./lis_c < input.txt
```
