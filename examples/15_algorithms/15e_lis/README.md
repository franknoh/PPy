# 15e — Longest increasing subsequence

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
| plain CPython | 836.4 ± 15.5 ms |
| `ppy run` | 1986.6 ± 48.5 ms |
| `ppy build` | 129.9 ± 4.0 ms |
| `ppy build --standalone` | **45.7 ± 1.7 ms** |
| C (`gcc -O3`, `scanf`) | 77.3 ± 2.2 ms |

`ppy run` compiles before it runs, which is most of its two seconds; it is
the development path, not the one to submit. `ppy build` produces a binary
that still starts an embedded CPython and imports the runtime: ~35 ms before
a line of the program runs, against C's ~1 ms. `--standalone` has no interpreter in it at all, which is where that
row comes from; the [folder README](../README.md) says what the subset costs.

## Run it

```bash
python  lis.ppy < input.txt
ppy run lis.ppy < input.txt
ppy build lis.ppy -o dist && ./dist/lis < input.txt
gcc -O3 lis.c -o lis_c && ./lis_c < input.txt
```
