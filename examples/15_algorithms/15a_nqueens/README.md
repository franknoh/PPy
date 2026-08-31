# 15a — N-Queens ([BOJ 9663](https://www.acmicpc.net/problem/9663))

Input: `N`. Output: how many ways N queens fit on an N×N board.

## Provenance

Generated, not hand-written. `nqueens.ppy` is exactly what
`ppy convert nqueens.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run. `nqueens.c` is the same solution hand-written in C, reading the
same input with `scanf`.

## What it shows

- Recursion lowers natively, and so do the operators a bitmask solution is
  made of: `~`, unary `-`, the shifts, and the masks.
- No input to speak of, so the wall clock here is startup plus compute.

## Numbers

Wall time of the whole process, measured from outside the way a judge does —
input, interpreter startup and all. Mean ± standard deviation over 5 runs;
`examples/15_algorithms/bench.py` reproduces it and
`scripts/refresh.py` says when these have drifted.

| path | wall |
|---|---:|
| plain | 351 ± 17 ms |
| ppy run | 2138 ± 100 ms |
| ppy build | 200 ± 9 ms |
| C scanf | 8 ± 1 ms |

`ppy run` compiles before it runs, which is most of its two seconds; it is
the development path, not the one to submit. `ppy build` produces a binary
that still starts an embedded CPython, and that startup is the ~170 ms floor
under every row of it.

## Run it

```bash
python  nqueens.ppy < input.txt
ppy run nqueens.ppy < input.txt
ppy build nqueens.ppy -o dist && ./dist/nqueens < input.txt
gcc -O3 nqueens.c -o nqueens_c && ./nqueens_c < input.txt
```
