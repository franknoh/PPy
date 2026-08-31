# 15a — N-Queens (Baekjoon 9663)

Input: `N`. Output: how many ways N queens fit on an N×N board.

## Provenance

Hand-written. The `.ppy` is written directly; there is no `.py` source and no
conversion step. The `.c` is the same solution hand-written in C, reading the
same input with `scanf`.

## What it shows

- Recursion lowers natively, and so do the operators a bitmask solution is
  made of: `~`, unary `-`, the shifts, and the masks.
- There is no input to speak of, so this row is pure compute.

## Numbers

Judge-style: the input is piped in and both halves are timed, mean ±
standard deviation over 5 runs. `examples/15_algorithms/bench.py` reproduces
the whole table.

| phase | plain | `ppy run` | `ppy build` | C (`scanf`) |
|---|---:|---:|---:|---:|
| solve | 122.4 ± 2.2 ms | 5.7 ± 0.3 ms | 6.4 ± 0.2 ms | 4.5 ± 0.3 ms |

## Run it

```bash
echo 12 | python  nqueens.ppy
echo 12 | ppy run nqueens.ppy
gcc -O3 nqueens.c -o nqueens_c && echo 12 | ./nqueens_c
```
