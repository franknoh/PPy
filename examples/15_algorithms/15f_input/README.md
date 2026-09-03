# 15f — Counting inversions, and what input costs

Input: `N`, then N integers. Output: how many pairs are out of order.
Half a million values here.

## Provenance

Generated, not hand-written. `inversions.ppy` is exactly what
`ppy convert inversions.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run. `inversions.c` is the same solution hand-written in C, reading the
same input with `scanf`.

## What it shows

- A Fenwick tree over half a million values, `add` and `prefix` both native.
- The read loop in the source converts into one bulk `ppy.read_ints`.

## Numbers

Wall time of the whole process, measured from outside the way a judge does —
input, interpreter startup and all. Mean ± standard deviation over 5 runs;
`examples/15_algorithms/bench.py` reproduces it and
`scripts/refresh.py` says when these have drifted.

| path | wall |
|---|---:|
| plain CPython | 1089.0 ± 60.8 ms |
| `ppy run` | 2079.4 ± 71.0 ms |
| `ppy build` | 107.2 ± 1.3 ms |
| `ppy build --standalone` | **42.6 ± 2.4 ms** |
| C (`gcc -O3`, `scanf`) | 55.6 ± 4.7 ms |

`ppy run` compiles before it runs, which is most of its two seconds; it is
the development path, not the one to submit. `ppy build` produces a binary
that still starts an embedded CPython and imports the runtime: ~35 ms before
a line of the program runs, against C's ~1 ms. `--standalone` has no interpreter in it at all, which is where that
row comes from; the [folder README](../README.md) says what the subset costs.

## Run it

```bash
python  inversions.ppy < input.txt
ppy run inversions.ppy < input.txt
ppy build inversions.ppy -o dist && ./dist/inversions < input.txt
gcc -O3 inversions.c -o inversions_c && ./inversions_c < input.txt
```
