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
| plain CPython | 630.0 ± 11.7 ms |
| `ppy run` | 175.0 ± 162.7 ms |
| `ppy build` | 94.6 ± 2.3 ms |
| `ppy build --standalone` | **37.6 ± 0.6 ms** |
| C (`gcc -O3`, `scanf`) | 46.0 ± 1.1 ms |
| C (`clang -O3`, `scanf`) | 45.4 ± 0.8 ms |

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
gcc   -O3 inversions.c -o inversions_c     && ./inversions_c     < input.txt
clang -O3 inversions.c -o inversions_clang && ./inversions_clang < input.txt
```

<!-- outputs:start -->
## What it prints

**`python  inversions.ppy < input.txt`**

```text
14
```

**`ppy run inversions.ppy < input.txt`**

```text
14
```

**`ppy build inversions.ppy -o dist && ./dist/inversions < input.txt`**

```text
14
```

**`gcc   -O3 inversions.c -o inversions_c     && ./inversions_c     < input.txt`**

```text
14
```

**`clang -O3 inversions.c -o inversions_clang && ./inversions_clang < input.txt`**

```text
14
```

<!-- outputs:end -->
