# 15c — Substring search ([BOJ 1786](https://www.acmicpc.net/problem/1786))

Input: the text, then the pattern. Output: how many times it occurs.
Four million characters here.

## Provenance

Hand-written. `kmp.ppy` is written directly; there is no `.py` source
and no conversion step. `kmp.c` is the same solution hand-written in C, reading the
same input with `scanf`.

## What it shows

- `ppy.read_token` fills the buffer without building a Python string.
- This is the one problem here whose `.ppy` is hand-written: the character
  buffer it wants has no plain-Python spelling that converts to it.

## Numbers

Wall time of the whole process, measured from outside the way a judge does —
input, interpreter startup and all. Mean ± standard deviation over 5 runs;
`examples/15_algorithms/bench.py` reproduces it and
`scripts/refresh.py` says when these have drifted.

| path | wall |
|---|---:|
| plain | 511 ± 10 ms |
| ppy run | 1930 ± 51 ms |
| ppy build | 79 ± 3 ms |
| C scanf | 11 ± 0 ms |

`ppy run` compiles before it runs, which is most of its two seconds; it is
the development path, not the one to submit. `ppy build` produces a binary
that still starts an embedded CPython and imports the runtime: ~35 ms before
a line of the program runs, against C's ~1 ms.

## Run it

```bash
python  kmp.ppy < input.txt
ppy run kmp.ppy < input.txt
ppy build kmp.ppy -o dist && ./dist/kmp < input.txt
gcc -O3 kmp.c -o kmp_c && ./kmp_c < input.txt
```
