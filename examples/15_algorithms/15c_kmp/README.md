# 15c — Substring search (Baekjoon 1786)

Input: the text on the first line, the pattern on the second. Output: how
many times the pattern occurs. Four million characters here.

## Provenance

Hand-written. The `.ppy` is written directly; there is no `.py` source and no
conversion step. The `.c` is the same solution hand-written in C, reading the
same input with `scanf`.

## What it shows

- `ppy.read_token` fills the buffer without building a Python string.
- This is the one problem where C's reader wins: `Buffer[int]` is 64-bit, so
  a character costs eight bytes to read where `scanf("%s")` costs one.

## Numbers

Judge-style: the input is piped in and both halves are timed, mean ±
standard deviation over 5 runs. `examples/15_algorithms/bench.py` reproduces
the whole table.

| phase | plain | `ppy run` | `ppy build` | C (`scanf`) |
|---|---:|---:|---:|---:|
| read | 26.4 ± 0.5 ms | 34.0 ± 1.7 ms | 26.3 ± 1.1 ms | 6.4 ± 0.4 ms |
| solve | 288.6 ± 4.3 ms | 5.2 ± 0.3 ms | 5.1 ± 0.2 ms | 2.8 ± 0.2 ms |

## Run it

```bash
python  kmp.ppy < input.txt
ppy run kmp.ppy < input.txt
gcc -O3 kmp.c -o kmp_c && ./kmp_c < input.txt
```
