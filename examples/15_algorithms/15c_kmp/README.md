# 15c: Substring search

Input: the text, then the pattern. Output: how many times the pattern
occurs. Four million characters at the judge size.

## Run it

`input.txt` searches `abracadabracadabra` for `abra`; the answer is 3.

```bash
python  kmp.ppy < input.txt
ppy run kmp.ppy < input.txt
ppy build kmp.ppy -o dist && ./dist/kmp < input.txt
gcc   -O3 kmp.c -o kmp_c     && ./kmp_c     < input.txt
clang -O3 kmp.c -o kmp_clang && ./kmp_clang < input.txt
```

<!-- outputs:start -->
## What it prints

**`python  kmp.ppy < input.txt`**, **`ppy run kmp.ppy < input.txt`**, **`ppy build kmp.ppy -o dist && ./dist/kmp < input.txt`**, **`gcc   -O3 kmp.c -o kmp_c     && ./kmp_c     < input.txt`**, **`clang -O3 kmp.c -o kmp_clang && ./kmp_clang < input.txt`**

```text
3
```

<!-- outputs:end -->

## Text as bytes

```python
haystack = array.array("b", bytes(4000064))
size = ppy.read_token(haystack)
```

Reading four million characters as a `str` would build a Python string
first, so text this large comes in as bytes. `ppy.read_token` fills a `Buffer[ppy.i8]` without
building a Python string, one byte per element. Four million characters
cost four megabytes rather than the thirty-two a 64-bit element would.

The width is storage, not type: reading a byte hands out an `int`, and a
value that does not fit falls back rather than wrapping. The failure table
and the scan are two native loops over that memory.

## Numbers

Wall time of the whole process, measured from outside the way a judge does:
input, interpreter startup and all. Mean ± standard deviation over 5 runs.
`bench.py` reproduces it and `scripts/refresh.py` says when these have
drifted.

| path | wall |
|---|---:|
| plain CPython | 292.4 ± 12.8 ms |
| `ppy run` | 145.5 ± 160.1 ms |
| `ppy build --unsafe` | 53.3 ± 1.3 ms |
| C (`gcc -O3`, `scanf`) | 9.3 ± 0.1 ms |
| C (`clang -O3`, `scanf`) | **9.0 ± 0.4 ms** |

- `ppy run` compiles before it runs, which is most of its time. It is the
  development path, not the one to submit.
- `ppy build` still starts an embedded CPython and imports the runtime,
  about 35 ms, before the program begins.

C is ahead by about the width of an interpreter start.

## Two exceptions in this folder

This is the one problem whose `.ppy` is hand-written. The character buffer
it wants has no plain-Python spelling that `ppy convert` could promote to
it.

It is also the one without a `--standalone` row: its text arrives as a
token, and `ppy.read_token` has no standalone lowering yet.

## Where the code comes from

`kmp.ppy` is hand-written; there is no `.py` source and no conversion step.
`kmp.c` is the same solution hand-written in C, reading the same input
with `scanf`.
