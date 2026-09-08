# 15c — Substring search

Input: the text, then the pattern. Output: how many times it occurs.
Four million characters here.

## Provenance

Hand-written. `kmp.ppy` is written directly; there is no `.py` source
and no conversion step. `kmp.c` is the same solution hand-written in C, reading the
same input with `scanf`.

## What it shows

- `ppy.read_token` fills the buffer without building a Python string.
- `Buffer[ppy.i8]` is one byte per element, so four million characters cost
  four megabytes rather than the thirty-two a 64-bit element would.
- This is the one problem here whose `.ppy` is hand-written: the character
  buffer it wants has no plain-Python spelling that converts to it.

## Numbers

Wall time of the whole process, measured from outside the way a judge does —
input, interpreter startup and all. Mean ± standard deviation over 5 runs;
`examples/15_algorithms/bench.py` reproduces it and
`scripts/refresh.py` says when these have drifted.

| path | wall |
|---|---:|
| plain CPython | 283.0 ± 2.9 ms |
| `ppy run` | 137.3 ± 153.8 ms |
| `ppy build` | 52.4 ± 2.0 ms |
| C (`gcc -O3`, `scanf`) | 9.4 ± 0.3 ms |
| C (`clang -O3`, `scanf`) | **9.0 ± 0.2 ms** |

`ppy run` compiles before it runs, which is most of its two seconds; it is
the development path, not the one to submit. `ppy build` produces a binary
that still starts an embedded CPython and imports the runtime: ~35 ms before
a line of the program runs, against C's ~1 ms. This is the one problem here
with no `--standalone` row: its text arrives as a token, and a standalone
build has no reader for one yet.

## Run it

```bash
python  kmp.ppy < input.txt
ppy run kmp.ppy < input.txt
ppy build kmp.ppy -o dist && ./dist/kmp < input.txt
gcc   -O3 kmp.c -o kmp_c     && ./kmp_c     < input.txt
clang -O3 kmp.c -o kmp_clang && ./kmp_clang < input.txt
```

<!-- outputs:start -->
## What it prints

**`python  kmp.ppy < input.txt`**

```text
1
```

**`ppy run kmp.ppy < input.txt`**

```text
1
```

**`ppy build kmp.ppy -o dist && ./dist/kmp < input.txt`**

```text
1
```

**`gcc   -O3 kmp.c -o kmp_c     && ./kmp_c     < input.txt`**

```text
1
```

**`clang -O3 kmp.c -o kmp_clang && ./kmp_clang < input.txt`**

```text
1
```

<!-- outputs:end -->
