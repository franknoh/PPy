# 15c — Substring search over four million characters

Input: the text, then the pattern. Output: how many times the pattern
occurs. Four million characters at the judge size, held in four megabytes:
`Buffer[ppy.i8]` is one byte per element, and `ppy.read_token` fills it
without ever building a Python string.

## Text as bytes

```python
haystack = array.array("b", bytes(4000064))
size = ppy.read_token(haystack)
```

A Python `str` has no native representation, so text that has to be fast
comes in as bytes. `Buffer[ppy.i8]` stores one byte and reads back an
`int`; the width is storage, not type, and a value that does not fit falls
back rather than wrapping. The KMP failure table and the scan are two
native loops over that memory.

This is the one problem here whose `.ppy` is hand-written: the character
buffer it wants has no plain-Python spelling that `ppy convert` could
promote to it. It is also the one without a `--standalone` row: its text
arrives as a token, and `ppy.read_token` has no standalone lowering yet.

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

`ppy run` compiles before it runs; it is the development path, not the one
to submit. `ppy build` still starts an embedded CPython and imports the
runtime — ~35 ms of the 52, which is why C is ahead by the width of an
interpreter start and not by the width of the scan.

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

**`python  kmp.ppy < input.txt`**

```text
3
```

**`ppy run kmp.ppy < input.txt`**

```text
3
```

**`ppy build kmp.ppy -o dist && ./dist/kmp < input.txt`**

```text
3
```

**`gcc   -O3 kmp.c -o kmp_c     && ./kmp_c     < input.txt`**

```text
3
```

**`clang -O3 kmp.c -o kmp_clang && ./kmp_clang < input.txt`**

```text
3
```

<!-- outputs:end -->

`kmp.ppy` is hand-written; there is no `.py` source and no conversion step.
`kmp.c` is the same solution hand-written in C, reading the same input
with `scanf`.
