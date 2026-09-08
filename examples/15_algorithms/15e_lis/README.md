# 15e — Longest increasing subsequence of a million values

Input: `N`, then N integers. Output: the length of the longest strictly
increasing subsequence. One million values at the judge size; the
standalone binary reads and solves them in 36 ms against gcc's 59 and
clang's 56.

## One loop nest, no allocation

```python
while low < high:
    middle: int = (low + high) // 2
```

The inner binary search is a `while` over indices with no allocation, so
the whole kernel is one native loop nest over a borrowed buffer. The read
loop in the original source — one `int(input())` per value — converted into
a single bulk `ppy.read_ints` over the same slots, which is where the
standalone row gets its margin: a million integers go into memory without
a Python object each.

## Numbers

Wall time of the whole process, measured from outside the way a judge does —
input, interpreter startup and all. Mean ± standard deviation over 5 runs;
`examples/15_algorithms/bench.py` reproduces it and
`scripts/refresh.py` says when these have drifted.

| path | wall |
|---|---:|
| plain CPython | 507.0 ± 10.7 ms |
| `ppy run` | 166.0 ± 151.2 ms |
| `ppy build` | 102.9 ± 4.2 ms |
| `ppy build --standalone` | **35.9 ± 0.8 ms** |
| C (`gcc -O3`, `scanf`) | 59.0 ± 2.8 ms |
| C (`clang -O3`, `scanf`) | 55.5 ± 1.7 ms |

`ppy run` compiles before it runs; it is the development path, not the one
to submit. `ppy build` still starts an embedded CPython and imports the
runtime, ~35 ms, before the program begins. `--standalone` has no
interpreter in it; the [folder README](../README.md) says what the subset
costs.

## Run it

`input.txt` holds ten values; the answer is 5.

```bash
python  lis.ppy < input.txt
ppy run lis.ppy < input.txt
ppy build lis.ppy -o dist && ./dist/lis < input.txt
gcc   -O3 lis.c -o lis_c     && ./lis_c     < input.txt
clang -O3 lis.c -o lis_clang && ./lis_clang < input.txt
```

<!-- outputs:start -->
## What it prints

**`python  lis.ppy < input.txt`**

```text
5
```

**`ppy run lis.ppy < input.txt`**

```text
5
```

**`ppy build lis.ppy -o dist && ./dist/lis < input.txt`**

```text
5
```

**`gcc   -O3 lis.c -o lis_c     && ./lis_c     < input.txt`**

```text
5
```

**`clang -O3 lis.c -o lis_clang && ./lis_clang < input.txt`**

```text
5
```

<!-- outputs:end -->

Generated, not hand-written: `lis.ppy` is exactly what
`ppy convert lis.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run. `lis.c` is the
same solution hand-written in C, reading the same input with `scanf`.
