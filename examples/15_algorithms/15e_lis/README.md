# 15e — Longest increasing subsequence

Input: `N`, then N integers. Output: the length of the longest strictly
increasing subsequence. One million values at the judge size.

## One loop nest, no allocation

```python
while low < high:
    middle: int = (low + high) // 2
```

The inner binary search is a `while` over indices with no allocation, so
the whole kernel is one native loop nest over a borrowed buffer.

## The read

The source reads its million values from one line with
`array.array("q", map(int, input().split()))`, and the conversion writes
that as `ppy.input[Buffer[int]]()`: the same line, the same `ValueError`
for a field that is not an integer, and no Python object per value on the
way into the buffer. The standalone variant in
[`standalone/lis.ppy`](../standalone/lis.ppy) reads the block with
`ppy.scan[Buffer[int]](count)` instead, which is what a standalone binary
lowers today. The [folder README](../README.md) says what the subset costs.

## Numbers

Wall time of the whole process, measured from outside the way a judge does —
input, interpreter startup and all. Mean ± standard deviation over 5 runs;
`bench.py` reproduces it and `scripts/refresh.py` says when these have
drifted. `ppy run` compiles before it runs, which is most of its time; it is
the development path, not the one to submit. `ppy build` still starts an
embedded CPython and imports the runtime, about 35 ms, before the program
begins.

| path | wall |
|---|---:|
| plain CPython | 505.3 ± 9.2 ms |
| `ppy run` | 166.6 ± 137.5 ms |
| `ppy build --unsafe` | 90.5 ± 1.6 ms |
| `ppy build --standalone --unsafe` | **38.2 ± 0.3 ms** |
| C (`gcc -O3`, `scanf`) | 56.1 ± 0.6 ms |
| C (`clang -O3`, `scanf`) | 53.9 ± 0.9 ms |

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
