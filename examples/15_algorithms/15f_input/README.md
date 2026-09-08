# 15f — Counting inversions, and what input costs

Input: `N`, then N integers. Output: how many pairs are out of order. Half
a million values at the judge size, counted with a Fenwick tree whose `add`
and `prefix` are both native. This is the problem that isolates the cost of
reading: 500k integers through `ppy.input` take 12.6 ms, through
`sys.stdin.read().split()` 49.8 ms, through C's `scanf` 20.6 ms.

## The reader is the difference

```python
count: int = ppy.input[int]()
ppy.read_ints(memoryview(values)[:count])
```

The original source read one value per `input()` call. `ppy convert`
rewrote the loop into a single bulk `ppy.read_ints` over the same slots —
the same buffer, filled by a small C scanner that owns file descriptor 0
and never builds a Python object per field. Under plain CPython the same
call runs through the same scanner, compiled once into the user cache, so
the three paths read the same way and print the same count.

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

`ppy run` compiles before it runs; it is the development path, not the one
to submit. `ppy build` still starts an embedded CPython and imports the
runtime, ~35 ms, before the program begins. `--standalone` has no
interpreter in it, and the 8 ms it holds over both C compilers is the
reader; the [folder README](../README.md) says what the subset costs.

## Run it

`input.txt` holds eight values; the answer is 14.

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

Generated, not hand-written: `inversions.ppy` is exactly what
`ppy convert inversions.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run. `inversions.c` is
the same solution hand-written in C, reading the same input with `scanf`.
