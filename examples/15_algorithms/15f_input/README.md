# 15f: Counting inversions, and what input costs

Input: `N`, then N integers. Output: how many pairs are out of order. Half
a million values at the judge size, counted with a Fenwick tree whose `add`
and `prefix` are both native.

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

**`python  inversions.ppy < input.txt`**, **`ppy run inversions.ppy < input.txt`**, **`ppy build inversions.ppy -o dist && ./dist/inversions < input.txt`**, **`gcc   -O3 inversions.c -o inversions_c     && ./inversions_c     < input.txt`**, **`clang -O3 inversions.c -o inversions_clang && ./inversions_clang < input.txt`**

```text
14
```

<!-- outputs:end -->

## The reader is the difference

```python
count: int = ppy.input[int]()
values: Buffer[int] = ppy.input[Buffer[int]]()
```

The original source reads the count with `int(input())` and the values
with `array.array("q", map(int, input().split()))`: one line each. The
conversion writes both as line reads that mean the same: the whole line,
and `ValueError` for a field that is not an integer.

The second one fills the buffer from a small C scanner that owns file
descriptor 0 and never builds a Python object per field. Under plain
CPython the same scanner is compiled once into the user cache, so the three
paths read the same way and print the same count.

## The standalone variant

The standalone variant, [`standalone/inversions.ppy`](../standalone/inversions.ppy),
is hand-written for the subset and reads the block as tokens:

```python
values: Buffer[int] = ppy.scan[Buffer[int]](count)
```

`ppy.scan` does not care where the line breaks fall, which is what a
standalone binary can lower today. It accepts input the Python would
reject, so it is written down as its own program rather than converted.

## What input costs

| reader | 500k integers |
|---|---:|
| a `ppy` buffer read | 12.6 ms |
| `sys.stdin.read().split()` | 49.8 ms |
| C's `scanf` | 20.6 ms |

The 8 ms the standalone binary holds over both C compilers is the reader.
The [folder README](../README.md) says what the subset costs.

## Numbers

Wall time of the whole process, measured from outside the way a judge does:
input, interpreter startup and all. Mean ± standard deviation over 5 runs.
`bench.py` reproduces it and `scripts/refresh.py` says when these have
drifted.

| path | wall |
|---|---:|
| plain CPython | 603.7 ± 11.2 ms |
| `ppy run` | 189.3 ± 166.5 ms |
| `ppy build --unsafe` | 98.4 ± 3.1 ms |
| `ppy build --standalone --unsafe` | **38.6 ± 0.5 ms** |
| C (`gcc -O3`, `scanf`) | 43.5 ± 0.8 ms |
| C (`clang -O3`, `scanf`) | 44.2 ± 0.4 ms |

- `ppy run` compiles before it runs, which is most of its time. It is the
  development path, not the one to submit.
- `ppy build` still starts an embedded CPython and imports the runtime,
  about 35 ms, before the program begins.

## Where the code comes from

Generated, not hand-written: `inversions.ppy` is exactly what
`ppy convert inversions.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run. `inversions.c` is
the same solution hand-written in C, reading the same input with `scanf`.
