# 15f — Counting inversions, and what input costs

Input: `N`, then N integers. Output: how many pairs are out of order. Half
a million values at the judge size, counted with a Fenwick tree whose `add`
and `prefix` are both native.

## The reader is the difference

```python
count: int = ppy.input[int]()
values: Buffer[int] = ppy.input[Buffer[int]]()
```

The original source reads the count with `int(input())` and the values
with `array.array("q", map(int, input().split()))`: one line each. The
conversion writes both as line reads that mean the same -- the whole line,
`ValueError` for a field that is not an integer -- and the second one fills
the buffer from a small C scanner that owns file descriptor 0 and never
builds a Python object per field. Under plain CPython the same scanner is
compiled once into the user cache, so the three paths read the same way and
print the same count.

The standalone variant, [`standalone/inversions.ppy`](../standalone/inversions.ppy),
is hand-written for the subset and reads the block as tokens:

```python
values: Buffer[int] = ppy.scan[Buffer[int]](count)
```

`ppy.scan` does not care where the line breaks fall, which is what a
standalone binary can lower today; it accepts input the Python would
reject, so it is written down as its own program rather than converted.

## What input costs

500k integers through a `ppy` buffer read take 12.6 ms, through
`sys.stdin.read().split()` 49.8 ms, through C's `scanf` 20.6 ms. The 8 ms
the standalone binary holds over both C compilers is the reader; the [folder
README](../README.md) says what the subset costs.

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
| plain CPython | 1056.3 ± 328.4 ms |
| `ppy run` | 270.6 ± 222.5 ms |
| `ppy build --unsafe` | 123.7 ± 12.4 ms |
| `ppy build --standalone --unsafe` | **45.5 ± 3.9 ms** |
| C (`gcc -O3`, `scanf`) | 50.7 ± 2.3 ms |
| C (`clang -O3`, `scanf`) | 56.4 ± 6.7 ms |

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

Generated, not hand-written: `inversions.ppy` is exactly what
`ppy convert inversions.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run. `inversions.c` is
the same solution hand-written in C, reading the same input with `scanf`.
