# 15f — Counting inversions, and what input costs (Baekjoon 1517)

Input: `N`, then N integers. Output: how many pairs are out of order.
Half a million values here.

## Provenance

Hand-written. `inversions.ppy` is written directly; there is no `.py` source
and no conversion step. `inversions.c` is the same solution hand-written in
C, reading the same input with `scanf`.

## The question this folder answers

Reading input is an IO effect, so it stays on the interpreter — which is why
`input()` and `sys.stdin.read().split()` cost exactly what they cost without
PPY. Reading 500k integers, mean ± standard deviation over 5 runs:

| how | plain CPython | `ppy run` |
|---|---:|---:|
| `input()` | 187.1 ± 4.0 ms | 199.2 ± 3.4 ms |
| `sys.stdin.read().split()` | 54.8 ± 2.4 ms | 53.0 ± 3.0 ms |
| **`ppy.input(Buffer[int], n)`** | **9.5 ± 0.3 ms** | **9.6 ± 0.4 ms** |

`ppy.input(T)` reads the next value the way `T` says to read it, straight
from file descriptor 0 through a small C reader, so the numbers never become
Python objects on the way in. It is 5.8× faster than the fast Python idiom
and 1.7× faster than C's `scanf` (16.5 ± 0.9 ms), and it works on every path including plain CPython — the
reader is compiled once and cached, with a pure-Python fallback where no C
compiler exists.

## Numbers

| phase | plain | `ppy run` | `ppy build` | C (`scanf`) |
|---|---:|---:|---:|---:|
| read | 9.5 ± 0.3 ms | 9.6 ± 0.4 ms | 9.0 ± 0.6 ms | 16.5 ± 0.9 ms |
| solve | 663.2 ± 21.7 ms | 35.9 ± 3.6 ms | 34.8 ± 0.9 ms | 31.5 ± 0.8 ms |

So the whole submission is 45 ms against C's 48 ms, and the part that used to
be untouchable is now the smaller half.

## Run it

```bash
python  inversions.ppy < input.txt
ppy run inversions.ppy < input.txt
gcc -O3 inversions.c -o inversions_c && ./inversions_c < input.txt

python ../bench.py 15f    # the tables above
```
