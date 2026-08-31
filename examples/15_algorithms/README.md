# Algorithms

Compute-heavy kernels from competitive programming.

## Provenance

Hand-written. `algorithms.ppy` is written directly; there is no `.py`
source and no conversion step involved. `algorithms.c` is the same eight
kernels hand-written in C — the reference the native path is measured
against, with the same workloads and the same printed answers.

## What it shows

- Eight problems: sieve, Collatz, knapsack, edit distance, Floyd-Warshall, matmul, union-find, Fermat.
- All three paths produce identical answers; the native path is 9x to 160x
  faster than plain CPython.
- Writes go through borrowed buffers, so the caller sees them.

## Against C

Same machine, same session, kernel wall time — mean ± standard deviation
over 7 runs, each a fresh process. Every row prints the same answer in every
column:

| kernel | plain | `ppy run` | `ppy build` | C (`gcc -O3`) |
|---|---:|---:|---:|---:|
| sieve 2e6 | 198.0 ± 8.2 | 9.8 ± 0.7 | 9.7 ± 0.7 | 15.5 ± 0.7 |
| collatz 3e5 | 1235.2 ± 17.6 | 44.5 ± 1.3 | 33.1 ± 0.6 | 45.8 ± 2.2 |
| knapsack 400×2e4 | 476.6 ± 9.5 | 5.3 ± 0.3 | 3.9 ± 0.2 | 2.8 ± 0.3 |
| edit 2000×2000 | 530.7 ± 7.1 | 3.2 ± 0.4 | 3.2 ± 0.3 | 4.1 ± 0.6 |
| floyd 220 | 496.3 ± 10.1 | 3.5 ± 0.2 | 3.2 ± 0.1 | 5.8 ± 0.5 |
| matmul 220 | 528.9 ± 6.7 | 4.0 ± 0.3 | 3.8 ± 0.1 | 2.7 ± 0.4 |
| union-find 5e5 | 184.6 ± 8.4 | 3.7 ± 0.6 | 3.7 ± 0.3 | 4.3 ± 0.5 |
| fermat 6e4 | 26.9 ± 0.8 | 2.4 ± 0.1 | 2.7 ± 0.2 | 1.9 ± 0.1 |

The `ppy run` column keeps Python-integer semantics: overflow is guarded and
falls back to arbitrary precision. `ppy build` is the wrap-semantics
artifact, which is where collatz picks up its remaining 11 ms and knapsack
its 1.4 ms; the other kernels are already guard-free in the loop and do not
move. `--host-cpu` is within the noise on all eight, so it is not shown.

Guard hoisting (`[tool.ppy.llvm] safeguards`) is what closed most of the old
gaps: a multiplied index like `i * n + k` proves its extreme cases once in a
guard block ahead of the loop, and the body runs plain `mul nsw` with no side
exits — which is what lets LLVM strength-reduce and vectorize. PPY wins
sieve, edit distance, floyd, and union-find, draws collatz, and gcc keeps
matmul, knapsack, and fermat, whose remaining guards live on data values no
range can prove.

## More problems

Each subfolder is one competitive-programming problem, read from standard
input the way a judge sends it and answered on stdout. Nothing inside the
programs is instrumented: the times below are wall time of the whole
process, measured from outside, input and interpreter startup included.
The C reference reads the same input with `scanf`.

| | problem | plain | `ppy build` | C (`scanf`) |
|---|---|---:|---:|---:|
| [15a](15a_nqueens/) | N-Queens ([BOJ 9663](https://www.acmicpc.net/problem/9663)) | 331 ms | 327 ms | 8 ms |
| [15b](15b_dijkstra/) | shortest path ([BOJ 1753](https://www.acmicpc.net/problem/1753)) | 2021 ms | 428 ms | 177 ms |
| [15c](15c_kmp/) | substring search ([BOJ 1786](https://www.acmicpc.net/problem/1786)) | 535 ms | 210 ms | 13 ms |
| [15d](15d_segment_tree/) | range sums ([BOJ 2042](https://www.acmicpc.net/problem/2042)) | 805 ms | 259 ms | 59 ms |
| [15e](15e_lis/) | longest increasing subsequence ([BOJ 12015](https://www.acmicpc.net/problem/12015)) | 763 ms | 250 ms | 68 ms |
| [15f](15f_input/) | counting inversions ([BOJ 1517](https://www.acmicpc.net/problem/1517)) | 948 ms | 241 ms | 52 ms |

`ppy run` is left out of the table because it compiles before it runs — a
flat ~2.1 s on every row, which is the development path rather than the one
to submit. What the `ppy build` column mostly shows at these input sizes is
that its binary still starts an embedded CPython: ~170 ms before any of the
program runs, against C's ~1 ms. The kernels themselves are at or ahead of C
(see the tables in each folder's README); the startup is what a judge would
charge PPY for.

Five of the six are written as ordinary Python and converted:
`ppy convert <name>.py --promote-buffers` writes the `.ppy` beside it, and
`examples/verify_conversions.py` checks that the committed file is exactly
that. [15c](15c_kmp/) is hand-written, because the character buffer it wants
has no plain-Python spelling that converts to it.

### Reading input

`ppy.input[T]()` reads the next value the way `T` says to read it, and types
the result the same way:

```python
n = ppy.input[int]()
a, b = ppy.input[tuple[int, int]]()
values = ppy.input[Buffer[int]](n)
```

It goes straight into memory rather than building a Python object per field
— 9.6 ms for 500k integers against 54.8 ms for `sys.stdin.read().split()`
and 16.5 ms for C's `scanf`. The conversion writes it for you: `int(input())`
becomes `ppy.input[int]()`, `a, b = map(int, input().split())` becomes the
tuple read, and a loop that fills a buffer one value at a time becomes one
bulk `ppy.read_ints`. A module that also touches `sys.stdin` keeps the
`input` it has, since the typed reader owns the file descriptor.
