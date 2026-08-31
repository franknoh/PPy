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
input the way a judge sends it and answered on stdout. The C reference reads
the same input with `scanf`, so the comparison includes what every
submission actually pays for. Totals below are read + solve, mean over 5
runs; `bench.py` prints both halves separately.

| | problem | plain | `ppy run` | C (`scanf`) |
|---|---|---:|---:|---:|
| [15a](15a_nqueens/) | N-Queens (9663) | 122.4 ms | 5.7 ms | 4.5 ms |
| [15b](15b_dijkstra/) | shortest path (1753) | 1718.6 ms | 139.5 ms | 168.9 ms |
| [15c](15c_kmp/) | substring search (1786) | 315.0 ms | 39.2 ms | 9.2 ms |
| [15d](15d_segment_tree/) | range sums (2042) | 538.1 ms | 34.0 ms | 55.2 ms |
| [15e](15e_lis/) | longest increasing subsequence (12015) | 523.1 ms | 43.0 ms | 64.3 ms |
| [15f](15f_input/) | counting inversions (1517) | 672.7 ms | 45.5 ms | 48.0 ms |

PPY wins four of the six outright. It loses N-Queens by a hair on pure
compute, and loses substring search on the read: `Buffer[int]` is 64-bit, so
a character costs eight bytes to read where `scanf("%s")` costs one.

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
and 16.5 ms for C's `scanf`. It works on every path, plain CPython included;
the reader is compiled once and cached, and falls back to pure Python where
there is no C compiler. A program that uses it must not also read `input()`
or `sys.stdin`, because the reader owns the file descriptor.
[15f](15f_input/) measures all three ways side by side.
