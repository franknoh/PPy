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

Each subfolder is a self-contained problem with its own C reference and its
own measurements:

| | problem | plain | `ppy run` | C |
|---|---|---:|---:|---:|
| [15a](15a_nqueens/) | n-queens, bitmask backtracking | 121.5 ms | 5.4 ms | 4.4 ms |
| [15b](15b_dijkstra/) | Dijkstra with a binary heap | 1409.7 ms | 59.6 ms | 79.7 ms |
| [15c](15c_kmp/) | KMP substring search | 265.0 ms | 5.1 ms | 4.4 ms |
| [15d](15d_segment_tree/) | segment tree, update and query | 489.8 ms | 6.1 ms | 5.7 ms |
| [15e](15e_lis/) | longest increasing subsequence | 1048.9 ms | 57.4 ms | 67.9 ms |
| [15f](15f_input/) | reading input, and what it costs | 683.3 ms | 33.3 ms | 30.9 ms |

[15f](15f_input/) answers the question this set of problems raises on a
judge: **PPY does not make reading input any faster.** Reading is an IO
effect and stays on the interpreter, so `input()` costs what it always did
and `sys.stdin.read().split()` is still the idiom to reach for. What gets
faster is everything after the parse.

## Run it

```bash
python  algorithms.ppy
ppy     algorithms.ppy
ppy run algorithms.ppy
gcc -O3 algorithms.c -o algorithms_c && ./algorithms_c
```
