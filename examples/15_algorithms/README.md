# Algorithms

Two things: `algorithms.ppy`, eight compute-heavy kernels measured against
the same eight in C, and six judge problems in `15a`–`15f`, each its own
folder, read from standard input the way a judge sends them.

## Provenance

Hand-written. `algorithms.ppy` is written directly; there is no `.py`
source and no conversion step involved. The six problem subfolders are
generated from their `.py` sources — each says so in its own README. `algorithms.c` is the same eight
kernels hand-written in C — the reference the native path is measured
against, with the same workloads and the same printed answers.

## What it shows

- Eight problems: sieve, Collatz, knapsack, edit distance, Floyd-Warshall, matmul, union-find, Fermat.
- All three paths produce identical answers; the native path is 9x to 160x
  faster than plain CPython on these eight.
- Writes go through borrowed buffers, so the caller sees them.

## Against C

Same machine, same session, kernel wall time — mean ± standard deviation
over 7 runs, each a fresh process. Every row prints the same answer in every
column:

| kernel | plain | `ppy run` | `ppy build` | C (`gcc -O3`) | C (`clang -O3`) |
|---|---:|---:|---:|---:|---:|
| sieve 2e6 | 173.1 ± 2.9 | 9.2 ± 0.5 | **8.8 ± 0.4** | 13.9 ± 0.6 | 13.8 ± 0.3 |
| collatz 3e5 | 1111.0 ± 13.1 | 39.9 ± 0.8 | **30.3 ± 0.4** | 41.4 ± 0.5 | 32.2 ± 0.3 |
| knapsack 400×2e4 | 423.1 ± 2.7 | 5.0 ± 0.1 | 3.6 ± 0.2 | 2.4 ± 0.1 | **2.1 ± 0.3** |
| edit 2000×2000 | 471.4 ± 6.5 | **2.9 ± 0.1** | **2.9 ± 0.3** | 3.6 ± 0.1 | 3.5 ± 0.2 |
| floyd 220 | 436.3 ± 5.3 | 3.4 ± 0.5 | 3.1 ± 0.2 | 5.0 ± 0.1 | **2.8 ± 0.0** |
| matmul 220 | 473.9 ± 5.7 | 3.6 ± 0.4 | 3.5 ± 0.1 | **2.1 ± 0.1** | 3.4 ± 0.1 |
| union-find 5e5 | 149.6 ± 3.8 | **3.2 ± 0.2** | **3.2 ± 0.4** | 3.6 ± 0.2 | 3.8 ± 0.3 |
| fermat 6e4 | 23.9 ± 0.3 | 2.3 ± 0.1 | 2.6 ± 0.2 | 1.8 ± 0.1 | **1.5 ± 0.1** |

The `ppy run` column keeps Python-integer semantics: overflow is guarded and
falls back to arbitrary precision. `ppy build` is the wrap-semantics
artifact, which is where collatz picks up its remaining 10 ms and knapsack
its 1.4 ms; the other kernels are already guard-free in the loop and do not
move. `--host-cpu` is within the noise on all eight, so it is not shown.
Bold is the fastest cell in the row.

Guard hoisting (`[tool.ppy.llvm] safeguards`) is what closed most of the old
gaps: a multiplied index like `i * n + k` proves its extreme cases once in a
guard block ahead of the loop, and the body runs plain `mul nsw` with no side
exits — which is what lets LLVM strength-reduce and vectorize. Against both
compilers PPY wins sieve, edit distance, and union-find, and collatz once the
artifact wraps; floyd beats gcc and loses to clang by a third of a
millisecond; gcc and clang keep knapsack, matmul, and fermat, whose remaining
guards live on data values no range can prove. gcc 13.3 and clang 22.1, both
`-O3`, no `-march`.

## More problems

Each subfolder is one competitive-programming problem — the shapes a judge
sets — read from standard input and answered on stdout. Each states its own
input and output format, so nothing here depends on an outside site being
up. Nothing inside the
programs is instrumented: the times below are wall time of the whole
process, measured from outside, input and interpreter startup included.
The C reference reads the same input with `scanf`.

| | problem | plain | `ppy build` | `--standalone` | C (`gcc`) | C (`clang`) |
|---|---|---:|---:|---:|---:|---:|
| [15a](15a_nqueens/) | N-Queens | 135.5 ms | 41.2 ms | 5.6 ms | 4.8 ms | 5.4 ms |
| [15b](15b_dijkstra/) | shortest path | 1511.0 ms | 251.9 ms | **104.8 ms** | 147.3 ms | 141.6 ms |
| [15c](15c_kmp/) | substring search | 283.0 ms | 52.4 ms | — | 9.4 ms | 9.0 ms |
| [15d](15d_segment_tree/) | range sums | 482.8 ms | 113.0 ms | **23.9 ms** | 50.7 ms | 51.8 ms |
| [15e](15e_lis/) | longest increasing subsequence | 507.0 ms | 102.9 ms | **35.9 ms** | 59.0 ms | 55.5 ms |
| [15f](15f_input/) | counting inversions | 630.0 ms | 94.6 ms | **37.6 ms** | 46.0 ms | 45.4 ms |

Every cell is the mean of five runs, recorded in
[`measurements.json`](measurements.json) with the machine it was measured
on; `bench.py` reproduces it, `scripts/refresh.py` says when a number here
has drifted, and both fail if the paths stop agreeing on the answer. Bold is
faster than the C reference.

`ppy run` is left out of the table because it compiles before it runs — a
flat two seconds or so on every row, which is the development path rather
than the one to submit.

The two `ppy build` columns are the same compiler with different amounts of
Python left in the artifact:

- **`ppy build`** is the hybrid: the kernels are native, but the glue around
  them — `main`, the buffers, `print` of a Python `int` — is the optimized
  Python the build wrote, so the binary embeds an interpreter and imports
  the runtime before the program begins. That is ~35 ms, and on the smaller
  problems it is most of what separates the column from C.
- **`--standalone`** is a binary with no CPython in it at all. `ldd` shows
  libc and nothing else, `ppy.input[int]()` lowers to the same buffered scan
  of standard input that `scanf` does, and the whole N-Queens executable is
  17.0 KB against the C one's 16.1 KB.

Four of the five standalone rows beat the C reference for one reason:
`ppy.input` reads into memory faster than `scanf` parses. Only N-Queens,
which reads a single integer and then computes, stays behind — there is
nothing there for a faster reader to win back. The table above is the
current margin; it is rendered from the record rather than written here, so
it does not go stale.

The column covers five of the six. A standalone build needs everything
`main` reaches to be native, which used to rule out every problem that
allocates a buffer; `ppy.buffer[T](n)` and `ppy.input[Buffer[int]](n)`
allocate natively, so it no longer does. Substring search is the exception:
its text arrives as a token and `ppy.read_token` has no standalone lowering
yet, which is why its row is a dash.

The committed solutions are not what that column times. They keep the
`try`/`except` and the `array.array` that let the same file run under plain
CPython, which is the point of a `.ppy` file; the standalone variants trade
both for the subset and live in [`standalone/`](standalone/), which says
what the trade costs. `bench.py` builds them from there and holds them to
the same answer as every other column.

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
— 12.6 ms for 500k integers against 49.8 ms for `sys.stdin.read().split()`
and 20.6 ms for C's `scanf`. The conversion writes it for you: `int(input())`
becomes `ppy.input[int]()`, `a, b = map(int, input().split())` becomes the
tuple read, and a loop that fills a buffer one value at a time becomes one
bulk `ppy.read_ints`. A module that also touches `sys.stdin` keeps the
`input` it has, since the typed reader owns the file descriptor.
