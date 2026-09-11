# Algorithms

Eight compute kernels measured against the same eight in C, and six judge
problems timed the way a judge times them.

`algorithms.ppy` is sieve, Collatz, knapsack, edit distance,
Floyd–Warshall, matmul, union-find, and Fermat, against `algorithms.c`
compiled with both gcc and clang. `15a`–`15f` are six competitive-programming
problems, each its own folder, read from standard input and timed as a whole
process, startup included.

## The eight kernels against C

Same machine, same session, kernel wall time — mean ± standard deviation
over 7 runs, each a fresh process. Every row prints the same answer in every
column:

| kernel | plain | `ppy run` | `ppy build --unsafe` | C (`gcc -O3`) | C (`clang -O3`) |
|---|---:|---:|---:|---:|---:|
| sieve 2e6 | 173.1 ± 2.9 | 9.2 ± 0.5 | **8.8 ± 0.4** | 13.9 ± 0.6 | 13.8 ± 0.3 |
| collatz 3e5 | 1111.0 ± 13.1 | 39.9 ± 0.8 | **30.3 ± 0.4** | 41.4 ± 0.5 | 32.2 ± 0.3 |
| knapsack 400×2e4 | 423.1 ± 2.7 | 5.0 ± 0.1 | 3.6 ± 0.2 | 2.4 ± 0.1 | **2.1 ± 0.3** |
| edit 2000×2000 | 471.4 ± 6.5 | **2.9 ± 0.1** | **2.9 ± 0.3** | 3.6 ± 0.1 | 3.5 ± 0.2 |
| floyd 220 | 436.3 ± 5.3 | 3.4 ± 0.5 | 3.1 ± 0.2 | 5.0 ± 0.1 | **2.8 ± 0.0** |
| matmul 220 | 473.9 ± 5.7 | 3.6 ± 0.4 | 3.5 ± 0.1 | **2.1 ± 0.1** | 3.4 ± 0.1 |
| union-find 5e5 | 149.6 ± 3.8 | **3.2 ± 0.2** | **3.2 ± 0.4** | 3.6 ± 0.2 | 3.8 ± 0.3 |
| fermat 6e4 | 23.9 ± 0.3 | 2.3 ± 0.1 | 2.6 ± 0.2 | 1.8 ± 0.1 | **1.5 ± 0.1** |

The native path is 9× to 160× faster than plain CPython on these eight, and
all three paths print identical answers. Bold is the fastest cell in the
row. `ppy run` keeps Python-integer semantics — overflow is guarded and
falls back to arbitrary precision — and `ppy build --unsafe` is the wrap-semantics
artifact, which is where collatz picks up its remaining 10 ms and knapsack
its 1.4 ms; the other kernels are already guard-free in the loop and do not
move. `--host-cpu` is within the noise on all eight.

Guard hoisting (`[tool.ppy.llvm] safeguards`) is what closed most of the old
gaps: a multiplied index like `i * n + k` proves its extreme cases once in a
guard block ahead of the loop, and the body runs plain `mul nsw` with no
side exits — which is what lets LLVM strength-reduce and vectorize. Against
both compilers PPY wins sieve, edit distance, and union-find, and collatz
once the artifact wraps; floyd beats gcc and loses to clang by a third of a
millisecond; gcc and clang keep knapsack, matmul, and fermat, whose
remaining guards live on data values no range can prove. gcc 13.3 and
clang 22.1, both `-O3`, no `-march`. Writes go through borrowed buffers,
so the caller sees them.

## The eight kernels against Numba, Mojo, and Codon

The same eight kernels ported to three tools that compile Python-shaped
code, in [`compare/`](compare/): [`algorithms_numba.py`](compare/algorithms_numba.py),
[`algorithms.mojo`](compare/algorithms.mojo), [`algorithms_codon.py`](compare/algorithms_codon.py).
Kernel wall time inside the program, milliseconds, mean and spread over
seven fresh processes; Numba is timed after one warm call so its compile is
not in the number. Collatz, as each of them spells it:

**PPY** -- annotated Python, and `ppy run` keeps Python's integers: the
multiply is overflow-checked and falls back to arbitrary precision.

```python
def collatz_longest(limit: int) -> int:
    best: int = 0
    for start in range(1, limit):
        n: int = start
        steps: int = 0
        while n != 1:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            steps += 1
        best = max(best, steps)
    return best
```

**Numba** is the same function with `@njit` in place of the annotations and
a NumPy array in place of `array.array`; **Codon** is the same source with
`List[int]` annotations, run with `codon run -release`. Both wrap at 64 bits
without a word.

```python
@njit
def collatz_longest(limit):
    best = 0
    for start in range(1, limit):
        n = start
        ...
```

**Mojo** is typed by hand -- `Int` for every index, `Int64` for every
element, `var` on every local -- and is the plain `List` version without
`UnsafePointer`:

```mojo
def collatz_longest(limit: Int) -> Int64:
    var best: Int64 = 0
    for start in range(1, limit):
        var n = Int64(start)
        var steps: Int64 = 0
        while n != 1:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            steps += 1
        best = max(best, steps)
    return best
```

<!-- compare:start -->
| | `ppy run` | `ppy build --unsafe` | Numba `@njit` | Mojo | Codon |
|---|---:|---:|---:|---:|---:|
| sieve 2e6 | 8.52 ± 0.33 | 9.02 ± 0.66 | **7.56 ± 0.17** | 11.84 ± 0.32 | 7.96 ± 0.34 |
| collatz 3e5 | 41.00 ± 0.93 | **31.68 ± 0.94** | 32.64 ± 0.64 | 69.37 ± 1.33 | 32.30 ± 0.38 |
| knapsack 400x2e4 | 5.04 ± 0.05 | 3.62 ± 0.08 | 4.52 ± 0.08 | **2.40 ± 0.41** | 5.62 ± 0.08 |
| edit 2000x2000 | 2.94 ± 0.05 | 2.80 ± 0.00 | **1.78 ± 0.08** | 8.40 ± 0.20 | 2.52 ± 0.13 |
| floyd 220 | 3.24 ± 0.15 | 3.06 ± 0.05 | 2.40 ± 0.00 | 3.01 ± 0.06 | **1.10 ± 0.00** |
| matmul 220 | 3.68 ± 0.11 | 3.64 ± 0.05 | **3.34 ± 0.05** | 8.01 ± 0.21 | 3.82 ± 0.04 |
| union-find 5e5 | 3.22 ± 0.28 | 3.20 ± 0.32 | **2.56 ± 0.09** | 7.28 ± 0.19 | 3.72 ± 0.41 |
| fermat 6e4 | 2.52 ± 0.29 | 2.58 ± 0.13 | 1.68 ± 0.04 | 2.25 ± 0.15 | **1.48 ± 0.04** |
<!-- compare:end -->

`ppy build --unsafe` is the wrap-semantics column and the one to read
against the other three; `ppy run` carries the overflow guards, which is
where collatz pays its 10 ms. Numba leads on six of the eight by ten to
thirty percent -- the same LLVM, a `for` over a NumPy array, and no guard
of any kind. Mojo's plain `List` loop is the slowest on five kernels and
the fastest on knapsack, where its bounds-free table indexing shows.
Numba 0.67.0 on CPython 3.12.13, Mojo 1.0.0 (`-O3`), Codon 0.19.6, PPY on
CPython 3.13.13; Intel Core Ultra 9 386H.

## The six problems

Each subfolder is one competitive-programming problem — the shapes a judge
sets — read from standard input and answered on stdout, with its own input
and output format so nothing depends on an outside site being up. Nothing
inside the programs is instrumented: the times below are wall time of the
whole process, measured from outside, input and interpreter startup
included. The C reference reads the same input with `scanf`.

| | problem | plain | `ppy build --unsafe` | `--standalone --unsafe` | C (`gcc`) | C (`clang`) |
|---|---|---:|---:|---:|---:|---:|
| [15a](15a_nqueens/) | N-Queens | 141.9 ms | 44.5 ms | 5.7 ms | 4.9 ms | 5.4 ms |
| [15b](15b_dijkstra/) | shortest path | 3112.1 ms | 2247.0 ms | **121.1 ms** | 146.4 ms | 144.5 ms |
| [15c](15c_kmp/) | substring search | 304.2 ms | 58.6 ms | — | 9.8 ms | 9.6 ms |
| [15d](15d_segment_tree/) | range sums | 1035.4 ms | 773.3 ms | **30.9 ms** | 52.2 ms | 51.3 ms |
| [15e](15e_lis/) | longest increasing subsequence | 536.1 ms | 95.8 ms | **41.6 ms** | 59.7 ms | 56.8 ms |
| [15f](15f_input/) | counting inversions | 660.2 ms | 91.5 ms | **41.4 ms** | 46.4 ms | 45.7 ms |

Every cell is the mean of five runs, recorded in
[`measurements.json`](measurements.json) with the machine it was measured
on; `bench.py` reproduces it, `scripts/refresh.py` says when a number here
has drifted, and both fail if the paths stop agreeing on the answer. Bold
is faster than both C references. `ppy run` is left out because it compiles
before it runs — a flat two seconds or so on every row, which is the
development path rather than the one to submit.

## `ppy build --unsafe` and `--standalone --unsafe`

- **`ppy build --unsafe`** is the hybrid: the kernels are native, but the glue around
  them — `main`, the buffers, `print` of a Python `int` — is the optimized
  Python the build wrote, so the binary embeds an interpreter and imports
  the runtime before the program begins. That is ~35 ms, and on the smaller
  problems it is most of what separates the column from C. Where `main`
  reads its input line by line -- 1.2 million edge lines in 15b, 400
  thousand command lines in 15d -- each `ppy.input[...]()` is a call into
  the runtime from the interpreter, about a microsecond apiece, and those
  reads are most of the plain and `ppy build --unsafe` columns there.
- **`--standalone`** is a binary with no CPython in it at all. `ldd` shows
  libc and nothing else, `ppy.scan[int]()` lowers to the same buffered scan
  of standard input that `scanf` does, and the whole N-Queens executable is
  17.0 KB against the C one's 16.1 KB.

Four of the five standalone rows beat both C references for one reason:
`ppy.scan` reads into memory faster than `scanf` parses. N-Queens, which
reads a single integer and then computes, stays behind — there is nothing
there for a faster reader to win back. The column covers five of the six: a
standalone build needs everything `main` reaches to be native, and
substring search reads its text as a token, for which `ppy.read_token` has
no standalone lowering yet. The standalone variants trade `try`/`except`
and `array.array` for the subset and live in [`standalone/`](standalone/);
`bench.py` builds them from there and holds them to the same answer as
every other column.

Five of the six problem solutions are written as ordinary Python and
converted: `ppy convert <name>.py --promote-buffers` writes the `.ppy`
beside it, and `examples/verify_conversions.py` checks that the committed
file is exactly that. [15c](15c_kmp/) is hand-written, because the
character buffer it wants has no plain-Python spelling that converts to it.

## Reading input

```python
n = ppy.input[int]()                 # one line, as int(input()) reads it
a, b = ppy.input[tuple[int, int]]()  # one line, exactly two fields
row = ppy.input[Buffer[int]]()       # one line of integers, into a buffer
values = ppy.scan[Buffer[int]](n)    # n integer tokens, whatever lines they are on
```

`ppy.input[T]()` reads one line the way the builtin `input()` does and
types the result from `T`; `ppy.scan[T]()` reads tokens across lines. Both
buffer reads go straight into memory rather than building a Python object
per field — 12.6 ms for 500k integers against 49.8 ms for
`sys.stdin.read().split()` and 20.6 ms for C's `scanf`. The conversion
writes the line reads for you: `int(input())` becomes `ppy.input[int]()`,
`a, b = map(int, input().split())` the tuple read,
`array.array("q", map(int, input().split()))` the buffer line read, and
each reads the line the original read. A module that also touches
`sys.stdin` keeps the `input` it has, since the typed reader owns the file
descriptor.

## Run it

```bash
python  algorithms.ppy
ppy run algorithms.ppy
gcc   -O3 algorithms.c -o algorithms_c     -lm && ./algorithms_c
clang -O3 algorithms.c -o algorithms_clang -lm && ./algorithms_clang
```

<!-- outputs:start -->
## What it prints

**`python  algorithms.ppy`**

```text
sieve 2e6              195.0 ms   -> 148933
collatz 3e5           1209.1 ms   -> 442
knapsack 400x2e4       480.0 ms   -> 199600
edit 2000x2000         515.0 ms   -> 1846
floyd 220              473.4 ms   -> 558837
matmul 220             505.6 ms   -> 18883
union-find 5e5         155.6 ms   -> 250000
fermat 6e4              26.3 ms   -> 6114
```

**`ppy run algorithms.ppy`**

```text
sieve 2e6                9.9 ms   -> 148933
collatz 3e5             41.8 ms   -> 442
knapsack 400x2e4         5.1 ms   -> 199600
edit 2000x2000           3.7 ms   -> 1846
floyd 220                3.2 ms   -> 558837
matmul 220               4.0 ms   -> 18883
union-find 5e5           3.3 ms   -> 250000
fermat 6e4               2.7 ms   -> 6114
```

**`gcc   -O3 algorithms.c -o algorithms_c     -lm && ./algorithms_c`**

```text
sieve 2e6               15.9 ms   -> 148933
collatz 3e5             43.6 ms   -> 442
knapsack 400x2e4         2.6 ms   -> 199600
edit 2000x2000           3.8 ms   -> 1846
floyd 220                5.8 ms   -> 558837
matmul 220               2.5 ms   -> 18883
union-find 5e5           3.7 ms   -> 250000
fermat 6e4               1.8 ms   -> 6114
```

**`clang -O3 algorithms.c -o algorithms_clang -lm && ./algorithms_clang`**

```text
sieve 2e6               18.7 ms   -> 148933
collatz 3e5             40.2 ms   -> 442
knapsack 400x2e4         2.0 ms   -> 199600
edit 2000x2000           3.6 ms   -> 1846
floyd 220                3.0 ms   -> 558837
matmul 220               3.9 ms   -> 18883
union-find 5e5           3.9 ms   -> 250000
fermat 6e4               1.5 ms   -> 6114
```

<!-- outputs:end -->

Each subfolder's README has its own commands, with a small `input.txt` so
they run as written; `bench.py` generates the judge-sized inputs.

Read on: [Reading input](../../docs/guide/input.md) ·
[Native lowering](../../docs/guide/native-lowering.md) ·
[Performance](../../docs/reference/performance.md) ·
[Buffers and JIT](../12_buffers_and_jit/README.md)

`algorithms.ppy` is hand-written; there is no `.py` source and no conversion
step. `algorithms.c` is the same eight kernels hand-written in C, with the
same workloads and the same printed answers. The six problem subfolders say
in their own READMEs which of them are generated.
