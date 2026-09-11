# 15b — Shortest path

Input: `V E`, the source `K`, then E lines of `u v w`. Output: the sum of
the reachable distances. 200k nodes and 1.2M edges at the judge size.

## Buffers handed between native functions

```python
def sift_down(keys: Buffer[int], nodes: Buffer[int], size: int, start: int) -> int:
```

`dijkstra` passes its heap to `sift_up` and `sift_down`, and the callee
writes through the caller's memory: a buffer handed on to another native
function stays native, with no copy and no boundary between them. `while
True:` with a `break` — how a sift loop is actually written — lowers as
written, and the adjacency is built natively too, so only the read is
Python's.

## Where the standalone margin comes from

The standalone binary reads 1.2 million edges through `ppy.input` faster
than `scanf` parses them, and that is its margin over both C compilers; the
[folder README](../README.md) says what the subset costs.

## Numbers

Wall time of the whole process, measured from outside the way a judge does —
input, interpreter startup and all. Mean ± standard deviation over 5 runs;
`bench.py` reproduces it and `scripts/refresh.py` says when these have
drifted. `ppy run` compiles before it runs, which is most of its time; it is
the development path, not the one to submit. `ppy build` still starts an
embedded CPython and imports the runtime, about 35 ms, before the program
begins. Here `main` reads the 1.2 million edge lines, one `ppy.input[tuple[int, int, int]]()` each, and that loop
runs in the interpreter on every path but the standalone one, which scans
the same input natively; the reads are most of the plain and `ppy build`
rows.

| path | wall |
|---|---:|
| plain CPython | 5481.1 ± 611.8 ms |
| `ppy run` | 4005.3 ± 316.8 ms |
| `ppy build --unsafe` | 4669.0 ± 69.9 ms |
| `ppy build --standalone --unsafe` | **123.6 ± 2.9 ms** |
| C (`gcc -O3`, `scanf`) | 156.5 ± 3.2 ms |
| C (`clang -O3`, `scanf`) | 144.5 ± 3.3 ms |

## Run it

`input.txt` is a five-node graph; the answer is 23.

```bash
python  dijkstra.ppy < input.txt
ppy run dijkstra.ppy < input.txt
ppy build dijkstra.ppy -o dist && ./dist/dijkstra < input.txt
gcc   -O3 dijkstra.c -o dijkstra_c     && ./dijkstra_c     < input.txt
clang -O3 dijkstra.c -o dijkstra_clang && ./dijkstra_clang < input.txt
```

<!-- outputs:start -->
## What it prints

**`python  dijkstra.ppy < input.txt`**, **`ppy run dijkstra.ppy < input.txt`**, **`ppy build dijkstra.ppy -o dist && ./dist/dijkstra < input.txt`**, **`gcc   -O3 dijkstra.c -o dijkstra_c     && ./dijkstra_c     < input.txt`**, **`clang -O3 dijkstra.c -o dijkstra_clang && ./dijkstra_clang < input.txt`**

```text
23
```

<!-- outputs:end -->

Generated, not hand-written: `dijkstra.ppy` is exactly what
`ppy convert dijkstra.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run. `dijkstra.c` is
the same solution hand-written in C, reading the same input with `scanf`.
