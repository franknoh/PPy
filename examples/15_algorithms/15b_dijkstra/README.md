# 15b — Shortest paths over 1.2 million edges

Input: `V E`, the source `K`, then E lines of `u v w`. Output: the sum of
the reachable distances. The judge-sized input is 200k nodes and 1.2M
edges, and the standalone binary answers it in 105 ms against gcc's 147 and
clang's 142 — a Dijkstra whose heap, adjacency, and read are all native,
and whose input arrives through `ppy.input` faster than `scanf` parses it.

## Buffers handed between native functions

```python
def sift_down(keys: Buffer[int], nodes: Buffer[int], size: int, start: int) -> int:
```

`dijkstra` passes its heap to `sift_up` and `sift_down`, and the callee
writes through the caller's memory — a buffer handed on to another native
function stays a native buffer, with no copy and no boundary between them.
`while True:` with a `break` is how a sift loop is actually written, and it
lowers as written. The adjacency is built natively too, so the only Python
on the path is the entry point.

## Numbers

Wall time of the whole process, measured from outside the way a judge does —
input, interpreter startup and all. Mean ± standard deviation over 5 runs;
`examples/15_algorithms/bench.py` reproduces it and
`scripts/refresh.py` says when these have drifted.

| path | wall |
|---|---:|
| plain CPython | 1511.0 ± 10.5 ms |
| `ppy run` | 286.5 ± 225.4 ms |
| `ppy build` | 251.9 ± 2.6 ms |
| `ppy build --standalone` | **104.8 ± 4.3 ms** |
| C (`gcc -O3`, `scanf`) | 147.3 ± 2.2 ms |
| C (`clang -O3`, `scanf`) | 141.6 ± 4.6 ms |

`ppy run` compiles before it runs; it is the development path, not the one
to submit. `ppy build` still starts an embedded CPython and imports the
runtime, ~35 ms, before the program begins. `--standalone` has no
interpreter in it, and reads 1.2 million edges into memory faster than
`scanf` does; the [folder README](../README.md) says what the subset costs.

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

**`python  dijkstra.ppy < input.txt`**

```text
23
```

**`ppy run dijkstra.ppy < input.txt`**

```text
23
```

**`ppy build dijkstra.ppy -o dist && ./dist/dijkstra < input.txt`**

```text
23
```

**`gcc   -O3 dijkstra.c -o dijkstra_c     && ./dijkstra_c     < input.txt`**

```text
23
```

**`clang -O3 dijkstra.c -o dijkstra_clang && ./dijkstra_clang < input.txt`**

```text
23
```

<!-- outputs:end -->

Generated, not hand-written: `dijkstra.ppy` is exactly what
`ppy convert dijkstra.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run. `dijkstra.c` is
the same solution hand-written in C, reading the same input with `scanf`.
