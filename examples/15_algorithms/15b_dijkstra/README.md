# 15b — Shortest path

Input: `V E`, the source `K`, then E lines of `u v w`. Output: the sum of
the reachable distances. 200k nodes and 1.2M edges here.

## Provenance

Generated, not hand-written. `dijkstra.ppy` is exactly what
`ppy convert dijkstra.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run. `dijkstra.c` is the same solution hand-written in C, reading the
same input with `scanf`.

## What it shows

- A buffer handed from one native function to another: `dijkstra` passes its
  heap to `sift_up` and `sift_down`, and the callee writes through the
  caller's memory.
- `break` inside `while True:` — how a sift loop is actually written.
- The adjacency is built natively too, so only the read is Python's.

## Numbers

Wall time of the whole process, measured from outside the way a judge does —
input, interpreter startup and all. Mean ± standard deviation over 5 runs;
`examples/15_algorithms/bench.py` reproduces it and
`scripts/refresh.py` says when these have drifted.

| path | wall |
|---|---:|
| plain CPython | 1842.8 ± 44.7 ms |
| `ppy run` | 2059.2 ± 100.4 ms |
| `ppy build` | 329.9 ± 8.2 ms |
| `ppy build --standalone` | **127.5 ± 3.5 ms** |
| C (`gcc -O3`, `scanf`) | 173.8 ± 2.2 ms |

`ppy run` compiles before it runs, which is most of its two seconds; it is
the development path, not the one to submit. `ppy build` produces a binary
that still starts an embedded CPython and imports the runtime: ~35 ms before
a line of the program runs, against C's ~1 ms. `--standalone` has no interpreter in it at all, which is where that
row comes from; the [folder README](../README.md) says what the subset costs.

## Run it

```bash
python  dijkstra.ppy < input.txt
ppy run dijkstra.ppy < input.txt
ppy build dijkstra.ppy -o dist && ./dist/dijkstra < input.txt
gcc -O3 dijkstra.c -o dijkstra_c && ./dijkstra_c < input.txt
```
