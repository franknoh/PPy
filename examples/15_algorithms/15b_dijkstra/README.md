# 15b — Dijkstra

Shortest paths over a 200k-node graph with a hand-written binary heap.

## Provenance

Hand-written. `dijkstra.ppy` is written directly; there is no `.py` source
and no conversion step. `dijkstra.c` is the same algorithm hand-written in C.

## What it shows

- A buffer handed from one native function to another: `dijkstra` passes its
  heap to `sift_up` and `sift_down`, and the callee writes through the
  caller's memory exactly as the Python semantics say it should.
- `break` inside `while True:` — how a sift loop is actually written.
- A `Final` module constant (`INFINITY`) folds into the native code instead
  of staying a global read that would keep the function boxed.

## Numbers

Kernel wall time, mean ± standard deviation over 7 runs:

| path | 200k nodes, 1.2M edges |
|---|---:|
| plain CPython | 1409.7 ± 24.6 ms |
| `ppy run` | 59.6 ± 7.5 ms |
| `ppy build` binary | 71.6 ± 4.6 ms |
| C (`gcc -O3`) | 79.7 ± 7.0 ms |

The heap chases pointers rather than doing arithmetic, so this is one where
PPY's bounds-checked native code lands ahead of the C reference.

## Run it

```bash
python  dijkstra.ppy
ppy run dijkstra.ppy
gcc -O3 dijkstra.c -o dijkstra_c && ./dijkstra_c
```
