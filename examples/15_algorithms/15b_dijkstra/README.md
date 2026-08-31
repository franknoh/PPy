# 15b — Shortest path (Baekjoon 1753)

Input: `V E`, the source `K`, then E lines of `u v w`. Output: the sum of
the reachable distances. 200k nodes and 1.2M edges here.

## Provenance

Hand-written. The `.ppy` is written directly; there is no `.py` source and no
conversion step. The `.c` is the same solution hand-written in C, reading the
same input with `scanf`.

## What it shows

- A buffer handed from one native function to another: `dijkstra` passes its
  heap to `sift_up` and `sift_down`, and the callee writes through the
  caller's memory.
- `break` inside `while True:` — how a sift loop is actually written.
- The adjacency is built natively too, so only the read is Python's.

## Numbers

Judge-style: the input is piped in and both halves are timed, mean ±
standard deviation over 5 runs. `examples/15_algorithms/bench.py` reproduces
the whole table.

| phase | plain | `ppy run` | `ppy build` | C (`scanf`) |
|---|---:|---:|---:|---:|
| read | 66.4 ± 2.3 ms | 73.3 ± 3.7 ms | 63.0 ± 1.8 ms | 93.3 ± 1.6 ms |
| solve | 1652.2 ± 9.0 ms | 66.2 ± 4.6 ms | 71.5 ± 6.1 ms | 75.6 ± 7.2 ms |

`ppy.input[Buffer[int]](n)` reads 3.6M integers faster than `scanf` does, so
the whole submission finishes in 140 ms against C's 169 ms.

## Run it

```bash
python  dijkstra.ppy < input.txt
ppy run dijkstra.ppy < input.txt
gcc -O3 dijkstra.c -o dijkstra_c && ./dijkstra_c < input.txt
```
