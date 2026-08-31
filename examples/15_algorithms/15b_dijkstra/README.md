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
| read | 52.5 ± 1.7 ms | 54.4 ± 4.2 ms | 51.9 ± 1.0 ms | 95.2 ± 2.6 ms |
| solve | 1654.3 ± 19.9 ms | 65.3 ± 6.8 ms | 78.8 ± 8.9 ms | 77.8 ± 2.8 ms |

`ppy.read_ints` reads 3.6M integers in half the time `scanf` takes, so the
whole submission finishes in 120 ms against C's 173 ms.

## Run it

```bash
python  dijkstra.ppy < input.txt
ppy run dijkstra.ppy < input.txt
gcc -O3 dijkstra.c -o dijkstra_c && ./dijkstra_c < input.txt
```
