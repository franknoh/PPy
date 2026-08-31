# 15d — Range sums (Baekjoon 2042)

Input: `N M K`, then N numbers, then M+K commands — `1 b c` assigns, `2 b c`
sums over [b, c). Output: the checksum of the answers.

## Provenance

Hand-written. The `.ppy` is written directly; there is no `.py` source and no
conversion step. The `.c` is the same solution hand-written in C, reading the
same input with `scanf`.

## What it shows

- A function whose writes all happen in a callee still lowers: `run_commands`
  never assigns into the tree itself, it calls `update`.
- `range(size - 1, 0, -1)` — a descending loop with a literal step.
- `query` is `@ppy.pure` while `update` is not, and both are native.

## Numbers

Judge-style: the input is piped in and both halves are timed, mean ±
standard deviation over 5 runs. `examples/15_algorithms/bench.py` reproduces
the whole table.

| phase | plain | `ppy run` | `ppy build` | C (`scanf`) |
|---|---:|---:|---:|---:|
| read | 20.1 ± 0.7 ms | 22.1 ± 1.8 ms | 19.7 ± 0.4 ms | 45.9 ± 0.5 ms |
| solve | 499.0 ± 15.5 ms | 8.1 ± 1.5 ms | 7.7 ± 1.5 ms | 7.6 ± 0.7 ms |

## Run it

```bash
python  segment_tree.ppy < input.txt
ppy run segment_tree.ppy < input.txt
gcc -O3 segment_tree.c -o segment_tree_c && ./segment_tree_c < input.txt
```
