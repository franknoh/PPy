# 15d — Segment tree

200k interleaved point updates and range-sum queries over 2^18 leaves.

## Provenance

Hand-written. `segment_tree.ppy` is written directly; there is no `.py`
source and no conversion step. `segment_tree.c` is the same data structure
hand-written in C.

## What it shows

- A function whose writes all happen in a callee still lowers: `workload`
  itself never assigns into the tree, it calls `update`, and the write lands
  in the caller's buffer just the same.
- `range(size - 1, 0, -1)` — a descending loop with a literal step.
- `query` is `@ppy.pure` while `update` is not, and both are native.

## Numbers

Kernel wall time, mean ± standard deviation over 7 runs:

| path | 2e5 update+query rounds |
|---|---:|
| plain CPython | 489.8 ± 7.9 ms |
| `ppy run` | 6.1 ± 0.7 ms |
| `ppy build` binary | 8.3 ± 1.7 ms |
| C (`gcc -O3`) | 5.7 ± 0.2 ms |

## Run it

```bash
python  segment_tree.ppy
ppy run segment_tree.ppy
gcc -O3 segment_tree.c -o segment_tree_c && ./segment_tree_c
```
