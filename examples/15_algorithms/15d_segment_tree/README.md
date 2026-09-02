# 15d — Range sums

Input: `N M K`, then N numbers, then M+K commands — `1 b c` assigns,
`2 b c` sums over [b, c). Output: the checksum of the answers.

## Provenance

Generated, not hand-written. `segment_tree.ppy` is exactly what
`ppy convert segment_tree.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run. `segment_tree.c` is the same solution hand-written in C, reading the
same input with `scanf`.

## What it shows

- A function whose writes all happen in a callee still lowers: `run_commands`
  never assigns into the tree itself, it calls `update`.
- `range(size - 1, 0, -1)` — a descending loop with a literal step.
- `query` is `@ppy.pure` while `update` is not, and both are native.

## Numbers

Wall time of the whole process, measured from outside the way a judge does —
input, interpreter startup and all. Mean ± standard deviation over 5 runs;
`examples/15_algorithms/bench.py` reproduces it and
`scripts/refresh.py` says when these have drifted.

| path | wall |
|---|---:|
| plain CPython | 859 ± 23 ms |
| `ppy run` | 2190 ± 45 ms |
| `ppy build` | 132 ± 4 ms |
| `ppy build --standalone` | **27.0 ± 1.7 ms** |
| C (`gcc -O3`, `scanf`) | 61 ± 4 ms |

`ppy run` compiles before it runs, which is most of its two seconds; it is
the development path, not the one to submit. `ppy build` produces a binary
that still starts an embedded CPython and imports the runtime: ~35 ms before
a line of the program runs, against C's ~1 ms. `--standalone` has no interpreter in it at all, which is where that
row comes from; the [folder README](../README.md) says what the subset costs.

## Run it

```bash
python  segment_tree.ppy < input.txt
ppy run segment_tree.ppy < input.txt
ppy build segment_tree.ppy -o dist && ./dist/segment_tree < input.txt
gcc -O3 segment_tree.c -o segment_tree_c && ./segment_tree_c < input.txt
```
