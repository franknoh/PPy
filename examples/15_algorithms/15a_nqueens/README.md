# 15a — N-Queens

Bitmask backtracking: the classic recursion, counting placements for n=12.

## Provenance

Hand-written. `nqueens.ppy` is written directly; there is no `.py` source
and no conversion step. `nqueens.c` is the same algorithm hand-written in C.

## What it shows

- Recursion lowers natively, and so do the operators a bitmask solution is
  made of: `~`, unary `-`, the shifts, and the masks.
- `available & -available` (lowest set bit) and `full & ~(...)` are the whole
  inner loop, and each becomes a single native instruction.

## Numbers

Kernel wall time, mean ± standard deviation over 7 runs:

| path | n=12 |
|---|---:|
| plain CPython | 121.5 ± 2.7 ms |
| `ppy run` | 5.4 ± 0.1 ms |
| `ppy build` binary | 6.2 ± 0.2 ms |
| C (`gcc -O3`) | 4.4 ± 0.4 ms |

## Run it

```bash
python  nqueens.ppy
ppy run nqueens.ppy
ppy build nqueens.ppy -o dist && ./dist/nqueens
gcc -O3 nqueens.c -o nqueens_c && ./nqueens_c
```
