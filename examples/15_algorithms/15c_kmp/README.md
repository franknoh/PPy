# 15c — KMP

Substring search over four million symbols, failure table and all.

## Provenance

Hand-written. `kmp.ppy` is written directly; there is no `.py` source and no
conversion step. `kmp.c` is the same algorithm hand-written in C.

## What it shows

- The `while` that walks the failure links is the whole trick, and it lowers
  with `break`-free control flow straight into the scan loop.
- Two native functions over the same borrowed buffers: the failure table is
  built once from the caller, then the scan reads it.

## Numbers

Kernel wall time, mean ± standard deviation over 7 runs:

| path | 4e6 symbols |
|---|---:|
| plain CPython | 265.0 ± 10.0 ms |
| `ppy run` | 5.1 ± 0.3 ms |
| `ppy build` binary | 5.3 ± 0.2 ms |
| C (`gcc -O3`) | 4.4 ± 0.2 ms |

## Run it

```bash
python  kmp.ppy
ppy run kmp.ppy
gcc -O3 kmp.c -o kmp_c && ./kmp_c
```
