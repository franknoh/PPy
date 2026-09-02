# 15a — N-Queens

Input: `N`. Output: how many ways N queens fit on an N×N board.

## Provenance

Generated, not hand-written. `nqueens.ppy` is exactly what
`ppy convert nqueens.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run. `nqueens.c` is the same solution hand-written in C, reading the
same input with `scanf`.

## What it shows

- Recursion lowers natively, and so do the operators a bitmask solution is
  made of: `~`, unary `-`, the shifts, and the masks.
- No input to speak of, so the wall clock here is startup plus compute.

## Numbers

Wall time of the whole process, measured from outside the way a judge does —
input, interpreter startup and all. Mean ± standard deviation over 5 runs;
`examples/15_algorithms/bench.py` reproduces it and
`scripts/refresh.py` says when these have drifted.

| path | wall |
|---|---:|
| plain CPython | 298.0 ± 10.0 ms |
| `ppy run` | 1735.8 ± 25.0 ms |
| `ppy build` | 45.6 ± 1.6 ms |
| `ppy build --standalone` | 7.0 ± 0.3 ms |
| C (`gcc -O3`, `scanf`) | **4.9 ± 0.1 ms** |

`ppy run` compiles before it runs, which is most of its two seconds; it is
the development path, not the one to submit. `ppy build` produces a binary
that still starts an embedded CPython and imports the runtime: ~35 ms before
a line of the program runs, against C's ~1 ms. `--standalone` has no interpreter in it at all, which is where that
row comes from; the [folder README](../README.md) says what the subset costs.

## Without CPython at all

The `ppy build` binary embeds an interpreter, because the program's glue —
`try`/`except`, the `print` of a Python `int` — is Python. Written so that
everything `main` reaches is native, the same solver builds standalone and
there is no interpreter under it:

```python
def main() -> None:
    n: int = ppy.input[int]()
    print(solve((1 << n) - 1, 0, 0, 0))
```

```bash
cd ../standalone
ppy build --standalone nqueens.ppy -o native
ldd native/nqueens      # linux-vdso, libc, ld-linux -- and nothing else
```

That row is the `--standalone` line in the table above, and it comes out of
a binary the size of the C one:

| path | binary |
|---|---:|
| `ppy build` (hybrid) | 16.3 KB + the runtime it imports |
| `ppy build --standalone` | 17.0 KB |
| C (`gcc -O3`, `scanf`) | 16.1 KB |

`ppy.input[int]()` lowers to the same buffered scan of standard input that
`scanf` does, and the ~35 ms of interpreter startup is simply not there —
which still leaves C ahead here, because this problem reads one integer and
then computes, so there is nothing for the faster reader to win back. What
standalone costs is the subset: no exceptions, no `array.array`, no Python
objects on the path from `main`, which is why the `try`/`except EOFError` of
the committed solution has to go. [`standalone/`](../standalone/) holds that
variant and the four others written the same way.

## Run it

```bash
python  nqueens.ppy < input.txt
ppy run nqueens.ppy < input.txt
ppy build nqueens.ppy -o dist && ./dist/nqueens < input.txt
gcc -O3 nqueens.c -o nqueens_c && ./nqueens_c < input.txt
```
