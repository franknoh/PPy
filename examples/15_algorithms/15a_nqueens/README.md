# 15a — N-Queens ([BOJ 9663](https://www.acmicpc.net/problem/9663))

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
| plain | 325 ± 8 ms |
| ppy run | 1877 ± 31 ms |
| ppy build | 50 ± 3 ms |
| C scanf | 5 ± 0 ms |

`ppy run` compiles before it runs, which is most of its two seconds; it is
the development path, not the one to submit. `ppy build` produces a binary
that still starts an embedded CPython and imports the runtime: ~35 ms before
a line of the program runs, against C's ~1 ms.

## Without CPython at all

The `ppy build` binary above embeds an interpreter, because the program's
glue — `try`/`except`, the `print` of a Python `int` — is Python. Written so
that everything `main` reaches is native, the same solver builds standalone
and there is no interpreter under it:

```python
def main() -> None:
    n: int = ppy.input[int]()
    print(solve((1 << n) - 1, 0, 0, 0))
```

```bash
ppy build --standalone queens.ppy -o dist
ldd dist/queens      # linux-vdso, libc, ld-linux -- and nothing else
```

Measured the same way, that binary runs in **7.2 ± 0.1 ms** against the C
reference's 8.5 ± 0.7 ms: `ppy.input[int]()` lowers to the same buffered
scan of standard input that `scanf` does, and the ~35 ms of interpreter
startup is simply not there. What it costs is the subset — no exceptions, no
`array.array`, no Python objects on the path from `main`.

## Run it

```bash
python  nqueens.ppy < input.txt
ppy run nqueens.ppy < input.txt
ppy build nqueens.ppy -o dist && ./dist/nqueens < input.txt
gcc -O3 nqueens.c -o nqueens_c && ./nqueens_c < input.txt
```
