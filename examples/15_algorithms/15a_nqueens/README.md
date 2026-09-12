# 15a — N-Queens

Input: `N`. Output: how many ways N queens fit on an N×N board.

## A bitmask solver

```python
def solve(full: int, columns: int, diagonal: int, antidiagonal: int) -> int:
```

Recursion lowers natively, and so do the operators a bitmask solution is
made of: `~`, unary `-`, the shifts, and the masks. There is almost no input
to read, so the wall clock is startup plus compute, which makes this the
clearest look at what an embedded interpreter costs and what disappears
when `--standalone` leaves it out.

## Numbers

Wall time of the whole process, measured from outside the way a judge does —
input, interpreter startup and all. Mean ± standard deviation over 5 runs;
`bench.py` reproduces it and `scripts/refresh.py` says when these have
drifted. `ppy run` compiles before it runs, which is most of its time; it is
the development path, not the one to submit. `ppy build` still starts an
embedded CPython and imports the runtime, about 35 ms, before the program
begins.

| path | wall |
|---|---:|
| plain CPython | 154.7 ± 11.9 ms |
| `ppy run` | 144.0 ± 174.3 ms |
| `ppy build --unsafe` | 48.7 ± 1.5 ms |
| `ppy build --standalone --unsafe` | 6.0 ± 0.2 ms |
| C (`gcc -O3`, `scanf`) | **5.2 ± 0.4 ms** |
| C (`clang -O3`, `scanf`) | 6.3 ± 0.6 ms |

## Without CPython at all

Written so that everything `main` reaches is native, the same solver
builds standalone and there is no interpreter under it:

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

It comes out of a binary the size of the C one:

| path | binary |
|---|---:|
| `ppy build --unsafe` (hybrid) | 16.3 KB + the runtime it imports |
| `ppy build --standalone --unsafe` | 17.0 KB |
| C (`gcc -O3`, `scanf`) | 16.1 KB |

`ppy.input[int]()` lowers to the same buffered scan of standard input that
`scanf` does, and the ~35 ms of interpreter startup is simply not there —
which still leaves C slightly ahead here, because this problem reads one
integer and then computes, so there is nothing for the faster reader to win
back. What standalone costs is the subset: no exceptions, no `array.array`,
no Python objects on the path from `main`, which is why the
`try`/`except EOFError` of the committed solution has to go.
[`standalone/`](../standalone/) holds that variant and the four others
written the same way.

## Run it

`input.txt` holds `12`; the answer is 14200.

```bash
python  nqueens.ppy < input.txt
ppy run nqueens.ppy < input.txt
ppy build nqueens.ppy -o dist && ./dist/nqueens < input.txt
gcc   -O3 nqueens.c -o nqueens_c     && ./nqueens_c     < input.txt
clang -O3 nqueens.c -o nqueens_clang && ./nqueens_clang < input.txt
```

<!-- outputs:start -->
## What it prints

**`python  nqueens.ppy < input.txt`**, **`ppy run nqueens.ppy < input.txt`**, **`ppy build nqueens.ppy -o dist && ./dist/nqueens < input.txt`**, **`gcc   -O3 nqueens.c -o nqueens_c     && ./nqueens_c     < input.txt`**, **`clang -O3 nqueens.c -o nqueens_clang && ./nqueens_clang < input.txt`**

```text
14200
```

<!-- outputs:end -->

Generated, not hand-written: `nqueens.ppy` is exactly what
`ppy convert nqueens.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run. `nqueens.c` is
the same solution hand-written in C, reading the same input with `scanf`.
