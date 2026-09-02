# The CPython-free variants

The five problems whose reachable graph is entirely native, written for
`ppy build --standalone`. Each one answers the same input as the problem it is
named after and prints the same number; `bench.py` builds them, times them,
and fails if any answer differs from the other paths'.

## Why they are separate files

`--standalone` links no interpreter, so what the program may touch is the
native subset and nothing else. Two things in the problem sources are outside
it, and neither is worth pretending about:

- `array.array` — a Python object. `ppy.buffer[int](n)` allocates the same
  memory natively, and is what these use.
- `try` / `except EOFError` — the problem sources fall back to a generated
  input when nothing is piped in, which is a convenience for running them by
  hand. A standalone binary at end of input has no exception to raise.

Everything else is the same algorithm, line for line. Keeping them here rather
than converting the problem sources in place is deliberate: the files beside
each problem are what `ppy convert` writes from ordinary Python, and that
claim is checked on every run.

They also stay out of `examples/run_all.py`, which compares plain CPython,
`ppy run`, and `ppy build` on every example and demands one answer. These
diverge from the Python path at end of input by design, so they are timed
against the problem sources instead of being held to that comparison.

## Which problems are here

| problem | file |
|---|---|
| [15a](../15a_nqueens/README.md) N-Queens | `nqueens.ppy` |
| [15b](../15b_dijkstra/README.md) Dijkstra | `dijkstra.ppy` |
| [15d](../15d_segment_tree/README.md) segment tree | `segment_tree.ppy` |
| [15e](../15e_lis/README.md) LIS | `lis.ppy` |
| [15f](../15f_input/README.md) inversions | `inversions.ppy` |

15c (KMP) is the one that is missing, and not for its memory — it already
holds its four million characters in a `Buffer[ppy.i8]`, one byte each. Its
text arrives as a token, and `ppy.read_token` has no standalone lowering
yet; `ppy.read_ints` does, which is why the other five are here.

## Build one

```bash
ppy build --standalone nqueens.ppy -o native
echo 12 | ./native/nqueens
```

The result needs no CPython, no `ppy`, and no shared library — `ldd` shows
libc and nothing else.
