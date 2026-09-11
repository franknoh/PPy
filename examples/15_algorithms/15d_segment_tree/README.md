# 15d — Range sums

Input: `N M K`, then N numbers, then M+K commands — `1 b c` assigns, `2 b c`
sums over [b, c). Output: the checksum of the answers.

## Writes in the callee still count

```python
def run_commands(tree: Buffer[int], commands: Buffer[int], size: int, rounds: int) -> int:
```

`run_commands` never assigns into the tree itself; it calls `update`, and
the write lands in the caller's memory either way. A function whose writes
all happen inside a callee it handed a buffer to lowers like any other.
`query` is `@ppy.pure` while `update` is not, and both are native.
`range(size - 1, 0, -1)`, the descending build of the tree with a literal
step, lowers as written.

## Where the standalone margin comes from

The standalone binary answers 400,000 commands in half the time of either C
compiler, because reading them through `ppy.input` costs half of what
`scanf` charges; the [folder README](../README.md) says what the subset
costs.

## Numbers

Wall time of the whole process, measured from outside the way a judge does —
input, interpreter startup and all. Mean ± standard deviation over 5 runs;
`bench.py` reproduces it and `scripts/refresh.py` says when these have
drifted. `ppy run` compiles before it runs, which is most of its time; it is
the development path, not the one to submit. `ppy build` still starts an
embedded CPython and imports the runtime, about 35 ms, before the program
begins. Here `main` reads the 400 thousand command lines, one `ppy.input[tuple[int, int, int]]()` each, and that loop
runs in the interpreter on every path but the standalone one, which scans
the same input natively; the reads are most of the plain and `ppy build`
rows.

| path | wall |
|---|---:|
| plain CPython | 1035.4 ± 12.7 ms |
| `ppy run` | 712.6 ± 185.3 ms |
| `ppy build --unsafe` | 773.3 ± 12.7 ms |
| `ppy build --standalone --unsafe` | **30.9 ± 1.6 ms** |
| C (`gcc -O3`, `scanf`) | 52.2 ± 0.7 ms |
| C (`clang -O3`, `scanf`) | 51.3 ± 0.2 ms |

## Run it

`input.txt` is eight values and four commands; the checksum is 28.

```bash
python  segment_tree.ppy < input.txt
ppy run segment_tree.ppy < input.txt
ppy build segment_tree.ppy -o dist && ./dist/segment_tree < input.txt
gcc   -O3 segment_tree.c -o segment_tree_c     && ./segment_tree_c     < input.txt
clang -O3 segment_tree.c -o segment_tree_clang && ./segment_tree_clang < input.txt
```

<!-- outputs:start -->
## What it prints

**`python  segment_tree.ppy < input.txt`**, **`ppy run segment_tree.ppy < input.txt`**, **`ppy build segment_tree.ppy -o dist && ./dist/segment_tree < input.txt`**, **`gcc   -O3 segment_tree.c -o segment_tree_c     && ./segment_tree_c     < input.txt`**, **`clang -O3 segment_tree.c -o segment_tree_clang && ./segment_tree_clang < input.txt`**

```text
28
```

<!-- outputs:end -->

Generated, not hand-written: `segment_tree.ppy` is exactly what
`ppy convert segment_tree.py --promote-buffers` writes, and
`examples/verify_conversions.py` checks that on every run. `segment_tree.c`
is the same solution hand-written in C, reading the same input with
`scanf`.
