# 15f — Reading input

Counting inversions with a Fenwick tree, and the question every competitive
programmer asks first: how fast is the input?

## Provenance

Hand-written. `inversions.ppy` is written directly; there is no `.py` source
and no conversion step. `inversions.c` is the same algorithm hand-written in
C. `read_bench.py` is a benchmark script, not part of the example's output.

## The short answer: PPY does not make input faster

Reading is an IO effect, so it stays on the interpreter — the same
`sys.stdin`, the same `int()`, the same objects. Reading 500k integers,
mean ± standard deviation over 5 runs:

| how | plain CPython | `ppy run` |
|---|---:|---:|
| `input()` | 187.1 ± 4.0 ms | 199.2 ± 3.4 ms |
| `sys.stdin.readline()` | 53.5 ± 4.0 ms | 57.1 ± 1.8 ms |
| `sys.stdin.read().split()` | 54.8 ± 2.4 ms | 53.0 ± 3.0 ms |

The 3.5× between `input()` and the other two is CPython's own — `input()`
handles prompts and the readline hook per call — and PPY inherits it
unchanged. **Use `sys.stdin.read().split()` for the same reason you would
without PPY.** `read_bench.py` reproduces this table.

## What PPY does make faster: everything after the parse

Once the numbers are in an `array.array`, the kernel is native. Same run,
split into its two halves:

| phase | plain CPython | `ppy run` | `ppy build` | C (`gcc -O3`) |
|---|---:|---:|---:|---:|
| read + parse 5e5 | 24.9 ± 1.0 ms | 28.7 ± 3.0 ms | 31.0 ± 1.2 ms | — |
| count inversions | 683.7 ± 17.4 ms | 36.7 ± 1.2 ms | 31.1 ± 2.3 ms | 30.9 ± 0.7 ms |

So a solution that reads 5e5 numbers and then does real work goes from
709 ms to 65 ms — and the ~25 ms of reading is the floor no amount of
compilation moves. The heavier the compute, the closer the speedup gets to
what the other examples here show; a program that only reads and sums gains
nothing at all.

## Run it

With no argument the program generates its own numbers, so it runs the same
way on all three paths. `-` reads standard input the way a judge would:

```bash
python  inversions.ppy
ppy run inversions.ppy
ppy run inversions.ppy - < numbers.txt
gcc -O3 inversions.c -o inversions_c && ./inversions_c

python read_bench.py          # the input table above
```
