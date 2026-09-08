# Lanes and the machine

`ppy.simd` is a few scalars operated on at once; `ppy.cpu` is the machine
as a facade with no instruction named. Both have a reference implementation
under CPython and a lowering to a dialect of the IR.

## Provenance

Hand-written. `lanes.ppy` is written directly; there is no `.py` source and
no conversion step involved.

## What it shows

- `simd.Vector[T, N]` is `N` lanes of `T`; `splat`, `load`, `store`,
  `insert`, `extract`, `shuffle`, `select`, and the reductions are the whole
  vocabulary, and `+ - * /`, `& | ^`, and the comparisons work lane by lane.
- Integer lanes wrap at their width: the ramp that starts three below the
  largest `int` wraps exactly as the machine would, on every path.
- A floating-point `reduce_add` folds in lane order, first to last, so the
  sum is the same number everywhere.
- `cpu.prefetch` and `cpu.pause` are hints and change no value;
  `cpu.vector_width[float]()` and `cpu.features()` are facts about the
  machine compiling, folded to constants natively. The two lines that print
  them start with `# `, which is how an example marks what is allowed to
  differ between machines.

## Run it

```bash
python  lanes.ppy
ppy run lanes.ppy
ppy emit ir lanes.ppy   # the simd and cpu dialects
```
