# Parallel ranges

`for i in parallel.range(n)` says the iterations may run at once. Under
CPython they run in order, which is one of the orders allowed; natively the
body becomes a function over a chunk of the range, and the build's parallel
backend decides how the chunks run.

## Provenance

Hand-written. `ranges.ppy` is written directly; there is no `.py` source and
no conversion step involved.

## What it shows

- A loop that writes through a pointer or a buffer splits freely.
- One `+=` into an outer `int` or `float` is a reduction. An integer
  reduction splits freely; a floating-point one keeps its order, because the
  answer would change otherwise -- `dot` prints twelve digits and they are
  the same on every path -- unless the function is `@ppy.fastmath`, which
  permits the reassociation and is why `dot_relaxed` prints six.
- `@ppy.parallel` on a function asks the same of its outermost `range`
  loops without spelling `parallel.range`; a loop that does not pass the
  analysis stays serial and says why in an optimization remark.
- `[tool.ppy.parallel] backend` chooses `threads`, `serial`, `simd`, or
  `openmp` (for `ppy emit c`), and every choice gives the same answer.

## Run it

```bash
python  ranges.ppy
ppy run ranges.ppy
ppy build --report-opt ranges.ppy    # which loops became parallel, and why the rest did not
ppy emit c ranges.ppy                # the loops as C; `backend = "openmp"` spells them as OpenMP
```
