# Parallelism

Splitting a fused kernel across threads.

## Provenance

Hand-written. `parallel.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- `@ppy.parallel` marks a loop as splittable; the compiler still has to prove it.
- A reassociating reduction is not split without `@ppy.fastmath`, because the answer would change.
- The split result is bit-identical to the serial kernel and to NumPy.

## Run it

```bash
python  parallel.ppy
ppy     parallel.ppy
ppy run parallel.ppy
```
