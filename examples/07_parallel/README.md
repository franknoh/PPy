# Parallelism

Splitting a fused kernel across threads.

## Provenance

Hand-written. `parallel.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- `@ppy.parallel` marks a loop as splittable; the compiler still has to prove it.
- A reassociating reduction is not split without `@ppy.fastmath`, because the answer would change.
- The split result is bit-identical to the serial kernel and to NumPy.
- `parallel.range` ([35_parallel_range](../35_parallel_range/)) is the 0.2.0
  spelling for a loop the program itself declares splittable, pointers and
  buffers included; `@ppy.parallel` stays the switch for the fused NumPy loops here.

## Run it

```bash
python  parallel.ppy
ppy     parallel.ppy
ppy run parallel.ppy
```

<!-- outputs:start -->
## What it prints

**`python  parallel.ppy`**

```text
fused serial         88.5 ms   sample=-0.249979000059
fused parallel      104.2 ms   sample=-0.249979000059
numpy               105.4 ms   sample=-0.249979000059
bit-identical: True
strict == numpy: True
relaxed close   : True
```

**`ppy     parallel.ppy`**

```text
fused serial         91.0 ms   sample=-0.249979000059
fused parallel       91.2 ms   sample=-0.249979000059
numpy                94.7 ms   sample=-0.249979000059
bit-identical: True
strict == numpy: True
relaxed close   : True
```

**`ppy run parallel.ppy`**

```text
fused serial         21.4 ms   sample=-0.249979000059
fused parallel       13.4 ms   sample=-0.249979000059
numpy                21.5 ms   sample=-0.249979000059
bit-identical: True
strict == numpy: True
relaxed close   : True
```

<!-- outputs:end -->
