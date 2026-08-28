# NumPy fusion

Elementwise NumPy expressions become one kernel.

## Provenance

Hand-written. `numpy_fusion.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- A chain of elementwise operations is fused into a single loop with no temporaries.
- Reductions fuse only at the root of the tree, because a nested reduction is not elementwise.
- Anything the guard rejects falls back to NumPy itself.

## Run it

```bash
python  numpy_fusion.ppy
ppy     numpy_fusion.ppy
ppy run numpy_fusion.ppy
```
