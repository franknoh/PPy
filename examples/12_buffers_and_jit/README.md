# Buffers and JIT

Borrowed memory, reassociation, and specialization.

## Provenance

Hand-written. `buffers_and_jit.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- A `Buffer[float]` is borrowed; a `list[float]` is copied in. At 8192 elements that is about 10x.
- `@ppy.fastmath` permits reassociation, which enables vectorization and changes the last bits.
- `@ppy.jit` specializes on observed argument values, with the guard compiled into C.

## Run it

```bash
python  buffers_and_jit.ppy
ppy     buffers_and_jit.ppy
ppy run buffers_and_jit.ppy
```
