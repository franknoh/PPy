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

<!-- outputs:start -->
## What it prints

**`python  buffers_and_jit.ppy`**

```text
list[float] copied         95.827 us   
Buffer[float] borrowed    151.539 us   0.63x, same sum: True
dot, strict order         244.469 us   
dot, @ppy.fastmath        243.773 us   1.00x, differs by 0.00e+00
digest, generic           249.708 us   
digest, @ppy.jit          246.653 us   1.01x, same: True
```

**`ppy     buffers_and_jit.ppy`**

```text
list[float] copied         95.258 us   
Buffer[float] borrowed    150.239 us   0.63x, same sum: True
dot, strict order         243.804 us   
dot, @ppy.fastmath        245.135 us   0.99x, differs by 0.00e+00
digest, generic           249.843 us   
digest, @ppy.jit          248.507 us   1.01x, same: True
```

**`ppy run buffers_and_jit.ppy`**

```text
list[float] copied         11.278 us   
Buffer[float] borrowed      3.728 us   3.03x, same sum: True
dot, strict order           3.592 us   
dot, @ppy.fastmath          0.773 us   4.65x, differs by 4.07e-10
digest, generic            11.093 us   
digest, @ppy.jit            7.213 us   1.54x, same: True
```

<!-- outputs:end -->
