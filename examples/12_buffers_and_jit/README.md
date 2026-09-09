# Buffers and JIT

Borrowed memory, reassociation, and specialization: three decisions that set
how fast a numeric function runs, each made once in a signature or a
decorator.

## `Buffer[T]` is borrowed, a list is copied

```python
@ppy.pure
@ppy.opt(3)
def total_view(values: Buffer[float]) -> float:
    result: float = 0.0
    for i in range(len(values)):
        result += values[i]
    return result
```

`total_list` takes a `list[float]` and is copied into a buffer on every
call; `total_view` takes a `Buffer[float]` — a `memoryview` over an
`array.array` — and reads the caller's memory in place. Same loop, same
sum, about 10× apart on 8,192 elements, because one of them pays for 8,192
boxed floats per call. A buffer is borrowed unless the signature says
otherwise, so the callee may not keep it (`E1612`) or write through it
without `Mut[...]` (`E1613`).

## Reassociation is opt-in

`dot_strict` and `dot_relaxed` are the same dot product. The strict one
keeps CPython's accumulation order and matches it bit for bit; the
`@ppy.fastmath` one lets LLVM vectorize the reduction and differs in the
last bits — the program prints the gap. The choice is per function, and the
default is the exact one.

## `@ppy.jit` specializes on the values it sees

```python
@ppy.jit(threshold=4, max_specializations=4)
@ppy.pure
@ppy.opt(3)
def digest_jit(values: Buffer[int], modulus: int) -> int:
```

After four calls with the same `modulus`, a version specialized to that
constant is compiled; `% 1000003` becomes multiply-and-shift instead of a
division. The guard on the value is in the generated C wrapper, and a call
with a different modulus falls back to the generic version or compiles
another specialization, up to four.

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
list[float] copied         97.306 us   
Buffer[float] borrowed    153.282 us   0.63x, same sum: True
dot, strict order         244.091 us   
dot, @ppy.fastmath        238.020 us   1.03x, differs by 0.00e+00
digest, generic           239.351 us   
digest, @ppy.jit          237.332 us   1.01x, same: True
```

**`ppy     buffers_and_jit.ppy`**

```text
list[float] copied         92.352 us   
Buffer[float] borrowed    146.138 us   0.63x, same sum: True
dot, strict order         238.365 us   
dot, @ppy.fastmath        236.017 us   1.01x, differs by 0.00e+00
digest, generic           246.093 us   
digest, @ppy.jit          245.886 us   1.00x, same: True
```

**`ppy run buffers_and_jit.ppy`**

```text
list[float] copied         10.684 us   
Buffer[float] borrowed      3.485 us   3.07x, same sum: True
dot, strict order           3.541 us   
dot, @ppy.fastmath          1.186 us   2.99x, differs by 4.07e-10
digest, generic            10.894 us   
digest, @ppy.jit            7.474 us   1.46x, same: True
```

<!-- outputs:end -->

Read on: [Directives and markers](../../docs/guide/directives.md) ·
[Algorithms](../15_algorithms/README.md)

`buffers_and_jit.ppy` is hand-written; there is no `.py` source and no
conversion step.
