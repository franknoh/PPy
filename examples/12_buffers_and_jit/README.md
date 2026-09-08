# Borrow the memory, keep the order, specialize the rest

Three decisions that decide how fast a numeric function runs, each made
once in a signature or a decorator. `Buffer[float]` lends native code the
caller's memory instead of copying a list — about 10× on 8,192 elements.
`@ppy.fastmath` lets a sum reassociate, which lets it vectorize.
`@ppy.jit` specializes on the argument values it actually sees, with the
guard compiled into C.

## `Buffer[T]` is zero-copy in both directions

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
sum, one of them paying for 8,192 boxed floats per call. A buffer is
borrowed unless the signature says otherwise, so the callee may not keep
it (`E1612`) or write through it without `Mut[...]` (`E1613`).

## Reassociation is opt-in, and it shows

`dot_strict` and `dot_relaxed` are the same dot product. The strict one
keeps CPython's accumulation order and matches it bit for bit; the
`@ppy.fastmath` one lets LLVM vectorize the reduction and differs in the
last bits — the program prints the gap. You choose per function, and the
default is the exact one.

## `@ppy.jit` learns the arguments

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
list[float] copied        118.966 us   
Buffer[float] borrowed    168.604 us   0.71x, same sum: True
dot, strict order         272.590 us   
dot, @ppy.fastmath        273.668 us   1.00x, differs by 0.00e+00
digest, generic           275.525 us   
digest, @ppy.jit          289.774 us   0.95x, same: True
```

**`ppy     buffers_and_jit.ppy`**

```text
list[float] copied        109.531 us   
Buffer[float] borrowed    173.688 us   0.63x, same sum: True
dot, strict order         272.337 us   
dot, @ppy.fastmath        277.740 us   0.98x, differs by 0.00e+00
digest, generic           290.856 us   
digest, @ppy.jit          289.047 us   1.01x, same: True
```

**`ppy run buffers_and_jit.ppy`**

```text
list[float] copied         13.007 us   
Buffer[float] borrowed      4.525 us   2.87x, same sum: True
dot, strict order           4.354 us   
dot, @ppy.fastmath          1.294 us   3.36x, differs by 4.07e-10
digest, generic            12.622 us   
digest, @ppy.jit            8.050 us   1.57x, same: True
```

<!-- outputs:end -->

## Read on

- [Directives and markers](../../docs/guide/directives.md) — `Buffer[T]`, `Owned`/`Borrowed`/`Mut`, `@ppy.jit`, `@ppy.fastmath`.
- [Algorithms](../15_algorithms/README.md) — buffers in eight kernels measured against C.

`buffers_and_jit.ppy` is hand-written; there is no `.py` source and no
conversion step.
