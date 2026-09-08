# One NumPy expression, one loop

`np.sin(a) * 2.0 + np.cos(b)` is three NumPy calls and two temporary arrays
of a million doubles each. Under `ppy run` it is one loop: the expression
tree lowers to the IR's tensor dialect, `tensor-fusion` folds the chain into
a single `tensor.fused` region, and `lower-tensor` writes one strided loop
with no temporaries, compiled through LLVM. The answer is NumPy's, to the
last bit.

## Elementwise fuses; a reduction fuses at the root

```python
@ppy.pure
@ppy.opt(3)
def normalize(x: np.ndarray) -> np.ndarray:
    scale: float = np.sqrt(np.sum(x * x))
    return x / scale
```

`x * x` feeds `np.sum`, so the multiply fuses into the reduction: one pass
over `x`, one accumulator, no squared array in between. `x / scale` is a
second loop, because it needs the reduction's result. A reduction can sit
only at the root of a fused tree — nested inside an elementwise expression
it would not be elementwise — and the fusion pass knows the difference.

## Guarded, not assumed

The kernel takes exactly what it was compiled for: `float64`, C-contiguous,
one shape across the operands. The generated boundary checks that on every
call. Anything else — a `float32` array, a transposed view, a broadcast —
runs NumPy itself. Reduction order is preserved bit for bit unless the
function is `@ppy.fastmath`, so `np.sum` here gives NumPy's number, not a
number close to it.

```bash
ppy inspect numpy_fusion.ppy --stage tensor     # the fused region, before it becomes loops
```

## Run it

```bash
python  numpy_fusion.ppy
ppy     numpy_fusion.ppy
ppy run numpy_fusion.ppy
```

<!-- outputs:start -->
## What it prints

**`python  numpy_fusion.ppy`**

```text
5.29e-05 2.22324428 1
```

**`ppy     numpy_fusion.ppy`**

```text
5.29e-05 2.22324428 1
```

**`ppy run numpy_fusion.ppy`**

```text
5.29e-05 2.22324428 1
```

<!-- outputs:end -->

## Read on

- [Plugins: NumPy](../../docs/internals/plugins.md) — what the plugin types, fuses, and leaves to NumPy.
- [Parallel fused kernels](../07_parallel/README.md) — the same loop split across threads.
- [The IR: the tensor dialect](../../docs/internals/ir.md) — `tensor.fused`, `lower-tensor`, and the shapes.

`numpy_fusion.ppy` is hand-written; there is no `.py` source and no conversion
step.
