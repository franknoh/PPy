# NumPy fusion

An elementwise NumPy expression becomes one loop with no temporaries.
`np.sin(a) * 2.0 + np.cos(b)` is three NumPy calls and two intermediate
arrays; under `ppy run` it is a single loop compiled through LLVM, and the
answer is NumPy's to the last bit.

## How the expression is fused

The expression tree lowers to the IR's tensor dialect — `numpy.sin` is
`tensor.unary {op = sin}`, `*` and `+` are `tensor.mul` and `tensor.add` —
and the `tensor-fusion` pass folds the chain into one `tensor.fused`
region: a computation of one output element from one element of each
input. `lower-tensor` writes it as one strided loop.

```python
@ppy.pure
@ppy.opt(3)
def normalize(x: np.ndarray) -> np.ndarray:
    scale: float = np.sqrt(np.sum(x * x))
    return x / scale
```

A reduction fuses only at the root of a tree. In `normalize`, `x * x` feeds
`np.sum`, so the multiply fuses into the reduction — one pass over `x`, one
accumulator, no squared array — and `x / scale` is a second loop because it
needs the reduction's result. Nested inside an elementwise expression a
reduction would not be elementwise, and the pass knows the difference.

```bash
ppy inspect numpy_fusion.ppy --stage tensor     # the fused region before it becomes loops
```

## What the guard checks

The kernel takes exactly what it was compiled for: `float64`, C-contiguous,
one shape across the operands. The generated boundary checks that on every
call. Anything else — a `float32` array, a transposed view, a broadcast —
runs NumPy itself. Reduction order is preserved bit for bit unless the
function is `@ppy.fastmath`, so `np.sum` here gives NumPy's number, not one
close to it.

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

Read on: [Plugins: NumPy](../../docs/internals/plugins.md) ·
[Parallel fused kernels](../07_parallel/README.md) ·
[The IR: the tensor dialect](../../docs/internals/ir.md)

`numpy_fusion.ppy` is hand-written; there is no `.py` source and no conversion
step.
