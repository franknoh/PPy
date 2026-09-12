# NumPy fusion

An elementwise NumPy expression becomes one loop with no temporaries.
`np.sin(a) * 2.0 + np.cos(b)` is three NumPy calls and two intermediate
arrays; under `ppy run` it is a single loop compiled through LLVM.

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

## Compared with NumPy, numexpr, Numba, and JAX

The two expressions over eight million doubles, in [`compare/`](compare/):
[`fusion_bench.ppy`](compare/fusion_bench.ppy),
[`fusion_numpy.py`](compare/fusion_numpy.py),
[`fusion_numexpr.py`](compare/fusion_numexpr.py),
[`fusion_numba.py`](compare/fusion_numba.py),
[`fusion_jax.py`](compare/fusion_jax.py). Milliseconds, best of five calls,
over five processes.

**PPY** is the NumPy expression as written, in a function:

```python
@ppy.pure
def blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sin(a) * 2.0 + np.cos(b)
```

**NumPy** is the same line with no function around it: three calls, two 64
MB temporaries. **numexpr** takes the expression as a string and evaluates
it in chunks across threads:

```python
ne.evaluate("sin(a) * 2.0 + cos(b)", local_dict={"a": a, "b": b})
```

**Numba** is an explicit loop under `@njit` writing into `np.empty_like`;
**JAX** is the expression under `jax.jit` on the CPU, with `jax_enable_x64`:

```python
@njit
def blend(a, b):
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        out[i] = math.sin(a[i]) * 2.0 + math.cos(b[i])
    return out
```

```python
@jax.jit
def blend(a, b):
    return jnp.sin(a) * 2.0 + jnp.cos(b)
```

<!-- compare:start -->
| | PPY | NumPy | numexpr | Numba `@njit` | JAX `jit` |
|---|---:|---:|---:|---:|---:|
| normalize | 22.25 ± 0.49 | 21.68 ± 0.54 | 13.03 ± 0.28 | 25.63 ± 0.79 | **8.03 ± 0.14** |
| blend | 73.52 ± 0.72 | 88.02 ± 0.73 | **10.07 ± 0.20** | 84.53 ± 0.84 | 22.84 ± 0.22 |
<!-- compare:end -->

`blend` is two transcendentals per element, and on one thread that is
what the time is: PPY, NumPy, and Numba each call `sin` and `cos` once per
element, and fusing away NumPy's two temporaries takes off the fifth of the
time that was memory traffic. numexpr and JAX evaluate `sin` and `cos`
across vector lanes and across threads, which is where their rows come
from. `normalize` is a reduction in NumPy's own order followed by a
division pass, on every tool that keeps the order; PPY's sum is NumPy's
sum, so the row is NumPy's time, and `@ppy.fastmath` is the permission to
reassociate it ([Parallel](../07_parallel/README.md) shows what that buys).

Intel Core Ultra 9 386H (16 threads); Numba 0.67.0, numexpr 2.14.2, NumPy 2.5.3 on CPython 3.12.13, JAX 0.11.1 on
CPython 3.13.13, PPY on CPython 3.14.5, from a checkout on a native
filesystem.

## Run it

```bash
python  numpy_fusion.ppy
ppy     numpy_fusion.ppy
ppy run numpy_fusion.ppy
```

<!-- outputs:start -->
## What it prints

**`python  numpy_fusion.ppy`**, **`ppy     numpy_fusion.ppy`**, **`ppy run numpy_fusion.ppy`**

```text
5.29e-05 2.22324428 1
```

<!-- outputs:end -->

Read on: [Plugins: NumPy](../../docs/internals/plugins.md) ·
[Parallel fused kernels](../07_parallel/README.md) ·
[The IR: the tensor dialect](../../docs/internals/ir.md)

`numpy_fusion.ppy` is hand-written; there is no `.py` source and no conversion
step.
