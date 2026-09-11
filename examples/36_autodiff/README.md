# Derivatives

`ppy.grad(f)` is the gradient of `f` with respect to its first parameter,
`ppy.value_and_grad(f, argnums=1)` the value and the gradient with respect
to another, and both follow one rule table on every path, so the nine
digits this program prints are the same nine everywhere.

## Reverse mode, twice

```python
def f(x: float, y: float) -> float:
    return math.sin(x) * y + x * x


df = ppy.grad(f)
both = ppy.value_and_grad(f, argnums=1)
```

Under CPython the derivative is made from `f`'s source the first time it is
called: the body restated one operation at a time, then every operation's
adjoint in reverse. Natively the `autodiff` transform does the same to the
IR — reverse mode over the core and tensor dialects — and `slope(x, y)`
compiles to a call to the derived function. There is no finite differencing
and no second implementation to drift.

## A derivative is an ordinary function

```python
def newton(x: float) -> float:
    for _ in range(6):
        x = x - g(x) / dg(x)
    return x
```

`dg` is used inside a loop like any callable; `newton` is native, and the
root it finds is a root — the last column prints `g` at that root, zero to
floating-point precision. What cannot be differentiated is refused: a
branch or a loop inside `f` is `E1660`, a non-`float` signature `E1661`, an
effect no derivative follows — I/O, a write, a thread — `E1662`.

## Compared with JAX and PyTorch

Newton's method on `g` from 100,000 starting points near 0.4, six steps
each, in [`compare/`](compare/): [`gradients_bench.ppy`](compare/gradients_bench.ppy),
[`gradients_jax.py`](compare/gradients_jax.py), and
[`gradients_torch.py`](compare/gradients_torch.py). Each prints the two
derivatives of the example to nine digits, then the mean root and the worst
residual, and the best of five runs; [`examples/compare.py`](../compare.py)
runs each five times and reports the mean and standard deviation across
processes, in milliseconds. All three print the same answers.

| | PPY | JAX `vmap` + `jit` | PyTorch `torch.func` |
|---|---:|---:|---:|
| newton, 100k starts | 12.25 ± 0.13 | **1.58 ± 0.45** | 7.17 ± 1.57 |

What each port asked for:

- **PPY** is the source above: `ppy.grad(g)` is a function, `newton` calls it
  in a loop, and `newton_all` calls `newton` in a loop over the starting
  points. Every call is native and scalar; nothing is batched.
- **JAX** is `jax.grad` over `jnp` functions; to run 100,000 solves it wants
  them batched -- `jax.vmap(newton)` under `jax.jit`, with the loop as
  `lax.fori_loop` so it traces -- and then XLA vectorizes the whole batch.
  `jax_enable_x64` is what makes the digits match.
- **PyTorch** is `torch.func.grad` and `vmap`, the functional API; the Newton
  loop stays a Python loop of six batched steps over a 100,000-element
  tensor, in float64 by `set_default_dtype`.

The derivative is the same in all three -- reverse mode, the same nine
digits. What differs is the shape of the program: JAX and PyTorch are fast
here because the problem batches, and a scalar `newton` in a Python loop
would cost them tens of microseconds a call; PPY's scalar loop is native and
pays nothing per call, and it is not vectorized across the batch.

Intel Core Ultra 9 386H (16 threads); JAX 0.11.1 and PyTorch 2.14.0 (CPU) on CPython 3.13.13, PPY on
CPython 3.13.13, from a checkout on a native filesystem.

## Run it

```bash
python  gradients.ppy
ppy run gradients.ppy
```

<!-- outputs:start -->
## What it prints

**`python  gradients.ppy`**

```text
2.929684375 4.201088436
-2.488585914 0.523598776 4.7e-17
```

**`ppy run gradients.ppy`**

```text
2.929684375 4.201088436
-2.488585914 0.523598776 4.7e-17
```

<!-- outputs:end -->

Read on: [Derivatives](../../docs/guide/autodiff.md) ·
[The IR](../../docs/internals/ir.md)

`gradients.ppy` is hand-written; there is no `.py` source and no conversion
step.
