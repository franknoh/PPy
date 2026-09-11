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
each, with the derivative from each framework's own autodiff. The programs
are in [`compare/`](compare/): [`gradients_bench.ppy`](compare/gradients_bench.ppy),
[`gradients_jax.py`](compare/gradients_jax.py), [`gradients_torch.py`](compare/gradients_torch.py).
Milliseconds for the whole batch, best of five, over five processes.

**PPY** -- `ppy.grad(g)` is a function; `newton` calls it in a loop, and a
loop over the starting points calls `newton`. Everything is scalar and
native; nothing is batched:

```python
dg = ppy.grad(g)


def newton(x: float) -> float:
    for _ in range(6):
        x = x - g(x) / dg(x)
    return x


def newton_all(count: int) -> float:
    total = 0.0
    for i in range(count):
        total += newton(0.4 + i * 1e-6)
    return total / count
```

**JAX** -- `jax.grad` over `jnp` functions, and to run 100,000 solves it
wants them batched: `vmap` under `jit`, the loop as `lax.fori_loop` so it
traces, `jax_enable_x64` so the digits match:

```python
def newton(x):
    def step(_, x):
        return x - g(x) / dg(x)

    return jax.lax.fori_loop(0, 6, step, x)


newton_all = jax.jit(jax.vmap(newton))
```

**PyTorch** -- `torch.func.grad` and `vmap`; the six steps stay a Python
loop over a 100,000-element tensor in float64:

```python
dg = grad(g)


def newton(x):
    for _ in range(6):
        x = x - g(x) / dg(x)
    return x


newton_all = vmap(newton)
```

<!-- compare:start -->
| | PPY | JAX `vmap` + `jit` | PyTorch `torch.func` |
|---|---:|---:|---:|
| newton, 100k starts | 12.61 ± 0.07 | **2.08 ± 0.06** | 459.27 ± 419.53 |
<!-- compare:end -->

The derivative is the same nine digits in all three -- reverse mode over
the same rule table. What differs is the shape of the program: JAX and
PyTorch are fast here because the problem batches and XLA and ATen
vectorize across the batch; a scalar `newton` in a Python loop would cost
them tens of microseconds per call. PPY's scalar loop pays nothing per
call and is not vectorized across the batch. Which one is faster depends
on whether your problem comes as 100,000 independent solves or as one.

Intel Core Ultra 9 386H; JAX 0.11.1 and PyTorch 2.14.0 (CPU) on CPython
3.13.13, PPY on CPython 3.13.13, from a checkout on a native filesystem.

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
