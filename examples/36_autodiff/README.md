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
