# A derivative that agrees to the last bit

`ppy.grad(f)` is the gradient of `f` with respect to its first parameter;
`ppy.value_and_grad(f, argnums=1)` is the value and the gradient with
respect to another. Under CPython the derivative is built from `f`'s
source; natively the compiler differentiates `f`'s IR. Both follow one rule
table in one order of accumulation, so the nine digits this program prints
are the same nine on every path.

## Reverse mode, twice

```python
def f(x: float, y: float) -> float:
    return math.sin(x) * y + x * x


df = ppy.grad(f)
both = ppy.value_and_grad(f, argnums=1)
```

The first time `df` is called under CPython, the body is restated one
operation at a time and every operation's adjoint is emitted in reverse.
Natively the `autodiff` transform does the same to the IR — reverse mode
over the core and tensor dialects — and `slope(x, y)` compiles to a call to
the derived function. There is no finite differencing anywhere and no
second implementation to drift.

## A derivative is an ordinary function

```python
def newton(x: float) -> float:
    for _ in range(6):
        x = x - g(x) / dg(x)
    return x
```

`dg` is used inside a loop like any callable; `newton` is native, and the
root it finds is a root — the last column prints `g` at that root, and it is
zero to floating-point precision. What cannot be differentiated is refused:
a branch or a loop inside `f` is `E1660`, a non-`float` signature `E1661`,
an effect no derivative follows — I/O, a write, a thread — `E1662`.

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

## Read on

- [Derivatives](../../docs/guide/autodiff.md) — what a differentiable body may contain.
- [The IR](../../docs/internals/ir.md) — the `autodiff` transform among the shared passes.

`gradients.ppy` is hand-written; there is no `.py` source and no conversion
step.
