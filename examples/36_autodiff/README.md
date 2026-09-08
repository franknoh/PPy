# Derivatives

`ppy.grad(f)` is the gradient of `f` with respect to its first parameter,
`ppy.value_and_grad(f, argnums=1)` the value and the gradient with respect
to another, and both follow one rule table on every path.

## Provenance

Hand-written. `gradients.ppy` is written directly; there is no `.py` source
and no conversion step involved.

## What it shows

- Under CPython the derivative is made from `f`'s source the first time it
  is called: the body restated one operation at a time, then every
  operation's adjoint in reverse.
- Natively the compiler differentiates `f`'s IR by the same rules in the
  same order -- the `autodiff` transform, reverse mode -- so `slope` and
  `newton` are native functions whose derivative calls are calls to derived
  functions, and the digits agree bit for bit.
- A derivative is an ordinary function: `newton` uses `dg` inside a loop,
  and the root it finds is a root, as the last column shows.
- A branch, a loop, or an effect inside the differentiated function is
  refused with `E1660`-`E1662` rather than differentiated wrongly.

## Run it

```bash
python  gradients.ppy
ppy run gradients.ppy
ppy emit ir gradients.ppy   # the derived functions beside their originals
```
