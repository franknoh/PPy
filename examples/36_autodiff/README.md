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

**`ppy emit ir gradients.ppy`**

```text
ppyir 1
module @gradients
dialect core 1
dialect math 1

func @gradients_f(%x: f64, %y: f64) -> f64 attrs {effects = [], ppy.abi = "ppy", ppy.qualname = "gradients.f", ppy.releases_gil = true, ppy.symbol = "ppy_gradients_f"} loc("examples/36_autodiff/gradients.ppy":6:0) {
^entry:
    %x_addr = core.alloca : ptr<f64, stack> loc("examples/36_autodiff/gradients.ppy":6:0)
    core.store %x, %x_addr
    %y_addr = core.alloca : ptr<f64, stack>
    core.store %y, %y_addr
    %0 = core.load %x_addr : f64 loc("examples/36_autodiff/gradients.ppy":7:4)
    %1 = math.sin %0 : f64
    %2 = core.load %y_addr : f64
    %3 = core.mul %1, %2 : f64
    %4 = core.load %x_addr : f64
    %5 = core.load %x_addr : f64
    %6 = core.mul %4, %5 : f64
    %7 = core.add %3, %6 : f64
    core.ret %7
}

func @gradients_g(%x: f64) -> f64 attrs {effects = ["may_raise"], ppy.abi = "ppy", ppy.qualname = "gradients.g", ppy.releases_gil = true, ppy.symbol = "ppy_gradients_g"} loc("examples/36_autodiff/gradients.ppy":10:0) {
^entry:
    %x_addr = core.alloca : ptr<f64, stack> loc("examples/36_autodiff/gradients.ppy":10:0)
    core.store %x, %x_addr
    %0 = core.load %x_addr : f64 loc("examples/36_autodiff/gradients.ppy":11:4)
    %1 = core.neg %0 : f64
    %2 = core.load %x_addr : f64
    %3 = core.mul %1, %2 : f64
    %4 = math.exp %3 : f64
    %5 = core.const 3.0 : f64
    %6 = core.load %x_addr : f64
    %7 = core.mul %5, %6 : f64
    %8 = math.cos %7 : f64
    %9 = core.mul %4, %8 : f64
    core.ret %9
}

func @gradients_slope(%x: f64, %y: f64) -> f64 attrs {effects = [], ppy.abi = "ppy", ppy.qualname = "gradients.slope", ppy.releases_gil = true, ppy.symbol = "ppy_gradients_slope"} loc("examples/36_autodiff/gradients.ppy":19:0) {
… 114 more lines
```

<!-- outputs:end -->
