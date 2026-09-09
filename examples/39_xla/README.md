# XLA

`@xla.jit` marks a function of floats, ints, and bools whose body is one
block of arithmetic and math. The compiler lowers it to the IR and emits
StableHLO itself — no trace, no JAX in the compiler — and the PJRT bridge
compiles the module once and runs each call on a device.

## What XLA takes today

```python
@xla.jit
def f(x: float, y: float) -> float:
    product = math.sin(x) * y
    return (product + (x if product > 0.0 else -x)) / 2.0
```

Arithmetic, `math.sin`, and a conditional expression that lowers to a
`select`: `f` is what the StableHLO backend accepts, and `ppy emit
stablehlo` prints the module (`ppy inspect --stage stablehlo` the same).
`branchy` is not, yet — an `if` statement is control flow — so the compiler
reports `W2007` and the function runs as written.

## Devices and the last bits

Under plain CPython, and wherever no device is present, every function runs
as written. XLA computes `sin` with its own library, so the last bits of a
result can differ from CPython's `math`; the printed digits are rounded so
the three paths compare equal where they should. `xla.devices()` is what the
bridge sees, and the line that prints it starts with `# `, the mark for
output that may differ between machines. The bridge needs JAX installed to
place arrays on a device; the compiler that wrote the StableHLO does not.

## Run it

```bash
python  device_math.ppy
ppy run device_math.ppy
ppy emit stablehlo device_math.ppy
```

<!-- outputs:start -->
## What it prints

**`python  device_math.ppy`**

```text
0.445520207 0.387753845 -4.485479898
2.0 1.5
# devices: ['cpu:0']
```

**`ppy run device_math.ppy`**

```text
0.445520207 0.387753845 -4.485479898
2.0 1.5
# devices: ['cpu:0']
```

**`ppy emit stablehlo device_math.ppy`**

```text
module @device_math {
  func.func public @device_math_f(%x: tensor<f64>, %y: tensor<f64>) -> (tensor<f64>) {
    %1 = stablehlo.sine %x : tensor<f64>
    %2 = stablehlo.multiply %1, %y : tensor<f64>
    %3 = stablehlo.constant dense<0.00000000000000000e+00> : tensor<f64>
    %4 = stablehlo.compare GT, %2, %3, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %5 = stablehlo.negate %x : tensor<f64>
    %6 = stablehlo.select %4, %x, %5 : tensor<i1>, tensor<f64>
    %7 = stablehlo.add %2, %6 : tensor<f64>
    %8 = stablehlo.constant dense<2.00000000000000000e+00> : tensor<f64>
    %9 = stablehlo.divide %7, %8 : tensor<f64>
    return %9 : tensor<f64>
  }
}
```

<!-- outputs:end -->

Read on: [XLA](../../docs/guide/xla.md) ·
[JAX export](../25_jax_export/README.md)

`device_math.ppy` is hand-written; there is no `.py` source and no conversion
step.
