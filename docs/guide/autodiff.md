# Derivatives: `ppy.grad`

`ppy.grad` and `ppy.value_and_grad` differentiate a scalar function. This
page covers how to call them, what the function body may contain, and how the
derivative is built on each path.

## Example

```python
import math
import ppy


def f(x: float, y: float) -> float:
    return math.sin(x) * y + x * x


df = ppy.grad(f)
both = ppy.value_and_grad(f, argnums=1)


def slope(x: float, y: float) -> float:
    return df(x, y)
```

## Calling `grad` and `value_and_grad`

- `ppy.grad(f)` is the gradient of `f` with respect to its first parameter.
- `argnums` names another parameter, or a tuple of them. With a tuple, the
  gradient is a tuple.
- `ppy.value_and_grad(f)` gives `f`'s value and the gradient as a pair.

## Rules for the function

`f` returns a `float`, and the parameters differentiated are `float`s. `f`'s
body is assignments and a return over:

- arithmetic
- the `math` functions
- `abs`
- under CPython, over NumPy arrays: `+ - * / ** @`, `sum`, `mean`, `.T`,
  `reshape`, `broadcast_to`

A branch, a loop, a call into anything else, or an effect no derivative
follows (I/O, a write, a thread) is refused.

## Diagnostics

| code | meaning |
|---|---|
| `E1660` | a misuse of `ppy.grad` itself |
| `E1661` | the types |
| `E1662` | an effect |

## How the derivative is built

Under CPython the derivative is made from `f`'s source the first time it is
called: the body restated one operation at a time, then each operation's
adjoint in reverse.

Natively the compiler differentiates `f`'s IR by the same rules in the same
order. This is the `autodiff` transform, reverse mode over the canonical and
tensor dialects. `df(x, y)` in a native function is a call to the derived
function.

Both follow one rule table, with one order of accumulation, so the three
paths agree bit for bit.

Examples: [Autodiff](../howto/36_autodiff.md).
