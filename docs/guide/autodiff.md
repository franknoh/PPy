# Derivatives: `ppy.grad`

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

`ppy.grad(f)` is the gradient of `f` with respect to its first parameter --
`argnums` names another, or a tuple of them, and then the gradient is a
tuple -- and `ppy.value_and_grad(f)` gives `f`'s value and the gradient as
a pair. `f` returns a `float` and the parameters differentiated are
`float`s; `f`'s body is assignments and a return over arithmetic, the
`math` functions, `abs`, and (under CPython, over NumPy arrays) `+ - * /
** @`, `sum`, `mean`, `.T`, `reshape`, `broadcast_to`. A branch, a loop,
a call into anything else, or an effect no derivative follows -- I/O, a
write, a thread -- is refused: `E1660` for a misuse of `ppy.grad` itself,
`E1661` for the types, `E1662` for an effect.

Under CPython the derivative is made from `f`'s source the first time it is
called: the body restated one operation at a time, then every operation's
adjoint in reverse. Natively the compiler differentiates `f`'s IR by the
same rules in the same order -- the `autodiff` transform, reverse mode
over the canonical and tensor dialects -- and `df(x, y)` in a native
function is a call to the derived function. Both follow one rule table,
with one order of accumulation, so the three paths agree bit for bit.

Examples: [Autodiff](../howto/36_autodiff.md).
