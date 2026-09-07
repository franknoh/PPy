"""What the numeric libraries share: the tensor operation a library call is.

`numpy.add`, `torch.add`, and `jax.numpy.add` are one operation, `tensor.add`;
`numpy.negative` and `torch.neg` are `tensor.unary {op = "neg"}`; `numpy.sum`
and `torch.sum` are `tensor.reduce {op = "add"}`. Each numeric plugin answers
`tensor_operation` from these tables, so the compiler reads one vocabulary --
the tensor dialect's -- whichever library a program spells it in, and one
fused kernel serves them all (spec 42).
"""

from __future__ import annotations

from .base import DialectOperationSpec

__all__ = ["BINARY", "REDUCTIONS", "UNARY", "converged", "special"]

#: Every library spelling of each unary operation, and the tensor dialect's name.
UNARY: dict[str, str] = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "exp": "exp",
    "exp2": "exp2",
    "log": "log",
    "log2": "log2",
    "log10": "log10",
    "sqrt": "sqrt",
    "floor": "floor",
    "ceil": "ceil",
    "trunc": "trunc",
    "abs": "abs",
    "absolute": "abs",
    "negative": "neg",
    "neg": "neg",
}
#: The binary operations, with NumPy's and torch's spellings.
BINARY: dict[str, str] = {
    "add": "add",
    "subtract": "sub",
    "sub": "sub",
    "multiply": "mul",
    "mul": "mul",
    "true_divide": "div",
    "divide": "div",
    "div": "div",
    "power": "pow",
    "pow": "pow",
    "minimum": "min",
    "maximum": "max",
}
#: Whole-array reductions; `mean` is a sum over the count.
REDUCTIONS: dict[str, str] = {
    "sum": "add",
    "prod": "mul",
    "product": "mul",
    "max": "max",
    "amax": "max",
    "min": "min",
    "amin": "min",
    "mean": "mean",
}
#: SciPy's names for the special functions the `special` dialect has.
SPECIAL: dict[str, str] = {
    "erf": "erf",
    "erfc": "erfc",
    "gamma": "gamma",
    "gammaln": "gammaln",
    "ndtr": "ndtr",
    "logit": "logit",
    "j0": "bessel_j0",
    "j1": "bessel_j1",
    "y0": "bessel_y0",
    "y1": "bessel_y1",
}


def converged(name: str) -> DialectOperationSpec | None:
    """The tensor operation a library function `name` is, or None."""
    if name in UNARY:
        return DialectOperationSpec("tensor", "unary", (("op", UNARY[name]),))
    if name in BINARY:
        return DialectOperationSpec("tensor", BINARY[name])
    if name in REDUCTIONS:
        return DialectOperationSpec("tensor", "reduce", (("op", REDUCTIONS[name]),))
    return None


def special(name: str) -> DialectOperationSpec | None:
    """The elementwise special function `name` as a tensor operation, or None."""
    if name in SPECIAL:
        return DialectOperationSpec("tensor", "unary", (("op", SPECIAL[name]),))
    return None
