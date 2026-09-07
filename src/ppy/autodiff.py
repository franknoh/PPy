"""`ppy.grad` and `ppy.value_and_grad`: the derivative of a function.

```python
import math
import ppy

def f(x: float, y: float) -> float:
    return math.sin(x) * y + x * x

df = ppy.grad(f)              # d f / d x
dy = ppy.grad(f, argnums=1)   # d f / d y
both = ppy.value_and_grad(f)  # (f(x, y), d f / d x)
```

Under CPython the derivative is made from the function's source: the body
-- assignments and a return, over arithmetic, `math`, and NumPy arrays --
is restated one operation at a time, then each operation's adjoint is
emitted in reverse, by the same rules in the same order the compiler's
`autodiff` transform uses, so the Python path and the native path agree
bit for bit. What the rules do not cover -- a branch, a loop, a call into
code that is not one of the known operations -- is refused with the
reason, as the compiler refuses it.
"""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
from collections.abc import Callable
from typing import Any

__all__ = ["grad", "value_and_grad"]

_LN2 = math.log(2.0)
_LN10 = math.log(10.0)
_TWO_OVER_ROOT_PI = 2.0 / math.sqrt(math.pi)

#: One-argument functions with a derivative rule, by their `math` name.
_UNARY = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "exp",
        "exp2",
        "log",
        "log2",
        "log10",
        "sqrt",
        "abs",
        "floor",
        "ceil",
        "trunc",
        "erf",
        "erfc",
        "neg",
    }
)
#: How `math` and NumPy spell them.
_MATH_NAMES = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "exp": "exp",
    "exp2": "exp2",
    "log": "log",
    "log2": "log2",
    "log10": "log10",
    "sqrt": "sqrt",
    "fabs": "abs",
    "floor": "floor",
    "ceil": "ceil",
    "trunc": "trunc",
    "erf": "erf",
    "erfc": "erfc",
}
_NUMPY_NAMES = {
    **_MATH_NAMES,
    "abs": "abs",
    "absolute": "abs",
    "negative": "neg",
}
_NUMPY_BINARY = {
    "add": "add",
    "subtract": "sub",
    "multiply": "mul",
    "divide": "div",
    "true_divide": "div",
    "power": "pow",
    "matmul": "matmul",
}
_OPERATORS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.Pow: "pow"}
_OPERATORS[ast.MatMult] = "matmul"


class _Unsupported(TypeError):
    """A function `ppy.grad` cannot differentiate, with the reason."""


def grad(function: Callable[..., Any], argnums: int | tuple[int, ...] = 0) -> Callable[..., Any]:
    """The gradient of `function` with respect to its `argnums` parameter(s)."""
    return _Derivative(function, _argnums(argnums), value=False)


def value_and_grad(
    function: Callable[..., Any], argnums: int | tuple[int, ...] = 0
) -> Callable[..., Any]:
    """`function`'s value and its gradient, as a pair."""
    return _Derivative(function, _argnums(argnums), value=True)


def _argnums(argnums: int | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(argnums, bool) or not isinstance(argnums, (int, tuple)):
        raise TypeError("`argnums` is a parameter position or a tuple of them")
    chosen = (argnums,) if isinstance(argnums, int) else tuple(argnums)
    if not chosen or any(isinstance(a, bool) or not isinstance(a, int) or a < 0 for a in chosen):
        raise TypeError("`argnums` is a parameter position or a tuple of them")
    return chosen


class _Derivative:
    """A callable made from `function`'s source the first time it is called."""

    def __init__(self, function: Callable[..., Any], argnums: tuple[int, ...], value: bool):
        self.function = function
        self.argnums = argnums
        self.value = value
        self._compiled: Callable[..., Any] | None = None
        self.__name__ = (
            f"{'value_and_grad' if value else 'grad'}({getattr(function, '__name__', 'f')})"
        )
        self.__qualname__ = self.__name__
        self.__wrapped__ = function

    def __call__(self, *args: Any) -> Any:
        if self._compiled is None:
            self._compiled = _build(self.function, self.argnums, self.value)
        return self._compiled(*args)

    def __repr__(self) -> str:
        return f"<ppy.{self.__name__}>"


# -- source transformation ---------------------------------------------------------------------


def _build(
    function: Callable[..., Any], argnums: tuple[int, ...], value: bool
) -> Callable[..., Any]:
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as error:
        raise _Unsupported(f"ppy.grad: the source of {function!r} is not available") from error
    tree = ast.parse(source)
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not definitions:
        raise _Unsupported("ppy.grad: only a `def` is differentiated")
    node = definitions[0]
    parameters = [a.arg for a in node.args.args]
    if node.args.vararg or node.args.kwarg or node.args.kwonlyargs or node.args.posonlyargs:
        raise _Unsupported("ppy.grad: the function takes positional parameters only")
    for index in argnums:
        if index >= len(parameters):
            raise _Unsupported(f"ppy.grad: `argnums` {index} names no parameter of {node.name}")
    namespace = dict(getattr(function, "__globals__", {}))
    emitter = _Emitter(parameters, namespace)
    emitter.forward(node.body)
    emitter.backward(argnums, value)
    code = emitter.render(f"{node.name}__grad", parameters)
    namespace.update(_HELPERS)
    exec(compile(code, f"<ppy.grad {node.name}>", "exec"), namespace)
    return namespace[f"{node.name}__grad"]


class _Emitter:
    """Restates a straight-line body as one operation per line, then its adjoints."""

    def __init__(self, parameters: list[str], namespace: dict[str, Any]) -> None:
        self.parameters = parameters
        self.namespace = namespace
        #: Each variable's current temp.
        self.current: dict[str, str] = {p: p for p in parameters}
        #: (result temp, operation, operands) in forward order.
        self.operations: list[tuple[str, str, tuple[str, ...]]] = []
        self.lines: list[str] = []
        self.counter = 0
        self.result: str | None = None

    # -- forward -------------------------------------------------------------------------

    def fresh(self) -> str:
        self.counter += 1
        return f"_t{self.counter}"

    def emit(self, operation: str, *operands: str, spelled: str) -> str:
        name = self.fresh()
        self.lines.append(f"    {name} = {spelled}")
        self.operations.append((name, operation, operands))
        return name

    def forward(self, body: list[ast.stmt]) -> None:
        for index, statement in enumerate(body):
            last = index == len(body) - 1
            if isinstance(statement, ast.Return) and last and statement.value is not None:
                self.result = self.expression(statement.value)
                return
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                continue  # a docstring
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                value = statement.value
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                target, value = statement.target, statement.value
            elif isinstance(statement, ast.AugAssign):
                target = statement.target
                value = ast.BinOp(
                    left=ast.Name(id=target.id, ctx=ast.Load()),
                    op=statement.op,
                    right=statement.value,
                )  # type: ignore[union-attr]
            else:
                raise _Unsupported(
                    f"ppy.grad: `{type(statement).__name__}` is not differentiated; a body is "
                    "assignments and a return"
                )
            if not isinstance(target, ast.Name):
                raise _Unsupported("ppy.grad: only a plain variable is assigned")
            self.current[target.id] = self.expression(value)
        raise _Unsupported("ppy.grad: the body ends in `return <expression>`")

    def expression(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            found = self.current.get(node.id)
            if found is not None:
                return found
            if node.id in self.namespace:
                return self.emit("const", spelled=node.id)
            raise _Unsupported(f"ppy.grad: `{node.id}` is not defined")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise _Unsupported(f"ppy.grad: {node.value!r} is not a number")
            return self.emit("const", spelled=repr(float(node.value)))
        if isinstance(node, ast.UnaryOp):
            operand = self.expression(node.operand)
            if isinstance(node.op, ast.USub):
                return self.emit("neg", operand, spelled=f"-{operand}")
            if isinstance(node.op, ast.UAdd):
                return operand
            raise _Unsupported("ppy.grad: only `-x` is differentiated among unary operators")
        if isinstance(node, ast.BinOp):
            operation = _OPERATORS.get(type(node.op))
            if operation is None:
                raise _Unsupported(f"ppy.grad: `{type(node.op).__name__}` has no derivative rule")
            left = self.expression(node.left)
            right = self.expression(node.right)
            symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**", "matmul": "@"}
            return self.emit(operation, left, right, spelled=f"{left} {symbol[operation]} {right}")
        if isinstance(node, ast.Attribute) and node.attr == "T":
            operand = self.expression(node.value)
            return self.emit("transpose", operand, spelled=f"_ppy_transpose({operand})")
        if isinstance(node, ast.Call):
            return self.call(node)
        raise _Unsupported(f"ppy.grad: `{type(node).__name__}` is not differentiated")

    def call(self, node: ast.Call) -> str:
        if node.keywords:
            raise _Unsupported("ppy.grad: keyword arguments are not differentiated")
        func = node.func
        if isinstance(func, ast.Attribute):
            owner = ast.unparse(func.value)
            module = self._module(func.value)
            name = func.attr
            if module == "math" and name in _MATH_NAMES and len(node.args) == 1:
                return self.unary(_MATH_NAMES[name], node.args[0], f"math.{name}")
            if module == "math" and name == "pow" and len(node.args) == 2:
                left, right = (self.expression(a) for a in node.args)
                return self.emit("pow", left, right, spelled=f"{left} ** {right}")
            if module == "numpy":
                if name in _NUMPY_NAMES and len(node.args) == 1:
                    return self.unary(_NUMPY_NAMES[name], node.args[0], f"{owner}.{name}")
                if name in _NUMPY_BINARY and len(node.args) == 2:
                    left, right = (self.expression(a) for a in node.args)
                    symbol = {
                        "add": "+",
                        "sub": "-",
                        "mul": "*",
                        "div": "/",
                        "pow": "**",
                        "matmul": "@",
                    }
                    operation = _NUMPY_BINARY[name]
                    return self.emit(
                        operation, left, right, spelled=f"{left} {symbol[operation]} {right}"
                    )
                if name in {"sum", "mean"} and len(node.args) == 1:
                    operand = self.expression(node.args[0])
                    return self.emit(name, operand, spelled=f"{owner}.{name}({operand})")
                if name == "transpose" and len(node.args) == 1:
                    operand = self.expression(node.args[0])
                    return self.emit("transpose", operand, spelled=f"_ppy_transpose({operand})")
                if name == "reshape" and len(node.args) == 2:
                    operand = self.expression(node.args[0])
                    shape = ast.unparse(node.args[1])
                    return self.emit(
                        "reshape", operand, spelled=f"{owner}.reshape({operand}, {shape})"
                    )
                if name == "broadcast_to" and len(node.args) == 2:
                    operand = self.expression(node.args[0])
                    shape = ast.unparse(node.args[1])
                    return self.emit(
                        "broadcast", operand, spelled=f"{owner}.broadcast_to({operand}, {shape})"
                    )
            if module is None and func.attr in {"sum", "mean"} and not node.args:
                operand = self.expression(func.value)
                return self.emit(func.attr, operand, spelled=f"{operand}.{func.attr}()")
            if module is None and func.attr == "reshape" and node.args:
                operand = self.expression(func.value)
                shape = ", ".join(ast.unparse(a) for a in node.args)
                return self.emit("reshape", operand, spelled=f"{operand}.reshape({shape})")
            raise _Unsupported(f"ppy.grad: `{ast.unparse(func)}` has no derivative rule")
        if isinstance(func, ast.Name):
            if func.id == "abs" and len(node.args) == 1:
                return self.unary("abs", node.args[0], "abs")
            if func.id == "float" and len(node.args) == 1:
                return self.expression(node.args[0])
            target = self.namespace.get(func.id)
            module = getattr(target, "__module__", None)
            if module == "math" and func.id in _MATH_NAMES and len(node.args) == 1:
                return self.unary(_MATH_NAMES[func.id], node.args[0], func.id)
            raise _Unsupported(f"ppy.grad: a call to `{func.id}` is not differentiated")
        raise _Unsupported("ppy.grad: this call is not differentiated")

    def unary(self, operation: str, argument: ast.expr, spelled_function: str) -> str:
        operand = self.expression(argument)
        return self.emit(operation, operand, spelled=f"{spelled_function}({operand})")

    def _module(self, node: ast.expr) -> str | None:
        """`math` or `numpy` when `node` names one of them in the function's globals."""
        if not isinstance(node, ast.Name):
            return None
        target = self.namespace.get(node.id)
        name = getattr(target, "__name__", None) if inspect.ismodule(target) else None
        return name if name in {"math", "numpy"} else None

    # -- backward ------------------------------------------------------------------------

    def backward(self, argnums: tuple[int, ...], value: bool) -> None:
        assert self.result is not None
        adjoint: dict[str, str] = {}
        self.lines.append("    # the adjoints, in reverse")
        seed = self.fresh()
        self.lines.append(f"    {seed} = _ppy_one_like({self.result})")
        adjoint[self.result] = seed

        def add(operand: str, contribution: str) -> None:
            if operand.startswith("_t") and self._is_constant(operand):
                return
            existing = adjoint.get(operand)
            name = self.fresh()
            if existing is None:
                self.lines.append(f"    {name} = {contribution}")
            else:
                self.lines.append(f"    {name} = _ppy_add({existing}, {contribution})")
            adjoint[operand] = name

        for result, operation, operands in reversed(self.operations):
            g = adjoint.get(result)
            if g is None:
                continue
            self.rule(operation, result, operands, g, add)
        outputs = []
        if value:
            outputs.append(self.result)
        for index in argnums:
            parameter = self.parameters[index]
            outputs.append(adjoint.get(parameter, f"_ppy_zero_like({parameter})"))
        self.lines.append(
            "    return " + (outputs[0] if len(outputs) == 1 else "(" + ", ".join(outputs) + ")")
        )

    def _is_constant(self, temp: str) -> bool:
        return any(name == temp and op == "const" for name, op, _ in self.operations)

    def rule(self, operation: str, r: str, operands: tuple[str, ...], g: str, add) -> None:  # type: ignore[no-untyped-def]
        if operation == "const":
            return
        if operation == "add":
            a, c = operands
            add(a, f"_ppy_unbroadcast({g}, {a})")
            add(c, f"_ppy_unbroadcast({g}, {c})")
        elif operation == "sub":
            a, c = operands
            add(a, f"_ppy_unbroadcast({g}, {a})")
            add(c, f"_ppy_unbroadcast(-{g}, {c})")
        elif operation == "mul":
            a, c = operands
            add(a, f"_ppy_unbroadcast({g} * {c}, {a})")
            add(c, f"_ppy_unbroadcast({g} * {a}, {c})")
        elif operation == "div":
            a, c = operands
            add(a, f"_ppy_unbroadcast({g} / {c}, {a})")
            add(c, f"_ppy_unbroadcast(-({g} * ({r} / {c})), {c})")
        elif operation == "pow":
            a, c = operands
            add(a, f"_ppy_unbroadcast({g} * ({c} * {a} ** ({c} - 1.0)), {a})")
            add(c, f"_ppy_unbroadcast({g} * ({r} * _ppy_log({a})), {c})")
        elif operation == "neg":
            add(operands[0], f"-{g}")
        elif operation == "matmul":
            a, c = operands
            add(a, f"{g} @ _ppy_transpose({c})")
            add(c, f"_ppy_transpose({a}) @ {g}")
        elif operation == "sum":
            add(operands[0], f"_ppy_broadcast({g}, {operands[0]})")
        elif operation == "mean":
            add(operands[0], f"_ppy_broadcast({g} / _ppy_size({operands[0]}), {operands[0]})")
        elif operation == "transpose":
            add(operands[0], f"_ppy_transpose({g})")
        elif operation == "reshape":
            add(operands[0], f"_ppy_reshape_like({g}, {operands[0]})")
        elif operation == "broadcast":
            add(operands[0], f"_ppy_unbroadcast({g}, {operands[0]})")
        elif operation in _UNARY:
            self.unary_rule(operation, r, operands[0], g, add)
        else:
            raise _Unsupported(f"ppy.grad: `{operation}` has no derivative rule")

    def unary_rule(self, operation: str, r: str, x: str, g: str, add) -> None:  # type: ignore[no-untyped-def]
        if operation == "sin":
            add(x, f"{g} * _ppy_cos({x})")
        elif operation == "cos":
            add(x, f"-({g} * _ppy_sin({x}))")
        elif operation == "tan":
            add(x, f"{g} * (1.0 + {r} * {r})")
        elif operation == "exp":
            add(x, f"{g} * {r}")
        elif operation == "exp2":
            add(x, f"{g} * ({r} * {_LN2!r})")
        elif operation == "log":
            add(x, f"{g} / {x}")
        elif operation == "log2":
            add(x, f"{g} / ({x} * {_LN2!r})")
        elif operation == "log10":
            add(x, f"{g} / ({x} * {_LN10!r})")
        elif operation == "sqrt":
            add(x, f"{g} / (2.0 * {r})")
        elif operation == "abs":
            add(x, f"_ppy_abs_adjoint({g}, {x})")
        elif operation == "erf":
            add(x, f"{g} * ({_TWO_OVER_ROOT_PI!r} * _ppy_exp(-({x} * {x})))")
        elif operation == "erfc":
            add(x, f"{g} * ({-_TWO_OVER_ROOT_PI!r} * _ppy_exp(-({x} * {x})))")
        elif operation in {"floor", "ceil", "trunc"}:
            return
        else:
            raise _Unsupported(f"ppy.grad: `{operation}` has no derivative rule")

    def render(self, name: str, parameters: list[str]) -> str:
        header = f"def {name}({', '.join(parameters)}):"
        return "\n".join([header, *self.lines]) + "\n"


# -- helpers the generated code calls ----------------------------------------------------------


def _numpy():  # type: ignore[no-untyped-def]
    import numpy

    return numpy


def _is_array(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _ppy_one_like(value: Any) -> Any:
    return _numpy().ones_like(value) if _is_array(value) else 1.0


def _ppy_zero_like(value: Any) -> Any:
    return _numpy().zeros_like(value) if _is_array(value) else 0.0


def _ppy_add(a: Any, c: Any) -> Any:
    return a + c


def _ppy_size(value: Any) -> Any:
    return float(value.size) if _is_array(value) else 1.0


def _ppy_log(value: Any) -> Any:
    return _numpy().log(value) if _is_array(value) else math.log(value)


def _ppy_exp(value: Any) -> Any:
    return _numpy().exp(value) if _is_array(value) else math.exp(value)


def _ppy_sin(value: Any) -> Any:
    return _numpy().sin(value) if _is_array(value) else math.sin(value)


def _ppy_cos(value: Any) -> Any:
    return _numpy().cos(value) if _is_array(value) else math.cos(value)


def _ppy_abs_adjoint(g: Any, x: Any) -> Any:
    """`g` where `x` is positive, `-g` where negative, and nothing at zero."""
    if _is_array(x):
        numpy = _numpy()
        return numpy.where(x > 0, g, numpy.where(x < 0, -g, 0.0))
    if x > 0:
        return g
    return -g if x < 0 else 0.0


def _ppy_transpose(value: Any) -> Any:
    return value.T if _is_array(value) else value


def _ppy_reshape_like(g: Any, like: Any) -> Any:
    return _numpy().reshape(g, like.shape) if _is_array(like) else g


def _ppy_broadcast(g: Any, like: Any) -> Any:
    if _is_array(like):
        return _numpy().broadcast_to(g, like.shape)
    return g


def _ppy_unbroadcast(g: Any, like: Any) -> Any:
    """`g` summed over the axes broadcasting grew, back to `like`'s shape."""
    if not _is_array(g):
        return g
    if not _is_array(like):
        return g.sum()
    numpy = _numpy()
    if g.shape == like.shape:
        return g
    leading = g.ndim - like.ndim
    if leading > 0:
        g = g.sum(axis=tuple(range(leading)))
    grown = tuple(axis for axis, dim in enumerate(like.shape) if dim == 1 and g.shape[axis] != 1)
    if grown:
        g = g.sum(axis=grown, keepdims=True)
    return numpy.asarray(g)


_HELPERS = {name: value for name, value in globals().items() if name.startswith("_ppy_")}
