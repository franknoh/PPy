"""Compiling a PPY function into one ATen C++ region (spec 20.3, 20.5).

A function whose body is entirely curated tensor operations becomes a single
C++ function calling the ATen API of the installed PyTorch build. Each `at::`
call still goes through the dispatcher, so autograd, device selection, and
backend keys behave exactly as they do from Python; what disappears is one
Python round trip per operation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..analysis import types as T
from ..analysis.checker import FunctionAnalysis, ModuleAnalysis
from ..analysis.symbols import FunctionInfo, ModuleSymbols

__all__ = ["ATEN_CALLS", "TorchRegion", "Unsupported", "emit_source", "find_regions"]

#: Curated operations and the ATen C++ function each maps to. Only calls with
#: a documented C++ counterpart of the same arity appear here (spec 18.3).
ATEN_CALLS: dict[str, tuple[str, int]] = {
    "add": ("at::add", 2),
    "sub": ("at::sub", 2),
    "mul": ("at::mul", 2),
    "div": ("at::div", 2),
    "pow": ("at::pow", 2),
    "matmul": ("at::matmul", 2),
    "mm": ("at::mm", 2),
    "bmm": ("at::bmm", 2),
    "maximum": ("at::maximum", 2),
    "minimum": ("at::minimum", 2),
    "relu": ("at::relu", 1),
    "sigmoid": ("at::sigmoid", 1),
    "tanh": ("at::tanh", 1),
    "exp": ("at::exp", 1),
    "log": ("at::log", 1),
    "sqrt": ("at::sqrt", 1),
    "rsqrt": ("at::rsqrt", 1),
    "abs": ("at::abs", 1),
    "neg": ("at::neg", 1),
    "erf": ("at::erf", 1),
    "sin": ("at::sin", 1),
    "cos": ("at::cos", 1),
    "silu": ("at::silu", 1),
    "gelu": ("at::gelu", 1),
    "sum": ("at::sum", 1),
    "mean": ("at::mean", 1),
    "t": ("at::t", 1),
    "contiguous": ("at::Tensor::contiguous", 1),
    "linear": ("at::linear", 3),
}

_BINARY_OPERATORS = {
    ast.Add: "at::add",
    ast.Sub: "at::sub",
    ast.Mult: "at::mul",
    ast.Div: "at::div",
    ast.MatMult: "at::matmul",
    ast.Pow: "at::pow",
}

_TENSOR = "torch.Tensor"


class Unsupported(Exception):
    """The function body has no ATen C++ translation."""


@dataclass(slots=True)
class TorchRegion:
    """One PPY function compiled to a C++ function over `at::Tensor`."""

    info: FunctionInfo
    symbol: str
    parameters: tuple[tuple[str, str], ...]
    #: `name = <expression>` bindings preceding the return, in source order.
    bindings: tuple[tuple[str, str], ...] = ()
    body: str = ""
    operations: tuple[str, ...] = ()
    reason: str = ""

    @property
    def module(self) -> str:
        return self.info.module

    @property
    def name(self) -> str:
        return self.info.name

    def declaration(self) -> str:
        rendered = ", ".join(
            f"const at::Tensor& {name}" if kind == "tensor" else f"double {name}"
            for name, kind in self.parameters
        )
        return f"at::Tensor {self.symbol}({rendered})"

    def source(self) -> str:
        lines = [f"    auto {name} = {value};" for name, value in self.bindings]
        lines.append(f"    return {self.body};")
        return self.declaration() + " {\n" + "\n".join(lines) + "\n}"


def _kind(t: T.Type) -> str:
    base = T.strip_literal(t)
    if isinstance(base, T.Instance):
        if base.name == _TENSOR:
            return "tensor"
        if base.name in {"int", "float", "bool"}:
            return "scalar"
    return "other"


def find_regions(
    symbols: ModuleSymbols,
    analysis: ModuleAnalysis,
) -> list[TorchRegion]:
    """Find functions that translate wholly into ATen C++ calls."""
    found: list[TorchRegion] = []
    for info in symbols.functions.values():
        function_analysis = analysis.functions.get(info.qualname)
        if function_analysis is None:
            continue
        region = _region_for(info, function_analysis, analysis)
        if region is not None:
            found.append(region)
    return found


def _region_for(
    info: FunctionInfo,
    function_analysis: FunctionAnalysis,
    analysis: ModuleAnalysis,
) -> TorchRegion | None:
    parameters: list[tuple[str, str]] = []
    for param in info.params:
        kind = _kind(param.type)
        if kind == "other" or param.kind in {"var_positional", "var_keyword"}:
            return None
        parameters.append((param.name, kind))
    if not parameters or not any(kind == "tensor" for _name, kind in parameters):
        return None
    if _kind(info.ret) != "tensor":
        return None

    body = [statement for statement in info.node.body if not _is_docstring(statement)]
    if not body or not isinstance(body[-1], ast.Return) or body[-1].value is None:
        return TorchRegion(
            info,
            _symbol(info),
            tuple(parameters),
            reason="a region ends in a `return` expression",
        )

    operations: list[str] = []
    names = {name for name, _kind in parameters}
    bindings: list[tuple[str, str]] = []
    try:
        # Intermediates are how model code is actually written, and each one
        # is a straight `auto name = ...;` ahead of the return.
        for statement in body[:-1]:
            name, value = _binding(statement, names)
            bindings.append((name, _render(value, analysis, names, operations)))
            names.add(name)
        rendered = _render(body[-1].value, analysis, names, operations)
    except Unsupported as exc:
        return TorchRegion(info, _symbol(info), tuple(parameters), reason=str(exc))
    return TorchRegion(
        info,
        _symbol(info),
        tuple(parameters),
        bindings=tuple(bindings),
        body=rendered,
        operations=tuple(dict.fromkeys(operations)),
    )


def _binding(statement: ast.stmt, names: set[str]) -> tuple[str, ast.expr]:
    """The name and value of a `name = <expression>` ahead of the return."""
    if isinstance(statement, ast.AnnAssign) and statement.value is not None:
        target: ast.expr = statement.target
        value = statement.value
    elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target, value = statement.targets[0], statement.value
    else:
        raise Unsupported("a region holds only assignments and a final `return`")
    if not isinstance(target, ast.Name):
        raise Unsupported("a region assigns only to plain local names")
    if target.id in names:
        raise Unsupported(f"`{target.id}` is assigned twice inside the region")
    return target.id, value


def _symbol(info: FunctionInfo) -> str:
    return "ppy_region_" + info.qualname.replace(".", "_")


def _is_docstring(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)


def _render(
    node: ast.expr,
    analysis: ModuleAnalysis,
    names: set[str],
    operations: list[str],
) -> str:
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise Unsupported(f"`{node.id}` is not a parameter of the region")
        return node.id
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise Unsupported("only numeric literals translate to a C++ scalar")
        return f"static_cast<double>({float(node.value)!r})"
    if isinstance(node, ast.BinOp):
        function = _BINARY_OPERATORS.get(type(node.op))
        if function is None:
            raise Unsupported("this operator has no ATen counterpart")
        operations.append(function)
        left = _render(node.left, analysis, names, operations)
        right = _render(node.right, analysis, names, operations)
        return f"{function}({left}, {right})"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operations.append("at::neg")
        return f"at::neg({_render(node.operand, analysis, names, operations)})"
    if isinstance(node, ast.Call):
        return _render_call(node, analysis, names, operations)
    raise Unsupported(f"`{type(node).__name__}` has no ATen counterpart")


def _render_call(
    node: ast.Call,
    analysis: ModuleAnalysis,
    names: set[str],
    operations: list[str],
) -> str:
    if node.keywords:
        raise Unsupported("keyword arguments are not translated")
    operation = _call_name(node.func)
    entry = ATEN_CALLS.get(operation or "")
    if entry is None:
        raise Unsupported(f"`{operation or ast.unparse(node.func)}` is not a curated operation")
    function, arity = entry
    if len(node.args) != arity:
        raise Unsupported(f"`{operation}` takes {arity} argument(s) in the C++ API")
    operations.append(function)

    if isinstance(node.func, ast.Attribute) and _is_method_call(node.func, analysis, names):
        receiver = _render(node.func.value, analysis, names, operations)
        rest = [_render(a, analysis, names, operations) for a in node.args]
        return f"({receiver}).{operation}({', '.join(rest)})"

    arguments = [_render(argument, analysis, names, operations) for argument in node.args]
    return f"{function}({', '.join(arguments)})"


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_method_call(func: ast.Attribute, analysis: ModuleAnalysis, names: set[str]) -> bool:
    """`x.relu()` is a method on a tensor, not `torch.relu(x)`."""
    owner = func.value
    if isinstance(owner, ast.Name) and owner.id in names:
        return True
    resolved = T.strip_literal(analysis.type_of(owner))
    return isinstance(resolved, T.Instance) and resolved.name == _TENSOR


_HEADER = """// generated by ppy; do not edit.
#include <torch/extension.h>
#include <ATen/ATen.h>
"""


def emit_source(regions: list[TorchRegion]) -> str:
    """The complete C++ translation unit for a module's regions."""
    parts = [_HEADER]
    parts.extend(region.source() for region in regions if region.body)
    return "\n\n".join(parts) + "\n"
