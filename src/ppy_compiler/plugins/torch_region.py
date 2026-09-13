"""Compiling a PPY function into one ATen C++ region (spec 20.3, 20.5).

A function whose body is entirely curated tensor operations becomes a single
C++ function calling the ATen API of the installed PyTorch build. Each `at::`
call still goes through the dispatcher, so autograd, device selection, and
backend keys behave exactly as they do from Python; what disappears is one
Python round trip per operation.

An operation is described by the C++ signature it is called through, not by
an arity alone: a dimension is an `int64_t`, a shape is an `IntArrayRef`, and
`is_causal` is a `bool`. Rendering is therefore directed by the slot an
argument fills, which is what lets a whole transformer block -- reshapes,
transposes, `layer_norm`, attention -- be one region instead of a dozen.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ..analysis import types as T
from ..analysis.checker import FunctionAnalysis, ModuleAnalysis
from ..analysis.symbols import FunctionInfo, ModuleSymbols

__all__ = ["ATEN_CALLS", "AtenCall", "TorchRegion", "Unsupported", "emit_source", "find_regions"]

#: The C++ slot an argument fills. `OPERAND` is the permissive one -- a tensor
#: expression, or a number where the C++ API takes an `at::Scalar`.
OPERAND = "operand"
#: `int64_t`: one dimension, or any other single integer.
DIM = "dim"
#: `at::IntArrayRef`: a shape or a list of dimensions, written as a tuple.
DIMS = "dims"
#: `at::TensorList`: the tuple `cat` and `stack` take.
TENSORS = "tensors"
#: `double`: an epsilon or a probability, never a tensor.
SCALAR = "scalar"
#: `bool`: `keepdim`, `is_causal`.
FLAG = "flag"
#: `c10::string_view`: the small closed set of mode strings ATen accepts.
TEXT = "text"


@dataclass(frozen=True, slots=True)
class AtenCall:
    """One C++ signature a curated operation can be called through."""

    cpp: str
    #: Every C++ parameter: the keyword PPY accepts for it, and its slot.
    params: tuple[tuple[str, str], ...]
    #: How many leading parameters a call must supply.
    required: int
    #: C++ text for each optional parameter, used when a later one is given.
    defaults: tuple[str, ...] = ()
    #: Whether `at::Tensor` carries the same operation as a method, with the
    #: first parameter as its receiver.
    method: bool = True
    #: Whether the free function exists in `at::`; `view` is a method only.
    free: bool = True

    def optional(self) -> int:
        return len(self.params) - self.required


def _unary(cpp: str) -> tuple[AtenCall, ...]:
    return (AtenCall(cpp, (("self", OPERAND),), 1),)


def _binary(cpp: str) -> tuple[AtenCall, ...]:
    return (AtenCall(cpp, (("self", OPERAND), ("other", OPERAND)), 2),)


def _reduction(cpp: str) -> tuple[AtenCall, ...]:
    """`sum`/`mean`: the whole tensor, or named dimensions with `keepdim`."""
    return (
        AtenCall(cpp, (("self", OPERAND),), 1),
        AtenCall(
            cpp,
            (("self", OPERAND), ("dim", DIMS), ("keepdim", FLAG)),
            2,
            defaults=("false",),
        ),
    )


#: Curated operations and the C++ signatures each may be called through. An
#: operation with several entries picks the first whose parameter count the
#: call fills (spec 18.3).
ATEN_CALLS: dict[str, tuple[AtenCall, ...]] = {
    # Elementwise arithmetic, the operators included.
    "add": _binary("at::add"),
    "sub": _binary("at::sub"),
    "mul": _binary("at::mul"),
    "div": _binary("at::div"),
    "pow": _binary("at::pow"),
    "maximum": _binary("at::maximum"),
    "minimum": _binary("at::minimum"),
    "remainder": _binary("at::remainder"),
    # Elementwise unary.
    "relu": _unary("at::relu"),
    "sigmoid": _unary("at::sigmoid"),
    "tanh": _unary("at::tanh"),
    "exp": _unary("at::exp"),
    "log": _unary("at::log"),
    "sqrt": _unary("at::sqrt"),
    "rsqrt": _unary("at::rsqrt"),
    "abs": _unary("at::abs"),
    "neg": _unary("at::neg"),
    "erf": _unary("at::erf"),
    "sin": _unary("at::sin"),
    "cos": _unary("at::cos"),
    "silu": _unary("at::silu"),
    # `gelu`'s second argument is the approximation, which GPT-2 needs as
    # "tanh"; ATen spells the exact form "none".
    "gelu": (
        AtenCall("at::gelu", (("self", OPERAND), ("approximate", TEXT)), 1, defaults=('"none"',)),
    ),
    # Matrix products.
    "matmul": _binary("at::matmul"),
    "mm": _binary("at::mm"),
    "bmm": _binary("at::bmm"),
    "linear": (
        AtenCall(
            "at::linear",
            (("input", OPERAND), ("weight", OPERAND), ("bias", OPERAND)),
            2,
            defaults=("{}",),
            method=False,
        ),
    ),
    # Reductions and normalization.
    "sum": _reduction("at::sum"),
    "mean": _reduction("at::mean"),
    "softmax": (AtenCall("at::softmax", (("self", OPERAND), ("dim", DIM)), 2),),
    "log_softmax": (AtenCall("at::log_softmax", (("self", OPERAND), ("dim", DIM)), 2),),
    "layer_norm": (
        AtenCall(
            "at::layer_norm",
            (
                ("input", OPERAND),
                ("normalized_shape", DIMS),
                ("weight", OPERAND),
                ("bias", OPERAND),
                ("eps", SCALAR),
            ),
            2,
            defaults=("{}", "{}", "1e-5"),
            method=False,
        ),
    ),
    # Attention: the fused kernel PyTorch itself dispatches to.
    "scaled_dot_product_attention": (
        AtenCall(
            "at::scaled_dot_product_attention",
            (
                ("query", OPERAND),
                ("key", OPERAND),
                ("value", OPERAND),
                ("attn_mask", OPERAND),
                ("dropout_p", SCALAR),
                ("is_causal", FLAG),
            ),
            3,
            defaults=("{}", "0.0", "false"),
            method=False,
        ),
    ),
    # Shape.
    "reshape": (AtenCall("at::reshape", (("self", OPERAND), ("shape", DIMS)), 2),),
    "view": (AtenCall("view", (("self", OPERAND), ("size", DIMS)), 2, free=False),),
    "transpose": (AtenCall("at::transpose", (("self", OPERAND), ("dim0", DIM), ("dim1", DIM)), 3),),
    "permute": (AtenCall("at::permute", (("self", OPERAND), ("dims", DIMS)), 2),),
    "unsqueeze": (AtenCall("at::unsqueeze", (("self", OPERAND), ("dim", DIM)), 2),),
    "squeeze": (
        AtenCall("at::squeeze", (("self", OPERAND),), 1),
        AtenCall("at::squeeze", (("self", OPERAND), ("dim", DIM)), 2),
    ),
    "flatten": (
        AtenCall(
            "at::flatten",
            (("self", OPERAND), ("start_dim", DIM), ("end_dim", DIM)),
            1,
            defaults=("0", "-1"),
        ),
    ),
    "t": _unary("at::t"),
    "contiguous": (AtenCall("contiguous", (("self", OPERAND),), 1, free=False),),
    "clone": _unary("at::clone"),
    "detach": _unary("at::detach"),
    # Joins take a tuple of tensors, not a tensor.
    "cat": (AtenCall("at::cat", (("tensors", TENSORS), ("dim", DIM)), 1, defaults=("0",)),),
    "stack": (AtenCall("at::stack", (("tensors", TENSORS), ("dim", DIM)), 1, defaults=("0",)),),
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

#: How a region parameter of each PPY type is declared in C++.
_DECLARATIONS = {
    "tensor": "const at::Tensor& {}",
    "int": "int64_t {}",
    "scalar": "double {}",
    "bool": "bool {}",
}


class Unsupported(Exception):
    """The function body has no ATen C++ translation."""


@dataclass(frozen=True, slots=True)
class _Context:
    """What rendering one region needs to know about its own scope."""

    analysis: ModuleAnalysis
    #: Every name in scope, and the kind it was declared or bound with.
    kinds: dict[str, str]
    operations: list[str] = field(default_factory=list)

    @property
    def names(self) -> set[str]:
        return set(self.kinds)


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
            _DECLARATIONS.get(kind, "double {}").format(name) for name, kind in self.parameters
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
        if base.name == "int":
            return "int"
        if base.name == "bool":
            return "bool"
        if base.name == "float":
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

    context = _Context(analysis, dict(parameters))
    bindings: list[tuple[str, str]] = []
    try:
        # Intermediates are how model code is actually written, and each one
        # is a straight `auto name = ...;` ahead of the return.
        for statement in body[:-1]:
            name, value = _binding(statement, context.names)
            bindings.append((name, _render(value, context)))
            context.kinds[name] = "tensor"
        rendered = _render(body[-1].value, context)
    except Unsupported as exc:
        return TorchRegion(info, _symbol(info), tuple(parameters), reason=str(exc))
    return TorchRegion(
        info,
        _symbol(info),
        tuple(parameters),
        bindings=tuple(bindings),
        body=rendered,
        operations=tuple(dict.fromkeys(context.operations)),
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


def _render(node: ast.expr, context: _Context) -> str:
    """An expression in an `OPERAND` slot: a tensor, or a number beside one."""
    if isinstance(node, ast.Name):
        if node.id not in context.kinds:
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
        context.operations.append(function)
        left = _render(node.left, context)
        right = _render(node.right, context)
        return f"{function}({left}, {right})"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        context.operations.append("at::neg")
        return f"at::neg({_render(node.operand, context)})"
    if isinstance(node, ast.Call):
        return _render_call(node, context)
    raise Unsupported(f"`{type(node).__name__}` has no ATen counterpart")


def _render_int(node: ast.expr, context: _Context) -> str:
    """A `int64_t` slot: an integer literal, or a region parameter typed `int`."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"-{_render_int(node.operand, context)}"
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return str(node.value)
    if isinstance(node, ast.Name) and context.kinds.get(node.id) == "int":
        return node.id
    raise Unsupported("a dimension is an integer literal or an `int` parameter of the region")


def _render_dims(node: ast.expr, context: _Context) -> str:
    """An `at::IntArrayRef` slot: a tuple of dimensions, or a single one."""
    if isinstance(node, (ast.Tuple, ast.List)):
        return "{" + ", ".join(_render_int(item, context) for item in node.elts) + "}"
    return "{" + _render_int(node, context) + "}"


def _render_tensors(node: ast.expr, context: _Context) -> str:
    """An `at::TensorList` slot: the tuple `cat` and `stack` take."""
    if not isinstance(node, (ast.Tuple, ast.List)):
        raise Unsupported("`cat` and `stack` take a tuple of tensors")
    return "{" + ", ".join(_render(item, context) for item in node.elts) + "}"


def _render_scalar(node: ast.expr, context: _Context) -> str:
    """A `double` slot: an epsilon or a probability, never a tensor."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"-{_render_scalar(node.operand, context)}"
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return repr(float(node.value))
    if isinstance(node, ast.Name) and context.kinds.get(node.id) in {"int", "scalar"}:
        return f"static_cast<double>({node.id})"
    raise Unsupported("this argument is a number, not a tensor")


def _render_flag(node: ast.expr, context: _Context) -> str:
    """A `bool` slot."""
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return "true" if node.value else "false"
    if isinstance(node, ast.Name) and context.kinds.get(node.id) == "bool":
        return node.id
    raise Unsupported("this argument is `True` or `False`, or a `bool` parameter")


def _render_text(node: ast.expr, context: _Context) -> str:
    """A `c10::string_view` slot: one of ATen's mode strings."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.isidentifier():
        return f'"{node.value}"'
    raise Unsupported("this argument is one of the mode strings ATen accepts")


_SLOTS = {
    OPERAND: _render,
    DIM: _render_int,
    DIMS: _render_dims,
    TENSORS: _render_tensors,
    SCALAR: _render_scalar,
    FLAG: _render_flag,
    TEXT: _render_text,
}


def _render_call(node: ast.Call, context: _Context) -> str:
    operation = _call_name(node.func)
    overloads = ATEN_CALLS.get(operation or "")
    if overloads is None:
        raise Unsupported(f"`{operation or ast.unparse(node.func)}` is not a curated operation")

    # A method call supplies the first parameter as its receiver, so it fills
    # one more slot than it writes arguments for.
    is_method = isinstance(node.func, ast.Attribute) and _is_method_call(node.func, context)
    supplied = len(node.args) + len(node.keywords) + (1 if is_method else 0)
    call = _select(operation or "", overloads, supplied, is_method)
    if not is_method and not call.free:
        raise Unsupported(f"`{operation}` exists on a tensor, not as a free `at::` function")

    offset = 1 if is_method else 0
    bound: list[ast.expr | None] = [None] * len(call.params)
    if isinstance(node.func, ast.Attribute) and is_method:
        bound[0] = node.func.value
    for index, argument in enumerate(node.args):
        bound[index + offset] = argument
    positions = {name: index for index, (name, _slot) in enumerate(call.params)}
    for keyword in node.keywords:
        position = positions.get(keyword.arg or "")
        if position is None:
            raise Unsupported(f"`{operation}` takes no `{keyword.arg}` argument in the C++ API")
        if bound[position] is not None:
            raise Unsupported(f"`{operation}` was given `{keyword.arg}` twice")
        bound[position] = keyword.value

    context.operations.append(call.cpp)
    arguments = _fill(operation or "", call, bound, context)
    if is_method:
        return f"({arguments[0]}).{operation}({', '.join(arguments[1:])})"
    return f"{call.cpp}({', '.join(arguments)})"


def _select(
    operation: str,
    overloads: tuple[AtenCall, ...],
    supplied: int,
    is_method: bool,
) -> AtenCall:
    """The first signature a call with this many arguments fills."""
    for call in overloads:
        if is_method and not call.method:
            continue
        if call.required <= supplied <= len(call.params):
            return call
    shapes = " or ".join(
        str(call.required) if not call.optional() else f"{call.required}-{len(call.params)}"
        for call in overloads
    )
    raise Unsupported(f"`{operation}` takes {shapes} argument(s) in the C++ API, not {supplied}")


def _fill(
    operation: str,
    call: AtenCall,
    bound: list[ast.expr | None],
    context: _Context,
) -> list[str]:
    """Render every argument through to the last one the call supplied."""
    last = max((i for i, value in enumerate(bound) if value is not None), default=-1)
    for index in range(call.required):
        if bound[index] is None:
            name = call.params[index][0]
            raise Unsupported(f"`{operation}` needs its `{name}` argument")
    rendered: list[str] = []
    for index in range(last + 1):
        value = bound[index]
        if value is None:
            rendered.append(call.defaults[index - call.required])
            continue
        slot = call.params[index][1]
        rendered.append(_SLOTS[slot](value, context))
    return rendered


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_method_call(func: ast.Attribute, context: _Context) -> bool:
    """`x.relu()` is a method on a tensor, not `torch.relu(x)`."""
    owner = func.value
    if isinstance(owner, ast.Name) and owner.id in context.kinds:
        return context.kinds[owner.id] == "tensor"
    resolved = T.strip_literal(context.analysis.type_of(owner))
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
