"""Fused elementwise loops over library arrays, built as tensor IR (spec 19.4).

A maximal expression tree of array operations that converge onto the tensor
dialect -- NumPy's ufuncs, torch's elementwise operators, SciPy's special
functions -- becomes one kernel: tensor IR over buffers of `float64` whose
length the kernel learns at the call, lowered by `lower-tensor` to a single
loop with no temporaries, and emitted by the backend like any function. The
plugin says which tensor operation a call is; the compiler builds the
kernel. Nothing here writes LLVM IR.

The loop runs only behind the guards the plugin demands: an exact array of
the library's own class, `float64` in native byte order, C-contiguous, one
shape across the operands (spec 19.3, 19.5).
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field

from ...analysis import types as T
from ...analysis.checker import ModuleAnalysis
from ...analysis.symbols import FunctionInfo
from ...ir import IRModule
from ...ir import shape as shapes
from ...ir.dialects import core
from ...ir.dialects import tensor as tensors
from ...ir.model import Builder, Value
from ...ir.types import F64, I64, BufferType, PtrType

__all__ = [
    "BINARY",
    "REDUCTIONS",
    "STORAGES",
    "UNARY",
    "FusedLoop",
    "FusionCandidate",
    "find_candidates",
    "find_module_candidates",
    "kernel_module",
]

#: Elementwise operations with two operands: the tensor dialect's.
BINARY = frozenset(tensors.ELEMENTWISE)
#: Elementwise operations with one operand: the tensor dialect's.
UNARY = frozenset(tensors.UNARY)
#: Whole-array reductions, and `mean`, which is a sum over the count.
REDUCTIONS = frozenset({*tensors.REDUCTIONS, "mean"})

#: Reductions whose fused form reassociates the accumulation. NumPy sums
#: pairwise and torch vectorizes, so a sequential loop is a different --
#: and observably different -- order. These fuse only where the program
#: permits reassociation (spec 19.8).
REASSOCIATING = frozenset({"add", "mul", "mean"})

#: The array classes a kernel runs over, by the library that owns them.
STORAGES: dict[str, str] = {"numpy.ndarray": "numpy", "torch.Tensor": "torch"}

_OPERATOR_OP = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.Pow: "pow",
}


@dataclass(slots=True)
class FusedLoop:
    """One fused kernel: an expression over N arrays producing one array.

    `expression` is a small prefix program over the tensor vocabulary,
    `(add a0 (sin a1))`, with leaves `a<i>` (an array), `s<i>` (a scalar
    argument), and `c<value>` (a constant); unary and binary names never
    collide, so arity tells them apart. `storage` names the library whose
    arrays the kernel runs over.
    """

    symbol: str
    arrays: tuple[str, ...]
    scalars: tuple[str, ...]
    reduction: str = ""
    expression: str = ""
    parallel: bool = False
    storage: str = "numpy"

    @property
    def returns_scalar(self) -> bool:
        return bool(self.reduction)


@dataclass(slots=True)
class FusionCandidate:
    """An expression a plugin placed on the tensor dialect and PPY can fuse."""

    function: FunctionInfo | None
    node: ast.expr | None
    loop: FusedLoop
    operations: tuple[str, ...] = ()
    reason: str = ""


class _Shape:
    """Tracks the array operands and scalar operands of one expression."""

    def __init__(self) -> None:
        self.arrays: list[str] = []
        self.scalars: list[str] = []
        self.storage: str = ""

    def array(self, name: str, storage: str) -> int:
        if self.storage and storage != self.storage:
            # One kernel, one library: a NumPy array and a torch tensor in
            # one expression is the library's own business.
            raise _Unsupported
        self.storage = storage
        if name not in self.arrays:
            self.arrays.append(name)
        return self.arrays.index(name)

    def scalar(self, name: str) -> int:
        if name not in self.scalars:
            self.scalars.append(name)
        return self.scalars.index(name)


class _Unsupported(Exception):
    pass


def _storage_of(node: ast.expr, module: ModuleAnalysis) -> str | None:
    """The library whose array this expression is, or None for a non-array."""
    base = T.strip_literal(module.type_of(node))
    if isinstance(base, T.Instance):
        return STORAGES.get(base.name)
    return None


def _operation_of(node: ast.expr, module: ModuleAnalysis) -> tuple[str, dict[str, object]] | None:
    """The tensor operation the plugin says this call is: (`add`, {}), (`unary`,
    {op: sin}), (`reduce`, {op: add}); None when it is not one."""
    note = module.lowerings.get(id(node))
    if note is None or not note.operation.startswith("tensor."):
        return None
    return note.operation.removeprefix("tensor."), dict(note.attributes)


def find_candidates(function: FunctionInfo, module: ModuleAnalysis) -> list[FusionCandidate]:
    """Find maximal fusible expression trees inside one function."""
    return _search(
        function.node,
        module,
        function=function,
        prefix=function.qualname,
        fastmath=function.directive("fastmath") is not None,
        parallel=function.directive("parallel") is not None,
    )


def find_module_candidates(tree: ast.Module, module: ModuleAnalysis) -> list[FusionCandidate]:
    """Find fusible expressions in module-level code, outside any definition."""
    body = ast.Module(
        body=[
            statement
            for statement in tree.body
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ],
        type_ignores=[],
    )
    return _search(body, module, function=None, prefix=module.name, fastmath=False)


def _search(
    root: ast.AST,
    module: ModuleAnalysis,
    *,
    function: FunctionInfo | None,
    prefix: str,
    fastmath: bool,
    parallel: bool = False,
) -> list[FusionCandidate]:
    found: list[FusionCandidate] = []
    claimed: set[int] = set()

    for node in ast.walk(root):
        if not isinstance(node, ast.expr) or id(node) in claimed:
            continue
        operation = _operation_of(node, module)
        if operation is None:
            continue
        name, attributes = operation
        shape = _Shape()
        operations: list[str] = []
        reduction = ""
        try:
            if name == "reduce":
                # A reduction is only fusible as the root of the tree: its
                # operand collapses to one value, so nothing can wrap it.
                kind = str(attributes.get("op", ""))
                if not isinstance(node, ast.Call) or len(node.args) != 1 or node.keywords:
                    continue
                if kind not in REDUCTIONS or (kind in REASSOCIATING and not fastmath):
                    continue
                reduction = kind
                operations.append(kind)
                expression = _render(node.args[0], module, shape, operations)
            else:
                expression = _render(node, module, shape, operations)
        except _Unsupported:
            continue
        if not shape.arrays:
            continue
        loop = FusedLoop(
            symbol="ppy_fused_" + prefix.replace(".", "_") + f"_{node.lineno}_{node.col_offset}",
            arrays=tuple(shape.arrays),
            scalars=tuple(shape.scalars),
            reduction=reduction,
            expression=expression,
            parallel=parallel,
            storage=shape.storage,
        )
        found.append(
            FusionCandidate(
                function=function,
                node=node,
                loop=loop,
                operations=tuple(dict.fromkeys(operations)),
                reason=f"{len(operations)} operation(s) fused into one strided loop",
            )
        )
        for child in ast.walk(node):
            claimed.add(id(child))
    return found


def _render(node: ast.expr, module: ModuleAnalysis, shape: _Shape, operations: list[str]) -> str:
    """Render one expression node into the kernel's small prefix program."""
    if isinstance(node, ast.Name):
        storage = _storage_of(node, module)
        if storage is not None:
            return f"a{shape.array(node.id, storage)}"
        base = T.strip_literal(module.type_of(node))
        if base in (T.FLOAT, T.INT, T.BOOL):
            return f"s{shape.scalar(node.id)}"
        raise _Unsupported
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return f"c{float(node.value)!r}"
    if isinstance(node, ast.BinOp):
        name = _OPERATOR_OP.get(type(node.op))
        if name is None:
            raise _Unsupported
        operations.append(name)
        left = _render(node.left, module, shape, operations)
        right = _render(node.right, module, shape, operations)
        return f"({name} {left} {right})"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operations.append("neg")
        return f"(neg {_render(node.operand, module, shape, operations)})"
    if isinstance(node, ast.Call):
        operation = _operation_of(node, module)
        if operation is None or node.keywords:
            raise _Unsupported
        name, attributes = operation
        if name == "reduce":
            # A nested reduction changes the shape, so the tree stops here.
            raise _Unsupported
        if name == "unary" and len(node.args) == 1:
            kind = str(attributes.get("op", ""))
            if kind not in UNARY:
                raise _Unsupported
            operations.append(kind)
            return f"({kind} {_render(node.args[0], module, shape, operations)})"
        if name in BINARY and len(node.args) == 2:
            operations.append(name)
            left = _render(node.args[0], module, shape, operations)
            right = _render(node.args[1], module, shape, operations)
            return f"({name} {left} {right})"
    raise _Unsupported


# -- the kernels, as tensor IR ------------------------------------------------------------


def kernel_module(loops: Iterable[FusedLoop], name: str) -> IRModule:
    """The IR module holding one function per fused loop.

    A kernel takes each array as a pointer, each scalar as an `f64`, and
    the element count `n` last: `sym(out, a0.., s0.., n)` for a map and
    `sym(a0.., s0.., n) -> f64` for a reduction. Inside, every array is a
    `tensor<f64, N>` loaded from a buffer of `n` elements, so `N` is `n`
    and the whole expression is tensor arithmetic `lower-tensor` turns
    into one loop.
    """
    module = IRModule(name)
    module.require("core", 1)
    module.require("tensor", 1)
    for loop in loops:
        _build(module, loop)
    return module


def _build(module: IRModule, loop: FusedLoop) -> None:
    params: list[tuple[str, object]] = []
    if not loop.returns_scalar:
        params.append(("out", PtrType(F64)))
    params.extend((f"a{i}", PtrType(F64)) for i in range(len(loop.arrays)))
    params.extend((f"s{i}", F64) for i in range(len(loop.scalars)))
    params.append(("n", I64))
    function = module.add_function(
        loop.symbol,
        params,
        [F64] if loop.returns_scalar else [],  # type: ignore[arg-type]
    )
    entry = function.add_entry_block()
    b = Builder(entry)
    arguments = list(entry.arguments)
    n = arguments[-1]
    offset = 0 if loop.returns_scalar else 1
    array_pointers = arguments[offset : offset + len(loop.arrays)]
    scalar_values = arguments[offset + len(loop.arrays) : -1]
    element = tensors.tensor_type(F64, (shapes.Symbol("N"),))

    def as_buffer(pointer: Value) -> Value:
        return core.call_intrinsic(
            b, "ppy.buffer_from_parts", (pointer, n), (BufferType(F64),)
        ).results[0]

    # The destination first, so the lowering can write the result straight into it.
    destination = None if loop.returns_scalar else as_buffer(arguments[0])
    arrays = [tensors.load(b, as_buffer(p), element, f"a{i}") for i, p in enumerate(array_pointers)]

    def emit(node: _Node) -> Value:
        if node.op == "array":
            return arrays[node.array]
        if node.op == "scalar":
            return tensors.fill(b, scalar_values[node.scalar], element)
        if node.op == "constant":
            return tensors.fill(b, core.const(b, node.constant, F64), element)
        operands = [emit(child) for child in node.operands]
        if node.op in UNARY and len(operands) == 1:
            return tensors.unary(b, node.op, operands[0])
        if node.op in BINARY and len(operands) == 2:
            return tensors.elementwise(b, node.op, operands[0], operands[1])
        raise ValueError(f"fused expression uses {node.op!r}, which the tensor dialect lacks")

    value = emit(_parse(loop.expression))
    if destination is not None:
        tensors.store(b, value, destination)
        core.ret(b)
        return
    kind = "add" if loop.reduction == "mean" else loop.reduction
    reduced = tensors.reduce(b, value, (0,), kind, name="total")
    slot = core.alloca(b, F64, name="result")
    tensors.store(
        b,
        reduced,
        core.call_intrinsic(
            b, "ppy.buffer_from_parts", (slot, core.const(b, 1, I64)), (BufferType(F64),)
        ).results[0],
    )
    total = core.load(b, slot)
    if loop.reduction == "mean":
        total = core.div(b, total, core.cast(b, n, F64))
    core.ret(b, total)


@dataclass(slots=True)
class _Node:
    op: str
    operands: list[_Node] = field(default_factory=list)
    array: int = -1
    scalar: int = -1
    constant: float = 0.0


def _parse(text: str) -> _Node:
    """Parse the prefix program rendered by `_render`."""
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()
    position = 0

    def parse() -> _Node:
        nonlocal position
        token = tokens[position]
        position += 1
        if token == "(":
            operation = tokens[position]
            position += 1
            operands = []
            while tokens[position] != ")":
                operands.append(parse())
            position += 1
            return _Node(operation, operands)
        if token.startswith("a"):
            return _Node("array", array=int(token[1:]))
        if token.startswith("s"):
            return _Node("scalar", scalar=int(token[1:]))
        return _Node("constant", constant=float(token[1:]))

    return parse()
