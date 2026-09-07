"""Fused elementwise loops over library arrays, built as tensor or columnar IR (spec 19.4).

A maximal expression tree of array operations that converge onto a shared
dialect -- NumPy's ufuncs, torch's elementwise operators, and SciPy's
special functions onto `tensor`; PyArrow's compute onto `columnar` --
becomes one kernel: IR over buffers whose length the kernel learns at the
call, lowered by `lower-tensor` to a single loop with no temporaries, and
emitted by the backend like any function. The plugin says which operation
a call is; the compiler builds the kernel. Nothing here writes LLVM IR.

The loop runs only behind the guards the plugin demands: an exact array of
the library's own class, `float64` in native byte order, C-contiguous, one
shape across the operands (spec 19.3, 19.5); for Arrow, an array of
`float64` or `bool` whose bitmaps start on a byte.
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
from ...ir.dialects import columnar, core
from ...ir.dialects import tensor as tensors
from ...ir.model import Builder, Value
from ...ir.types import BOOL, F64, I64, U8, BufferType, IRType, PtrType

__all__ = [
    "BINARY",
    "COLUMNAR",
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

#: The columnar operations a kernel takes, with the element kinds they read
#: and give: `f64` values, `bool` masks, or the operand's own (`same`).
COLUMNAR: dict[str, tuple[tuple[str, ...], str]] = {
    **dict.fromkeys(columnar.ARITHMETIC, (("f64", "f64"), "f64")),
    **dict.fromkeys(columnar.COMPARISON, (("f64", "f64"), "bool")),
    **dict.fromkeys(columnar.BOOLEAN, (("bool", "bool"), "bool")),
    "negate": (("f64",), "f64"),
    "abs": (("f64",), "f64"),
    "invert": (("bool",), "bool"),
    "is_null": (("same",), "bool"),
    "is_valid": (("same",), "bool"),
    "fill_null": (("f64", "scalar"), "f64"),
    "select": (("bool", "f64", "f64"), "f64"),
}
#: Columnar operations whose result has no nulls.
_NEVER_NULL = frozenset({"is_null", "is_valid", "fill_null"})

#: Reductions whose fused form reassociates the accumulation. NumPy sums
#: pairwise and torch vectorizes, so a sequential loop is a different --
#: and observably different -- order. These fuse only where the program
#: permits reassociation (spec 19.8).
REASSOCIATING = frozenset({"add", "mul", "mean"})

#: The array classes a kernel runs over, by the library that owns them.
STORAGES: dict[str, str] = {
    "numpy.ndarray": "numpy",
    "torch.Tensor": "torch",
    "pyarrow.Array": "pyarrow",
}

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

    `expression` is a small prefix program over the dialect's vocabulary,
    `(add a0 (sin a1))`, with leaves `a<i>` (an array), `s<i>` (a scalar
    argument), and `c<value>` (a constant); unary and binary names never
    collide, so arity tells them apart. `storage` names the library whose
    arrays the kernel runs over; for Arrow, `kinds` says whether each array
    holds `f64` values or `bool` masks, `result` what the kernel gives, and
    `nullable` whether the result carries a validity bitmap.
    """

    symbol: str
    arrays: tuple[str, ...]
    scalars: tuple[str, ...]
    reduction: str = ""
    expression: str = ""
    parallel: bool = False
    storage: str = "numpy"
    kinds: tuple[str, ...] = ()
    result: str = "f64"
    nullable: bool = True

    @property
    def returns_scalar(self) -> bool:
        return bool(self.reduction)


@dataclass(slots=True)
class FusionCandidate:
    """An expression a plugin placed on a shared dialect and PPY can fuse."""

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
        #: What each array holds, once an operation has said.
        self.kinds: dict[str, str] = {}

    def array(self, name: str, storage: str, kind: str | None) -> int:
        if self.storage and storage != self.storage:
            # One kernel, one library: a NumPy array and a torch tensor in
            # one expression is the library's own business.
            raise _Unsupported
        self.storage = storage
        if name not in self.arrays:
            self.arrays.append(name)
        if kind is not None:
            known = self.kinds.setdefault(name, kind)
            if known != kind:
                raise _Unsupported
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


def _operation_of(
    node: ast.expr, module: ModuleAnalysis
) -> tuple[str, str, dict[str, object]] | None:
    """The shared operation the plugin says this call is: (`tensor`, `add`,
    {}), (`tensor`, `unary`, {op: sin}), (`columnar`, `select`, {}); None
    when it is not one a kernel takes."""
    note = module.lowerings.get(id(node))
    if note is None:
        return None
    dialect, _, name = note.operation.partition(".")
    if dialect not in {"tensor", "columnar"} or not name:
        return None
    return dialect, name, dict(note.attributes)


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
        dialect, name, attributes = operation
        shape = _Shape()
        operations: list[str] = []
        reduction = ""
        try:
            if dialect == "tensor" and name == "reduce":
                # A reduction is only fusible as the root of the tree: its
                # operand collapses to one value, so nothing can wrap it.
                kind = str(attributes.get("op", ""))
                if not isinstance(node, ast.Call) or len(node.args) != 1 or node.keywords:
                    continue
                if kind not in REDUCTIONS or (kind in REASSOCIATING and not fastmath):
                    continue
                reduction = kind
                operations.append(kind)
                expression, result = _render(node.args[0], module, shape, operations, "f64")
            else:
                expression, result = _render(node, module, shape, operations, None)
        except _Unsupported:
            continue
        if not shape.arrays:
            continue
        nullable = shape.storage == "pyarrow" and operations[0] not in _NEVER_NULL
        loop = FusedLoop(
            symbol="ppy_fused_" + prefix.replace(".", "_") + f"_{node.lineno}_{node.col_offset}",
            arrays=tuple(shape.arrays),
            scalars=tuple(shape.scalars),
            reduction=reduction,
            expression=expression,
            parallel=parallel,
            storage=shape.storage,
            kinds=tuple(shape.kinds.get(name, "f64") for name in shape.arrays),
            result=result,
            nullable=nullable,
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


def _render(
    node: ast.expr,
    module: ModuleAnalysis,
    shape: _Shape,
    operations: list[str],
    expect: str | None,
) -> tuple[str, str]:
    """Render one expression node into the kernel's small prefix program.

    Returns the text and what it computes: `f64` or `bool`. `expect` is
    what the enclosing operation reads, which decides what an array leaf
    holds; None leaves the leaf to whoever reads it next.
    """
    if isinstance(node, ast.Name):
        storage = _storage_of(node, module)
        if storage is not None:
            kind = None if expect in (None, "same") else expect
            index = shape.array(node.id, storage, kind)
            return f"a{index}", kind or shape.kinds.get(node.id, "f64")
        base = T.strip_literal(module.type_of(node))
        if base in (T.FLOAT, T.INT, T.BOOL) and expect != "bool":
            return f"s{shape.scalar(node.id)}", "f64"
        raise _Unsupported
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and expect != "bool"
    ):
        return f"c{float(node.value)!r}", "f64"
    if isinstance(node, ast.BinOp):
        name = _OPERATOR_OP.get(type(node.op))
        if name is None or expect == "bool":
            raise _Unsupported
        operations.append(name)
        left, _ = _render(node.left, module, shape, operations, "f64")
        right, _ = _render(node.right, module, shape, operations, "f64")
        return f"({name} {left} {right})", "f64"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and expect != "bool":
        operations.append("neg")
        inner, _ = _render(node.operand, module, shape, operations, "f64")
        return f"(neg {inner})", "f64"
    if isinstance(node, ast.Call):
        operation = _operation_of(node, module)
        if operation is None or node.keywords:
            raise _Unsupported
        dialect, name, attributes = operation
        if dialect == "columnar":
            return _render_columnar(node, name, module, shape, operations, expect)
        if name == "reduce" or expect == "bool":
            # A nested reduction changes the shape, so the tree stops here.
            raise _Unsupported
        if name == "unary" and len(node.args) == 1:
            kind = str(attributes.get("op", ""))
            if kind not in UNARY:
                raise _Unsupported
            operations.append(kind)
            inner, _ = _render(node.args[0], module, shape, operations, "f64")
            return f"({kind} {inner})", "f64"
        if name in BINARY and len(node.args) == 2:
            operations.append(name)
            left, _ = _render(node.args[0], module, shape, operations, "f64")
            right, _ = _render(node.args[1], module, shape, operations, "f64")
            return f"({name} {left} {right})", "f64"
    raise _Unsupported


def _render_columnar(
    node: ast.Call,
    name: str,
    module: ModuleAnalysis,
    shape: _Shape,
    operations: list[str],
    expect: str | None,
) -> tuple[str, str]:
    described = COLUMNAR.get(name)
    if described is None:
        raise _Unsupported
    reads, gives = described
    if len(node.args) != len(reads) or (expect not in (None, "same") and expect != gives):
        raise _Unsupported
    operations.append(name)
    parts = []
    for argument, wanted in zip(node.args, reads, strict=True):
        if wanted == "scalar":
            if isinstance(argument, ast.Constant) and isinstance(argument.value, (int, float)):
                parts.append(f"c{float(argument.value)!r}")
                continue
            if isinstance(argument, ast.Name) and _storage_of(argument, module) is None:
                base = T.strip_literal(module.type_of(argument))
                if base in (T.FLOAT, T.INT):
                    parts.append(f"s{shape.scalar(argument.id)}")
                    continue
            raise _Unsupported
        text, _kind = _render(argument, module, shape, operations, wanted)
        parts.append(text)
    return f"({name} {' '.join(parts)})", gives


# -- the kernels, as tensor or columnar IR ------------------------------------------------


def kernel_module(loops: Iterable[FusedLoop], name: str) -> IRModule:
    """The IR module holding one function per fused loop.

    A tensor kernel takes each array as a pointer, each scalar as an `f64`,
    and the element count `n` last: `sym(out, a0.., s0.., n)` for a map and
    `sym(a0.., s0.., n) -> f64` for a reduction. Inside, every array is a
    `tensor<f64, N>` loaded from a buffer of `n` elements, so `N` is `n`
    and the whole expression is tensor arithmetic `lower-tensor` turns
    into one loop. A columnar kernel takes the result's values and validity
    buffers, then each array's, then the scalars and `n`, and is columnar
    arithmetic over columns of `n` rows.
    """
    module = IRModule(name)
    module.require("core", 1)
    for loop in loops:
        if loop.storage == "pyarrow":
            module.require("columnar", 1)
            _build_columnar(module, loop)
        else:
            module.require("tensor", 1)
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


_KINDS: dict[str, IRType] = {"f64": F64, "bool": BOOL}


def _build_columnar(module: IRModule, loop: FusedLoop) -> None:
    kinds = loop.kinds or ("f64",) * len(loop.arrays)
    result_type = _KINDS[loop.result]
    params: list[tuple[str, object]] = [
        ("out", PtrType(columnar.values_element(result_type))),
        ("outv", PtrType(U8)),
    ]
    for i, kind in enumerate(kinds):
        params.append((f"a{i}", PtrType(columnar.values_element(_KINDS[kind]))))
        params.append((f"v{i}", PtrType(U8)))
    params.extend((f"s{i}", F64) for i in range(len(loop.scalars)))
    params.append(("n", I64))
    function = module.add_function(loop.symbol, params, [])  # type: ignore[arg-type]
    entry = function.add_entry_block()
    b = Builder(entry)
    arguments = list(entry.arguments)
    n = arguments[-1]
    bits = b.create(
        "core.shr",
        (core.add(b, n, core.const(b, 7, I64), overflow="wrap"), core.const(b, 3, I64)),
        (I64,),
    ).result

    def as_buffer(pointer: Value, count: Value, element: IRType) -> Value:
        return core.call_intrinsic(
            b, "ppy.buffer_from_parts", (pointer, count), (BufferType(element),)
        ).results[0]

    def rows_of(dtype: IRType) -> Value:
        return bits if dtype is BOOL else n

    scalar_values = arguments[2 + 2 * len(kinds) : -1]
    columns = []
    for i, kind in enumerate(kinds):
        dtype = _KINDS[kind]
        values = as_buffer(arguments[2 + 2 * i], rows_of(dtype), columnar.values_element(dtype))
        validity = as_buffer(arguments[3 + 2 * i], bits, U8)
        columns.append(columnar.from_parts(b, values, validity, n, dtype, f"a{i}"))

    def scalar_of(node: _Node) -> Value:
        if node.op == "scalar":
            return scalar_values[node.scalar]
        return core.const(b, node.constant, F64)

    def emit(node: _Node) -> Value:
        if node.op == "array":
            return columns[node.array]
        if node.op in {"scalar", "constant"}:
            return columnar.fill(b, scalar_of(node), n)
        if node.op == "fill_null":
            return columnar.fill_null(b, emit(node.operands[0]), scalar_of(node.operands[1]))
        operands = [emit(child) for child in node.operands]
        if node.op in {"negate", "abs", "invert"}:
            return columnar.unary(b, node.op, operands[0])
        if node.op == "is_null":
            return columnar.is_null(b, operands[0])
        if node.op == "is_valid":
            return columnar.is_valid(b, operands[0])
        if node.op == "select":
            return columnar.select(b, operands[0], operands[1], operands[2])
        if (
            node.op in columnar.ARITHMETIC
            or node.op in columnar.COMPARISON
            or node.op in columnar.BOOLEAN
        ):
            return columnar.binary(b, node.op, operands[0], operands[1])
        raise ValueError(f"fused expression uses {node.op!r}, which the columnar dialect lacks")

    value = emit(_parse(loop.expression))
    out = as_buffer(arguments[0], rows_of(result_type), columnar.values_element(result_type))
    info = columnar.describe(value.type)
    assert info is not None
    if info.nullable:
        columnar.store(b, value, out, as_buffer(arguments[1], bits, U8))
    else:
        columnar.store(b, value, out)
    core.ret(b)


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
