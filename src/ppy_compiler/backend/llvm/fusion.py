"""Fused NumPy elementwise loops (spec 19.4).

A recognized elementwise expression tree becomes one broadcast-free strided
loop over `float64` data instead of one Python dispatch and one temporary array
per node. The loop runs only behind the guards the plugin demands: exact
`numpy.ndarray`, a supported dtype, native byte order, C-contiguity, and
matching shapes (spec 19.3, 19.5).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ...analysis import types as T
from ...analysis.checker import ModuleAnalysis
from ...analysis.symbols import FunctionInfo
from ...plugins.numpy_plugin import FUSIBLE_BINARY, FUSIBLE_REDUCTIONS, FUSIBLE_UNARY

__all__ = [
    "FusedLoop",
    "FusionCandidate",
    "find_candidates",
    "find_module_candidates",
    "lower_candidate",
    "UNARY",
    "BINARY",
]

#: Unary ufuncs lowered to an LLVM intrinsic over one element.
UNARY = {
    "sin": "llvm.sin.f64",
    "cos": "llvm.cos.f64",
    "exp": "llvm.exp.f64",
    "log": "llvm.log.f64",
    "log2": "llvm.log2.f64",
    "log10": "llvm.log10.f64",
    "sqrt": "llvm.sqrt.f64",
    "absolute": "llvm.fabs.f64",
    "abs": "llvm.fabs.f64",
    "negative": "",
}

#: Binary ufuncs lowered to a single machine instruction per element.
BINARY = set(FUSIBLE_BINARY)

#: Reductions lowered to a sequential accumulation.
REDUCTIONS = set(FUSIBLE_REDUCTIONS)

#: Reductions whose fused form reassociates the accumulation. NumPy sums
#: pairwise, so a sequential loop is a different -- and observably different --
#: order. These fuse only where the program permits reassociation (spec 19.8).
REASSOCIATING = frozenset({"sum", "prod", "product", "mean"})

_OPERATOR_UFUNC = {
    ast.Add: "add",
    ast.Sub: "subtract",
    ast.Mult: "multiply",
    ast.Div: "true_divide",
    ast.Pow: "power",
}


@dataclass(slots=True)
class FusedLoop:
    """One fused kernel: an expression over N arrays producing one array."""

    symbol: str
    arrays: tuple[str, ...]
    scalars: tuple[str, ...]
    reduction: str = ""
    expression: str = ""
    parallel: bool = False

    @property
    def returns_scalar(self) -> bool:
        return bool(self.reduction)


@dataclass(slots=True)
class FusionCandidate:
    """An expression the NumPy plugin marked `Intrinsic` and PPY can fuse."""

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

    def array(self, name: str) -> int:
        if name not in self.arrays:
            self.arrays.append(name)
        return self.arrays.index(name)

    def scalar(self, name: str) -> int:
        if name not in self.scalars:
            self.scalars.append(name)
        return self.scalars.index(name)


class _Unsupported(Exception):
    pass


def _is_array(node: ast.expr, module: ModuleAnalysis) -> bool:
    base = T.strip_literal(module.type_of(node))
    return isinstance(base, T.Instance) and base.name == "numpy.ndarray"


def _ufunc_of(node: ast.expr, module: ModuleAnalysis) -> str | None:
    note = module.lowerings.get(id(node))
    if note is None or note.lowering != "Intrinsic":
        return None
    return note.qualname.rpartition(".")[2]


def find_candidates(
    function: FunctionInfo,
    module: ModuleAnalysis,
) -> list[FusionCandidate]:
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
            statement for statement in tree.body
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
        ufunc = _ufunc_of(node, module)
        if ufunc is None:
            continue
        if ufunc not in UNARY and ufunc not in BINARY and ufunc not in REDUCTIONS:
            continue
        shape = _Shape()
        operations: list[str] = []
        reduction = ""
        try:
            if ufunc in REDUCTIONS:
                # A reduction is only fusible as the root of the tree: its
                # operand collapses to one value, so nothing can wrap it.
                if not isinstance(node, ast.Call) or len(node.args) != 1 or node.keywords:
                    continue
                if ufunc in REASSOCIATING and not fastmath:
                    continue
                reduction = ufunc
                operations.append(ufunc)
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
    """Render one expression node into a small postfix program."""
    if isinstance(node, ast.Name):
        if _is_array(node, module):
            return f"a{shape.array(node.id)}"
        base = T.strip_literal(module.type_of(node))
        if base in (T.FLOAT, T.INT, T.BOOL):
            return f"s{shape.scalar(node.id)}"
        raise _Unsupported
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return f"c{float(node.value)!r}"
    if isinstance(node, ast.BinOp):
        ufunc = _OPERATOR_UFUNC.get(type(node.op))
        if ufunc is None or ufunc not in BINARY:
            raise _Unsupported
        operations.append(ufunc)
        left = _render(node.left, module, shape, operations)
        right = _render(node.right, module, shape, operations)
        return f"({ufunc} {left} {right})"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operations.append("negative")
        return f"(negative {_render(node.operand, module, shape, operations)})"
    if isinstance(node, ast.Call):
        ufunc = _ufunc_of(node, module)
        if ufunc is None:
            raise _Unsupported
        if node.keywords:
            raise _Unsupported
        if ufunc in REDUCTIONS:
            # A nested reduction changes the shape, so the tree stops here.
            raise _Unsupported
        if ufunc in UNARY and len(node.args) == 1:
            operations.append(ufunc)
            return f"({ufunc} {_render(node.args[0], module, shape, operations)})"
        if ufunc in BINARY and len(node.args) == 2:
            operations.append(ufunc)
            left = _render(node.args[0], module, shape, operations)
            right = _render(node.args[1], module, shape, operations)
            return f"({ufunc} {left} {right})"
    raise _Unsupported


def lower_candidate(ir, llvm_module, candidate: FusionCandidate):  # type: ignore[no-untyped-def]
    """Emit the LLVM function implementing one fused loop.

    Signature: `void sym(double* out, double* a0, .., double s0, .., i64 n)`,
    or `double sym(double* a0, .., double s0, .., i64 n)` for a reduction.
    """
    loop = candidate.loop
    double = ir.DoubleType()
    i64 = ir.IntType(64)

    array_args = [double.as_pointer() for _ in loop.arrays]
    scalar_args = [double for _ in loop.scalars]
    if loop.returns_scalar:
        signature = ir.FunctionType(double, [*array_args, *scalar_args, i64])
    else:
        signature = ir.FunctionType(
            ir.VoidType(), [double.as_pointer(), *array_args, *scalar_args, i64]
        )

    function = ir.Function(llvm_module, signature, name=loop.symbol)
    offset = 0 if loop.returns_scalar else 1
    out = None if loop.returns_scalar else function.args[0]
    arrays = list(function.args[offset : offset + len(loop.arrays)])
    scalars = list(function.args[offset + len(loop.arrays) : -1])
    length = function.args[-1]

    entry = function.append_basic_block("entry")
    builder = ir.IRBuilder(entry)

    accumulator = None
    index = builder.alloca(i64, name="i")
    if loop.returns_scalar:
        accumulator = builder.alloca(double, name="acc")
        if loop.reduction in {"max", "min"}:
            # An empty reduction has no identity to start from; NumPy raises,
            # so the caller's fallback must handle it.
            builder.store(builder.load(builder.gep(arrays[0], [ir.Constant(i64, 0)])), accumulator)
            builder.store(ir.Constant(i64, 1), index)
        else:
            initial = 1.0 if loop.reduction in {"prod", "product"} else 0.0
            builder.store(ir.Constant(double, initial), accumulator)
            builder.store(ir.Constant(i64, 0), index)
    else:
        builder.store(ir.Constant(i64, 0), index)

    header = function.append_basic_block("head")
    body = function.append_basic_block("body")
    exit_block = function.append_basic_block("end")
    builder.branch(header)

    builder.position_at_end(header)
    current = builder.load(index)
    builder.cbranch(builder.icmp_signed("<", current, length), body, exit_block)

    builder.position_at_end(body)
    position = builder.load(index)
    emitter = _Emitter(ir, llvm_module, builder, arrays, scalars, position)
    value = emitter.evaluate(_parse(loop.expression))

    if loop.returns_scalar:
        carried = builder.load(accumulator)
        if loop.reduction in {"prod", "product"}:
            updated = builder.fmul(carried, value)
        elif loop.reduction in {"max", "min"}:
            # NumPy's min/max propagate NaN, and doing the same keeps the
            # reduction associative, so no reassociation contract is needed.
            symbol = ">" if loop.reduction == "max" else "<"
            better = builder.fcmp_ordered(symbol, value, carried)
            unordered = builder.fcmp_unordered("!=", value, value)
            updated = builder.select(
                builder.or_(better, unordered), value, carried
            )
        else:
            # `sum` and `mean` accumulate in strict index order: no
            # reassociation without an explicit directive (spec 19.8).
            updated = builder.fadd(carried, value)
        builder.store(updated, accumulator)
    else:
        builder.store(value, builder.gep(out, [position]))

    builder.store(builder.add(builder.load(index), ir.Constant(i64, 1)), index)
    builder.branch(header)

    builder.position_at_end(exit_block)
    if loop.returns_scalar:
        total = builder.load(accumulator)
        if loop.reduction == "mean":
            total = builder.fdiv(total, builder.sitofp(length, double))
        builder.ret(total)
    else:
        builder.ret_void()
    return function


@dataclass(slots=True)
class _Node:
    op: str
    operands: list["_Node"] = field(default_factory=list)
    array: int = -1
    scalar: int = -1
    constant: float = 0.0


def _parse(text: str) -> _Node:
    """Parse the postfix program rendered by `_render`."""
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


class _Emitter:
    def __init__(self, ir, module, builder, arrays, scalars, position):  # type: ignore[no-untyped-def]
        self.ir = ir
        self.module = module
        self.builder = builder
        self.arrays = arrays
        self.scalars = scalars
        self.position = position

    def evaluate(self, node: _Node):  # type: ignore[no-untyped-def]
        ir = self.ir
        if node.op == "array":
            pointer = self.builder.gep(self.arrays[node.array], [self.position])
            return self.builder.load(pointer)
        if node.op == "scalar":
            return self.scalars[node.scalar]
        if node.op == "constant":
            return ir.Constant(ir.DoubleType(), node.constant)

        operands = [self.evaluate(child) for child in node.operands]
        if node.op == "negative":
            return self.builder.fneg(operands[0])
        if node.op in UNARY:
            return self.builder.call(self._intrinsic(UNARY[node.op], 1), operands)
        if node.op == "add":
            return self.builder.fadd(*operands)
        if node.op == "subtract":
            return self.builder.fsub(*operands)
        if node.op == "multiply":
            return self.builder.fmul(*operands)
        if node.op in {"true_divide", "divide"}:
            return self.builder.fdiv(*operands)
        if node.op == "power":
            return self.builder.call(self._intrinsic("llvm.pow.f64", 2), operands)
        if node.op in {"minimum", "maximum"}:
            symbol = "<" if node.op == "minimum" else ">"
            keep = self.builder.fcmp_ordered(symbol, operands[0], operands[1])
            return self.builder.select(keep, operands[0], operands[1])
        raise _Unsupported(node.op)

    def _intrinsic(self, name: str, arity: int):  # type: ignore[no-untyped-def]
        ir = self.ir
        existing = self.module.globals.get(name)
        if existing is not None:
            return existing
        double = ir.DoubleType()
        return ir.Function(self.module, ir.FunctionType(double, [double] * arity), name=name)
