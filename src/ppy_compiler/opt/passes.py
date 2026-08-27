"""Optimization passes (spec 14).

Every pass here preserves Python-observable semantics. Nothing in this module
may weaken arithmetic, aliasing, exception behavior, or floating-point
ordering: those require an explicit directive (spec 3.4).
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field

from ..analysis import types as T
from ..analysis.checker import FunctionAnalysis, ModuleAnalysis
from ..analysis.symbols import FunctionInfo
from .annotate import PURE_NODES, const_of, has_const, type_of

__all__ = [
    "PassContext",
    "Pass",
    "StripDirectives",
    "ConstantFold",
    "BranchFold",
    "UnreachableCode",
    "CopyPropagation",
    "UnusedLocals",
    "Peephole",
    "CommonSubexpression",
    "LoopInvariantMotion",
    "InlineSmallFunctions",
    "LoopUnroll",
    "FuseLibraryCalls",
    "AdjustLibraryCalls",
    "FUSED_BINDER",
]

#: Name the LLVM backend injects to supply a fused kernel to a module.
FUSED_BINDER = "__ppy_bind_fused__"

_MAX_UNROLL = 8
_INLINE_BUDGET = {2: 16, 3: 48}


@dataclass(slots=True)
class PassContext:
    module: ModuleAnalysis
    level: int
    functions: dict[str, FunctionAnalysis] = field(default_factory=dict)
    remarks: list[tuple[int, str]] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    fastmath: bool = False

    def remark(self, node: ast.AST, message: str) -> None:
        self.remarks.append((getattr(node, "lineno", 0), message))

    def count(self, name: str, amount: int = 1) -> None:
        self.stats[name] = self.stats.get(name, 0) + amount


class Pass(ast.NodeTransformer):
    """Base class for a transformation pass."""

    name = "pass"
    min_level = 1

    def __init__(self, context: PassContext) -> None:
        self.context = context

    def run(self, tree: ast.Module) -> ast.Module:
        result = self.visit(tree)
        ast.fix_missing_locations(result)
        return result


def _is_load(node: ast.expr) -> bool:
    """Only a value being read may be replaced; a store target may not."""
    context = getattr(node, "ctx", None)
    return context is None or isinstance(context, ast.Load)


def _is_pure_expr(node: ast.expr) -> bool:
    for child in ast.walk(node):
        if isinstance(child, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)):
            return False
        if isinstance(child, ast.Attribute):
            return False
        if isinstance(child, ast.Subscript):
            return False
        if not isinstance(child, (*PURE_NODES, ast.expr_context, ast.operator, ast.unaryop,
                                  ast.boolop, ast.cmpop, ast.comprehension, ast.arguments, ast.arg)):
            return False
    return True


def _assigned_names(nodes: list[ast.stmt]) -> set[str]:
    found: set[str] = set()
    for statement in nodes:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                found.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.add(node.name)
            elif isinstance(node, ast.Global):
                found.update(node.names)
    return found


def _loaded_names(nodes: list[ast.stmt] | ast.AST) -> set[str]:
    roots = nodes if isinstance(nodes, list) else [nodes]
    found: set[str] = set()
    for statement in roots:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                found.add(node.id)
    return found


class StripDirectives(Pass):
    """Remove PPY decorators: they are inert by contract (spec 6.1, 15.4)."""

    name = "strip-directives"
    min_level = 0

    def __init__(self, context: PassContext, directive_names: set[str]) -> None:
        super().__init__(context)
        self.directive_names = directive_names

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        return self._strip(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        return self._strip(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        return self._strip(node)

    def _strip(self, node):  # type: ignore[no-untyped-def]
        kept = []
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if ast.unparse(target) in self.directive_names:
                self.context.count("directives_removed")
                continue
            kept.append(decorator)
        node.decorator_list = kept
        return node

    def visit_With(self, node: ast.With) -> ast.AST:
        self.generic_visit(node)
        remaining = [
            item for item in node.items
            if ast.unparse(item.context_expr) not in self.directive_names
        ]
        if remaining or not node.items:
            node.items = remaining or node.items
            return node
        self.context.count("dynamic_blocks_flattened")
        return node.body  # type: ignore[return-value]


class ConstantFold(Pass):
    """Replace provably constant pure expressions with their value."""

    name = "constant-fold"
    min_level = 1

    def visit(self, node: ast.AST) -> ast.AST:
        node = super().visit(node)
        if isinstance(node, ast.expr) and not isinstance(node, ast.Constant) and _is_load(node):
            if has_const(node) and _is_pure_expr(node):
                value = const_of(node)
                if isinstance(value, (int, float, complex, str, bytes, bool, type(None))):
                    replacement = ast.Constant(value=value)
                    ast.copy_location(replacement, node)
                    self.context.count("constants_folded")
                    return replacement
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and has_const(node):
            value = const_of(node)
            if isinstance(value, (int, float, str, bytes, bool, type(None))):
                replacement = ast.Constant(value=value)
                ast.copy_location(replacement, node)
                self.context.count("constants_propagated")
                return replacement
        return node


class BranchFold(Pass):
    """Fold branches whose test is a compile-time constant."""

    name = "branch-fold"
    min_level = 1

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        value = self._test_value(node.test)
        if value is None:
            return node
        taken = node.body if value else node.orelse
        self.context.count("branches_folded")
        self.context.remark(node, f"branch folded: test is always {bool(value)}")
        return taken or [ast.copy_location(ast.Pass(), node)]

    def visit_While(self, node: ast.While) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        value = self._test_value(node.test)
        if value is False:
            self.context.count("loops_removed")
            return node.orelse or [ast.copy_location(ast.Pass(), node)]
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        value = self._test_value(node.test)
        if value is None:
            return node
        self.context.count("branches_folded")
        return node.body if value else node.orelse

    def _test_value(self, test: ast.expr) -> bool | None:
        if isinstance(test, ast.Constant):
            return bool(test.value)
        if has_const(test) and _is_pure_expr(test):
            return bool(const_of(test))
        return None


class UnreachableCode(Pass):
    """Drop statements that follow an unconditional transfer of control."""

    name = "unreachable-code"
    min_level = 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._truncate(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._truncate(node.body)
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)
        node.body = self._truncate(node.body)
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        self.generic_visit(node)
        node.body = self._truncate(node.body)
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        node.body = self._truncate(node.body)
        node.orelse = self._truncate(node.orelse)
        return node

    def _truncate(self, body: list[ast.stmt]) -> list[ast.stmt]:
        for index, statement in enumerate(body):
            if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                if index + 1 < len(body):
                    self.context.count("unreachable_removed", len(body) - index - 1)
                return body[: index + 1]
        return body


class Peephole(Pass):
    """Local simplifications that never change observable behavior."""

    name = "peephole"
    min_level = 1

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        node.values = self._drop_settled_operands(node)
        if not node.values:
            return ast.copy_location(ast.Constant(value=isinstance(node.op, ast.And)), node)
        if len(node.values) == 1:
            return node.values[0]
        return node

    def _drop_settled_operands(self, node: ast.BoolOp) -> list[ast.expr]:
        """Drop operands a constant has already settled, keeping short-circuiting."""
        neutral = isinstance(node.op, ast.And)
        kept: list[ast.expr] = []
        for index, value in enumerate(node.values):
            settled = isinstance(value, ast.Constant) and isinstance(value.value, bool)
            if settled and value.value is neutral and index < len(node.values) - 1:
                self.context.count("peepholes")
                continue
            kept.append(value)
            if settled and value.value is not neutral:
                self.context.count("peepholes")
                break
        return kept

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.UnaryOp):
            if isinstance(node.operand.op, ast.Not):
                inner = node.operand.operand
                if type_of(inner) == T.BOOL:
                    self.context.count("peepholes")
                    return inner
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        # Inlining can substitute a literal into `x is None`. Identity against a
        # literal is what CPython raises SyntaxWarning about, and the answer is
        # already known, so settle it here instead of emitting the comparison.
        if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Is, ast.IsNot)):
            return node
        left, right = node.left, node.comparators[0]
        if not (isinstance(left, ast.Constant) and isinstance(right, ast.Constant)):
            return node
        if left.value is None or right.value is None:
            matches = left.value is None and right.value is None
            settled = matches == isinstance(node.ops[0], ast.Is)
            self.context.count("peepholes")
            return ast.copy_location(ast.Constant(value=settled), node)
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        if (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Pass)
            and not node.orelse
            and _is_pure_expr(node.test)
        ):
            self.context.count("peepholes")
            return ast.copy_location(ast.Pass(), node)
        return node


class CopyPropagation(Pass):
    """Forward `x = y` where both are locals and neither is reassigned."""

    name = "copy-propagation"
    min_level = 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        return self._propagate(node)

    def _propagate(self, node: ast.FunctionDef) -> ast.FunctionDef:
        assigned = _assigned_names(node.body)
        counts: dict[str, int] = {}
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                counts[child.id] = counts.get(child.id, 0) + 1
        substitutions: dict[str, str] = {}
        keep: list[ast.stmt] = []
        for statement in node.body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.Name)
                and counts.get(statement.targets[0].id) == 1
                and counts.get(statement.value.id, 0) <= 1
                and statement.value.id in assigned | {p.arg for p in node.args.args}
            ):
                substitutions[statement.targets[0].id] = statement.value.id
                self.context.count("copies_propagated")
                continue
            keep.append(statement)
        if not substitutions:
            return node
        node.body = keep
        _Rename(substitutions).visit(node)
        return node


class _Rename(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        replacement = self.mapping.get(node.id)
        if replacement is not None and isinstance(node.ctx, ast.Load):
            node.id = replacement
        return node


class UnusedLocals(Pass):
    """Remove assignments to locals that are never read, when the value is pure."""

    name = "unused-locals"
    min_level = 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        loaded = _loaded_names(node.body)
        kept: list[ast.stmt] = []
        for statement in node.body:
            if (
                isinstance(statement, (ast.Assign, ast.AnnAssign))
                and _single_name_target(statement) is not None
                and _single_name_target(statement) not in loaded
                and statement.value is not None
                and _is_pure_expr(statement.value)
            ):
                self.context.count("unused_locals_removed")
                continue
            kept.append(statement)
        node.body = kept or [ast.copy_location(ast.Pass(), node)]
        return node


def _single_name_target(statement: ast.Assign | ast.AnnAssign) -> str | None:
    if isinstance(statement, ast.AnnAssign):
        return statement.target.id if isinstance(statement.target, ast.Name) else None
    if len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
        return statement.targets[0].id
    return None


class CommonSubexpression(Pass):
    """Hoist a repeated pure subexpression in one statement into a temporary."""

    name = "cse"
    min_level = 2

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        new_body: list[ast.stmt] = []
        counter = 0
        for statement in node.body:
            if isinstance(statement, (ast.Assign, ast.Return)) and statement.value is not None:
                repeated = self._repeated(statement.value)
                if repeated is not None:
                    counter += 1
                    name = f"_ppy_cse{counter}"
                    temporary = ast.Assign(
                        targets=[ast.Name(id=name, ctx=ast.Store())],
                        value=copy.deepcopy(repeated),
                    )
                    ast.copy_location(temporary, statement)
                    ast.fix_missing_locations(temporary)
                    _ReplaceExpr(ast.dump(repeated), name).visit(statement)
                    new_body.append(temporary)
                    self.context.count("subexpressions_eliminated")
                    self.context.remark(statement, f"common subexpression hoisted into `{name}`")
            new_body.append(statement)
        node.body = new_body
        return node

    def _repeated(self, root: ast.expr) -> ast.expr | None:
        seen: dict[str, tuple[int, ast.expr]] = {}
        for node in ast.walk(root):
            if not isinstance(node, (ast.BinOp, ast.Compare)):
                continue
            if not _is_pure_expr(node):
                continue
            key = ast.dump(node)
            count, first = seen.get(key, (0, node))
            seen[key] = (count + 1, first)
        candidates = [
            (len(ast.dump(expr)), expr)
            for key, (count, expr) in seen.items()
            if count >= 2
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda pair: pair[0])[1]


class _ReplaceExpr(ast.NodeTransformer):
    def __init__(self, dump: str, name: str) -> None:
        self.dump = dump
        self.name = name

    def visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.expr) and ast.dump(node) == self.dump:
            return ast.copy_location(ast.Name(id=self.name, ctx=ast.Load()), node)
        return super().visit(node)


class LoopInvariantMotion(Pass):
    """Hoist pure loop-invariant expressions out of a loop body."""

    name = "licm"
    min_level = 2

    def visit_For(self, node: ast.For) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        return self._hoist(node, node.body, {_target_names(node.target)})

    def _hoist(self, node: ast.For, body: list[ast.stmt], bound: set[frozenset[str]]) -> ast.AST | list[ast.stmt]:
        varying = set().union(*bound) | _assigned_names(body)
        hoisted: list[ast.stmt] = []
        kept: list[ast.stmt] = []
        for statement in body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and _is_pure_expr(statement.value)
                and not (_loaded_names(statement.value) & varying)
                and statement.targets[0].id not in _loaded_names(node.iter)
            ):
                hoisted.append(statement)
                self.context.count("invariants_hoisted")
                self.context.remark(statement, "loop-invariant expression hoisted")
                continue
            kept.append(statement)
        if not hoisted:
            return node
        node.body = kept or [ast.copy_location(ast.Pass(), node)]
        return [*hoisted, node]


def _target_names(target: ast.expr) -> frozenset[str]:
    return frozenset(
        node.id for node in ast.walk(target)
        if isinstance(node, ast.Name)
    )


class InlineSmallFunctions(Pass):
    """Inline a small verified-pure function whose body is a single return."""

    name = "inline"
    min_level = 2

    def __init__(
        self, context: PassContext, candidates: dict[str, tuple[FunctionInfo, ast.AST]]
    ) -> None:
        super().__init__(context)
        self.candidates = candidates
        self.budget = _INLINE_BUDGET.get(context.level, 0)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.keywords:
            return node
        entry = self.candidates.get(node.func.id)
        if entry is None:
            return node
        info, definition = entry
        body = _single_return_expr(definition)
        if body is None:
            return node
        if _expr_size(body) > self.budget:
            return node
        params = [p.name for p in info.params]
        if len(params) != len(node.args):
            return node
        if any(not _is_pure_expr(a) for a in node.args):
            return node
        mapping = dict(zip(params, node.args))
        inlined = _Substitute(mapping).visit(copy.deepcopy(body))
        ast.copy_location(inlined, node)
        ast.fix_missing_locations(inlined)
        self.context.count("calls_inlined")
        self.context.remark(node, f"inlined call to `{info.name}`")
        return inlined


class _Substitute(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, ast.expr]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        replacement = self.mapping.get(node.id)
        if replacement is not None and isinstance(node.ctx, ast.Load):
            return copy.deepcopy(replacement)
        return node


def _single_return_expr(node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.expr | None:
    body = [s for s in node.body if not isinstance(s, ast.Pass)]
    if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None:
        return body[0].value
    if len(body) == 2 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[1], ast.Return) and body[1].value is not None:
            return body[1].value
    return None


def _expr_size(node: ast.expr) -> int:
    return sum(1 for _ in ast.walk(node))


class LoopUnroll(Pass):
    """Fully unroll `for i in range(n)` for a small constant `n`."""

    name = "unroll"
    min_level = 3

    def visit_For(self, node: ast.For) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        if node.orelse or not isinstance(node.target, ast.Name):
            return node
        bounds = _constant_range(node.iter)
        if bounds is None:
            return node
        values = list(range(*bounds))
        if not values or len(values) > _MAX_UNROLL:
            return node
        if len(values) * len(node.body) > 32:
            return node
        if any(isinstance(n, (ast.Break, ast.Continue)) for n in ast.walk(node)):
            return node
        unrolled: list[ast.stmt] = []
        for value in values:
            for statement in node.body:
                clone = copy.deepcopy(statement)
                _Substitute({node.target.id: ast.Constant(value=value)}).visit(clone)
                ast.copy_location(clone, statement)
                ast.fix_missing_locations(clone)
                unrolled.append(clone)
        self.context.count("loops_unrolled")
        self.context.remark(node, f"loop unrolled {len(values)} times")
        return unrolled


def _constant_range(node: ast.expr) -> tuple[int, int, int] | None:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"):
        return None
    values: list[int] = []
    for argument in node.args:
        if isinstance(argument, ast.Constant) and isinstance(argument.value, int):
            values.append(argument.value)
        elif has_const(argument) and isinstance(const_of(argument), int):
            values.append(const_of(argument))  # type: ignore[arg-type]
        else:
            return None
    match values:
        case [stop]:
            return 0, stop, 1
        case [start, stop]:
            return start, stop, 1
        case [start, stop, step] if step != 0:
            return start, stop, step
    return None


class FuseLibraryCalls(Pass):
    """Replace a plugin-fused expression with a call to its generated kernel.

    The kernel is bound once per module; the original expression becomes the
    fallback the binding falls back to whenever a guard fails (spec 19.3).
    """

    name = "fuse"
    min_level = 2

    def __init__(self, context: PassContext, plan: dict[tuple[int, int], object]) -> None:
        super().__init__(context)
        self.plan = plan
        self.bound: dict[str, tuple[object, ast.expr]] = {}

    def visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.expr):
            loop = self.plan.get((getattr(node, "lineno", -1), getattr(node, "col_offset", -1)))
            if loop is not None:
                return self._replace(node, loop)
        return super().visit(node)

    def _replace(self, node: ast.expr, loop) -> ast.AST:  # type: ignore[no-untyped-def]
        helper = f"_ppy_fused_{len(self.bound)}"
        self.bound[helper] = (loop, copy.deepcopy(node))
        arguments = [
            ast.Name(id=name, ctx=ast.Load())
            for name in (*loop.arrays, *loop.scalars)
        ]
        call = ast.Call(func=ast.Name(id=helper, ctx=ast.Load()), args=arguments, keywords=[])
        ast.copy_location(call, node)
        ast.fix_missing_locations(call)
        self.context.count("expressions_fused")
        self.context.remark(node, f"NumPy expression fused into one strided loop ({loop.symbol})")
        return call

    def bindings(self) -> list[ast.stmt]:
        """Module-level statements binding each kernel to its fallback lambda."""
        statements: list[ast.stmt] = []
        for helper, (loop, original) in self.bound.items():
            parameters = [ast.arg(arg=name) for name in (*loop.arrays, *loop.scalars)]
            fallback = ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[], args=parameters, kwonlyargs=[],
                    kw_defaults=[], defaults=[],
                ),
                body=original,
            )
            statements.append(
                ast.Assign(
                    targets=[ast.Name(id=helper, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id=FUSED_BINDER, ctx=ast.Load()),
                        args=[ast.Constant(value=loop.symbol), fallback],
                        keywords=[],
                    ),
                )
            )
        for statement in statements:
            ast.fix_missing_locations(statement)
        return statements


class AdjustLibraryCalls(Pass):
    """Apply a plugin's framework-level rewrite to one call's arguments."""

    name = "adjust-calls"
    min_level = 1

    def __init__(self, context: PassContext, plan: dict[tuple[int, int], object]) -> None:
        super().__init__(context)
        self.plan = plan

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        adjustment = self.plan.get(
            (getattr(node, "lineno", -1), getattr(node, "col_offset", -1))
        )
        if adjustment is None:
            return node

        if adjustment.replace_first_argument is not None and node.args:
            node.args[0] = ast.copy_location(
                ast.Name(id=adjustment.replace_first_argument, ctx=ast.Load()), node.args[0]
            )
        present = {keyword.arg for keyword in node.keywords}
        for name, source in adjustment.add_keywords:
            if name in present:
                continue
            node.keywords.append(ast.keyword(arg=name, value=ast.parse(source, mode="eval").body))
        ast.fix_missing_locations(node)
        self.context.count("calls_adjusted")
        self.context.remark(node, f"{adjustment.qualname}: {adjustment.reason}")
        return node
