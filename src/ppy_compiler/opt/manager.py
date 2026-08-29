"""Pass manager and optimization-level policy (spec 14)."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ..analysis.checker import ModuleAnalysis, ProjectAnalysis
from ..analysis.symbols import FunctionInfo, ModuleSymbols
from ..diagnostics import Diagnostic, Severity, Span
from .annotate import annotate
from .passes import (
    AdjustLibraryCalls,
    BranchFold,
    CommonSubexpression,
    ConstantFold,
    CopyPropagation,
    FuseLibraryCalls,
    InlineSmallFunctions,
    LoopInvariantMotion,
    LoopUnroll,
    Pass,
    PassContext,
    Peephole,
    StripDirectives,
    UnreachableCode,
    UnusedLocals,
)

__all__ = ["OptimizationResult", "Optimizer", "level_description"]

_DIRECTIVE_NAMES = {
    "ppy.pure",
    "ppy.opt",
    "ppy.jit",
    "ppy.parallel",
    "ppy.native",
    "ppy.inline",
    "ppy.noinline",
    "ppy.specialize",
    "ppy.fastmath",
    "ppy.dynamic",
    "ppy.jax",
    "ppy.reflective",
    "pure",
    "opt",
    "jit",
    "parallel",
    "native",
    "inline",
    "noinline",
    "specialize",
    "fastmath",
    "dynamic",
    "reflective",
}

_LEVELS = {
    0: "parse, check, and canonicalize only",
    1: "local folding, branch folding, dead code and unused-local removal",
    2: "O1 plus CSE, loop-invariant motion, and small-function inlining",
    3: "O2 plus aggressive inlining and loop unrolling",
}

_MAX_ROUNDS = 3


def level_description(level: int) -> str:
    return _LEVELS.get(level, "unknown level")


@dataclass(slots=True)
class OptimizationResult:
    tree: ast.Module
    remarks: list[Diagnostic] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    function_levels: dict[str, int] = field(default_factory=dict)
    fused_symbols: tuple[str, ...] = ()


class Optimizer:
    """Applies the pass pipeline to one module."""

    def __init__(
        self,
        symbols: ModuleSymbols,
        module: ModuleAnalysis,
        project: ProjectAnalysis,
        *,
        level: int = 2,
        fusion: dict[tuple[int, int], object] | None = None,
        adjustments: dict[tuple[int, int], object] | None = None,
    ) -> None:
        self.symbols = symbols
        self.module = module
        self.project = project
        self.level = level
        self.fusion = fusion or {}
        self.adjustments = adjustments or {}

    def run(self) -> OptimizationResult:
        tree = annotate(self.symbols.module.tree, self.module)
        context = PassContext(module=self.module, level=self.level)
        result = OptimizationResult(tree=tree)

        self._run_pass(StripDirectives(context, _DIRECTIVE_NAMES), tree)

        if self.adjustments and self.level >= 1:
            self._run_pass(AdjustLibraryCalls(context, self.adjustments), tree)

        fuse: FuseLibraryCalls | None = None
        if self.fusion and self.level >= 2:
            # Fusion runs before every other pass, while nodes still carry the
            # source positions the analysis recorded them at.
            fuse = FuseLibraryCalls(context, self.fusion)
            self._run_pass(fuse, tree)

        for node in self._function_nodes(tree):
            info = self._info_for(node)
            level = info.opt_level if info and info.opt_level is not None else self.level
            result.function_levels[info.qualname if info else node.name] = level
            function_context = PassContext(
                module=self.module,
                level=level,
                fastmath=bool(info and info.directive("fastmath")),
            )
            self._optimize_subtree(node, function_context, level, tree)
            context.remarks.extend(function_context.remarks)
            for key, value in function_context.stats.items():
                context.stats[key] = context.stats.get(key, 0) + value
            node._ppy_done = True

        self._optimize_module_level(tree, context)

        preamble: list[ast.stmt] = []
        if fuse is not None and fuse.bound:
            preamble.extend(fuse.bindings())
            result.fused_symbols = tuple(loop.symbol for loop, _ in fuse.bound.values())
        if preamble:
            index = _after_imports(tree)
            tree.body[index:index] = preamble

        result.tree = tree
        result.stats = context.stats
        result.remarks = [
            Diagnostic("R3001", Severity.REMARK, message, Span(self.symbols.path, line, 0))
            for line, message in context.remarks
        ]
        ast.fix_missing_locations(tree)
        return result

    def _function_nodes(self, tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
        found: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.append(node)
            elif isinstance(node, ast.ClassDef):
                found.extend(
                    child
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
        return found

    def _info_for(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionInfo | None:
        direct = self.symbols.functions.get(node.name)
        if direct is not None:
            return direct
        for cls in self.symbols.classes.values():
            method = cls.methods.get(node.name)
            if method is not None and method.node.lineno == node.lineno:
                return method
        return None

    def _optimize_subtree(
        self, node: ast.AST, context: PassContext, level: int, tree: ast.Module | None = None
    ) -> None:
        if level <= 0:
            return
        for _ in range(_MAX_ROUNDS):
            before = context.stats.copy()
            for optimization in self._pipeline(context, level, tree):
                optimization.visit(node)
                ast.fix_missing_locations(node)
            if context.stats == before:
                break

    def _optimize_module_level(self, tree: ast.Module, context: PassContext) -> None:
        if self.level <= 0:
            return
        module_body = ast.Module(
            body=[s for s in tree.body if not getattr(s, "_ppy_done", False)],
            type_ignores=[],
        )
        if not module_body.body:
            return
        self._optimize_subtree(module_body, context, self.level, tree)
        optimized = iter(module_body.body)
        rebuilt: list[ast.stmt] = []
        for statement in tree.body:
            if getattr(statement, "_ppy_done", False):
                rebuilt.append(statement)
            else:
                try:
                    rebuilt.append(next(optimized))
                except StopIteration:
                    continue
        rebuilt.extend(optimized)
        tree.body = rebuilt

    def _pipeline(
        self, context: PassContext, level: int, tree: ast.Module | None = None
    ) -> list[Pass]:
        pipeline: list[Pass] = [
            ConstantFold(context),
            BranchFold(context),
            UnreachableCode(context),
            CopyPropagation(context),
            Peephole(context),
        ]
        if level >= 2:
            pipeline.insert(0, InlineSmallFunctions(context, self._inline_candidates(tree)))
            pipeline.append(LoopInvariantMotion(context))
            pipeline.append(CommonSubexpression(context))
        if level >= 3:
            pipeline.append(LoopUnroll(context))
        pipeline.append(UnusedLocals(context))
        return pipeline

    def _inline_candidates(
        self, tree: ast.Module | None
    ) -> dict[str, tuple[FunctionInfo, ast.FunctionDef]]:
        """Small functions with verified purity and no `@ppy.noinline`.

        The body comes from the tree being transformed, not the original
        source, so an already-optimized body is what gets inlined.
        """
        live: dict[str, ast.FunctionDef] = {}
        if tree is not None:
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    live[node.name] = node

        candidates: dict[str, tuple[FunctionInfo, ast.FunctionDef]] = {}
        for name, info in self.symbols.functions.items():
            if info.directive("noinline") is not None:
                continue
            analysis = self.module.functions.get(info.qualname)
            if analysis is None or not analysis.verified_pure:
                continue
            if info.is_generator or info.is_async:
                continue
            candidates[name] = (info, live.get(name, info.node))
        return candidates

    def _run_pass(self, optimization: Pass, tree: ast.Module) -> None:
        optimization.visit(tree)
        ast.fix_missing_locations(tree)


def _after_imports(tree: ast.Module) -> int:
    """The index just past the module's leading import statements."""
    index = 0
    for position, statement in enumerate(tree.body):
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            index = position + 1
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            index = position + 1
            continue
        break
    return index
