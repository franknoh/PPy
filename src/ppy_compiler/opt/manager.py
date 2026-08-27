"""Pass manager and optimization-level policy (spec 14)."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ..analysis.checker import ModuleAnalysis, ProjectAnalysis
from ..analysis.symbols import FunctionInfo, ModuleSymbols
from ..diagnostics import Diagnostic, Severity, Span
from .annotate import annotate
from .passes import (
    BranchFold,
    CommonSubexpression,
    ConstantFold,
    CopyPropagation,
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
    "ppy.pure", "ppy.opt", "ppy.jit", "ppy.parallel", "ppy.native",
    "ppy.inline", "ppy.noinline", "ppy.specialize", "ppy.fastmath",
    "ppy.dynamic", "ppy.jax",
    "pure", "opt", "jit", "parallel", "native", "inline", "noinline",
    "specialize", "fastmath", "dynamic",
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


class Optimizer:
    """Applies the pass pipeline to one module."""

    def __init__(
        self,
        symbols: ModuleSymbols,
        module: ModuleAnalysis,
        project: ProjectAnalysis,
        *,
        level: int = 2,
    ) -> None:
        self.symbols = symbols
        self.module = module
        self.project = project
        self.level = level

    def run(self) -> OptimizationResult:
        tree = annotate(self.symbols.module.tree, self.module)
        context = PassContext(module=self.module, level=self.level)
        result = OptimizationResult(tree=tree)

        self._run_pass(StripDirectives(context, _DIRECTIVE_NAMES), tree)

        for node in self._function_nodes(tree):
            info = self._info_for(node)
            level = info.opt_level if info and info.opt_level is not None else self.level
            result.function_levels[info.qualname if info else node.name] = level
            function_context = PassContext(
                module=self.module,
                level=level,
                fastmath=bool(info and info.directive("fastmath")),
            )
            self._optimize_subtree(node, function_context, level)
            context.remarks.extend(function_context.remarks)
            for key, value in function_context.stats.items():
                context.stats[key] = context.stats.get(key, 0) + value
            setattr(node, "_ppy_done", True)

        self._optimize_module_level(tree, context)

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
                    child for child in node.body
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

    def _optimize_subtree(self, node: ast.AST, context: PassContext, level: int) -> None:
        if level <= 0:
            return
        for _ in range(_MAX_ROUNDS):
            before = context.stats.copy()
            for optimization in self._pipeline(context, level):
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
        self._optimize_subtree(module_body, context, self.level)
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

    def _pipeline(self, context: PassContext, level: int) -> list[Pass]:
        pipeline: list[Pass] = [
            ConstantFold(context),
            BranchFold(context),
            UnreachableCode(context),
            CopyPropagation(context),
            Peephole(context),
        ]
        if level >= 2:
            pipeline.insert(0, InlineSmallFunctions(context, self._inline_candidates()))
            pipeline.append(LoopInvariantMotion(context))
            pipeline.append(CommonSubexpression(context))
        if level >= 3:
            pipeline.append(LoopUnroll(context))
        pipeline.append(UnusedLocals(context))
        return pipeline

    def _inline_candidates(self) -> dict[str, FunctionInfo]:
        """Small functions with verified purity and no `@ppy.noinline`."""
        candidates: dict[str, FunctionInfo] = {}
        for name, info in self.symbols.functions.items():
            if info.directive("noinline") is not None:
                continue
            analysis = self.module.functions.get(info.qualname)
            if analysis is None or not analysis.verified_pure:
                continue
            if info.is_generator or info.is_async:
                continue
            candidates[name] = info
        return candidates

    def _run_pass(self, optimization: Pass, tree: ast.Module) -> None:
        optimization.visit(tree)
        ast.fix_missing_locations(tree)
