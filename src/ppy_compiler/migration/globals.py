"""Rewrites for module-namespace writes that name their target.

At module scope, `globals()["LIMIT"] = 128` is `LIMIT = 128` with extra
steps -- the same binding in the same namespace. Inside a function the two
differ (`globals()` writes past the local scope), so this pass rewrites only
statements that sit in the module's own body.
"""

from __future__ import annotations

import keyword

import libcst as cst

from .pipeline import bound_names
from .report import Rewrite, rewrites_at

__all__ = ["ModuleNamespaceWrites"]


class _NamespaceRewriter(cst.CSTTransformer):
    def __init__(self) -> None:
        self.pending: list[tuple[cst.CSTNode, str, str]] = []
        self._depth = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self._depth += 1
        return True

    def leave_FunctionDef(
        self, original: cst.FunctionDef, updated: cst.FunctionDef
    ) -> cst.FunctionDef:
        self._depth -= 1
        return updated

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._depth += 1
        return True

    def leave_ClassDef(self, original: cst.ClassDef, updated: cst.ClassDef) -> cst.ClassDef:
        self._depth -= 1
        return updated

    def leave_SimpleStatementLine(
        self, original: cst.SimpleStatementLine, updated: cst.SimpleStatementLine
    ) -> cst.SimpleStatementLine:
        if self._depth or len(updated.body) != 1 or not isinstance(updated.body[0], cst.Assign):
            return updated
        assign = updated.body[0]
        if len(assign.targets) != 1:
            return updated
        target = assign.targets[0].target
        if not isinstance(target, cst.Subscript) or not isinstance(target.value, cst.Call):
            return updated
        call = target.value
        if not isinstance(call.func, cst.Name) or call.func.value != "globals" or call.args:
            return updated
        if len(target.slice) != 1:
            return updated
        element = target.slice[0].slice
        if not isinstance(element, cst.Index) or not isinstance(element.value, cst.SimpleString):
            return updated
        name = element.value.evaluated_value
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
            return updated
        self.pending.append(
            (
                original,
                "module-namespace-writes",
                f'`globals()["{name}"] = ...` became `{name} = ...`',
            )
        )
        return updated.with_changes(
            body=[cst.Assign(targets=[cst.AssignTarget(cst.Name(name))], value=assign.value)]
        )


class ModuleNamespaceWrites:
    """`globals()["NAME"] = value` in the module body becomes `NAME = value`."""

    name = "module-namespace-writes"

    def apply(self, module: cst.Module, source: str) -> tuple[cst.Module, list[Rewrite]]:
        if "globals" in bound_names(source):
            return module, []
        rewriter = _NamespaceRewriter()
        rewritten = module.visit(rewriter)
        return rewritten, rewrites_at(module, rewriter.pending)
