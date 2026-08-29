"""Rewrites for dynamic constructs that were static all along.

`setattr(obj, "status", v)` names its attribute as surely as `obj.status = v`
does; only the spelling is dynamic. Each rewrite here is proven equivalent
before it happens -- a literal, identifier-shaped name, the builtin actually
meaning the builtin -- and anything short of that proof is left for the
checker to report as the dynamic feature it still is.
"""

from __future__ import annotations

import ast
import keyword

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from .pipeline import bound_names
from .report import Rewrite

__all__ = ["LiteralAttributes", "StaticImports"]


def _literal_name(node: cst.BaseExpression) -> str | None:
    """The attribute name a call spells statically, if it spells one."""
    if not isinstance(node, cst.SimpleString):
        return None
    value = node.evaluated_value
    if not isinstance(value, str):
        return None
    if not value.isidentifier() or keyword.iskeyword(value):
        return None
    return value


def _plain_base(node: cst.BaseExpression) -> bool:
    if isinstance(node, cst.Name):
        return True
    return isinstance(node, cst.Attribute) and _plain_base(node.value)


def _arguments(call: cst.Call) -> list[cst.BaseExpression] | None:
    """Positional arguments, or None if anything keyword/star-shaped appears."""
    found: list[cst.BaseExpression] = []
    for argument in call.args:
        if argument.keyword is not None or argument.star:
            return None
        found.append(argument.value)
    return found


class _AttributeRewriter(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, shadowed: frozenset[str]) -> None:
        self.shadowed = shadowed
        self.rewrites: list[Rewrite] = []

    def _line(self, node: cst.CSTNode) -> int:
        return self.get_metadata(PositionProvider, node).start.line

    def leave_SimpleStatementLine(
        self, original: cst.SimpleStatementLine, updated: cst.SimpleStatementLine
    ) -> cst.SimpleStatementLine:
        if len(updated.body) != 1 or not isinstance(updated.body[0], cst.Expr):
            return updated
        call = updated.body[0].value
        if not isinstance(call, cst.Call) or not isinstance(call.func, cst.Name):
            return updated
        builtin = call.func.value
        if builtin not in {"setattr", "delattr"} or builtin in self.shadowed:
            return updated
        arguments = _arguments(call)
        if not arguments or not _plain_base(arguments[0]):
            return updated
        line = self._line(original)
        if builtin == "setattr" and len(arguments) == 3:
            name = _literal_name(arguments[1])
            if name is None:
                return updated
            target = cst.Attribute(value=arguments[0], attr=cst.Name(name))
            self.rewrites.append(
                Rewrite(line, "literal-attributes", f"`setattr` became `.{name} = ...`")
            )
            return updated.with_changes(
                body=[cst.Assign(targets=[cst.AssignTarget(target)], value=arguments[2])]
            )
        if builtin == "delattr" and len(arguments) == 2:
            name = _literal_name(arguments[1])
            if name is None:
                return updated
            target = cst.Attribute(value=arguments[0], attr=cst.Name(name))
            self.rewrites.append(
                Rewrite(line, "literal-attributes", f"`delattr` became `del .{name}`")
            )
            return updated.with_changes(body=[cst.Del(target=target)])
        return updated

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.BaseExpression:
        if not isinstance(updated.func, cst.Name) or updated.func.value != "getattr":
            return updated
        if "getattr" in self.shadowed:
            return updated
        arguments = _arguments(updated)
        # Two arguments exactly: a default (the third) changes behavior when
        # the attribute is missing, and that is not a spelling difference.
        if arguments is None or len(arguments) != 2 or not _plain_base(arguments[0]):
            return updated
        name = _literal_name(arguments[1])
        if name is None:
            return updated
        self.rewrites.append(
            Rewrite(self._line(original), "literal-attributes", f"`getattr` became `.{name}`")
        )
        return cst.Attribute(value=arguments[0], attr=cst.Name(name))


class LiteralAttributes:
    """`setattr`/`delattr`/`getattr` with constant names become plain access."""

    name = "literal-attributes"

    def apply(self, module: cst.Module, source: str) -> tuple[cst.Module, list[Rewrite]]:
        rewriter = _AttributeRewriter(bound_names(source) & {"setattr", "delattr", "getattr"})
        rewritten = MetadataWrapper(module, unsafe_skip_copy=True).visit(rewriter)
        return rewritten, rewriter.rewrites


def _only_plainly_imported(source: str, name: str) -> bool:
    """`name` is bound by `import name` statements and nothing else."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.asname for alias in node.names if alias.name == name):
                return False
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if node.id == name:
                return False
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
        ):
            return False
    return True


def _module_string(node: cst.BaseExpression) -> str | None:
    if not isinstance(node, cst.SimpleString):
        return None
    value = node.evaluated_value
    if not isinstance(value, str) or not value:
        return None
    parts = value.split(".")
    if all(part.isidentifier() and not keyword.iskeyword(part) for part in parts):
        return value
    return None


class _ImportRewriter(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self) -> None:
        self.rewrites: list[Rewrite] = []

    def leave_SimpleStatementLine(
        self, original: cst.SimpleStatementLine, updated: cst.SimpleStatementLine
    ) -> cst.SimpleStatementLine:
        if len(updated.body) != 1 or not isinstance(updated.body[0], cst.Assign):
            return updated
        assign = updated.body[0]
        if len(assign.targets) != 1 or not isinstance(assign.targets[0].target, cst.Name):
            return updated
        call = assign.value
        if not isinstance(call, cst.Call) or not isinstance(call.func, cst.Attribute):
            return updated
        func = call.func
        if not isinstance(func.value, cst.Name) or func.value.value != "importlib":
            return updated
        if func.attr.value != "import_module":
            return updated
        arguments = _arguments(call)
        if arguments is None or len(arguments) != 1:
            return updated
        module_name = _module_string(arguments[0])
        if module_name is None:
            return updated
        alias = assign.targets[0].target.value
        line = self.get_metadata(PositionProvider, original).start.line
        spelled = (
            f"import {module_name}" if alias == module_name else f"import {module_name} as {alias}"
        )
        self.rewrites.append(
            Rewrite(
                line,
                "static-imports",
                f"`importlib.import_module({module_name!r})` became `{spelled}`",
            )
        )
        asname = None if alias == module_name else cst.AsName(cst.Name(alias))
        return updated.with_changes(
            body=[cst.Import(names=[cst.ImportAlias(name=_dotted_cst(module_name), asname=asname)])]
        )


def _dotted_cst(dotted: str) -> cst.Name | cst.Attribute:
    parts = dotted.split(".")
    node: cst.Name | cst.Attribute = cst.Name(parts[0])
    for part in parts[1:]:
        node = cst.Attribute(value=node, attr=cst.Name(part))
    return node


class _NameCounter(cst.CSTVisitor):
    def __init__(self, name: str) -> None:
        self.name = name
        self.count = 0

    def visit_Name(self, node: cst.Name) -> None:
        if node.value == self.name:
            self.count += 1


class _ImportDropper(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self) -> None:
        self.rewrites: list[Rewrite] = []

    def leave_SimpleStatementLine(
        self, original: cst.SimpleStatementLine, updated: cst.SimpleStatementLine
    ) -> cst.SimpleStatementLine | cst.RemovalSentinel:
        if len(updated.body) != 1 or not isinstance(updated.body[0], cst.Import):
            return updated
        names = updated.body[0].names
        if len(names) != 1 or names[0].asname is not None:
            return updated
        if not isinstance(names[0].name, cst.Name) or names[0].name.value != "importlib":
            return updated
        line = self.get_metadata(PositionProvider, original).start.line
        self.rewrites.append(
            Rewrite(line, "static-imports", "`import importlib` is no longer needed")
        )
        return cst.RemoveFromParent()


class StaticImports:
    """`m = importlib.import_module("pkg.mod")` becomes `import pkg.mod as m`."""

    name = "static-imports"

    def apply(self, module: cst.Module, source: str) -> tuple[cst.Module, list[Rewrite]]:
        if not _only_plainly_imported(source, "importlib"):
            return module, []
        rewriter = _ImportRewriter()
        rewritten = MetadataWrapper(module, unsafe_skip_copy=True).visit(rewriter)
        if not rewriter.rewrites:
            return rewritten, rewriter.rewrites
        # When the rewrites consumed every use, the import that fed them is
        # freight with no cargo; a module the interpreter has loaded anyway
        # imports the same with or without it.
        counter = _NameCounter("importlib")
        rewritten.visit(counter)
        if counter.count == 1:
            dropper = _ImportDropper()
            rewritten = MetadataWrapper(rewritten, unsafe_skip_copy=True).visit(dropper)
            rewriter.rewrites.extend(dropper.rewrites)
        return rewritten, rewriter.rewrites
