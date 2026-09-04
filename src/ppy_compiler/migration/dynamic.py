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
from collections.abc import Sequence

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from .pipeline import bound_names
from .report import Rewrite, rewrites_at

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
    def __init__(self, shadowed: frozenset[str]) -> None:
        self.shadowed = shadowed
        self.pending: list[tuple[cst.CSTNode, str, str]] = []

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
        if builtin == "setattr" and len(arguments) == 3:
            name = _literal_name(arguments[1])
            if name is None:
                return updated
            target = cst.Attribute(value=arguments[0], attr=cst.Name(name))
            self.pending.append(
                (original, "literal-attributes", f"`setattr` became `.{name} = ...`")
            )
            return updated.with_changes(
                body=[cst.Assign(targets=[cst.AssignTarget(target)], value=arguments[2])]
            )
        if builtin == "delattr" and len(arguments) == 2:
            name = _literal_name(arguments[1])
            if name is None:
                return updated
            target = cst.Attribute(value=arguments[0], attr=cst.Name(name))
            self.pending.append((original, "literal-attributes", f"`delattr` became `del .{name}`"))
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
        self.pending.append((original, "literal-attributes", f"`getattr` became `.{name}`"))
        return cst.Attribute(value=arguments[0], attr=cst.Name(name))


class LiteralAttributes:
    """`setattr`/`delattr`/`getattr` with constant names become plain access."""

    name = "literal-attributes"

    def apply(self, module: cst.Module, source: str) -> tuple[cst.Module, list[Rewrite]]:
        rewriter = _AttributeRewriter(bound_names(source) & {"setattr", "delattr", "getattr"})
        rewritten = module.visit(rewriter)
        return rewritten, rewrites_at(module, rewriter.pending)


def _module_string(value: object) -> str | None:
    if not isinstance(value, str) or not value or value.startswith("."):
        return None
    parts = value.split(".")
    if all(part.isidentifier() and not keyword.iskeyword(part) for part in parts):
        return value
    return None


def _import_sites(source: str) -> dict[int, tuple[str, str, str]]:
    """line -> (alias, module name, feeder name) for each static import call.

    The callee is resolved through the shared lexical bindings, so
    `importlib.import_module`, `il.import_module`, and a `from importlib
    import import_module as imp` alias are all one thing -- and a local
    function that happens to share the name is not.
    """
    from ..analysis.lexical import scan_module

    tree = ast.parse(source)
    bindings = scan_module(tree, "<migration>")
    sites: dict[int, tuple[str, str, str]] = {}
    for statement in ast.walk(tree):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        value = statement.value
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            continue
        if bindings.targets_at(value.func) != frozenset({"importlib.import_module"}):
            continue
        if len(value.args) != 1 or value.keywords:
            continue
        argument = value.args[0]
        if not isinstance(argument, ast.Constant):
            continue
        module_name = _module_string(argument.value)
        if module_name is None:
            continue
        probe: ast.expr = value.func
        while isinstance(probe, ast.Attribute):
            probe = probe.value
        if not isinstance(probe, ast.Name):
            continue
        sites[statement.lineno] = (target.id, module_name, probe.id)
    return sites


class _ImportRewriter(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, sites: dict[int, tuple[str, str, str]]) -> None:
        self.sites = sites
        self.rewrites: list[Rewrite] = []
        self.feeders: set[str] = set()

    def leave_SimpleStatementLine(
        self, original: cst.SimpleStatementLine, updated: cst.SimpleStatementLine
    ) -> cst.SimpleStatementLine:
        line = self.get_metadata(PositionProvider, original).start.line
        site = self.sites.get(line)
        if site is None or len(updated.body) != 1 or not isinstance(updated.body[0], cst.Assign):
            return updated
        alias, module_name, feeder = site
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
        self.feeders.add(feeder)
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


class _FeederReferences(cst.CSTVisitor):
    """Count uses of a name outside import statements."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.count = 0
        self._importing = 0

    def visit_Import(self, node: cst.Import) -> bool:
        self._importing += 1
        return True

    def leave_Import(self, original: cst.Import) -> None:
        self._importing -= 1

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        self._importing += 1
        return True

    def leave_ImportFrom(self, original: cst.ImportFrom) -> None:
        self._importing -= 1

    def visit_Name(self, node: cst.Name) -> None:
        if not self._importing and node.value == self.name:
            self.count += 1


class _FeederDropper(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, feeders: frozenset[str]) -> None:
        self.feeders = feeders
        self.rewrites: list[Rewrite] = []

    def leave_SimpleStatementLine(
        self, original: cst.SimpleStatementLine, updated: cst.SimpleStatementLine
    ) -> cst.SimpleStatementLine | cst.RemovalSentinel:
        if len(updated.body) != 1:
            return updated
        statement = updated.body[0]
        spelled = self._feeding_import(statement)
        if spelled is None:
            return updated
        line = self.get_metadata(PositionProvider, original).start.line
        self.rewrites.append(Rewrite(line, "static-imports", f"`{spelled}` is no longer needed"))
        return cst.RemoveFromParent()

    def _feeding_import(self, statement: cst.BaseSmallStatement) -> str | None:
        if isinstance(statement, cst.Import) and len(statement.names) == 1:
            imported = statement.names[0]
            if not isinstance(imported.name, cst.Name) or imported.name.value != "importlib":
                return None
            bound = imported.asname.name.value if imported.asname else "importlib"  # type: ignore[union-attr]
            if bound in self.feeders:
                return f"import importlib as {bound}" if imported.asname else "import importlib"
            return None
        if isinstance(statement, cst.ImportFrom) and isinstance(statement.names, Sequence):
            if not isinstance(statement.module, cst.Name) or statement.module.value != "importlib":
                return None
            if len(statement.names) != 1:
                return None
            imported = statement.names[0]
            if not isinstance(imported.name, cst.Name) or imported.name.value != "import_module":
                return None
            bound = imported.asname.name.value if imported.asname else "import_module"  # type: ignore[union-attr]
            if bound in self.feeders:
                suffix = f" as {bound}" if imported.asname else ""
                return f"from importlib import import_module{suffix}"
        return None


class StaticImports:
    """`m = importlib.import_module("pkg.mod")` becomes `import pkg.mod as m`.

    Every spelling the lexical bindings resolve to importlib's importer is
    caught, and an import that fed only rewritten calls is removed with them.
    """

    name = "static-imports"

    def apply(self, module: cst.Module, source: str) -> tuple[cst.Module, list[Rewrite]]:
        sites = _import_sites(source)
        if not sites:
            return module, []
        rewriter = _ImportRewriter(sites)
        rewritten = MetadataWrapper(module, unsafe_skip_copy=True).visit(rewriter)
        if not rewriter.rewrites:
            return rewritten, rewriter.rewrites
        # An import that fed only the rewritten calls is freight with no
        # cargo; a module the interpreter has loaded anyway imports the same
        # with or without it.
        unused = frozenset(
            feeder
            for feeder in rewriter.feeders
            if _references_outside_imports(rewritten, feeder) == 0
        )
        if unused:
            dropper = _FeederDropper(unused)
            rewritten = MetadataWrapper(rewritten, unsafe_skip_copy=True).visit(dropper)
            rewriter.rewrites.extend(dropper.rewrites)
        return rewritten, rewriter.rewrites


def _references_outside_imports(module: cst.Module, name: str) -> int:
    counter = _FeederReferences(name)
    module.visit(counter)
    return counter.count
