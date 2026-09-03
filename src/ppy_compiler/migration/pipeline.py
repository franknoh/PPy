"""The migration pass framework: match, rewrite, and account for it.

A pass is deterministic and semantics-preserving on its own -- each rewrites
only what it can prove equivalent, and reports every site it touched. The
pipeline runs them in a fixed order over libcst so untouched source keeps its
bytes, which is what lets `examples/verify_conversions.py` hold `ppy migrate`
to the same regeneration standard as `ppy convert`.
"""

from __future__ import annotations

import ast
from typing import Protocol

import libcst as cst

from .report import Rewrite

__all__ = ["MigrationPass", "apply_passes", "bound_names"]


class MigrationPass(Protocol):
    """One automatic migration rewrite."""

    name: str

    def apply(self, module: cst.Module, source: str) -> tuple[cst.Module, list[Rewrite]]:
        """Rewrite what this pass owns; report each site changed."""


def bound_names(source: str) -> frozenset[str]:
    """Every name the module binds anywhere, in any scope.

    The passes rewrite calls to builtins (`setattr`, `globals`,
    `importlib.import_module`); a module that binds any of those names
    anywhere -- an assignment, a parameter, an import alias -- may not mean
    the builtin, and the pass that cannot tell stays its hand entirely.
    """
    found: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            found.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = node.args
                for arg in [
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                    *([arguments.vararg] if arguments.vararg else []),
                    *([arguments.kwarg] if arguments.kwarg else []),
                ]:
                    found.add(arg.arg)
        elif isinstance(node, ast.alias):
            found.add((node.asname or node.name).partition(".")[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            found.update(node.names)
    return frozenset(found)


#: What each pass rewrites, as the text it must find to have anything to do.
#: A pass walks the whole tree with metadata resolved, which is most of a
#: migration's time on a large module; a module that never spells the name
#: cannot contain the call, so the walk is skipped. Aliases still spell the
#: name at their definition (`from importlib import import_module as imp`).
_TRIGGERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("literal-attributes", ("getattr", "setattr", "delattr")),
    ("static-imports", ("import_module",)),
    ("module-namespace-writes", ("globals",)),
)


def passes_for(source: str) -> list:  # type: ignore[type-arg]
    """The migration passes that could rewrite something in this source."""
    from .dynamic import LiteralAttributes, StaticImports
    from .globals import ModuleNamespaceWrites

    available = {
        "literal-attributes": LiteralAttributes,
        "static-imports": StaticImports,
        "module-namespace-writes": ModuleNamespaceWrites,
    }
    return [
        available[name]()
        for name, spellings in _TRIGGERS
        if any(spelling in source for spelling in spellings)
    ]


def apply_passes(source: str) -> tuple[str, list[Rewrite]]:
    """Run every migration pass that could apply over one module's source."""
    passes = passes_for(source)
    if not passes:
        return source, []
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError:
        # The frontend will report the syntax error with a real diagnostic.
        return source, []
    rewrites: list[Rewrite] = []
    for migration_pass in passes:
        module, found = migration_pass.apply(module, source)
        rewrites.extend(found)
        if found:
            source = module.code
    return module.code, rewrites
