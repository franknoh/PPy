"""Plugin-driven source rewrites shared by both backends (spec 15.4, 22.2).

A framework integration often needs the call itself adjusted rather than
replaced -- resolving an application by name instead of by import string, or
teaching a reloader which files to watch.
"""

from __future__ import annotations

import ast

from ..analysis.symbols import ModuleSymbols, ProjectSymbols
from ..plugins.base import CallAdjustment, PluginRegistry

__all__ = ["AdjustPlan", "find_adjustments", "adjustments_for_project"]

#: Positions, per module, of calls whose arguments a plugin rewrites.
AdjustPlan = dict[tuple[int, int], CallAdjustment]


def find_adjustments(
    symbols: ModuleSymbols,
    project: ProjectSymbols,
    plugins: PluginRegistry,
) -> AdjustPlan:
    """Ask each plugin which calls in this module need argument rewriting."""
    plan: AdjustPlan = {}
    resolver = project.resolver(symbols)
    for node in ast.walk(symbols.module.tree):
        if not isinstance(node, ast.Call):
            continue
        qualname = resolver.canonical(node.func)
        if qualname is None:
            continue
        plugin = plugins.for_qualname(qualname)
        if plugin is None:
            continue
        adjust = getattr(plugin, "adjust_call", None)
        if adjust is None:
            continue
        found = adjust(qualname, node, symbols)
        if found is not None:
            plan[(node.lineno, node.col_offset)] = found
    return plan


def adjustments_for_project(bundle) -> dict[str, AdjustPlan]:  # type: ignore[no-untyped-def]
    plans: dict[str, AdjustPlan] = {}
    for name, symbols in bundle.symbols.modules.items():
        found = find_adjustments(symbols, bundle.symbols, bundle.project.plugins)
        if found:
            plans[name] = found
    return plans
