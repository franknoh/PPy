"""Attach analysis results to AST nodes so passes can consume them."""

from __future__ import annotations

import ast
import copy

from ..analysis import types as T
from ..analysis.checker import ModuleAnalysis

__all__ = ["annotate", "const_of", "type_of", "has_const", "PURE_NODES"]

_CONST_ATTR = "_ppy_const"
_HAS_CONST_ATTR = "_ppy_has_const"
_TYPE_ATTR = "_ppy_type"

#: Constants that are safe to materialize in generated source.
_FOLDABLE = (int, float, complex, str, bytes, bool, type(None))

#: Node types whose evaluation cannot have an observable effect on its own.
PURE_NODES = (
    ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Tuple, ast.List, ast.Set, ast.Dict, ast.IfExp, ast.Slice, ast.Starred,
    ast.JoinedStr, ast.FormattedValue,
)


def annotate(tree: ast.Module, analysis: ModuleAnalysis) -> ast.Module:
    """Copy the tree with per-node types and constants attached."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.expr):
            continue
        facts = analysis.node_facts.get(id(node))
        node_type = analysis.node_types.get(id(node))
        if node_type is not None:
            setattr(node, _TYPE_ATTR, node_type)
        if facts is not None and facts.has_constant and isinstance(facts.constant, _FOLDABLE):
            setattr(node, _CONST_ATTR, facts.constant)
            setattr(node, _HAS_CONST_ATTR, True)
    return copy.deepcopy(tree)


def const_of(node: ast.AST) -> object:
    return getattr(node, _CONST_ATTR, None)


def has_const(node: ast.AST) -> bool:
    return bool(getattr(node, _HAS_CONST_ATTR, False))


def type_of(node: ast.AST) -> T.Type:
    return getattr(node, _TYPE_ATTR, T.UNKNOWN)
