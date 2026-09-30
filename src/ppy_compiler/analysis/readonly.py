"""Does a function only read a parameter?

`def bucket_sort(my_list: list[int | float])` that never writes `my_list`
can be handed a `list[int]`: every value it reads is one the declaration
allows. A function that appends a float to it cannot. The answer comes from
the function's syntax alone, so it is the same on every pass of the checker:
each use of the parameter must be one that reads the object and keeps no
other reference to it.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .symbols import FunctionInfo

__all__ = ["reads_only"]

#: Builtins that read an argument and keep nothing of it.
_READING_CALLS = frozenset(
    {
        "len",
        "sum",
        "min",
        "max",
        "sorted",
        "any",
        "all",
        "list",
        "tuple",
        "set",
        "frozenset",
        "dict",
        "str",
        "repr",
        "print",
        "bool",
        "abs",
        "round",
        "enumerate",
        "zip",
        "reversed",
        "iter",
        "map",
        "filter",
        "isinstance",
        "id",
        "hash",
    }
)

#: Methods that read a container.
_READING_METHODS = frozenset(
    {"count", "index", "copy", "get", "keys", "values", "items", "issubset", "issuperset"}
)


def reads_only(info: FunctionInfo, name: str, *, nested: bool = False) -> bool:
    """Does `info`'s body use parameter `name` only to read it?

    `nested` says the elements are containers too (`list[list[int]]`): then
    an element the body takes out must only be read as well."""
    return _reads_only(info.node, name, nested)


@lru_cache(maxsize=4096)
def _reads_only(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str, nested: bool) -> bool:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    for found in ast.walk(node):
        if isinstance(found, (ast.Global, ast.Nonlocal)) and name in found.names:
            return False
        if not (isinstance(found, ast.Name) and found.id == name):
            continue
        if isinstance(found.ctx, ast.Store):
            continue  # `arr = sorted(arr)` rebinds the name, not the object
        if isinstance(found.ctx, ast.Del):
            return False
        if not _reading_use(node, found, parents, nested):
            return False
    return True


def _reading_use(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    use: ast.expr,
    parents: dict[int, ast.AST],
    nested: bool,
) -> bool:
    """Is `use`, the container or (when `nested`) one of its elements, only read here?"""
    parent = parents.get(id(use))
    if isinstance(parent, ast.Subscript) and parent.value is use:
        # `xs[i]` reads; `xs[i] = v` and `del xs[i]` write.
        if not isinstance(parent.ctx, ast.Load):
            return False
        return not nested or _reading_use(function, parent, parents, False)
    if isinstance(parent, (ast.For, ast.AsyncFor, ast.comprehension)) and parent.iter is use:
        if not nested:
            return True
        # `for row in grid`: each `row` is an element, read or not.
        return isinstance(parent.target, ast.Name) and _reads_only(
            function, parent.target.id, False
        )
    if isinstance(parent, (ast.Compare, ast.BinOp, ast.FormattedValue)):
        return True
    if isinstance(parent, ast.Return):
        # `return my_list` hands the caller back what it passed in; the
        # caller's own type for it is the narrower, true one.
        return not nested
    if isinstance(parent, (ast.BoolOp, ast.UnaryOp, ast.If, ast.While, ast.Assert)):
        return True
    if isinstance(parent, ast.IfExp):
        return parent.test is use
    if isinstance(parent, ast.Call) and use in parent.args:
        return isinstance(parent.func, ast.Name) and parent.func.id in _READING_CALLS
    if isinstance(parent, ast.keyword):
        call = parents.get(id(parent))
        return (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in _READING_CALLS
        )
    if isinstance(parent, ast.Attribute) and parent.value is use:
        call = parents.get(id(parent))
        return (
            isinstance(call, ast.Call) and call.func is parent and parent.attr in _READING_METHODS
        )
    return False
