"""Which names a nested function or a lambda shares with the function it is in.

CPython's rule, which native code follows: a name a nested scope uses and does
not bind itself (or declares `nonlocal`) is the enclosing function's variable,
seen as it is when the nested code runs, not when it was made. The enclosing
function keeps such a variable in a cell both sides read and write.

A comprehension is a scope of its own for its targets and nothing else here:
what its body reads of the function is read in place, which is how both the
checker and the lowering treat it.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

from . import types as T

if TYPE_CHECKING:
    from .results import FunctionAnalysis
    from .symbols import FunctionInfo

__all__ = [
    "Scope",
    "callable_spelled",
    "captured_names",
    "cell_captures",
    "closure_nodes",
    "free_names",
    "is_plain_callable",
    "own_names",
    "rebound_by_closures",
    "shared_with_closures",
]

Scope = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda


def _parameters(node: Scope) -> set[str]:
    args = node.args
    names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


def _scope_nodes(node: Scope) -> Iterator[ast.AST]:
    """Every node of `node`'s own scope: not inside a nested function, lambda,
    or class (their names are their own), but inside comprehensions."""
    body = [node.body] if isinstance(node, ast.Lambda) else list(node.body)
    pending: list[ast.AST] = list(body)
    while pending:
        current = pending.pop()
        yield current
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Its decorators and defaults run here; its body does not.
            pending.extend(getattr(current, "decorator_list", []))
            if not isinstance(current, ast.ClassDef):
                pending.extend(current.args.defaults)
                pending.extend(d for d in current.args.kw_defaults if d is not None)
            continue
        if isinstance(current, ast.Lambda):
            pending.extend(current.args.defaults)
            continue
        pending.extend(ast.iter_child_nodes(current))


def _declared(node: Scope, kind: type) -> set[str]:
    return {
        name
        for child in _scope_nodes(node)
        if isinstance(child, kind)
        for name in child.names  # type: ignore[attr-defined]
    }


def _comprehension_targets(node: Scope) -> set[str]:
    found: set[str] = set()
    for child in _scope_nodes(node):
        if isinstance(child, ast.comprehension):
            found |= {n.id for n in ast.walk(child.target) if isinstance(n, ast.Name)}
    return found


def own_names(node: Scope) -> set[str]:
    """The names `node` binds for itself: its parameters, what it assigns,
    deletes, imports, or defines, less what it declares `global` or `nonlocal`."""
    bound = _parameters(node)
    targets = _comprehension_targets(node)
    for child in _scope_nodes(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            if child.id not in targets:
                bound.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(child, ast.ExceptHandler) and child.name:
            bound.add(child.name)
    return bound - _declared(node, ast.Global) - _declared(node, ast.Nonlocal)


def free_names(node: Scope) -> set[str]:
    """The names `node` uses and does not bind: its own reads and `nonlocal`
    writes, and what its nested functions and lambdas use of the same kind.
    Which of them are the enclosing function's, and which are globals or
    builtins, the caller decides."""
    own = own_names(node)
    used: set[str] = set(_declared(node, ast.Nonlocal))
    targets = _comprehension_targets(node)
    for child in _scope_nodes(node):
        if isinstance(child, ast.Name) and child.id not in targets:
            used.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            used |= free_names(child)
    return used - own - _declared(node, ast.Global)


#: Calls whose `key=` lambda is lowered in place, its body run per element.
_INLINE_KEYS = frozenset({"sorted", "min", "max"})


def _inline_key(call: ast.Call, lambda_: ast.Lambda) -> bool:
    if not any(k.arg == "key" and k.value is lambda_ for k in call.keywords):
        return False
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in _INLINE_KEYS
    return isinstance(func, ast.Attribute) and func.attr == "sort"


def closure_nodes(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Scope]:
    """The nested functions and lambdas directly in `node`'s scope that become
    closures: every one but a sort key's lambda, whose body runs in place."""
    inline: set[int] = set()
    found: list[Scope] = []
    pending: list[ast.AST] = list(node.body)
    while pending:
        current = pending.pop()
        if isinstance(current, ast.Call):
            for keyword in current.keywords:
                if isinstance(keyword.value, ast.Lambda) and _inline_key(current, keyword.value):
                    inline.add(id(keyword.value))
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append(current)
            pending.extend(current.decorator_list)
            pending.extend(current.args.defaults)
            continue
        if isinstance(current, ast.Lambda):
            if id(current) not in inline:
                found.append(current)
            pending.extend(current.args.defaults)
            if id(current) in inline:
                pending.append(current.body)
            continue
        if isinstance(current, ast.ClassDef):
            continue
        pending.extend(ast.iter_child_nodes(current))
    return found


def shared_with_closures(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """The names of `node` that the closures it makes share with it."""
    shared: set[str] = set()
    for child in closure_nodes(node):
        shared |= free_names(child)
    return shared & own_names(node)


def cell_captures(node: Scope, outer: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """The variables of `outer` that `node`, a function defined directly in it,
    reads and that no one rebinds while `node` runs: `node` does not, no
    function nested in `outer` declares them `nonlocal`, and they are not a
    function or class `outer` defines. A call of `node` may be handed each one
    as it is in its cell when the call starts, and see what reading the cell
    would have shown for the whole call: the frame of `outer` is waiting on
    the call, and nothing else binds the name."""
    shared = free_names(node) & own_names(outer)
    defined = {
        child.name
        for child in _scope_nodes(outer)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    return shared - defined - rebound_by_closures(outer)


def rebound_by_closures(node: Scope) -> set[str]:
    """The names of `node` a function nested in it rebinds with `nonlocal`: a
    call can change them, so what `node` knows of their values does not last."""
    found: set[str] = set()
    for child in ast.walk(node):
        if child is not node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found |= _declared(child, ast.Nonlocal)
    return found & own_names(node)


def captured_names(
    info: FunctionInfo, analyses: Mapping[str, FunctionAnalysis]
) -> dict[str, T.Type]:
    """The names a nested function shares with the function it is in, and the
    types that function gave them."""
    outer = analyses.get(info.enclosing) if info.enclosing else None
    if outer is None:
        return {}
    known = outer.locals
    return {name: known[name] for name in sorted(free_names(info.node)) if name in known}


def is_plain_callable(t: T.Type) -> bool:
    """A function value native code can hold: called with positional arguments
    only, neither a coroutine nor a generator."""
    base = T.strip_literal(t)
    return (
        isinstance(base, T.Callable_)
        and not base.is_async
        and not base.is_generator
        and all(p.kind in {"positional_only", "positional_or_keyword"} for p in base.params)
    )


def callable_spelled(t: T.Callable_) -> str:
    """A function value's type written out, as a parameter's element spells it:
    `Callable[[int, float], bool]`. Two values are passed for one another only
    when these agree, which is when their native calls agree."""
    params = ", ".join(str(p.type) for p in t.params)
    return f"Callable[[{params}], {t.ret}]"
