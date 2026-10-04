"""Which settled module globals native code passes each function.

A settled global (`ModuleSymbols.settled_globals`) is bound once by the
module's body and never rebound, so the object a name refers to when a
function is called is the one it will refer to throughout the call. Native
code takes each one the function reads as one more parameter: Python's
boundary reads the global when it calls the function, and a native caller
passes the one it was given. A caller therefore takes every global its native
callees read, and this pass closes each function's set over the calls it
makes.
"""

from __future__ import annotations

import ast

from . import types as T
from .closures import cell_captures
from .results import FunctionAnalysis, ImplicitGlobal, ProjectAnalysis

__all__ = ["cell_scope", "close_settled_globals", "implicit_name", "implicit_parameter_name"]


def implicit_name(module: str, name: str) -> str:
    """The parameter a global the function does not read by name is passed as."""
    return f"__global_{module.replace('.', '_')}_{name}"


def cell_scope(enclosing: str) -> str:
    """The `module` of an `ImplicitGlobal` that is a variable of the function
    `enclosing`, read from the cell of the function object Python calls."""
    return f"{enclosing}.<locals>"


def _plain(analysis: FunctionAnalysis, held: dict[tuple[str, str], ImplicitGlobal]) -> bool:
    """A function of the module that can take more parameters than it spells:
    not a method, a closure, a generic, a generator, or a coroutine, whose
    callers do not pass arguments the way a native call to a function does.

    A function defined in another one is, where all it takes are the cells
    of that one: Python's boundary reads them from the function object it
    calls, and its one native caller is itself."""
    info = analysis.info
    if info.type_params or info.is_generator or info.is_async or info.owner is not None:
        return False
    if info.enclosing is None:
        return True
    scope = cell_scope(info.enclosing)
    return all(module == scope for module, _name in held)


def _cells(
    analysis: FunctionAnalysis, functions: dict[str, FunctionAnalysis]
) -> dict[tuple[str, str], ImplicitGlobal] | None:
    """The variables of the function around `analysis` its entry is handed
    (`closures.cell_captures`), or None where one has no type to pass."""
    info = analysis.info
    outer = functions.get(info.enclosing or "")
    node = info.node
    if outer is None or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {}
    around = outer.info.node
    if not isinstance(around, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {}
    scope = cell_scope(outer.info.qualname)
    found: dict[tuple[str, str], ImplicitGlobal] = {}
    params = {p.name: p.type for p in outer.info.params}
    for name in sorted(cell_captures(node, around) - {info.name}):
        typed = outer.locals.get(name, params.get(name))
        if typed is None:
            return None
        written = not _immutable(typed) and not _only_read(node, name)
        found[(scope, name)] = ImplicitGlobal(scope, name, typed, written)
    return found


#: Builtins that read a collection they are given and keep nothing of it.
_READERS = frozenset(
    {
        "len", "sum", "min", "max", "sorted", "any", "all", "list", "tuple", "set",
        "frozenset", "dict", "enumerate", "zip", "reversed", "str", "repr", "print",
        "abs", "bool", "int", "float", "isinstance",
    }
)  # fmt: skip


def _immutable(t: T.Type) -> bool:
    """A value no one can change in place: a number, a string, None, or a tuple of them."""
    base = T.strip_literal(t)
    if base in (T.INT, T.FLOAT, T.BOOL, T.STR, T.NONE):
        return True
    if isinstance(base, T.Tuple_):
        return all(_immutable(item) for item in base.items)
    return False


def _only_read(node: ast.AST, name: str) -> bool:
    """Whether every use of `name` in `node` only reads what it holds: an item
    read, a comparison, a loop over it, or a builtin that reads it. Anything
    else (a method call, an item stored, handing it on) may write it."""
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    for use in ast.walk(node):
        if not (isinstance(use, ast.Name) and use.id == name):
            continue
        parent = parents.get(id(use))
        if isinstance(parent, ast.Subscript) and parent.value is use:
            if isinstance(parent.ctx, ast.Load):
                continue
            return False
        if isinstance(parent, ast.Compare):
            continue
        if isinstance(parent, (ast.For, ast.comprehension)) and parent.iter is use:
            continue
        if (
            isinstance(parent, ast.Call)
            and use in parent.args
            and isinstance(parent.func, ast.Name)
            and parent.func.id in _READERS
        ):
            continue
        return False
    return True


def close_settled_globals(project: ProjectAnalysis) -> None:
    """Fill `implicit_globals` and `globals_native` for every function."""
    functions: dict[str, FunctionAnalysis] = {}
    for module in project.modules.values():
        functions.update(module.functions)
    found: dict[str, dict[tuple[str, str], ImplicitGlobal]] = {}
    native: dict[str, bool] = {}
    for qualname, analysis in functions.items():
        module = analysis.info.module
        writes = analysis.mutated_params | analysis.delegated_writes
        found[qualname] = {
            (module, name): ImplicitGlobal(module, name, kind, name in writes)
            for name, kind in analysis.settled_globals.items()
        }
        native[qualname] = not analysis.unsettled_global
        if analysis.info.enclosing is not None:
            cells = _cells(analysis, functions)
            if cells is None:
                native[qualname] = False
            else:
                found[qualname].update(cells)
    changed = True
    while changed:
        changed = False
        for qualname, analysis in functions.items():
            own = found[qualname]
            ok = native[qualname]
            for callee in analysis.calls:
                if callee == qualname or callee not in found:
                    continue
                enclosing = functions[callee].info.enclosing
                scope = cell_scope(enclosing) if enclosing is not None else None
                if enclosing != qualname and any(
                    module == scope for module, _name in found[callee]
                ):
                    # A function given cells: the one it is defined in calls
                    # it as the closure it is there; any other caller has no
                    # cells to give it.
                    ok = False
                    continue
                ok = ok and native[callee]
                for key, wanted in found[callee].items():
                    if key[0] == scope:
                        continue
                    held = own.get(key)
                    if held is None or (wanted.written and not held.written):
                        own[key] = ImplicitGlobal(
                            wanted.module,
                            wanted.name,
                            wanted.type,
                            wanted.written or (held is not None and held.written),
                        )
                        changed = True
            if own and not _plain(analysis, own):
                ok = False
            if ok != native[qualname]:
                native[qualname] = ok
                changed = True
    for qualname, analysis in functions.items():
        analysis.implicit_globals = tuple(found[qualname][key] for key in sorted(found[qualname]))
        analysis.globals_native = native[qualname]


def implicit_parameter_name(analysis: FunctionAnalysis, held: ImplicitGlobal) -> str:
    """The name `held` has inside the function: the global's own when the body
    reads it by that name, a spelled-out one when only a callee does."""
    if held.module == analysis.info.module and held.name in analysis.settled_globals:
        return held.name
    if analysis.info.enclosing is not None and held.module == cell_scope(analysis.info.enclosing):
        return held.name
    return implicit_name(held.module, held.name)
