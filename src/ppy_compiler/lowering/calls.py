"""Calls between native functions spelled with keywords or with defaults left out.

A native function takes its arguments in order: there is no keyword in its
ABI. A call that names an argument, or leaves one to its default, is bound
here, at compile time, the way Python binds it: each keyword goes to the
parameter it names, and each parameter left out takes its default. The
call then goes on as a positional one.

Python evaluates the arguments in the order they are written, before it
binds any. Binding moves a keyword argument to its parameter's place, so
a keyword argument may move only where its order cannot be seen: when at
most one of the keywords runs any code (the others are names, constants, or
attributes of names).
A default is evaluated once, when Python runs the `def`. Put in at the call
it is the same value only if it is a constant (a number, a string, `None`,
a tuple of those), so a default of any other kind keeps the call as it is.
"""

from __future__ import annotations

import ast

from ..analysis import types as T
from ..analysis.closures import free_names, own_names
from ..analysis.symbols import FunctionInfo, ParamInfo
from ..backend.llvm.lowering import Unsupported

__all__ = ["CallBinding", "constant_default", "nested_entry_refusal"]

#: Parameter kinds a call binds by name or position.
_BINDABLE = frozenset({"positional_only", "positional_or_keyword", "keyword_only"})


def constant_default(node: ast.expr | None) -> bool:
    """Whether a default is a value the call can spell again: a constant, a
    signed number, or a tuple of those."""
    if isinstance(node, ast.Constant):
        return not isinstance(node.value, type(...))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return isinstance(node.operand, ast.Constant) and isinstance(
            node.operand.value, (int, float, complex)
        )
    if isinstance(node, ast.Tuple):
        return all(constant_default(item) for item in node.elts)
    return False


def _constant_type(node: ast.expr) -> T.Type:
    """What the checker would say of a constant default."""
    if isinstance(node, ast.Tuple):
        return T.Tuple_(tuple(_constant_type(item) for item in node.elts))
    value = node.operand.value if isinstance(node, ast.UnaryOp) else node.value  # type: ignore[attr-defined]
    if value is None:
        return T.NONE
    named = {bool: T.BOOL, int: T.INT, float: T.FLOAT, str: T.STR}.get(type(value))
    return named if named is not None else T.UNKNOWN


def _quiet(node: ast.expr) -> bool:
    """Whether evaluating `node` earlier or later than written could be seen:
    a name, a constant, or an attribute of a name runs no code of the program."""
    if isinstance(node, ast.Constant | ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return _quiet(node.value)
    return constant_default(node)


def bind_arguments(
    info: FunctionInfo, args: list[ast.expr], keywords: list[ast.keyword], skip: int = 0
) -> list[ast.expr]:
    """The arguments of a call to `info`, one per parameter in order, after the
    first `skip` (a bound method's receiver). Raises `Unsupported` where the
    call cannot be bound at compile time."""
    params: list[ParamInfo] = [p for p in info.params if not p.global_of][skip:]
    shown = info.name
    if any(p.kind not in _BINDABLE for p in params):
        raise Unsupported(f"`{shown}` takes `*args` or `**kwargs`, which a native call cannot bind")
    if any(isinstance(a, ast.Starred) for a in args):
        raise Unsupported(f"a call to `{shown}` with `*arguments` has no native lowering")
    positional = [p for p in params if p.kind != "keyword_only"]
    if len(args) > len(positional):
        raise Unsupported(f"`{shown}` called with the wrong number of arguments")
    bound: dict[str, ast.expr] = {p.name: a for p, a in zip(positional, args, strict=False)}
    order: list[str] = []
    for keyword in keywords:
        name = keyword.arg
        if name is None:
            raise Unsupported(f"a call to `{shown}` with `**keywords` has no native lowering")
        param = next((p for p in params if p.name == name), None)
        if param is None or param.kind == "positional_only" or name in bound:
            raise Unsupported(f"`{shown}` called with the wrong number of arguments")
        bound[name] = keyword.value
        order.append(name)
    spelled: list[ast.expr] = []
    defaulted: list[int] = []
    for param in params:
        given = bound.get(param.name)
        if given is not None:
            spelled.append(given)
            continue
        if not param.has_default:
            raise Unsupported(f"`{shown}` called with the wrong number of arguments")
        if not constant_default(param.default):
            raise Unsupported(
                f"`{shown}` leaves `{param.name}` to a default that is not a constant, "
                "which Python made once"
            )
        assert param.default is not None
        spelled.append(param.default)
        defaulted.append(len(spelled) - 1)
    placed = [name for name in (p.name for p in params) if name in order]
    # One argument that runs code moves past ones that run none unseen.
    if placed != order and sum(not _quiet(keyword.value) for keyword in keywords) > 1:
        raise Unsupported(
            f"a call to `{shown}` whose keyword arguments, moved into place, would run "
            "in another order"
        )
    for index in defaulted:
        # The default's node is the `def`'s; the call gets a copy of its own,
        # which the caller's module is told the type of.
        copy = ast.parse(ast.unparse(spelled[index]), mode="eval").body
        ast.copy_location(copy, spelled[index])
        spelled[index] = copy
    return spelled


class CallBinding:  # pylint: disable=too-few-public-methods
    """Binding a call's keywords and defaults; mixed into `_FunctionLowering`."""

    def _function_info(self, qualname: str) -> FunctionInfo | None:
        """The function or method of the project called by `qualname`."""
        known = self.__dict__.get("_infos_by_qualname")
        if known is None:
            symbols = self.frontend.analysis.symbols  # type: ignore[attr-defined]
            known = {info.qualname: info for info in symbols.functions.values()}
            for owner in symbols.classes.values():
                for method in owner.methods.values():
                    known.setdefault(method.qualname, method)
            self.__dict__["_infos_by_qualname"] = known
        found = known.get(qualname)
        if found is None and self.frontend.imports is not None:  # type: ignore[attr-defined]
            imported = self.frontend.imports(qualname)  # type: ignore[attr-defined]
            found = imported[0] if imported is not None else None
        return found  # type: ignore[no-any-return]

    def _spelled(self, qualname: str, node: ast.Call, skip: int = 0) -> list[ast.expr]:
        """`node`'s arguments, one per parameter of `qualname` after `skip`."""
        return self._spelled_of(qualname, list(node.args), list(node.keywords), skip)

    def _typed_defaults(self, spelled: list[ast.expr], given: list[ast.expr]) -> None:
        """Tell the module's analysis what each default put in a call is."""
        analysis = self.frontend.analysis  # type: ignore[attr-defined]
        kept = self.__dict__.setdefault("_defaults_kept", [])
        ids = {id(node) for node in given}
        for node in spelled:
            if id(node) in ids:
                continue
            kept.append(node)  # alive as long as the lowering, so its id is not reused
            for inner in ast.walk(node):
                if isinstance(inner, ast.expr):
                    analysis.node_types[id(inner)] = _constant_type(inner)

    def _spelled_of(
        self, qualname: str, args: list[ast.expr], keywords: list[ast.keyword], skip: int = 0
    ) -> list[ast.expr]:
        info = self._function_info(qualname)
        if info is None:
            if keywords:
                raise Unsupported("keyword arguments have no native ABI")
            return args
        needed = len([p for p in info.params if not p.global_of]) - skip
        if not keywords and len(args) == needed:
            return args
        spelled = bind_arguments(info, args, keywords, skip)
        self._typed_defaults(spelled, [*args, *(k.value for k in keywords)])
        return spelled

    def _scoped_callee(self, node: ast.Call) -> str | None:
        """The function a call by plain name reaches by Python's scopes, where a
        function around this one, or this one, defines that name: its
        qualname, or "" where the name is a local of another kind there. None
        where the name is the module's, for the usual lookup."""
        if not isinstance(node.func, ast.Name):
            return None
        name = node.func.id
        sources = self.frontend.sources  # type: ignore[attr-defined]
        current: FunctionInfo | None = self.info  # type: ignore[attr-defined]
        if current is not None and current.qualname.endswith(".<value>"):
            current = None
        while current is not None:
            if name in own_names(current.node):
                defined = _defined_in(current.node).get(name)
                if isinstance(defined, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return f"{current.qualname}.<locals>.{name}"
                return ""
            found = sources.get(current.enclosing or "")
            current = found[0] if found is not None else None
        return None

    def _binds_keywords(self, node: ast.Call) -> bool:
        """Whether `node` calls a function of the project, whose parameters
        the call's keywords can be bound to."""
        lexical = self.frontend.analysis.symbols.lexical  # type: ignore[attr-defined]
        targets = getattr(lexical, "targets_at", None)
        if targets is None:
            return False
        found = targets(node.func)
        return bool(found) and all(self._function_info(q) is not None for q in found)


def _defined_in(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, ast.AST]:
    """The functions and classes `node` defines in its own scope, by name."""
    found: dict[str, ast.AST] = {}
    pending: list[ast.AST] = list(node.body)
    while pending:
        current = pending.pop()
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.setdefault(current.name, current)
            continue
        if isinstance(current, ast.Lambda):
            continue
        pending.extend(ast.iter_child_nodes(current))
    return found


def nested_entry_refusal(
    info: FunctionInfo, enclosing: dict[str, FunctionInfo]
) -> str | None:
    """Why a function defined inside another cannot have a native entry of its
    own, or None where it can: it shares no variable with the functions around
    it, so every function object its `def` makes behaves the same, and one
    native entry serves them all. Its own name, which it calls itself by, is
    the one name it may use of the function around it, and only to call."""
    node = info.node
    if node.decorator_list:
        return "a nested function with decorators is lowered with the function around it"
    free = free_names(node)
    outer = enclosing.get(info.enclosing or "")
    first = True
    while outer is not None:
        shared = free & own_names(outer.node)
        if first and info.name in shared and _defined_in(outer.node).get(info.name) is node:
            if not _only_called(node, info.name):
                return f"`{info.name}` uses itself as a value, which it shares with the function around it"
            shared.discard(info.name)
        if shared:
            names = ", ".join(f"`{n}`" for n in sorted(shared))
            return f"the function around it stays in Python and shares {names} with it"
        free -= own_names(outer.node)
        first = False
        outer = enclosing.get(outer.enclosing or "")
    return None


def _only_called(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """Whether every use of `name` in `node` is a call by that name."""
    called = {
        id(inner.func)
        for inner in ast.walk(node)
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
    }
    return all(
        id(inner) in called
        for inner in ast.walk(node)
        if isinstance(inner, ast.Name) and inner.id == name
    )
