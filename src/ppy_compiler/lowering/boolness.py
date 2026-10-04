"""Where a function would store a `bool` in a slot native code holds as an `int`.

Python keeps `True` a `bool` wherever it goes: `x: int = True` prints `True`,
`[1, True]` holds a `bool`, a dataclass `int` field given `True` prints
`True`. Native code holds an `int` slot as a 64-bit word, so the `bool`
becomes `1` the moment it is stored, and nothing after can tell. Arithmetic
is not a store: `True + 1` is `2` in both.

`hidden_bool` finds the first such store in a function body -- an
assignment, a return, a `yield`, an element of a container literal, an
argument to a container's `append`/`add`/`insert`/`push`, to a class, or to
a function called through a value -- and the lowering keeps that function
in Python (a standalone build reports it). A call of a module function by
name is left to `lowering.intness`, which refuses it only where the callee
shows the difference.
"""

from __future__ import annotations

import ast
from collections.abc import Callable

from ..analysis import types as T

#: Container methods whose argument at this index is kept as an element.
_STORING = {
    "append": 0,
    "add": 0,
    "appendleft": 0,
    "push": 0,
    "insert": 1,
    "setdefault": 1,
}


def _is_bool(t: T.Type) -> bool:
    return T.strip_literal(t) == T.BOOL


def _holds_int(t: T.Type) -> bool:
    """A slot an `int` is stored in and a `bool` is not told apart in."""
    t = T.strip_literal(t)
    if t == T.INT:
        return True
    if isinstance(t, T.Union_):
        members = [T.strip_literal(m) for m in t.members]
        return T.INT in members and T.BOOL not in members
    return False


def bool_in_int(value: T.Type, slot: T.Type) -> bool:
    """Whether a value of type `value` stored in a slot of type `slot` may be a
    `bool` the slot holds as an `int`, at the top or inside."""
    value, slot = T.strip_literal(value), T.strip_literal(slot)
    if isinstance(value, T.Union_):
        return any(bool_in_int(member, slot) for member in value.members)
    if _is_bool(value):
        return _holds_int(slot)
    if isinstance(value, T.Tuple_) and isinstance(slot, T.Tuple_):
        if slot.homogeneous and slot.items:
            return any(bool_in_int(item, slot.items[0]) for item in value.items)
        return any(
            bool_in_int(item, kept) for item, kept in zip(value.items, slot.items, strict=False)
        )
    if (
        isinstance(value, T.Instance)
        and isinstance(slot, T.Instance)
        and value.name == slot.name
        and len(value.args) == len(slot.args)
    ):
        return any(bool_in_int(a, b) for a, b in zip(value.args, slot.args, strict=True))
    return False


def _carries_int(t: T.Type) -> bool:
    t = T.strip_literal(t)
    if _holds_int(t):
        return True
    if isinstance(t, T.Tuple_):
        return any(_carries_int(item) for item in t.items)
    if isinstance(t, T.Instance):
        return any(_carries_int(arg) for arg in t.args)
    return False


def _element(slot: T.Type, index: int = 0) -> T.Type | None:
    slot = T.strip_literal(slot)
    if isinstance(slot, T.Instance) and len(slot.args) > index:
        return slot.args[index]
    return None


def _yielded(ret: T.Type) -> T.Type | None:
    ret = T.strip_literal(ret)
    if isinstance(ret, T.Instance) and ret.args:
        return ret.args[0]
    return None


class _Finder:
    def __init__(
        self,
        type_of: Callable[[ast.expr], T.Type],
        local_type: Callable[[str], T.Type | None],
        returns: T.Type,
        classes: dict[str, object],
        direct: Callable[[ast.Call], bool],
        module: str,
    ) -> None:
        self.module = module
        self.slots: dict[str, T.Type] = {}
        self.type_of = type_of
        self.local_type = local_type
        self.returns = returns
        self.classes = classes
        self.direct = direct

    def hides(self, value: ast.expr | None, slot: T.Type | None) -> bool:
        """Whether storing `value` in a slot of type `slot` loses a `bool`."""
        if value is None or slot is None:
            return False
        match value:
            case ast.List() | ast.Set():
                element = _element(slot)
                return (
                    any(
                        self.hides(e.value if isinstance(e, ast.Starred) else e, element)
                        for e in value.elts
                    )
                    if element is not None
                    else False
                )
            case ast.Tuple():
                stripped = T.strip_literal(slot)
                if isinstance(stripped, T.Tuple_) and stripped.items:
                    if stripped.homogeneous:
                        return any(self.hides(e, stripped.items[0]) for e in value.elts)
                    return any(
                        self.hides(e, s) for e, s in zip(value.elts, stripped.items, strict=False)
                    )
                return False
            case ast.Dict():
                key, item = _element(slot, 0), _element(slot, 1)
                return any(self.hides(k, key) for k in value.keys if k is not None) or any(
                    self.hides(v, item) for v in value.values
                )
            case ast.ListComp() | ast.SetComp() | ast.GeneratorExp():
                return self.hides(value.elt, _element(slot))
            case ast.DictComp():
                return self.hides(value.key, _element(slot, 0)) or self.hides(
                    value.value, _element(slot, 1)
                )
            case ast.IfExp():
                return self.hides(value.body, slot) or self.hides(value.orelse, slot)
            case ast.BoolOp():
                return any(self.hides(v, slot) for v in value.values)
            case ast.NamedExpr():
                return self.hides(value.value, slot)
        return bool_in_int(self.type_of(value), slot)

    def bind_slots(self, nodes: list[ast.AST], params: dict[str, T.Type]) -> None:
        """The slot each local is held in: the checker narrows `y` to `bool`
        after `y = flag`, but a name any binding holds as an `int` (or a
        container of them) is one `int` slot for every binding."""
        bound: dict[str, list[T.Type]] = {name: [t] for name, t in params.items()}
        for node in nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.setdefault(node.id, []).append(self.type_of(node))
        for name, types in bound.items():
            holding = [t for t in types if _carries_int(t)]
            if holding:
                self.slots[name] = holding[0]

    def target_type(self, target: ast.expr) -> T.Type | None:
        if isinstance(target, ast.Name):
            if target.id in self.slots:
                return self.slots[target.id]
            declared = self.local_type(target.id)
            return declared if declared is not None else self.type_of(target)
        if isinstance(target, ast.Subscript):
            # What the container holds, not what the checker narrowed the place to.
            held = T.strip_literal(self.type_of(target.value))
            if isinstance(held, T.Instance) and held.args:
                mapping = held.name in {"dict", "defaultdict", "OrderedDict", "Counter"}
                return held.args[1] if mapping and len(held.args) > 1 else held.args[0]
            return self.type_of(target)
        if isinstance(target, ast.Attribute):
            owner = T.strip_literal(self.type_of(target.value))
            if isinstance(owner, T.Instance):
                info = self.classes.get(owner.name) or self.classes.get(
                    owner.name.rpartition(".")[2]
                )
                fields = getattr(info, "fields", {})
                if target.attr in fields:
                    return fields[target.attr]
            return self.type_of(target)
        return None

    def assigned(self, target: ast.expr, value: ast.expr) -> bool:
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
            if len(target.elts) == len(value.elts):
                return any(
                    self.assigned(t, v) for t, v in zip(target.elts, value.elts, strict=True)
                )
        if isinstance(target, (ast.Tuple, ast.List)):
            whole = self.type_of(value)
            return any(
                isinstance(t, ast.Name) and bool_in_int(whole, self._unpacked(whole, i, t))
                for i, t in enumerate(target.elts)
            )
        return self.hides(value, self.target_type(target))

    def _unpacked(self, whole: T.Type, index: int, target: ast.Name) -> T.Type:
        """The slot an unpacked element lands in, in the shape of `whole`, so
        that `bool_in_int` compares the element with the target's type."""
        slot = self.target_type(target) or T.UNKNOWN
        whole = T.strip_literal(whole)
        if isinstance(whole, T.Tuple_) and not whole.homogeneous:
            items = [T.UNKNOWN] * len(whole.items)
            if index < len(items):
                items[index] = slot
            return T.Tuple_(tuple(items), False)
        if isinstance(whole, T.Instance) and whole.args:
            return T.Instance(whole.name, (slot, *whole.args[1:]))  # type: ignore[call-arg]
        return T.UNKNOWN

    def parameters(self, call: ast.Call) -> list[tuple[ast.expr, T.Type]] | None:
        """Each argument and the type of the parameter it is stored in."""
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr in _STORING:
            receiver = T.strip_literal(self.type_of(func.value))
            if isinstance(receiver, T.Instance) and receiver.name not in self.classes:
                index = _STORING[func.attr]
                element = _element(receiver, 1 if receiver.name in {"dict", "defaultdict"} else 0)
                if element is not None and len(call.args) > index:
                    return [(call.args[index], element)]
                return []
        called = T.strip_literal(self.type_of(func))
        params: tuple[T.Param, ...]
        if isinstance(called, T.ClassObject):
            info = self.classes.get(called.name) or self.classes.get(called.name.rpartition(".")[2])
            if info is None:
                return None
            init = info.methods.get("__init__")  # type: ignore[attr-defined]
            if init is not None:
                params = tuple(T.Param(p.name, p.type, p.has_default, p.kind) for p in init.params)
                params = params[1:]
            elif info.is_dataclass:  # type: ignore[attr-defined]
                params = tuple(
                    T.Param(name, t)
                    for name, t in info.fields.items()  # type: ignore[attr-defined]
                    if name not in info.class_vars  # type: ignore[attr-defined]
                )
            else:
                return []
        elif isinstance(called, T.Callable_):
            if self.direct(call):
                return []
            params = called.params
            if (
                isinstance(func, ast.Attribute)
                and params
                and params[0].name in {"self", "cls"}
                and called.qualname
            ):
                params = params[1:]
        else:
            return []
        if not self.user(called, func):
            return []
        pairs: list[tuple[ast.expr, T.Type]] = []
        positional = [p for p in params if p.kind in {"positional_or_keyword", "positional_only"}]
        for argument, param in zip(call.args, positional, strict=False):
            if not isinstance(argument, ast.Starred):
                pairs.append((argument, param.type))
        named = {p.name: p.type for p in params}
        for keyword in call.keywords:
            if keyword.arg is not None and keyword.arg in named:
                pairs.append((keyword.value, named[keyword.arg]))
        return pairs

    def user(self, called: T.Type, func: ast.expr) -> bool:
        """A class or function of the program, or a function held as a value:
        a builtin's parameters read an `int` (`range(True)`) where these may
        keep it."""
        if isinstance(called, T.ClassObject):
            return True
        if not isinstance(called, T.Callable_):
            return False
        if called.qualname:
            return called.qualname.startswith(f"{self.module}.")
        if isinstance(func, ast.Attribute):
            # A callable field of an object of the program.
            receiver = T.strip_literal(self.type_of(func.value))
            return isinstance(receiver, T.Instance) and (
                receiver.name in self.classes or receiver.name.rpartition(".")[2] in self.classes
            )
        return True

    def statement_hides(self, node: ast.AST) -> bool:
        match node:
            case ast.Assign():
                return any(self.assigned(t, node.value) for t in node.targets)
            case ast.AnnAssign() if node.value is not None:
                return self.assigned(node.target, node.value)
            case ast.Return():
                return self.hides(node.value, self.returns)
            case ast.Yield():
                return self.hides(node.value, _yielded(self.returns))
            case ast.Call():
                pairs = self.parameters(node)
                return pairs is not None and any(self.hides(a, t) for a, t in pairs)
        return False


def _own_nodes(function: ast.AST) -> list[ast.AST]:
    """The nodes of `function`'s own body, without nested functions, lambdas,
    and classes, which are lowered (or not) on their own."""
    found: list[ast.AST] = []
    pending = list(ast.iter_child_nodes(function))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        found.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return found


def hidden_bool(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    returns: T.Type,
    type_of: Callable[[ast.expr], T.Type],
    local_type: Callable[[str], T.Type | None],
    classes: dict[str, object],
    direct: Callable[[ast.Call], bool],
    module: str,
    params: dict[str, T.Type] | None = None,
) -> ast.AST | None:
    """The first place `function` stores a `bool` where native code holds an
    `int`, or None."""
    finder = _Finder(type_of, local_type, returns, classes, direct, module)
    nodes = sorted(
        _own_nodes(function), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))
    )
    finder.bind_slots(nodes, params or {})
    for node in nodes:
        if finder.statement_hides(node):
            return node
    return None
