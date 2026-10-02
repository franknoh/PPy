"""Parameter types from the calls a project makes, under `--no-strict`.

A function written without annotations, `def count_divisors(n)`, says nothing
about `n`; the program that calls it as `count_divisors(28)` does. This pass
reads every call the project makes to a function or method, binds each
argument to the parameter it reaches, and gives the parameter the type every
call agrees on. The checker then sees the function as if it were annotated.

Inferring is not trusting. What makes a type taken from call sites safe:

* A native caller is checked against the inferred signature like any other,
  so a call that passes something else is a type error the checker reports,
  and the pass below takes the inference back rather than report it.
* A Python caller -- one in the project the analysis could not type, a
  doctest, another program importing the module, a call through `getattr` --
  reaches the native code through its Python-ABI wrapper, which checks the
  exact type of every argument (`PyLong_CheckExact`, an exact class, a list
  of exact ints) and runs the Python body when one does not match. A type
  inferred here is guarded the same way a declared one is; nothing outside
  the project can bring a value native code was not built for.
* A function the program uses as a value (`map(f, xs)`, `key=f`) is called by
  whatever it is passed to, with arguments no call site shows. Native code
  calls a sort key directly, past the wrapper, so such a function is left
  alone, as is one with a decorator, one called with `*args` or `**kwargs`,
  and a dunder method Python calls implicitly.

A method may be reached through any class of its family: `shape.area(k)`
with `shape: Shape` runs `Circle.area` when the shape is a circle. So the
methods one name overrides across a hierarchy take evidence together.

Where no call in the project gives a type, the default value does
(`def f(x, lo=0)`), and so do the calls in the module's doctests that pass
literals, which are how many functions in the wild are exercised.
"""

from __future__ import annotations

import ast
import doctest
from dataclasses import dataclass, field

from . import types as T
from .binding import bind_call, positional_values

__all__ = ["CallSiteInference", "new_errors"]

#: Decorators that keep a method a method; any other wraps the function in
#: something the analysis cannot see through.
_PLAIN_DECORATORS = frozenset({"builtins.staticmethod", "builtins.classmethod"})

#: Parameter kinds a call writes one argument into.
_SINGLE = frozenset({"positional_only", "positional_or_keyword", "keyword_only"})

#: How many sites an explanation names before it counts the rest.
_SHOWN_SITES = 3


@dataclass(slots=True)
class _Evidence:
    """What the calls reaching one parameter pass."""

    seen: T.Type | None = None
    sites: list[str] = field(default_factory=list)
    #: A call passed something no type here describes, or more kinds of
    #: value than one parameter type holds: the parameter stays unknown.
    blocked: bool = False

    def add(self, observed: T.Type, site: str) -> None:
        self.seen = observed if self.seen is None else T.join(self.seen, observed)
        self.sites.append(site)


@dataclass(frozen=True, slots=True)
class _Target:
    """A function a call reaches, and how many parameters the call leaves
    unwritten (the receiver of a bound method)."""

    qualname: str
    offset: int


class CallSiteInference:
    """The candidates of one project, and the rounds that type them."""

    def __init__(self, symbols) -> None:  # type: ignore[no-untyped-def]
        self.symbols = symbols
        self.family = _families(symbols)
        self._groups: dict[str, list[str]] = {}
        for qualname, key in self.family.items():
            self._groups.setdefault(key, []).append(qualname)
        #: qualname -> the parameter indices with no annotation and no type.
        self.candidates = self._candidates()
        #: Families whose inference made the checker report an error.
        self.retracted: set[str] = set()
        self._doctests: list[tuple[str, ast.Call, _Target, dict[str, T.Type]]] | None = None
        self._fields: dict[str, dict[str, T.Type]] = {}

    # -- which functions -------------------------------------------------

    def _candidates(self) -> dict[str, list[int]]:
        rebound = _rebound(self.symbols)
        found: dict[str, list[int]] = {}
        for qualname, info in self.symbols.functions.items():
            if info.enclosing is not None or info.is_async or info.is_property:
                continue
            if any(d not in _PLAIN_DECORATORS for d in info.decorators) or info.directives:
                continue
            name = info.name
            if name.startswith("__") and name.endswith("__") and name != "__init__":
                continue  # called by Python's protocols, with arguments no call shows
            if not info.is_method and name in rebound.get(info.module, set()):
                continue
            receiver = 1 if info.is_method and not info.is_static else 0
            unknown = [
                index
                for index, param in enumerate(info.params)
                if index >= receiver
                and not param.annotated
                and param.kind in _SINGLE
                and isinstance(param.type, T.UnknownType)
            ]
            if unknown:
                found[qualname] = unknown
        # A family moves together: a member that cannot be inferred keeps the
        # rest from being typed by calls that may reach it.
        blocked = {
            self.family.get(q, q)
            for q, info in self.symbols.functions.items()
            if info.is_method and q not in found and _has_untyped(info)
        }
        return {q: v for q, v in found.items() if self.family.get(q, q) not in blocked}

    # -- one round ---------------------------------------------------------

    def step(self, analysis) -> bool:  # type: ignore[no-untyped-def]
        """Type each candidate parameter from the calls the last analysis
        typed. True if any parameter moved."""
        evidence, spread = self._evidence(analysis)
        valued = self._values(analysis)
        changed = False
        for qualname, indices in self.candidates.items():
            info = self.symbols.functions.get(qualname)
            key = self.family.get(qualname, qualname)
            if info is None or key in self.retracted:
                continue
            closed = key in spread or key in valued
            for index in indices:
                param = info.params[index]
                found = evidence.get((qualname, index))
                settled, origin = (None, "") if closed else _settle(param, found)
                if settled is None:
                    settled, origin = T.UNKNOWN, ""
                if param.type != settled:
                    param.type = settled
                    changed = True
                param.inferred = not isinstance(settled, T.UnknownType)
                param.origin = origin
        return changed

    def _evidence(self, analysis):  # type: ignore[no-untyped-def]
        evidence: dict[tuple[str, int], _Evidence] = {}
        #: Families a call reaches with `*args` or `**kwargs`.
        spread: set[str] = set()
        reached: set[tuple[str, int]] = set()
        for module_name, module in self.symbols.modules.items():
            checked = analysis.modules.get(module_name)
            if checked is None:
                continue
            owners = _method_owners(module)
            for node in module.module.nodes:
                if not isinstance(node, ast.Call):
                    continue
                for target in self._targets(module, checked, node, owners):
                    key = self.family.get(target.qualname, target.qualname)
                    if any(isinstance(a, ast.Starred) for a in node.args) or any(
                        k.arg is None for k in node.keywords
                    ):
                        spread.add(key)
                        continue
                    site = f"{_short(module.path)}:{node.lineno}"
                    for member in self._members(target.qualname):
                        info = self.symbols.functions[member]
                        offset = _offset_for(info, target.offset)
                        for bound in bind_call(
                            info.params,
                            positional_values(node.args),
                            [(k.arg, k.value) for k in node.keywords],
                            offset=offset,
                        ):
                            observed = T.strip_literal(checked.type_of(bound.value))
                            if isinstance(observed, (T.UnknownType, T.AnyType, T.NeverType)):
                                continue
                            slot = evidence.setdefault((member, bound.index), _Evidence())
                            reached.add((member, bound.index))
                            slot.add(observed, site)
        self._add_doctests(evidence, reached)
        return evidence, spread

    def _values(self, analysis) -> set[str]:  # type: ignore[no-untyped-def]
        """The families of the functions the program takes as a value where
        native code may hold it: `key=f`, `map(f, xs)`, `g = obj.method`.

        Only a typed reference counts. A function value native code calls is
        one the analysis typed; a reference it could not type is in code
        that runs as Python, and a call made from there crosses the wrapper.
        """
        found: set[str] = set()
        for module_name, module in self.symbols.modules.items():
            checked = analysis.modules.get(module_name)
            if checked is None:
                continue
            called = {id(n.func) for n in module.module.nodes if isinstance(n, ast.Call)}
            for node in module.module.nodes:
                if not isinstance(node, (ast.Name, ast.Attribute)) or id(node) in called:
                    continue
                if not isinstance(node.ctx, ast.Load):
                    continue
                seen = T.strip_literal(checked.type_of(node))
                for member in T.members_of(seen):
                    if not isinstance(member, T.Callable_):
                        continue
                    if member.qualname in self.symbols.functions:
                        found.add(self.family.get(member.qualname, member.qualname))
        return found

    def _members(self, qualname: str) -> list[str]:
        key = self.family.get(qualname)
        return [qualname] if key is None else self._groups[key]

    def _targets(self, module, checked, node: ast.Call, owners) -> list[_Target]:  # type: ignore[no-untyped-def]
        """The project functions a call may run."""
        functions = self.symbols.functions
        func = node.func
        called = T.strip_literal(checked.type_of(func))
        if isinstance(called, T.Callable_) and called.qualname in functions:
            info = functions[called.qualname]
            # `Point.scale(p, 2)` names the receiver; `p.scale(2)` does not.
            through_class = isinstance(func, ast.Attribute) and isinstance(
                T.strip_literal(checked.type_of(func.value)), T.ClassObject
            )
            bound = info.is_method and not info.is_static and not through_class
            if info.is_classmethod:
                bound = True
            return [_Target(called.qualname, 1 if bound else 0)]
        if isinstance(called, T.ClassObject):
            return self._initializers(called.name)
        if isinstance(func, ast.Attribute):
            if (
                isinstance(func.value, ast.Call)
                and isinstance(func.value.func, ast.Name)
                and func.value.func.id == "super"
            ):
                owner = owners.get(id(node))
                cls = self.symbols.classes.get(owner) if owner else None
                if cls is not None:
                    for entry in cls.mro[1:]:
                        base = self.symbols.classes.get(entry)
                        if base is not None and func.attr in base.methods:
                            return [_Target(base.methods[func.attr].qualname, 1)]
                return []
            receiver = T.strip_literal(checked.type_of(func.value))
            found = []
            for member in T.members_of(receiver):
                if isinstance(member, T.Instance) and member.name in self.symbols.classes:
                    method = self.symbols.classes[member.name].find_method(func.attr, self.symbols)
                    if method is not None:
                        found.append(_Target(method.qualname, 1))
                elif isinstance(member, T.ClassObject) and member.name in self.symbols.classes:
                    method = self.symbols.classes[member.name].find_method(func.attr, self.symbols)
                    if method is not None:
                        found.append(_Target(method.qualname, 1 if method.is_classmethod else 0))
            return found
        if isinstance(called, (T.UnknownType, T.AnyType)):
            # A call the checker did not type: Python runs it, through the
            # wrapper. The name still says which function, if it is one.
            qualname = self.symbols.resolver(module).canonical(func)
            if qualname is None:
                return []
            if qualname in functions and not functions[qualname].is_method:
                return [_Target(qualname, 0)]
            if qualname in self.symbols.classes:
                return self._initializers(qualname)
        return []

    def _initializers(self, class_name: str) -> list[_Target]:
        cls = self.symbols.classes.get(class_name)
        if cls is None:
            return []
        method = cls.find_method("__init__", self.symbols)
        return [] if method is None else [_Target(method.qualname, 1)]

    # -- doctests --------------------------------------------------------

    def _add_doctests(self, evidence, reached) -> None:  # type: ignore[no-untyped-def]
        """Calls in doctests count only for a parameter no call of the
        project typed: they are a function's documented use, and the type a
        doctest passes is the type the function is meant for."""
        if self._doctests is None:
            self._doctests = list(_doctest_calls(self.symbols))
        for site, node, target, scope in self._doctests:
            for member in self._members(target.qualname):
                info = self.symbols.functions[member]
                for bound in bind_call(
                    info.params,
                    positional_values(node.args),
                    [(k.arg, k.value) for k in node.keywords],
                    offset=_offset_for(info, target.offset),
                ):
                    if (member, bound.index) in reached:
                        continue
                    observed = _literal_type(bound.value, scope)
                    if observed is None:
                        continue
                    evidence.setdefault((member, bound.index), _Evidence()).add(
                        observed, f"doctest {site}"
                    )

    # -- taking it back ----------------------------------------------------

    def retract(self, errors, analysis) -> bool:  # type: ignore[no-untyped-def]
        """Take back every inference an error points at: the function the
        error is in, and the functions called on its line. True if any
        inference was taken back."""
        inferred = {
            q
            for q in self.candidates
            if any(self.symbols.functions[q].params[i].inferred for i in self.candidates[q])
        }
        blamed: set[str] = set()
        for path, line in errors:
            for qualname in inferred:
                info = self.symbols.functions[qualname]
                end = info.node.end_lineno or info.node.lineno
                if info.path == path and info.node.lineno <= line <= end:
                    blamed.add(qualname)
            for module_name, module in self.symbols.modules.items():
                if module.path != path:
                    continue
                checked = analysis.modules.get(module_name)
                if checked is None:
                    continue
                owners = _method_owners(module)
                for node in module.module.nodes:
                    if isinstance(node, ast.Call) and node.lineno <= line <= (
                        node.end_lineno or node.lineno
                    ):
                        for target in self._targets(module, checked, node, owners):
                            blamed.update(m for m in self._members(target.qualname))
        blamed &= inferred
        if not blamed:
            # An error no inferred function or call on its line explains
            # came through a field or a return type: take back what the
            # module of the error infers, and what its calls reach.
            paths = {path for path, _line in errors}
            blamed = {q for q in inferred if self.symbols.functions[q].path in paths}
            for module in self.symbols.modules.values():
                checked = analysis.modules.get(module.name)
                if module.path not in paths or checked is None:
                    continue
                owners = _method_owners(module)
                for node in module.module.nodes:
                    if isinstance(node, ast.Call):
                        for target in self._targets(module, checked, node, owners):
                            blamed.update(self._members(target.qualname))
            blamed &= inferred
        if not blamed:
            return False
        for qualname in blamed:
            self.retracted.add(self.family.get(qualname, qualname))
        self._reset()
        return True

    def retract_all(self) -> None:
        self.retracted.update(self.family.get(q, q) for q in self.candidates)
        self._reset()

    def remember_fields(self) -> None:
        """What the classes' fields were before anything was inferred: a
        field typed from an inferred parameter (`self.n = n`) only ever
        widens, so taking the parameter back must take the field back too."""
        self._fields = {name: dict(cls.fields) for name, cls in self.symbols.classes.items()}

    def _reset(self) -> None:
        for name, fields in self._fields.items():
            cls = self.symbols.classes.get(name)
            if cls is not None:
                cls.fields.clear()
                cls.fields.update(fields)
        for qualname, indices in self.candidates.items():
            if self.family.get(qualname, qualname) not in self.retracted:
                continue
            info = self.symbols.functions[qualname]
            for index in indices:
                param = info.params[index]
                param.type = T.UNKNOWN
                param.inferred = False
                param.origin = ""


def new_errors(before, after) -> set[tuple[object, int]]:  # type: ignore[no-untyped-def]
    """The places (path, line) of the errors `after` reports that `before`
    did not; an error with no place is `(None, 0)`."""
    fresh = _errors(after) - _errors(before)
    return {(None, 0) if span is None else (span.path, span.line) for _c, _m, span in fresh}


def _errors(analysis) -> set[tuple[str, str, object]]:  # type: ignore[no-untyped-def]
    return {
        (d.code, d.message, d.span)
        for module in analysis.modules.values()
        for d in module.diagnostics.items
        if d.is_error
    }


def _settle(param, found: _Evidence | None) -> tuple[T.Type | None, str]:  # type: ignore[no-untyped-def]
    """The type a parameter takes from its evidence, or None, and where it came from."""
    seen = None if found is None or found.blocked else found.seen
    sites = [] if found is None else list(found.sites)
    if param.has_default and param.default is not None:
        default = _literal_type(param.default, {})
        if default is None:
            return None, ""
        seen = default if seen is None else T.join(seen, default)
        sites.append("the default value")
    if seen is None:
        return None, ""
    settled = _usable(seen)
    if settled is None:
        return None, ""
    return settled, _describe(sites)


def _describe(sites: list[str]) -> str:
    calls = [s for s in sites if not s.startswith("doctest ") and s != "the default value"]
    doctests = [s.removeprefix("doctest ") for s in sites if s.startswith("doctest ")]
    parts = []
    if calls:
        shown = ", ".join(list(dict.fromkeys(calls))[:_SHOWN_SITES])
        unshown = len(dict.fromkeys(calls)) - _SHOWN_SITES
        more = f", and {unshown} more places" if unshown > 0 else ""
        plural = "s" if len(calls) != 1 else ""
        parts.append(f"{len(calls)} call{plural} ({shown}{more})")
    if doctests:
        shown = ", ".join(list(dict.fromkeys(doctests))[:_SHOWN_SITES])
        unshown = len(dict.fromkeys(doctests)) - _SHOWN_SITES
        more = f", and {unshown} more places" if unshown > 0 else ""
        plural = "s" if len(doctests) != 1 else ""
        parts.append(f"{len(doctests)} doctest call{plural} ({shown}{more})")
    if "the default value" in sites:
        parts.append("the default value")
    return " and ".join(parts)


def _usable(t: T.Type) -> T.Type | None:
    """The join as a parameter type, or None where it is not one.

    One type is; so is a type or None (`memo=None` and a dict). An int on one
    call and a float on another is a float: the checker keeps a native call
    that passes an int to a float whose int-ness the body shows in Python,
    and the wrapper refuses that int, so the int-ness rules decide exactly
    as they do for a declared `float`. Other unions are left alone.
    """
    t = T.strip_literal(t)
    if isinstance(t, T.Union_):
        members = [m for m in t.members if m != T.NONE]
        if set(members) == {T.INT, T.FLOAT}:
            members = [T.FLOAT]
        if len(members) != 1:
            return None
        if len(t.members) > 1 and T.is_exact_builtin(members[0]):
            # `int | None` has no native form, and the arithmetic on it the
            # body does after a test for `None` is an error to the checker.
            return None
        t = members[0] if len(members) == len(t.members) else T.union(members[0], T.NONE)
        if not _concrete(members[0]):
            return None
        return t
    if t == T.NONE or not _concrete(t):
        return None
    return t


def _concrete(t: T.Type) -> bool:
    """A type with nothing left open in it: no unknown, no `Any`, no empty
    display's `Never`, no function or class object."""
    if isinstance(t, (T.UnknownType, T.AnyType, T.NeverType, T.DynamicType)):
        return False
    if isinstance(t, (T.Callable_, T.ClassObject, T.Module_, T.TypeVar_)):
        return False
    if isinstance(t, T.Instance):
        return all(_concrete(a) for a in t.args)
    if isinstance(t, T.Tuple_):
        return all(_concrete(i) for i in t.items)
    if isinstance(t, T.Union_):
        return all(_concrete(m) for m in t.members)
    return True


def _offset_for(info, offset: int) -> int:  # type: ignore[no-untyped-def]
    """A static method in a family of instance methods takes no receiver."""
    if info.is_static:
        return 0
    return offset


def _has_untyped(info) -> bool:  # type: ignore[no-untyped-def]
    receiver = 1 if info.is_method and not info.is_static else 0
    return any(
        not p.annotated and isinstance(p.type, T.UnknownType)
        for p in info.params[receiver:]
        if p.kind in _SINGLE
    )


def _families(symbols) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """Each method that overrides or is overridden, to a key its family shares."""
    parent: dict[str, str] = {}

    def find(q: str) -> str:
        while parent.get(q, q) != q:
            q = parent[q]
        return q

    for cls in symbols.classes.values():
        for name, method in cls.methods.items():
            for entry in cls.mro[1:]:
                base = symbols.classes.get(entry)
                if base is None or name not in base.methods:
                    continue
                a, b = find(method.qualname), find(base.methods[name].qualname)
                if a != b:
                    parent[a] = b
                parent.setdefault(method.qualname, method.qualname)
                parent.setdefault(base.methods[name].qualname, base.methods[name].qualname)
    return {q: find(q) for q in parent}


def _rebound(symbols) -> dict[str, set[str]]:  # type: ignore[no-untyped-def]
    """Per module, the names its code binds again, declares `global`, or
    defines twice: a call of such a name may reach another function."""
    found: dict[str, set[str]] = {}
    for module_name, module in symbols.modules.items():
        rebound = found.setdefault(module_name, set())
        defined: dict[str, int] = {}
        for node in module.module.nodes:
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                rebound.add(node.id)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                rebound.update(node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined[node.name] = defined.get(node.name, 0) + 1
        rebound.update(name for name, count in defined.items() if count > 1)
    return found


def _method_owners(module) -> dict[int, str]:  # type: ignore[no-untyped-def]
    """Each call inside a method, to the class the method is defined in:
    what `super()` starts from."""
    owners: dict[int, str] = {}
    for cls in module.classes.values():
        for method in cls.methods.values():
            for node in method.nodes:
                if isinstance(node, ast.Call):
                    owners.setdefault(id(node), cls.qualname)
    return owners


def _short(path) -> str:  # type: ignore[no-untyped-def]
    return getattr(path, "name", str(path))


# -- literals ------------------------------------------------------------------


def _literal_type(node: ast.expr, scope: dict[str, T.Type]) -> T.Type | None:
    """The type of a literal written in the source, or None if it is not one."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (bool, int, float, str)) or node.value is None:
            return T.strip_literal(T.type_of_constant(node.value))
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _literal_type(node.operand, scope)
        return inner if inner in (T.INT, T.FLOAT) else None
    if isinstance(node, ast.Name):
        return scope.get(node.id)
    if isinstance(node, (ast.List, ast.Set)):
        if not node.elts:
            return None
        items = [_literal_type(e, scope) for e in node.elts]
        if any(i is None for i in items):
            return None
        element = _usable(T.join(*items))  # type: ignore[arg-type]
        if element is None or isinstance(element, T.Union_):
            return None
        return T.list_of(element) if isinstance(node, ast.List) else T.set_of(element)
    if isinstance(node, ast.Tuple):
        items: list[T.Type] = []
        for element in node.elts:
            found = _literal_type(element, scope)
            if found is None:
                return None
            items.append(found)
        return T.Tuple_(tuple(items)) if items else None
    if isinstance(node, ast.Dict):
        if not node.keys or any(k is None for k in node.keys):
            return None
        keys = [_literal_type(k, scope) for k in node.keys]  # type: ignore[arg-type]
        values = [_literal_type(v, scope) for v in node.values]
        if any(k is None for k in keys) or any(v is None for v in values):
            return None
        key = _usable(T.join(*keys))  # type: ignore[arg-type]
        value = _usable(T.join(*values))  # type: ignore[arg-type]
        if key is None or value is None or isinstance(key, T.Union_):
            return None
        if isinstance(value, T.Union_):
            return None
        return T.dict_of(key, value)
    return None


def _doctest_calls(symbols):  # type: ignore[no-untyped-def]
    """(site, call, target, scope) for each call a docstring example makes of
    a function or method of its module, with the names bound before it."""
    parser = doctest.DocTestParser()
    for module in symbols.modules.values():
        holders: list[ast.AST] = [module.module.tree]
        holders.extend(
            n
            for n in module.module.nodes
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        for holder in holders:
            text = ast.get_docstring(holder, clean=False)  # type: ignore[arg-type]
            if not text or ">>>" not in text:
                continue
            try:
                examples = parser.get_examples(text)
            except ValueError:
                continue
            start = getattr(holder, "lineno", 1)
            scope: dict[str, T.Type] = {}
            for example in examples:
                try:
                    tree = ast.parse(example.source)
                except SyntaxError:
                    continue
                site = f"{_short(module.path)}:{start + example.lineno}"
                raises = example.exc_msg is not None
                for statement in tree.body:
                    if not raises:
                        for node in ast.walk(statement):
                            if isinstance(node, ast.Call):
                                target = _doctest_target(symbols, module, node, scope)
                                if target is not None:
                                    yield site, node, target, dict(scope)
                    _bind_doctest_names(symbols, module, statement, scope)


def _doctest_target(symbols, module, node: ast.Call, scope) -> _Target | None:  # type: ignore[no-untyped-def]
    func = node.func
    if isinstance(func, ast.Name):
        qualname = symbols.resolver(module).canonical(func)
        if qualname is None:
            return None
        if qualname in symbols.functions and not symbols.functions[qualname].is_method:
            return _Target(qualname, 0)
        if qualname in symbols.classes:
            method = symbols.classes[qualname].find_method("__init__", symbols)
            return None if method is None else _Target(method.qualname, 1)
        return None
    if isinstance(func, ast.Attribute):
        owner = _doctest_receiver(symbols, module, func.value, scope)
        if isinstance(owner, T.Instance) and owner.name in symbols.classes:
            method = symbols.classes[owner.name].find_method(func.attr, symbols)
            if method is not None:
                return _Target(method.qualname, 1)
        if isinstance(owner, T.ClassObject) and owner.name in symbols.classes:
            method = symbols.classes[owner.name].find_method(func.attr, symbols)
            if method is not None and (method.is_static or method.is_classmethod):
                return _Target(method.qualname, 1 if method.is_classmethod else 0)
    return None


def _doctest_receiver(symbols, module, node: ast.expr, scope) -> T.Type | None:  # type: ignore[no-untyped-def]
    if isinstance(node, ast.Name):
        if node.id in scope:
            return scope[node.id]
        qualname = symbols.resolver(module).canonical(node)
        if qualname is not None and qualname in symbols.classes:
            return T.ClassObject(qualname, symbols.classes[qualname].instance())
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        qualname = symbols.resolver(module).canonical(node.func)
        if qualname in symbols.classes and not symbols.classes[qualname].type_params:
            return symbols.classes[qualname].instance()
    return None


def _bind_doctest_names(symbols, module, statement: ast.stmt, scope) -> None:  # type: ignore[no-untyped-def]
    """`>>> s = Stack()` makes `s` a `Stack` for the examples after it;
    `>>> xs = [3, 1, 2]` makes `xs` a `list[int]`. Any other binding of a
    name forgets it."""
    for node in ast.walk(statement):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            scope.pop(node.id, None)
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        name = statement.targets[0].id
        value = statement.value
        found = _literal_type(value, scope)
        if found is None:
            found = _doctest_receiver(symbols, module, value, scope)
            if not isinstance(found, T.Instance):
                found = None
        if found is not None:
            scope[name] = found
