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
#: something the analysis cannot see through, unless `_keeps_signature` vouches
#: for it.
_PLAIN_DECORATORS = frozenset({"builtins.staticmethod", "builtins.classmethod"})

#: Known decorators that hand back something called with the decorated
#: function's own parameters, whatever else they do around the call.
_SIGNATURE_DECORATORS = frozenset(
    {
        "abc.abstractmethod",
        "typing.final",
        "typing.override",
        "functools.cache",
        "functools.lru_cache",
    }
)

#: An operator's method and the reflected one Python tries on the right
#: operand; `_INPLACE` is the augmented assignment's own method.
_BINARY = {
    ast.Add: ("__add__", "__radd__"),
    ast.Sub: ("__sub__", "__rsub__"),
    ast.Mult: ("__mul__", "__rmul__"),
    ast.Div: ("__truediv__", "__rtruediv__"),
    ast.FloorDiv: ("__floordiv__", "__rfloordiv__"),
    ast.Mod: ("__mod__", "__rmod__"),
    ast.MatMult: ("__matmul__", "__rmatmul__"),
    ast.BitAnd: ("__and__", "__rand__"),
    ast.BitOr: ("__or__", "__ror__"),
    ast.BitXor: ("__xor__", "__rxor__"),
    ast.LShift: ("__lshift__", "__rlshift__"),
    ast.RShift: ("__rshift__", "__rrshift__"),
    ast.Pow: ("__pow__", "__rpow__"),
}
_INPLACE = {op: f"__i{names[0][2:]}" for op, names in _BINARY.items()}
#: A comparison's method, the reflected one, and for `!=` the `__eq__` it
#: falls back to.
_COMPARE = {
    ast.Eq: ("__eq__", "__eq__"),
    ast.NotEq: ("__ne__", "__ne__", "__eq__"),
    ast.Lt: ("__lt__", "__gt__"),
    ast.Gt: ("__gt__", "__lt__"),
    ast.LtE: ("__le__", "__ge__"),
    ast.GtE: ("__ge__", "__le__"),
}
#: The comparisons the runtime also calls itself, with two objects of the
#: class (ordering a collection, hashing a key, `in` on a list): their other
#: operand may only be inferred as the class itself.
_COMPARISONS = frozenset({"__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"})
#: Dunders whose every call is an operator the program spells, so the
#: operator uses are their call sites. Any other dunder is called by Python's
#: protocols with arguments no use shows.
_OPERATOR_DUNDERS = (
    frozenset(n for names in _BINARY.values() for n in names)
    | frozenset(_INPLACE.values())
    | _COMPARISONS
    | frozenset({"__getitem__", "__setitem__", "__delitem__", "__contains__"})
)

#: Methods a string has and a list, a dict, a number, or a tuple has not: a
#: parameter whose only attribute uses are these is taken to be a `str` (a
#: `bytes` argument has some of them too, and crosses the boundary's check
#: into the Python body).
_STR_METHODS = frozenset(
    {
        "capitalize",
        "casefold",
        "center",
        "encode",
        "endswith",
        "find",
        "format",
        "isalnum",
        "isalpha",
        "isdecimal",
        "isdigit",
        "islower",
        "isnumeric",
        "isspace",
        "isupper",
        "join",
        "ljust",
        "lower",
        "lstrip",
        "partition",
        "replace",
        "rfind",
        "rjust",
        "rpartition",
        "rsplit",
        "rstrip",
        "split",
        "splitlines",
        "startswith",
        "strip",
        "swapcase",
        "title",
        "upper",
        "zfill",
    }
)

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
        #: (qualname, index) -> the declared type of a parameter written with
        #: an open element type (`xs: list`), which calls may refine.
        self.declared: dict[tuple[str, int], T.Type] = {}
        #: qualname -> the parameter indices with no annotation and no type,
        #: or with an open element type.
        self.candidates = self._candidates()
        #: Families whose inference made the checker report an error.
        self.retracted: set[str] = set()
        self._doctests: list[tuple[str, ast.Call, _Target, dict[str, T.Type]]] | None = None
        self._doctest_operators: list[tuple[str, str, T.Type, T.Type | None]] | None = None
        self._parametrized: dict[tuple[str, int], _Evidence] | None = None
        self._argparse: dict[int, T.Type] | None = None
        self._constants: dict[str, dict[str, T.Type]] | None = None
        self._rebound: dict[str, set[str]] | None = None
        self._fields: dict[str, dict[str, T.Type]] = {}

    # -- which functions -------------------------------------------------

    def _candidates(self) -> dict[str, list[int]]:
        rebound = _rebound(self.symbols)
        found: dict[str, list[int]] = {}
        for qualname, info in self.symbols.functions.items():
            if info.enclosing is not None or info.is_async or info.is_property:
                continue
            if info.directives or not all(
                _keeps_signature(self.symbols, d) for d in info.decorators
            ):
                continue
            name = info.name
            if (
                name.startswith("__")
                and name.endswith("__")
                and name != "__init__"
                and not (info.is_method and name in _OPERATOR_DUNDERS)
            ):
                continue  # called by Python's protocols, with arguments no call shows
            if not info.is_method and name in rebound.get(info.module, set()):
                continue
            receiver = 1 if info.is_method and not info.is_static else 0
            unknown = []
            for index, param in enumerate(info.params):
                if index < receiver or param.kind not in _SINGLE:
                    continue
                if not param.annotated and isinstance(param.type, T.UnknownType):
                    unknown.append(index)
                elif param.annotated and _open(param.type):
                    # `def mean(xs: list)`: the calls may say which list.
                    self.declared[(qualname, index)] = param.type
                    unknown.append(index)
            if unknown:
                found[qualname] = unknown
        # A family moves together: a member that cannot be inferred keeps the
        # rest from being typed by calls that may reach it.
        blocked = {
            self.family.get(q, q)
            for q, info in self.symbols.functions.items()
            if info.is_method and q not in found and _has_untyped(info)
        }
        found = {q: v for q, v in found.items() if self.family.get(q, q) not in blocked}
        self.declared = {k: v for k, v in self.declared.items() if k[0] in found}
        return found

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
                declared = self.declared.get((qualname, index))
                found = evidence.get((qualname, index))
                settled, origin = (None, "")
                if not closed:
                    default = self._default_type(info, param)
                    settled, origin = _settle(param, found, default)
                    if settled is not None and not self._fits(info, settled, declared):
                        settled, origin = None, ""
                    if (
                        settled is None
                        and found is None
                        and declared is None
                        and not param.has_default
                        and qualname not in self.family
                    ):
                        settled, origin = _from_use(info, param)
                reset = T.UNKNOWN if declared is None else declared
                if settled is None:
                    settled, origin = reset, ""
                if param.type != settled:
                    param.type = settled
                    changed = True
                param.inferred = settled != reset
                param.origin = origin if param.inferred else ""
        return changed

    def _fits(self, info, settled: T.Type, declared: T.Type | None) -> bool:  # type: ignore[no-untyped-def]
        """Whether a settled type may be taken: it fills in what a declared
        open type leaves open, and a comparison the runtime calls with two of
        the class's own objects takes one of them."""
        if declared is not None and not _fills_open(declared, settled):
            return False
        if info.name in _COMPARISONS:
            return isinstance(settled, T.Instance) and settled.name == info.owner
        return True

    def _default_type(self, info, param) -> T.Type | None:  # type: ignore[no-untyped-def]
        """What a default the source does not write as a literal always
        evaluates to: a module constant written as one (`tol=EPSILON`), or a
        builtin conversion (`seed=int(time())`, `size=len(NAMES)`)."""
        if not param.has_default or param.default is None:
            return None
        module = self.symbols.modules.get(info.module)
        if module is None:
            return None
        if self._constants is None:
            self._constants = {}
        scope = self._constants.get(info.module)
        if scope is None:
            scope = {}
            for name, value in _module_constants(module).items():
                written = _literal_type(value, scope)
                if written is not None:
                    scope[name] = written
            self._constants[info.module] = scope
        default = param.default
        found = _literal_type(default, scope)
        if found is None and isinstance(default, ast.Call) and not default.keywords:
            spelled = self.symbols.resolver(module).canonical(default.func)
            if spelled is None and isinstance(default.func, ast.Name):
                # A builtin function the module does not bind itself.
                if self._rebound is None:
                    self._rebound = _rebound(self.symbols)
                if default.func.id not in self._rebound.get(info.module, set()):
                    spelled = f"builtins.{default.func.id}"
            converted = _CONVERSIONS.get(spelled or "")
            if converted is not None and len(default.args) == 1:
                found = converted
        if found is None or found == T.NONE or not _concrete(found):
            return None
        return found

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
                            observed = self._observed(checked, bound.value)
                            if observed is None:
                                continue
                            slot = evidence.setdefault((member, bound.index), _Evidence())
                            reached.add((member, bound.index))
                            slot.add(observed, site)
            self._add_operators(module, checked, evidence, reached)
        self._add_parametrized(evidence, reached)
        self._add_doctests(evidence, reached)
        return evidence, spread

    def _observed(self, checked, value: ast.expr) -> T.Type | None:  # type: ignore[no-untyped-def]
        """The type a call passes, or None where the checker could not tell
        (the call then runs as Python, through the wrapper). A namespace
        `argparse` filled is read through the parser's `type=`."""
        observed = T.strip_literal(checked.type_of(value))
        if isinstance(observed, T.UnknownType):
            if self._argparse is None:
                self._argparse = {}
                for module in self.symbols.modules.values():
                    self._argparse.update(_argparse_values(self.symbols, module))
            found = self._argparse.get(id(value))
            if found is not None:
                return found
        if isinstance(observed, (T.UnknownType, T.AnyType, T.NeverType)):
            return None
        if _open(observed):
            # `list[Any]`: a value the checker types no better says nothing
            # about the elements, and native code holds no such list, so the
            # call runs as Python, through the wrapper, like an unknown one.
            return None
        return observed

    # -- operators ---------------------------------------------------------

    def _add_operators(self, module, checked, evidence, reached) -> None:  # type: ignore[no-untyped-def]
        """`a + b` calls `type(a).__add__(a, b)`, or `type(b).__radd__(b, a)`;
        `a < b`, `a[k]`, `a[k] = v`, and `x in a` call theirs. Each such use
        the checker typed is a call site of the methods it may run."""
        site = _short(module.path)

        def reach(receiver: ast.expr, name: str, arguments: list[ast.expr | None]) -> None:
            owner = T.strip_literal(checked.type_of(receiver))
            for member in T.members_of(owner):
                if not isinstance(member, T.Instance) or member.name not in self.symbols.classes:
                    continue
                method = self.symbols.classes[member.name].find_method(name, self.symbols)
                if method is None:
                    continue
                for qualname in self._members(method.qualname):
                    for index, value in enumerate(arguments, start=1):
                        observed = None if value is None else self._observed(checked, value)
                        if value is not None and observed is None:
                            continue
                        slot = evidence.setdefault((qualname, index), _Evidence())
                        reached.add((qualname, index))
                        if observed is None:
                            slot.blocked = True
                        else:
                            slot.add(observed, f"{site}:{receiver.lineno}")

        stored: set[int] = set()
        for node in module.module.nodes:
            if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
                forward, reflected = _BINARY[type(node.op)]
                reach(node.left, forward, [node.right])
                reach(node.right, reflected, [node.left])
            elif isinstance(node, ast.AugAssign) and type(node.op) in _BINARY:
                forward, reflected = _BINARY[type(node.op)]
                target = node.target
                if isinstance(target, ast.Subscript):
                    # `a[k] += v` stores what `+` made, which no use types here.
                    stored.add(id(target))
                    reach(target.value, "__setitem__", [target.slice, None])
                    reach(target.value, "__getitem__", [target.slice])
                    continue
                reach(target, _INPLACE[type(node.op)], [node.value])
                reach(target, forward, [node.value])
                reach(node.value, reflected, [target])
            elif isinstance(node, ast.Compare):
                left = node.left
                for op, right in zip(node.ops, node.comparators, strict=True):
                    if isinstance(op, (ast.In, ast.NotIn)):
                        reach(right, "__contains__", [left])
                    elif type(op) in _COMPARE:
                        forward, reflected, *fallback = _COMPARE[type(op)]
                        reach(left, forward, [right])
                        reach(right, reflected, [left])
                        for name in fallback:
                            reach(left, name, [right])
                    left = right
            elif isinstance(node, ast.Subscript) and not isinstance(node.ctx, ast.Store):
                key: ast.expr | None = None if isinstance(node.slice, ast.Slice) else node.slice
                name = "__getitem__" if isinstance(node.ctx, ast.Load) else "__delitem__"
                reach(node.value, name, [key])
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Subscript):
                        stored.add(id(target))
                        key = None if isinstance(target.slice, ast.Slice) else target.slice
                        reach(target.value, "__setitem__", [key, node.value])
        for node in module.module.nodes:
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Store)
                and id(node) not in stored
            ):
                # Unpacked into, a loop target, `with ... as a[k]`: a store whose
                # value no use here types.
                reach(node.value, "__setitem__", [None, None])

    def _add_parametrized(self, evidence, reached) -> None:  # type: ignore[no-untyped-def]
        """`@pytest.mark.parametrize("a, b", [(1, 2), (3, 4)])`: pytest calls
        the test with each case's values. Like a doctest, a case counts only
        for a parameter no call of the project typed."""
        if self._parametrized is None:
            self._parametrized = {}
            constants = {
                name: _module_constants(module) for name, module in self.symbols.modules.items()
            }
            for qualname in self.candidates:
                info = self.symbols.functions[qualname]
                cases = _parametrize_cases(info, constants.get(info.module, {}))
                for key, slot in cases.items():
                    self._parametrized[(qualname, key)] = slot
        for key, slot in self._parametrized.items():
            if key in reached:
                continue
            target = evidence.setdefault(key, _Evidence())
            target.blocked = target.blocked or slot.blocked
            if slot.seen is not None:
                for site in slot.sites:
                    target.add(slot.seen, site)

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
                param.type = self.declared.get((qualname, index), T.UNKNOWN)
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


def _settle(  # type: ignore[no-untyped-def]
    param, found: _Evidence | None, default_type: T.Type | None = None
) -> tuple[T.Type | None, str]:
    """The type a parameter takes from its evidence, or None, and where it came from."""
    if found is not None and found.blocked:
        return None, ""
    seen = None if found is None else found.seen
    sites = [] if found is None else list(found.sites)
    if param.has_default and param.default is not None:
        default = _literal_type(param.default, {})
        if default is None:
            default = default_type
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
        if T.NONE in t.members and T.is_exact_builtin(members[0]):
            # `int | None` has no native form, and the arithmetic on it the
            # body does after a test for `None` is an error to the checker.
            return None
        t = members[0] if T.NONE not in t.members else T.union(members[0], T.NONE)
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


def _open(t: T.Type) -> bool:
    """A declared container type with `Any` for what it holds: `list`,
    `dict[str, Any]`, `list[list]`."""
    if not isinstance(t, T.Instance) or not t.args:
        return False
    if t.name not in {"list", "set", "dict"}:
        return False
    return any(
        (isinstance(a, T.AnyType) and not isinstance(a, T.DynamicType)) or _open(a) for a in t.args
    )


def _fills_open(declared: T.Type, seen: T.Type) -> bool:
    """Whether `seen` is `declared` with each `Any` written in by a concrete type."""
    if isinstance(declared, T.AnyType):
        return _concrete(seen) and not isinstance(seen, T.Union_)
    if isinstance(declared, T.Instance):
        return (
            isinstance(seen, T.Instance)
            and seen.name == declared.name
            and len(seen.args) == len(declared.args)
            and all(_fills_open(d, a) for d, a in zip(declared.args, seen.args, strict=True))
        )
    return declared == seen


def _keeps_signature(symbols, decorator: str) -> bool:  # type: ignore[no-untyped-def]
    """Whether a decorated function is still called with its own parameters:
    `@staticmethod`, `@lru_cache`, a pytest mark (the function itself, marked),
    or a project decorator that wraps with `functools.wraps` and passes the
    wrapper's `*args, **kwargs` through unchanged."""
    if decorator in _PLAIN_DECORATORS or decorator in _SIGNATURE_DECORATORS:
        return True
    if decorator.startswith("pytest.mark.") and decorator.count(".") == 2:
        return True
    info = symbols.functions.get(decorator)
    return info is not None and _forwards(symbols, info)


def _forwards(symbols, info) -> bool:  # type: ignore[no-untyped-def]
    """`def logged(fn): @functools.wraps(fn) def inner(*args, **kwargs): ...
    return fn(*args, **kwargs) ... return inner`: a decorator whose wrapper
    calls the function only with the arguments it was given."""
    node = info.node
    arguments = node.args
    if (
        info.is_method
        or node.decorator_list
        or len(arguments.args) != 1
        or arguments.posonlyargs
        or arguments.kwonlyargs
        or arguments.vararg
        or arguments.kwarg
        or arguments.defaults
    ):
        return False
    fn = arguments.args[0].arg
    inner = [s for s in node.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(inner) != 1 or not isinstance(node.body[-1], ast.Return):
        return False
    wrapper = inner[0]
    returned = node.body[-1].value
    if not (isinstance(returned, ast.Name) and returned.id == wrapper.name):
        return False
    resolver = symbols.resolver(symbols.modules[info.module])
    if (
        not any(
            isinstance(d, ast.Call)
            and resolver.decorator_identity(d) == "functools.wraps"
            and len(d.args) == 1
            and isinstance(d.args[0], ast.Name)
            and d.args[0].id == fn
            for d in wrapper.decorator_list
        )
        or len(wrapper.decorator_list) != 1
    ):
        return False
    signature = wrapper.args
    if (
        signature.args
        or signature.posonlyargs
        or signature.kwonlyargs
        or signature.vararg is None
        or signature.kwarg is None
    ):
        return False
    star, double = signature.vararg.arg, signature.kwarg.arg
    forwarded: set[int] = set()
    for sub in ast.walk(wrapper):
        if (
            isinstance(sub, ast.Name)
            and not isinstance(sub.ctx, ast.Load)
            and sub.id in {fn, star, double}
        ):
            return False
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == fn
            and len(sub.args) == 1
            and isinstance(sub.args[0], ast.Starred)
            and isinstance(sub.args[0].value, ast.Name)
            and sub.args[0].value.id == star
            and len(sub.keywords) == 1
            and sub.keywords[0].arg is None
            and isinstance(sub.keywords[0].value, ast.Name)
            and sub.keywords[0].value.id == double
        ):
            forwarded.update({id(sub.func), id(sub.args[0].value), id(sub.keywords[0].value)})
    # The function, the arguments, and the keywords go nowhere else.
    for sub in ast.walk(wrapper):
        if (
            isinstance(sub, ast.Name)
            and sub.id in {fn, star, double}
            and id(sub) not in forwarded
            and not any(sub is d.args[0] for d in wrapper.decorator_list if isinstance(d, ast.Call))
        ):
            return False
    # Nor does the outer function do anything else with `fn`.
    uses = [n for n in ast.walk(node) if isinstance(n, ast.Name) and n.id == fn]
    return all(any(u is w for w in ast.walk(wrapper)) for u in uses) and bool(forwarded)


def _module_constants(module) -> dict[str, ast.expr]:  # type: ignore[no-untyped-def]
    """The names a module binds once, at its top level, to a value."""
    stores: dict[str, int] = {}
    for node in module.module.nodes:
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            stores[node.id] = stores.get(node.id, 0) + 1
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                stores[name] = stores.get(name, 0) + 2
    found: dict[str, ast.expr] = {}
    for statement in module.module.tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and stores.get(statement.targets[0].id) == 1
        ):
            found[statement.targets[0].id] = statement.value
    return found


def _parametrize_cases(info, constants: dict[str, ast.expr]) -> dict[int, _Evidence]:  # type: ignore[no-untyped-def]
    """Per parameter index, what `@pytest.mark.parametrize` passes it. The
    cases may be a module constant (`TEST_CASES = (...)`), and a value a
    constant written as a literal."""
    found: dict[int, _Evidence] = {}
    scope: dict[str, T.Type] = {}
    for name, value in constants.items():
        written = _literal_type(value, scope)
        if written is not None and not (
            isinstance(written, T.Instance) and T.is_empty_container(written)
        ):
            scope[name] = written
    if not any(d == "pytest.mark.parametrize" for d in info.decorators):
        return found
    index_of = {p.name: i for i, p in enumerate(info.params)}
    for decorator in info.node.decorator_list:
        if not (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "parametrize"
        ):
            continue
        site = f"parametrize {_short(info.path)}:{decorator.lineno}"
        names = _parametrize_names(decorator.args[0]) if decorator.args else None
        if names is None or any(k.arg == "indirect" for k in decorator.keywords):
            for index in index_of.values():
                found.setdefault(index, _Evidence()).blocked = True
            continue
        cases = decorator.args[1] if len(decorator.args) > 1 else None
        if isinstance(cases, ast.Name):
            cases = constants.get(cases.id)
        slots = [found.setdefault(index_of[n], _Evidence()) for n in names if n in index_of]
        if len(slots) != len(names) or not isinstance(cases, (ast.List, ast.Tuple)):
            for slot in slots:
                slot.blocked = True
            continue
        for case in cases.elts:
            if (
                isinstance(case, ast.Call)
                and isinstance(case.func, ast.Attribute)
                and case.func.attr == "param"
            ):
                values = list(case.args)
            elif len(names) == 1:
                values = [case]
            elif isinstance(case, (ast.Tuple, ast.List)):
                values = list(case.elts)
            else:
                values = []
            if len(values) != len(names):
                for slot in slots:
                    slot.blocked = True
                continue
            for slot, value in zip(slots, values, strict=True):
                observed = _literal_type(value, scope)
                if observed is None:
                    slot.blocked = True
                else:
                    slot.add(observed, site)
    return found


def _parametrize_names(node: ast.expr) -> list[str] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [n.strip() for n in node.value.split(",") if n.strip()]
    if isinstance(node, (ast.List, ast.Tuple)) and all(
        isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts
    ):
        return [e.value for e in node.elts]  # type: ignore[attr-defined]
    return None


def _from_use(info, param) -> tuple[T.Type | None, str]:  # type: ignore[no-untyped-def]
    """A parameter nothing calls with a type, typed by what the body does with
    it, where that admits one builtin: `range(n)` takes an `int`, and
    `s.split()` is a `str` method. Any other use of the name -- another
    attribute, a subscript, a test against `None`, a new binding -- leaves it
    alone. A caller with another type crosses the boundary's check."""
    name = param.name
    kinds: set[str] = set()
    line = 0
    explained: set[int] = set()
    for node in info.nodes:
        if isinstance(node, ast.Name) and node.id == name:
            if not isinstance(node.ctx, ast.Load):
                return None, ""
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            return None, ""
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and (
            node is not info.node
        ):
            return None, ""  # a closure may see the name
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Name)
                and func.id == "range"
                and not node.keywords
                and any(isinstance(a, ast.Name) and a.id == name for a in node.args)
            ):
                kinds.add("int")
                line = line or node.lineno
                explained.update(id(a) for a in node.args if isinstance(a, ast.Name))
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == name
                and func.attr in _STR_METHODS
            ):
                kinds.add("str")
                line = line or node.lineno
                explained.add(id(func.value))
        elif isinstance(node, ast.Compare):
            compared = [node.left, *node.comparators]
            if any(isinstance(c, ast.Constant) and c.value is None for c in compared) and any(
                isinstance(c, ast.Name) and c.id == name for c in compared
            ):
                return None, ""
    if len(kinds) != 1 or not explained:
        return None, ""
    for node in info.nodes:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == name
            and id(node.value) not in explained
        ):
            return None, ""  # another attribute: not only a string's
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == name
            and "int" in kinds
        ):
            return None, ""
    kind = kinds.pop()
    if kind == "int":
        return T.INT, f"its use in `range` ({_short(info.path)}:{line})"
    return T.STR, f"its use as a string ({_short(info.path)}:{line})"


def _argparse_values(symbols, module) -> dict[int, T.Type]:  # type: ignore[no-untyped-def]
    """Each `args.name` of a namespace `parser.parse_args()` returned, to the
    type the parser's `add_argument` gives it: `type=int` an `int`, no `type`
    a `str`, `action="store_true"` a `bool`. Only where every use of the
    parser and the namespace in the scope is one of these, and the value is
    there whether or not the option is given (a positional, `required=True`,
    or a default of the type)."""
    resolver = symbols.resolver(module)
    found: dict[int, T.Type] = {}
    scopes: list[ast.AST] = [module.module.tree]
    scopes.extend(
        n for n in module.module.nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    for scope in scopes:
        body = list(_own_nodes(scope))
        parsers: dict[str, dict[str, T.Type] | None] = {}
        namespaces: dict[str, str] = {}
        assigned: dict[str, int] = {}
        for node in body:
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                assigned[node.id] = assigned.get(node.id, 0) + 1
        for node in body:
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
            ):
                continue
            name, call = node.targets[0].id, node.value
            if assigned.get(name) != 1:
                continue
            if resolver.canonical(call.func) == "argparse.ArgumentParser":
                parsers[name] = {}
            elif (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "parse_args"
                and isinstance(call.func.value, ast.Name)
                and not call.args
                and not call.keywords
            ):
                namespaces[name] = call.func.value.id
        if not parsers or not namespaces:
            continue
        used: set[int] = set()
        for node in body:
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in parsers
            ):
                parser = node.func.value.id
                used.add(id(node.func.value))
                if node.func.attr == "add_argument":
                    dests = parsers[parser]
                    if dests is not None:
                        dest, kind = _argument(resolver, node)
                        if dest is None:
                            parsers[parser] = None
                        elif kind is not None:
                            dests[dest] = kind
                elif node.func.attr not in {"parse_args", "print_help", "print_usage"}:
                    parsers[parser] = None  # `set_defaults`, groups, subparsers
        for node in body:
            if (
                isinstance(node, ast.Name)
                and node.id in parsers
                and isinstance(node.ctx, ast.Load)
                and id(node) not in used
            ):
                parsers[node.id] = None  # the parser goes somewhere else
        for node in body:
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in namespaces
                and not isinstance(node.ctx, ast.Load)
            ):
                namespaces.pop(node.value.id, None)
        for node in body:
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name)
                and node.value.id in namespaces
            ):
                dests = parsers.get(namespaces[node.value.id])
                if dests and node.attr in dests:
                    found[id(node)] = dests[node.attr]
    return found


def _own_nodes(scope: ast.AST):  # type: ignore[no-untyped-def]
    """The nodes of a module or function body, without the bodies of the
    functions, classes, and lambdas defined in it."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))


#: Builtins whose result has one type whatever they are given (or they raise).
_CONVERSIONS = {
    "builtins.int": T.INT,
    "builtins.float": T.FLOAT,
    "builtins.str": T.STR,
    "builtins.bool": T.BOOL,
    "builtins.len": T.INT,
}

_ARGUMENT_TYPES = {"builtins.int": T.INT, "builtins.float": T.FLOAT, "builtins.str": T.STR}


def _argument(resolver, call: ast.Call) -> tuple[str | None, T.Type | None]:  # type: ignore[no-untyped-def]
    """An `add_argument` call's destination and the type its value always
    has, or None for the type where it may be `None` or something else.
    None for the destination where the call cannot be read."""
    flags = [a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    if not flags or len(flags) != len(call.args):
        return None, None
    keywords = {k.arg: k.value for k in call.keywords}
    if None in keywords:
        return None, None
    positional = not flags[0].startswith("-")
    if "dest" in keywords:
        dest_node = keywords["dest"]
        if not (isinstance(dest_node, ast.Constant) and isinstance(dest_node.value, str)):
            return None, None
        dest = dest_node.value
    elif positional:
        dest = flags[0]
    else:
        long = [f for f in flags if f.startswith("--")]
        dest = (long[0] if long else flags[0]).lstrip("-").replace("-", "_")
    if "nargs" in keywords or "const" in keywords:
        return dest, None
    action = keywords.get("action")
    if action is not None:
        if isinstance(action, ast.Constant) and action.value in {"store_true", "store_false"}:
            return dest, T.BOOL if "default" not in keywords else None
        return dest, None
    kind = T.STR
    if "type" in keywords:
        kind = _ARGUMENT_TYPES.get(resolver.canonical(keywords["type"]) or "")
        if kind is None:
            return dest, None
    required = keywords.get("required")
    always = positional or (isinstance(required, ast.Constant) and required.value is True)
    default = keywords.get("default")
    if default is not None:
        written = _literal_type(default, {})
        # A string default goes through `type=` as a given value would.
        if written not in (kind, T.STR):
            return dest, None
        always = True
    return dest, kind if always else None


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
    if isinstance(node, ast.BinOp):
        return _literal_arithmetic(node, scope)
    if isinstance(node, (ast.List, ast.Set)):
        if not node.elts:
            # Empty: what it holds is what the displays beside it hold
            # (`{1: [2], 2: []}`); alone, it says nothing.
            return T.list_of(T.NEVER) if isinstance(node, ast.List) else T.set_of(T.NEVER)
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
        if not node.keys:
            return T.dict_of(T.NEVER, T.NEVER)
        if any(k is None for k in node.keys):
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


def _literal_arithmetic(node: ast.BinOp, scope: dict[str, T.Type]) -> T.Type | None:
    """`2 << 31`, `10**9 + 7`, `1 / 3`, `"ab" * 3`: arithmetic on literals."""
    left, right = _literal_type(node.left, scope), _literal_type(node.right, scope)
    numbers = {T.INT, T.FLOAT}
    if left in numbers and right in numbers:
        if isinstance(node.op, ast.Div):
            return T.FLOAT
        if T.FLOAT in (left, right):
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
                return T.FLOAT
            return None
        if isinstance(node.op, ast.Pow):
            # `2 ** -1` is a float.
            negative = isinstance(node.right, ast.UnaryOp) and isinstance(node.right.op, ast.USub)
            return None if negative or not isinstance(node.right, ast.Constant) else T.INT
        return T.INT if not isinstance(node.op, ast.MatMult) else None
    if left == T.STR and right == T.STR and isinstance(node.op, ast.Add):
        return T.STR
    if {left, right} == {T.STR, T.INT} and isinstance(node.op, ast.Mult):
        return T.STR
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
                        inner = _loop_scope(statement, scope)
                        for node in ast.walk(statement):
                            if isinstance(node, ast.Call):
                                target = _doctest_target(symbols, module, node, inner)
                                if target is not None:
                                    yield site, node, target, dict(inner)
                            for call, target in _doctest_operators(symbols, module, node, inner):
                                yield site, call, target, dict(inner)
                    _bind_doctest_names(symbols, module, statement, scope)


def _loop_scope(statement: ast.stmt, scope: dict[str, T.Type]) -> dict[str, T.Type]:
    """The names a doctest's `for` loops bind, added to those bound before:
    `for value in [17, 20, 31]:` makes `value` an `int` inside the loop, as
    `for i in range(10):` does. A name bound any other way inside the
    statement is forgotten for it."""
    if not isinstance(statement, (ast.For, ast.While, ast.If, ast.With)):
        return scope
    inner = dict(scope)
    loops: dict[str, T.Type | None] = {}
    for node in ast.walk(statement):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            name = node.target.id
            element = _element_of(node.iter, inner)
            loops[name] = element if name not in loops or loops[name] == element else None
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            if not any(isinstance(f, ast.For) and f.target is node for f in ast.walk(statement)):
                loops[node.id] = None
    for name, element in loops.items():
        if element is None:
            inner.pop(name, None)
        else:
            inner[name] = element
    return inner


def _element_of(node: ast.expr, scope: dict[str, T.Type]) -> T.Type | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and not node.keywords
        and all(_literal_type(a, scope) == T.INT for a in node.args)
    ):
        return T.INT
    found = _literal_type(node, scope)
    if isinstance(found, T.Instance) and found.name in {"list", "set"} and found.args:
        element = found.args[0]
        return element if _concrete(element) else None
    if isinstance(found, T.Tuple_) and found.items:
        element = _usable(T.join(*found.items))
        return element if element is not None and not isinstance(element, T.Union_) else None
    return None


def _doctest_operators(symbols, module, node: ast.AST, scope):  # type: ignore[no-untyped-def]
    """`>>> heap["B"]`, `>>> a + 2`, `>>> a < b`: an operator a doctest applies
    to an object it made, as the call of the method it runs."""
    uses: list[tuple[ast.expr, str, ast.expr]] = []
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        forward, reflected = _BINARY[type(node.op)]
        uses += [(node.left, forward, node.right), (node.right, reflected, node.left)]
    elif isinstance(node, ast.Compare) and len(node.ops) == 1:
        op, right = node.ops[0], node.comparators[0]
        if isinstance(op, (ast.In, ast.NotIn)):
            uses.append((right, "__contains__", node.left))
        elif type(op) in _COMPARE:
            forward, reflected, *_rest = _COMPARE[type(op)]
            uses += [(node.left, forward, right), (right, reflected, node.left)]
    elif (
        isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Load)
        and not isinstance(node.slice, ast.Slice)
    ):
        uses.append((node.value, "__getitem__", node.slice))
    for receiver, name, argument in uses:
        owner = _doctest_receiver(symbols, module, receiver, scope)
        if not isinstance(owner, T.Instance) or owner.name not in symbols.classes:
            continue
        method = symbols.classes[owner.name].find_method(name, symbols)
        if method is not None:
            call = ast.Call(func=receiver, args=[argument], keywords=[])
            yield call, _Target(method.qualname, 1)


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
        if found is not None and not (
            isinstance(found, T.Instance) and T.is_empty_container(found)
        ):
            scope[name] = found
