"""Which `float` parameters show whether they were given an `int`.

Python lets an `int` stand where a `float` is declared, and keeps it an
`int`: `f(0)` of `def f(x: float) -> float: return x` is `0`, not `0.0`.
Native code converts the `int` to a double at the boundary, which is only
right where nothing the function does lets the difference show. It shows
where the value, or a value computed from it the way an `int` would stay an
`int`, is returned, printed or formatted, stored where the caller sees it,
asked its class, or read for an attribute a `float` has and an `int` spells
differently.

`exact_params` follows each float-carrying parameter through the body,
flow-insensitively (a name is tainted if any assignment taints it), and
names the parameters that reach one of those places. The boundary then
takes only a real `float` for them: an `int` keeps the call in Python.
Every other `float` parameter still takes an `int`, converted.

`bool_exact_params` does the same for `int` parameters and a `bool`: Python
keeps `f(True)` of `def f(n: int)` a `bool`, which prints `True` and is an
instance of `bool`, where native code holds `1`. The boundary already takes
only a real `int` for an `int` parameter, so the parameters it names only
keep a native caller that passes a `bool` from compiling the call. A
parameter neither names is an exact `int` in the body, which is how
`isinstance(n, bool)` of one is decided natively.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable

from ..analysis import types as T

#: Calls whose result is an `int`'s kind when their argument is: `abs(x)` of
#: an `int` is an `int`.
_PASSING = frozenset({"abs", "min", "max", "sum", "sorted", "list", "tuple", "reversed"})
#: Calls that show the value as text or ask its class.
_SHOWING = frozenset(
    {"print", "str", "repr", "format", "isinstance", "type", "ascii", "id", "divmod"}
)
#: Calls whose result is the same for an `int` and the `float` it converts to.
_HIDING = frozenset(
    {"float", "int", "bool", "len", "round", "hash", "math", "range", "enumerate", "zip"}
)


def _carries_float(t: T.Type) -> bool:
    t = T.strip_literal(t)
    if t == T.FLOAT:
        return True
    if isinstance(t, T.Tuple_):
        return any(_carries_float(item) for item in t.items)
    if isinstance(t, T.Instance):
        return any(_carries_float(arg) for arg in t.args)
    if isinstance(t, T.Union_):
        return any(_carries_float(member) for member in t.members)
    return False


def _carries_int(t: T.Type) -> bool:
    t = T.strip_literal(t)
    if t == T.INT:
        return True
    if isinstance(t, T.Tuple_):
        return any(_carries_int(item) for item in t.items)
    if isinstance(t, T.Instance):
        return any(_carries_int(arg) for arg in t.args)
    if isinstance(t, T.Union_):
        return any(_carries_int(member) for member in t.members)
    return False


def gives_bool(t: T.Type) -> bool:
    """Whether a value of type `t` may hold a bool where an int is taken."""
    t = T.strip_literal(t)
    if t == T.BOOL:
        return True
    if isinstance(t, T.Tuple_):
        return any(gives_bool(item) for item in t.items)
    if isinstance(t, T.Instance):
        return any(gives_bool(arg) for arg in t.args)
    if isinstance(t, T.Union_):
        return any(gives_bool(member) for member in t.members)
    return False


def gives_int(t: T.Type) -> bool:
    """Whether a value of type `t` may hold an int where a float is taken."""
    t = T.strip_literal(t)
    if t in (T.INT, T.BOOL):
        return True
    if isinstance(t, T.Tuple_):
        return any(gives_int(item) for item in t.items)
    if isinstance(t, T.Instance):
        return any(gives_int(arg) for arg in t.args)
    if isinstance(t, T.Union_):
        return any(gives_int(member) for member in t.members)
    return False


def _is_text(t: T.Type) -> bool:
    return T.strip_literal(t) in (T.STR, T.BYTES)


def _is_float(t: T.Type) -> bool:
    return T.strip_literal(t) == T.FLOAT


def _names(target: ast.expr) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _names(element)
    elif isinstance(target, ast.Starred):
        yield from _names(target.value)


def _root(node: ast.expr) -> str | None:
    """The name a subscript or attribute place is reached from."""
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


class _Taint:
    """One parameter followed through one function body."""

    def __init__(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        types: dict[int, T.Type],
        params: set[str],
        start: set[str],
        callee: Callable[[ast.Call], tuple[list[str], frozenset[str]] | None],
        ambient: _Taint | None = None,
        boolean: bool = False,
    ) -> None:
        self.callee = callee
        #: Following a `bool` held as an `int` rather than an `int` held as a
        #: `float`: arithmetic makes it an `int` again.
        self.boolean = boolean
        #: Every float parameter followed at once: what may be an int at all.
        self.ambient = ambient
        self.function = function
        self.types = types
        self.params = params
        self.tainted = set(start)
        self.shown = False
        #: Types of the nodes this walk spells itself, kept alive with them.
        self.extra: dict[int, T.Type] = {}
        self.made: list[ast.AST] = []
        self.catches = any(isinstance(n, ast.Try) for n in ast.walk(function))

    def type_of(self, node: ast.expr) -> T.Type:
        found = self.extra.get(id(node))
        return found if found is not None else self.types.get(id(node), T.UNKNOWN)

    # -- expressions ---------------------------------------------------------------

    def carries(self, node: ast.expr | None) -> bool:
        """Whether `node`'s value may be an `int` where native code has a `float`."""
        if node is None:
            return False
        match node:
            case ast.Name():
                return node.id in self.tainted
            case ast.Constant():
                return False
            case ast.BinOp() if self.boolean:
                left, right = self.carries(node.left), self.carries(node.right)
                if isinstance(node.op, ast.Mod) and _is_text(self.type_of(node.left)):
                    self.shown = self.shown or right  # `"%s" % x` formats it
                    return False
                # `True & True` is a `bool`; `True + 1` and `-True` are `int`s.
                bitwise = isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor))
                return bitwise and (left or right)
            case ast.UnaryOp() if self.boolean:
                self.visit(node.operand)
                return False
            case ast.BinOp():
                left, right = self.carries(node.left), self.carries(node.right)
                division = isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod))
                if division and (left or right) and self.catches:
                    # `0 / x` raises with words that name `float` for one and
                    # not the other, which a handler can print.
                    self.shown = True
                if isinstance(node.op, ast.Mod) and _is_text(self.type_of(node.left)):
                    self.shown = self.shown or right  # `"%s" % x` formats it
                    return False
                if isinstance(node.op, ast.Div) or not (left or right):
                    return False
                # A real float on the other side makes the result a float in Python too.
                other = node.right if left else node.left
                return (left and right) or not self.real_float(other)
            case ast.UnaryOp():
                inner = self.carries(node.operand)
                return inner and not isinstance(node.op, ast.Not)
            case ast.BoolOp():
                return self.any_of(node.values)
            case ast.IfExp():
                self.visit(node.test)
                return self.any_of([node.body, node.orelse])
            case ast.Compare():
                parts = [self.carries(part) for part in (node.left, *node.comparators)]
                if self.boolean and any(parts):
                    # `n is True` tells `True` from `1`.
                    self.shown = self.shown or any(
                        isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops
                    )
                return False
            case ast.Tuple() | ast.List() | ast.Set():
                return self.any_of(node.elts)
            case ast.Dict():
                keys = self.any_of([k for k in node.keys if k is not None])
                values = self.any_of(node.values)
                return keys or values
            case ast.Starred():
                return self.carries(node.value)
            case ast.Subscript():
                self.visit(node.slice)
                return self.carries(node.value)
            case ast.Attribute():
                if self.carries(node.value):
                    if self.boolean and T.strip_literal(self.type_of(node.value)) == T.INT:
                        return False  # `n.real` of `True` is `1`
                    # `x.real`, `x.is_integer()`, `p.x` of a tainted tuple or object.
                    if _is_float(self.type_of(node.value)):
                        self.shown = True
                        return False
                    return True
                return False
            case ast.NamedExpr():
                value = self.carries(node.value)
                if value:
                    self.tainted.add(node.target.id)
                return value
            case ast.JoinedStr():
                if self.any_of(node.values):
                    self.shown = True
                return False
            case ast.FormattedValue():
                return self.carries(node.value)
            case ast.Call():
                return self.call(node)
            case ast.ListComp() | ast.SetComp() | ast.GeneratorExp():
                self.comprehension(node.generators)
                return self.carries(node.elt)
            case ast.DictComp():
                self.comprehension(node.generators)
                return self.any_of([node.key, node.value])
            case ast.Lambda():
                if self.mentions(node.body):
                    self.shown = True
                return False
            case ast.Await() | ast.Yield() | ast.YieldFrom():
                if self.carries(node.value):
                    self.shown = True
                return False
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr) and self.carries(child):
                self.shown = True
        return False

    def real_float(self, node: ast.expr) -> bool:
        """A float in Python too: typed `float`, and not from a float parameter."""
        if not _is_float(self.type_of(node)):
            return False
        return self.ambient is None or not self.ambient.carries(node)

    def any_of(self, nodes: list[ast.expr]) -> bool:
        """Whether any carries it, having looked at every one."""
        found = False
        for node in nodes:
            found = self.carries(node) or found
        return found

    def visit(self, node: ast.expr | None) -> None:
        self.carries(node)

    def mentions(self, node: ast.AST) -> bool:
        return any(isinstance(n, ast.Name) and n.id in self.tainted for n in ast.walk(node))

    def comprehension(self, generators: list[ast.comprehension]) -> None:
        for generator in generators:
            if self.carries(generator.iter):
                self.tainted.update(_names(generator.target))
            for condition in generator.ifs:
                self.visit(condition)

    def call(self, node: ast.Call) -> bool:
        arguments = [*node.args, *(k.value for k in node.keywords)]
        carried = [self.carries(a) for a in arguments]
        func = node.func
        name = func.id if isinstance(func, ast.Name) else ""
        if isinstance(func, ast.Attribute):
            receiver = self.carries(func.value)
            root = _root(func.value)
            if isinstance(func.value, ast.Name) and func.value.id in {"math", "cmath"}:
                return False
            if receiver and self.boolean and T.strip_literal(self.type_of(func.value)) == T.INT:
                return False  # `n.bit_length()` of `True` is `1`
            if receiver and _is_float(self.type_of(func.value)):
                self.shown = True  # `x.is_integer()`, `x.hex()`
                return False
            if any(carried):
                # `xs.append(x)`: the collection now carries it; a parameter's
                # is the caller's.
                if root is not None:
                    if root in self.params:
                        self.shown = True
                    self.tainted.add(root)
                else:
                    self.shown = True
            if receiver and func.attr in {"pop", "copy", "get", "popleft", "values", "items"}:
                return True
            return receiver and func.attr not in {"index", "count", "__len__"}
        if name in _SHOWING and any(carried):
            self.shown = True
            return False
        if name in _HIDING:
            # `round(x, 2)` of an int is an int.
            return name == "round" and len(arguments) > 1 and carried[0]
        if name in _PASSING:
            return any(carried)
        if any(carried):
            # Another function of the module given it shows it where that
            # function's own parameter does, which includes returning it; any
            # other callee may do anything with it.
            known = self.callee(node)
            if known is None:
                self.shown = True
                return False
            names, exact = known
            given = [n for n, c in zip(names, carried[: len(node.args)], strict=False) if c]
            given += [k.arg for k in node.keywords if k.arg is not None and self.carries(k.value)]
            if len(node.args) > len(names) or any(n in exact for n in given):
                self.shown = True
        return False

    # -- statements ------------------------------------------------------------------------

    def body(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self.statement(statement)

    def assign(self, target: ast.expr, carried: bool) -> None:
        if not carried:
            if not isinstance(target, ast.Name):
                self.place(target)
            return
        if isinstance(target, (ast.Subscript, ast.Attribute)):
            root = _root(target)
            if root is None or root in self.params or isinstance(target, ast.Attribute):
                self.shown = True
            else:
                self.tainted.add(root)
            self.place(target)
            return
        self.tainted.update(_names(target))

    def place(self, target: ast.expr) -> None:
        if isinstance(target, ast.Subscript):
            self.visit(target.slice)
            self.place(target.value)
        elif isinstance(target, ast.Attribute):
            self.place(target.value)

    def statement(self, node: ast.stmt) -> None:
        match node:
            case ast.Assign():
                carried = self.carries(node.value)
                for target in node.targets:
                    self.assign(target, carried)
            case ast.AnnAssign():
                self.assign(node.target, self.carries(node.value))
            case ast.AugAssign():
                spelled = ast.BinOp(node.target, node.op, node.value)
                ast.copy_location(spelled, node)
                self.made.append(spelled)
                self.extra[id(spelled)] = self.type_of(node.target)
                self.assign(node.target, self.carries(spelled))
            case ast.Return():
                if self.carries(node.value):
                    self.shown = True
            case ast.Expr():
                self.visit(node.value)
            case ast.If() | ast.While():
                self.visit(node.test)
                self.body(node.body)
                self.body(node.orelse)
            case ast.For() | ast.AsyncFor():
                if self.carries(node.iter):
                    self.tainted.update(_names(node.target))
                self.body(node.body)
                self.body(node.orelse)
            case ast.Raise():
                if self.carries(node.exc) or self.carries(node.cause):
                    self.shown = True
            case ast.Assert():
                self.visit(node.test)
                if self.carries(node.msg):
                    self.shown = True
            case ast.Try():
                self.body(node.body)
                for handler in node.handlers:
                    self.body(handler.body)
                self.body(node.orelse)
                self.body(node.finalbody)
            case ast.With() | ast.AsyncWith():
                for item in node.items:
                    self.visit(item.context_expr)
                self.body(node.body)
            case ast.Match():
                if self.carries(node.subject):
                    self.shown = True
                for case in node.cases:
                    self.body(case.body)
            case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
                if self.mentions(node):
                    self.shown = True
            case ast.Global() | ast.Nonlocal():
                if any(name in self.tainted for name in node.names):
                    self.shown = True
            case ast.Delete() | ast.Pass() | ast.Break() | ast.Continue() | ast.Import():
                pass
            case _:
                if self.mentions(node):
                    self.shown = True

    def run(self, to_fixpoint: bool = False) -> bool:
        """Whether the parameter shows, to a fixed point over the body."""
        while True:
            before = len(self.tainted)
            self.body(self.function.body)
            if self.shown and not to_fixpoint:
                return True
            if len(self.tainted) == before:
                return self.shown


class ModuleIntness:
    """The exact parameters of every function of one module, to a fixed point
    over the calls between them: a function that passes a parameter to one
    whose parameter shows shows it too."""

    def __init__(
        self,
        functions: dict[str, object],
        types: dict[int, T.Type],
        tree: ast.Module | None = None,
    ) -> None:
        self.functions = functions
        self.types = types
        self.tree = tree
        self._exact: dict[str, frozenset[str]] | None = None
        self._bool_exact: dict[str, frozenset[str]] | None = None
        self._direct: frozenset[str] | None = None

    def called_directly(self, name: str) -> bool:
        """Whether a module-level function of this name is only ever called by
        name: never decorated, rebound, or taken as a value, so every native
        call of it passes its arguments where `_call_arguments` sees them."""
        if self._direct is None:
            self._direct = self._direct_functions()
        return name in self._direct

    def _direct_functions(self) -> frozenset[str]:
        if self.tree is None:
            return frozenset()
        defined: dict[str, int] = {}
        for statement in self.tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                plain = not statement.decorator_list
                defined[statement.name] = defined.get(statement.name, 0) + (1 if plain else 2)
        direct = {name for name, count in defined.items() if count == 1}
        called = {id(node.func) for node in ast.walk(self.tree) if isinstance(node, ast.Call)}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and node.id in direct:
                if not isinstance(node.ctx, ast.Load) or id(node) not in called:
                    direct.discard(node.id)
            elif isinstance(node, (ast.Global, ast.Nonlocal, ast.alias)):
                names = node.names if not isinstance(node, ast.alias) else [node.asname]
                direct.difference_update(n for n in names if n)
        return frozenset(direct)

    def exact(self, qualname: str) -> frozenset[str]:
        """The `float` parameters that show whether they were given an `int`."""
        if self._exact is None:
            self._exact = self._solve()
        return self._exact.get(qualname, frozenset())

    def bool_exact(self, qualname: str) -> frozenset[str]:
        """The `int` parameters that show whether they were given a `bool`."""
        if self._bool_exact is None:
            self._bool_exact = self._solve(boolean=True)
        return self._bool_exact.get(qualname, frozenset())

    def _params(self, qualname: str) -> list[str] | None:
        analysis = self.functions.get(qualname)
        if analysis is None:
            return None
        info = analysis.info  # type: ignore[attr-defined]
        if any(p.kind in {"var_positional", "var_keyword"} for p in info.params):
            return None
        return [p.name for p in info.params]

    def _solve(self, boolean: bool = False) -> dict[str, frozenset[str]]:
        exact: dict[str, frozenset[str]] = {q: frozenset() for q in self.functions}

        def callee(call: ast.Call) -> tuple[list[str], frozenset[str]] | None:
            typed = T.strip_literal(self.types.get(id(call.func), T.UNKNOWN))
            qualname = typed.qualname if isinstance(typed, T.Callable_) else ""
            names = self._params(qualname) if qualname else None
            if names is None:
                return None
            if isinstance(call.func, ast.Attribute) and names and names[0] in {"self", "cls"}:
                names = names[1:]
            return names, exact[qualname]

        while True:
            changed = False
            for qualname, analysis in self.functions.items():
                info = analysis.info  # type: ignore[attr-defined]
                node = getattr(info, "node", None)
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                params = {p.name: p.type for p in info.params}
                carries = _carries_int if boolean else _carries_float
                carrying = {name for name, t in params.items() if carries(t)}
                ambient = _Taint(node, self.types, set(params), carrying, callee, boolean=boolean)
                ambient.run(to_fixpoint=True)
                found = frozenset(
                    name
                    for name in carrying
                    if _Taint(node, self.types, set(params), {name}, callee, ambient, boolean).run()
                )
                if found != exact[qualname]:
                    exact[qualname] = found | exact[qualname]
                    changed = True
            if not changed:
                return exact


#: Operators whose result is a `float` when either operand is one.
_FLOATING = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)


def real_float_locals(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    params: Iterable[str],
    exact: frozenset[str],
) -> frozenset[str]:
    """The names of a function that only ever hold a real `float`: every
    binding is a float literal, `float(...)`, a division, or arithmetic with a
    real float on one side -- `t = (2, 1e308 * x); n, f = t` makes `f` one.
    `exact` are the parameters that are real floats on entry; any other
    parameter, a name bound by a loop, a `with`, a handler, a comprehension, a
    nested scope, or a `global`, is not one."""
    params = set(params)
    #: name -> the values bound to it; None where one is not followed.
    bound: dict[str, list[ast.expr | None]] = {}
    augmented: set[int] = set()

    def bind(target: ast.expr, value: ast.expr | None) -> None:
        if isinstance(target, ast.Name):
            bound.setdefault(target.id, []).append(value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            items: list[ast.expr | None]
            if isinstance(value, ast.Tuple) and len(value.elts) == len(target.elts):
                items = list(value.elts)
            elif isinstance(value, ast.Name):
                items = [
                    ast.Subscript(value, ast.Constant(i), ast.Load())
                    for i in range(len(target.elts))
                ]
            else:
                items = [None] * len(target.elts)
            for element, item in zip(target.elts, items, strict=True):
                if isinstance(element, ast.Starred):
                    bind(element.value, None)
                else:
                    bind(element, item)

    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bind(node.target, node.value)
        elif isinstance(node, ast.AugAssign):
            augmented.add(id(node.target))
            if not isinstance(node.op, (*_FLOATING, ast.Div)):
                bind(node.target, None)
        elif isinstance(node, ast.NamedExpr):
            bind(node.target, node.value)
    handled = {
        id(n)
        for node in ast.walk(function)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
        for t in ([*node.targets] if isinstance(node, ast.Assign) else [node.target])
        for n in ast.walk(t)
    }
    for node in ast.walk(function):
        if node is not function and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            # A nested scope's names may be this one's through `nonlocal`.
            for inner in ast.walk(node):
                if isinstance(inner, ast.Name):
                    bound.setdefault(inner.id, []).append(None)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            if id(node) not in handled and id(node) not in augmented:
                bound.setdefault(node.id, []).append(None)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                bound.setdefault(name, []).append(None)
        elif isinstance(node, ast.comprehension):
            for inner in ast.walk(node.target):
                if isinstance(inner, ast.Name):
                    bound.setdefault(inner.id, []).append(None)

    real = {name for name in bound if name not in params} | set(exact)

    def is_real(node: ast.expr | None) -> bool:
        match node:
            case ast.Constant():
                return type(node.value) is float
            case ast.Name():
                return node.id in real
            case ast.BinOp():
                if isinstance(node.op, ast.Div):
                    return is_real(node.left) or is_real(node.right)
                if isinstance(node.op, _FLOATING):
                    return is_real(node.left) or is_real(node.right)
                return False
            case ast.UnaryOp():
                return isinstance(node.op, (ast.USub, ast.UAdd)) and is_real(node.operand)
            case ast.IfExp():
                return is_real(node.body) and is_real(node.orelse)
            case ast.Call():
                return (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "float"
                    and len(node.args) == 1
                    and not node.keywords
                    and "float" not in bound
                )
            case ast.Subscript():
                # `t[i]` of a name only ever bound to tuple literals.
                if not (
                    isinstance(node.value, ast.Name)
                    and isinstance(node.slice, ast.Constant)
                    and type(node.slice.value) is int
                    and node.value.id not in params
                ):
                    return False
                index = node.slice.value
                literals = bound.get(node.value.id, [None])
                return all(
                    isinstance(literal, ast.Tuple)
                    and -len(literal.elts) <= index < len(literal.elts)
                    and not isinstance(literal.elts[index], ast.Starred)
                    and is_real(literal.elts[index])
                    for literal in literals
                )
        return False

    while True:
        kept = {
            name
            for name in real
            if all(is_real(value) for value in bound.get(name, [] if name in exact else [None]))
        }
        if kept == real:
            return frozenset(real)
        real = kept
