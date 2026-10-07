"""Small expressions natively: `isinstance`, chained comparisons, membership
in a tuple, a literal set, or a `range`, and integer `**` and `pow`.

Each is written in terms of what the lowering already has: a chain is its
comparisons joined by short circuits, `x in (a, b)` is `x == a or x == b`,
and so on, with every operand evaluated once and in CPython's order.
"""

from __future__ import annotations

import ast

from ..analysis import types as T
from ..backend.llvm.lowering import Unsupported
from ..ir import BOOL, F64, I64, Successor, Value
from ..ir.dialects import core
from ..ir.dialects import math as math_dialect
from .collections import HANDLE, Shape
from .exceptions import ARGS_NONE, ARGS_ONE_TEXT
from .intness import exact_locals, gives_bool, gives_int

#: The builtin classes `isinstance` is asked of, by the name a program spells.
_BUILTIN_CLASSES = frozenset(
    {
        "int",
        "float",
        "bool",
        "complex",
        "str",
        "bytes",
        "list",
        "tuple",
        "dict",
        "set",
        "frozenset",
        "object",
        "range",
    }
)

#: The classes a value of a builtin type may be at run time: an `int` may be a
#: `bool`, and a `float` or a `complex` may be given an `int` (the numeric tower).
_RUNTIME_CLASSES = {
    "int": ("int", "bool"),
    "bool": ("bool",),
    "float": ("float", "int", "bool"),
    "complex": ("complex", "float", "int", "bool"),
}


def _cheap(node: ast.expr) -> bool:
    """Whether evaluating `node` twice is evaluating it once: a name, a
    constant, a negated constant, or an attribute of a name."""
    if isinstance(node, (ast.Name, ast.Constant)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return isinstance(node.operand, ast.Constant)
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name)
    return False


class ExpressionLowering:  # pylint: disable=attribute-defined-outside-init
    """Mixed into `_FunctionLowering` ahead of the others."""

    #: How many hidden temporaries this lowering has named.
    _hidden_count: int = 0

    # -- typed synthetic nodes -------------------------------------------------------

    def _typed(self, node: ast.expr, t: T.Type, like: ast.AST) -> ast.expr:
        """A node the lowering makes, typed as the checker would have typed it,
        and kept alive for as long as the lowering (its id is the key)."""
        ast.copy_location(node, like)
        self.frontend.analysis.node_types[id(node)] = t  # type: ignore[attr-defined]
        self.__dict__.setdefault("_made_nodes", []).append(node)
        return node

    def _once(self, node: ast.expr) -> ast.expr:
        """`node`, or a hidden local holding its value when evaluating it again
        would run it again: a scalar only."""
        if _cheap(node):
            return node
        t = T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]
        if t not in (T.INT, T.FLOAT, T.BOOL):
            raise Unsupported("a chained comparison's operand is a name or a number here")
        value = self._expr(node)  # type: ignore[attr-defined]
        self._hidden_count += 1
        name = f".t{self._hidden_count}"
        self._store(ast.Name(name, ast.Store()), value)  # type: ignore[attr-defined]
        return self._typed(ast.Name(name, ast.Load()), t, node)

    def _pair(self, left: ast.expr, op: ast.cmpop, right: ast.expr, like: ast.AST) -> Value:
        pair = self._typed(ast.Compare(left, [op], [right]), T.BOOL, like)
        return self._compare(pair)  # type: ignore[attr-defined,arg-type]

    def _any_of(self, tests: list, stop_on: bool) -> Value:
        """Short-circuit over thunks giving truths: the first equal to `stop_on`
        decides, else the last."""
        done = self._block("any.end")  # type: ignore[attr-defined]
        result = done.add_argument(BOOL, "any")
        for index, test in enumerate(tests):
            truth = test()
            if index == len(tests) - 1:
                core.br(self.b, Successor(done, [truth]))  # type: ignore[attr-defined]
                break
            following = self._block("any.next")  # type: ignore[attr-defined]
            if stop_on:
                core.cond_br(self.b, truth, Successor(done, [truth]), Successor(following))  # type: ignore[attr-defined]
            else:
                core.cond_br(self.b, truth, Successor(following), Successor(done, [truth]))  # type: ignore[attr-defined]
            self.b.at_end(following)  # type: ignore[attr-defined]
        self.b.at_end(done)  # type: ignore[attr-defined]
        return result

    def _negated(self, truth: Value) -> Value:
        return core.bitwise(self.b, "xor", truth, core.const(self.b, True, BOOL))  # type: ignore[attr-defined]

    # -- chained comparisons -------------------------------------------------------------

    def _chained(self, node: ast.Compare) -> Value:
        """`a < b < c`: `a < b and b < c` with `b` evaluated once, and `c` only
        where `a < b` held."""
        operands = [node.left, *node.comparators]
        state = {"left": operands[0]}

        def step(index: int):  # type: ignore[no-untyped-def]
            def run() -> Value:
                right = operands[index + 1]
                if index + 1 < len(operands) - 1:
                    right = self._once(right)
                left = state["left"]
                state["left"] = right
                return self._pair(left, node.ops[index], right, node)

            return run

        tests = [step(index) for index in range(len(node.ops))]
        return self._any_of(tests, stop_on=False)

    # -- membership --------------------------------------------------------------------

    def _member_of(self, node: ast.Compare) -> Value | None:
        """`x in (a, b)`, `x in {a, b}`, `x in [a, b]`, `x in t` for a tuple `t`,
        and `x in range(...)`; `not in` is its negation."""
        operator = node.ops[0]
        if isinstance(operator, (ast.Is, ast.IsNot)):
            return self._bool_identity(node)
        if not isinstance(operator, (ast.In, ast.NotIn)):
            return None
        container = node.comparators[0]
        found: Value | None = None
        if isinstance(container, (ast.Tuple, ast.Set, ast.List)) and not any(
            isinstance(e, ast.Starred) for e in container.elts
        ):
            found = self._in_display(node.left, list(container.elts), node)
        elif isinstance(container, ast.Call) and self._is_range(container):
            found = self._in_range(node.left, container)
        elif self._is_tuple_value(container):
            found = self._in_tuple_value(node.left, container, node)
        if found is None:
            return None
        return self._negated(found) if isinstance(operator, ast.NotIn) else found

    def _bool_identity(self, node: ast.Compare) -> Value | None:
        """`flag is True`, `a is not b` of two bools: there is one `True` and one
        `False`, so identity is equality. An `int` may be a `bool` or not, so
        only values the checker says are bools compare this way."""
        left, right = node.left, node.comparators[0]
        if self._plain_type(left) != T.BOOL or self._plain_type(right) != T.BOOL:
            return None
        found = self._pair(left, ast.Eq(), right, node)
        return self._negated(found) if isinstance(node.ops[0], ast.IsNot) else found

    def _is_range(self, node: ast.Call) -> bool:
        t = T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]
        return (
            isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and 1 <= len(node.args) <= 3
            and not node.keywords
            and isinstance(t, T.Instance)
            and t.name == "range"
        )

    def _plain_type(self, node: ast.expr) -> T.Type:
        return T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]

    def _in_display(
        self, item: ast.expr, elements: list[ast.expr], like: ast.Compare
    ) -> Value | None:
        """`x in (a, b, c)`: the display is made before any comparison, then
        `x == a or x == b or x == c`, where `x is a` counts too."""
        if not elements:
            if not _cheap(item):
                self._expr(item)  # type: ignore[attr-defined]
            return core.const(self.b, False, BOOL)  # type: ignore[attr-defined]
        optional = self._optional_in_display(item, elements, like)  # type: ignore[attr-defined]
        if optional is not None:
            return optional
        item_type = self._plain_type(item)
        kinds = {self._plain_type(e) for e in elements} | {item_type}
        if not kinds <= {T.INT, T.FLOAT, T.BOOL, T.STR}:
            return None
        if T.STR in kinds and len(kinds) > 1:
            # `1 in ("a", 1)` compares across types, which stays in Python.
            return None
        if T.STR in kinds:
            if not (_cheap(item) and all(_cheap(e) for e in elements)):
                return None
            left = item
        else:
            # The item, then each element, as CPython makes the display.
            left = self._once(item)
            elements = [self._once(e) for e in elements]
        if item_type == T.FLOAT:
            self._not_nan(left)
        tests = [
            (lambda e=e: self._pair(left, ast.Eq(), e, like))  # type: ignore[misc]
            for e in elements
        ]
        return self._any_of(tests, stop_on=True)

    def _not_nan(self, node: ast.expr) -> None:
        """A NaN is in a tuple that holds that very object, which only Python knows."""
        value = self._expr(node)  # type: ignore[attr-defined]
        core.guard(
            self.b,  # type: ignore[attr-defined]
            core.cmp(self.b, "eq", value, value),  # type: ignore[attr-defined]
            "contract",
            "a NaN's membership is decided by identity",
        )

    def _is_tuple_value(self, node: ast.expr) -> bool:
        return isinstance(node, ast.Name) and node.id in getattr(self, "tuples", {})

    def _in_tuple_value(
        self, item: ast.expr, container: ast.expr, like: ast.Compare
    ) -> Value | None:
        t = self._plain_type(container)
        if not isinstance(t, T.Tuple_) or t.homogeneous or not t.items:
            return None
        items = [T.strip_literal(i) for i in t.items]
        kinds = set(items) | {self._plain_type(item)}
        if not kinds <= {T.INT, T.FLOAT, T.BOOL}:
            return None
        left = self._once(item)
        if self._plain_type(item) == T.FLOAT:
            self._not_nan(left)
        elements = [
            self._typed(ast.Subscript(container, ast.Constant(i), ast.Load()), items[i], like)
            for i in range(len(items))
        ]
        tests = [
            (lambda e=e: self._pair(left, ast.Eq(), e, like))  # type: ignore[misc]
            for e in elements
        ]
        return self._any_of(tests, stop_on=True)

    def _range_bounds(self, node: ast.Call) -> tuple[Value, Value, Value, int | None]:
        """A `range(...)`'s start, stop, and step, evaluated in order, and the
        step when it is a constant; a zero step raises CPython's `ValueError`."""
        bounds = [self._coerce(self._expr(a), "int") for a in node.args]  # type: ignore[attr-defined]
        constant: int | None = 1
        if len(bounds) == 1:
            start, stop, step = self._int_constant(0), bounds[0], self._int_constant(1)  # type: ignore[attr-defined]
        elif len(bounds) == 2:
            start, stop = bounds
            step = self._int_constant(1)  # type: ignore[attr-defined]
        else:
            start, stop, step = bounds
            literal = node.args[2]
            constant = None
            if isinstance(literal, ast.Constant) and type(literal.value) is int:
                constant = literal.value
            elif (
                isinstance(literal, ast.UnaryOp)
                and isinstance(literal.op, ast.USub)
                and isinstance(literal.operand, ast.Constant)
                and type(literal.operand.value) is int
            ):
                constant = -literal.operand.value
            if constant != 0 and constant is not None:
                pass
            else:
                self._guard(  # type: ignore[attr-defined]
                    core.cmp(self.b, "ne", step, self._int_constant(0)),  # type: ignore[attr-defined]
                    "bounds",
                    "range() arg 3 must not be zero",
                    raises="ValueError: range() arg 3 must not be zero",
                )
        return start, stop, step, constant

    def _in_range(self, item: ast.expr, container: ast.Call) -> Value | None:
        """`x in range(a, b, s)` for an int `x`: a bounds check and, for a step
        other than one, a remainder; no walk."""
        if self._plain_type(item) not in (T.INT, T.BOOL):
            return None
        value = self._coerce(self._expr(item), "int")  # type: ignore[attr-defined]
        start, stop, step, constant = self._range_bounds(container)
        b = self.b  # type: ignore[attr-defined]
        up_inside = core.bitwise(
            b, "and", core.cmp(b, "le", start, value), core.cmp(b, "lt", value, stop)
        )
        down_inside = core.bitwise(
            b, "and", core.cmp(b, "lt", stop, value), core.cmp(b, "le", value, start)
        )
        if constant is not None:
            inside = up_inside if constant > 0 else down_inside
        else:
            upward = core.cmp(b, "gt", step, self._int_constant(0))  # type: ignore[attr-defined]
            inside = core.select(b, upward, up_inside, down_inside)
        if constant in (1, -1):
            return inside
        # Inside the bounds `x - start` has the step's sign and fits the word
        # unless the range spans more than half of it.
        result = self._alloca(BOOL, "in.range")  # type: ignore[attr-defined]
        core.store(b, core.const(b, False, BOOL), result)
        there = self._block("in.range.there")  # type: ignore[attr-defined]
        done = self._block("in.range.done")  # type: ignore[attr-defined]
        core.cond_br(b, inside, Successor(there), Successor(done))
        b.at_end(there)
        distance, overflowed = core.checked(b, "sub", value, start)
        core.guard(b, self._negated(overflowed), "overflow", "range membership")
        remainder = core.mod(b, distance, step, overflow="wrap", rounding="trunc")
        core.store(b, core.cmp(b, "eq", remainder, self._int_constant(0)), result)  # type: ignore[attr-defined]
        core.br(b, Successor(done))
        b.at_end(done)
        return core.load(b, result)

    # -- isinstance --------------------------------------------------------------------

    def _class_names(self, node: ast.expr) -> list[str] | None:
        """The classes `isinstance`'s second argument names: a class, `type(None)`,
        or a tuple of these."""
        if isinstance(node, ast.Tuple):
            names: list[str] = []
            for element in node.elts:
                inner = self._class_names(element)
                if inner is None:
                    return None
                names.extend(inner)
            return names
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "type"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value is None
        ):
            return ["NoneType"]
        called = T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]
        if isinstance(called, T.ClassObject):
            return [called.name]
        if isinstance(node, ast.Name) and node.id in _BUILTIN_CLASSES:
            return [node.id]
        return None

    def _runtime_classes(self, t: T.Type) -> set[str] | None:
        """The classes a value the checker typed `t` may be when it runs, where
        they are all builtin; None for an object class or anything unknown."""
        t = T.strip_literal(t)
        if isinstance(t, T.Union_):
            found: set[str] = set()
            for member in t.members:
                inner = self._runtime_classes(member)
                if inner is None:
                    return None
                found |= inner
            return found
        if isinstance(t, T.Tuple_):
            return {"tuple"}
        if t == T.NONE:
            return {"NoneType"}
        if not isinstance(t, T.Instance) or t.name not in T.BUILTIN_MRO:
            return None
        if t.name in {"object", "BaseException"} or t.name.endswith(("Error", "Exception")):
            return None
        if t.name in self._module_classes():  # type: ignore[attr-defined]
            return None
        return set(_RUNTIME_CLASSES.get(t.name, (t.name,)))

    def _is_instance(self, node: ast.Call) -> Value | None:
        """`isinstance(x, T)` where the checker's type of `x` decides it: an `int`
        is an `int` (a `bool` may be one too), a `str` is never a `list`."""
        if len(node.args) == 2 and not node.keywords:
            wanted = self._class_names(node.args[1])
            possible = self._runtime_classes(self._type_of(node.args[0]))  # type: ignore[attr-defined]
            subject = node.args[0]
            if possible is not None and isinstance(subject, ast.Name):
                # The checker narrows `flag: bool` by `isinstance(flag, int)` to
                # `int`; what the name was declared keeps what it may be.
                declared = self._local_type(subject.id)  # type: ignore[attr-defined]
                known = self._runtime_classes(declared) if declared is not None else None
                if known is not None:
                    possible &= known
                exact = self._exact_parameter(subject.id)
                if exact is not None:
                    possible &= {exact}
                else:
                    local = self._exact_local(subject.id)
                    if local is not None:
                        possible &= {local}
            if wanted is not None and possible is not None:
                answers = {
                    any(name in T.BUILTIN_MRO.get(runtime, (runtime,)) for name in wanted)
                    for runtime in possible
                }
                if not possible:
                    answers = {False}  # a branch no value reaches
                if len(answers) == 1:
                    if not _cheap(node.args[0]):
                        self._expr(node.args[0])  # type: ignore[attr-defined]
                    return core.const(self.b, answers.pop(), BOOL)  # type: ignore[attr-defined]
                decided = self._optional_isinstance(subject, wanted, possible)  # type: ignore[attr-defined]
                if decided is not None:
                    return decided
                raise Unsupported(
                    f"`isinstance` of a `{self._type_of(node.args[0])}` depends on the value"  # type: ignore[attr-defined]
                )
            if (
                wanted is not None
                and "NoneType" in wanted
                and self._object_of(node.args[0]) is not None
            ):  # type: ignore[attr-defined]
                return self._is_none_or(node, [n for n in wanted if n != "NoneType"])
        return super()._is_instance(node)  # type: ignore[misc]

    def _exact_local(self, name: str) -> str | None:
        """`int` or `float` for a name of this function that only ever holds a
        real one of that class (`lowering.intness.exact_locals`)."""
        info = self.info  # type: ignore[attr-defined]
        cache = self.__dict__.setdefault("_exact_local_names", {})
        found = cache.get(info.qualname)
        if found is None:
            node = info.node
            found = {}
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = [p.name for p in info.params]
                for kind in ("int", "float"):
                    exact = frozenset(n for n in params if self._exact_parameter(n) == kind)
                    for local in exact_locals(node, params, exact, kind, self._type_of):  # type: ignore[attr-defined]
                        found[local] = kind
            cache[info.qualname] = found
        return found.get(name)

    def _exact_parameter(self, name: str) -> str | None:
        """The one class a parameter of a module-level function only ever called
        by name is when it runs: `int` for an `int` one, which the boundary
        passes only a real `int` and a native caller never a `bool` once the
        body shows the difference (`lowering.intness`), and
        `float` for a `float` one that shows whether it is an `int`; None when
        the body rebinds it or anything else."""
        info = self.info  # type: ignore[attr-defined]
        if f"{self.frontend.analysis.name}.{info.name}" != info.qualname or not (  # type: ignore[attr-defined]
            self.frontend.called_directly(info.name)  # type: ignore[attr-defined]
        ):
            return None  # a method, a nested function, or one called through a value
        parameter = next((p for p in info.params if p.name == name), None)
        if parameter is None or parameter.kind in {"var_positional", "var_keyword"}:
            return None
        declared = T.strip_literal(parameter.type)
        if declared == T.INT:
            exact = "int"
        elif declared == T.FLOAT and name in self.frontend.exact_params(info.qualname):  # type: ignore[attr-defined]
            exact = "float"
        else:
            return None
        if parameter.default is not None:
            default = self._type_of(parameter.default)  # type: ignore[attr-defined]
            if gives_bool(default) if exact == "int" else gives_int(default):
                return None
        augmented = {id(n.target) for n in ast.walk(info.node) if isinstance(n, ast.AugAssign)}
        for n in ast.walk(info.node):
            if (
                isinstance(n, ast.Name)
                and n.id == name
                and not isinstance(n.ctx, ast.Load)
                and id(n) not in augmented
            ):
                return None  # rebound: the checker's type at the use decides
            if isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names:
                return None
        return exact

    def _is_none_or(self, node: ast.Call, others: list[str]) -> Value | None:
        """`isinstance(node, (Node, type(None)))` of a `Node | None`."""
        handle, owned = self._handle(node.args[0])  # type: ignore[attr-defined]
        absent = self._negated(self._present(handle))  # type: ignore[attr-defined]
        self._done_with(handle, owned)  # type: ignore[attr-defined]
        if not others:
            return absent
        classes = node.args[1]
        rest = [
            e
            for e in (classes.elts if isinstance(classes, ast.Tuple) else [classes])
            if not (
                isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "type"
            )
        ]
        narrowed = ast.Call(node.func, [node.args[0], ast.Tuple(rest, ast.Load())], [])
        ast.copy_location(narrowed, node)
        self.__dict__.setdefault("_made_nodes", []).append(narrowed)
        present = super()._is_instance(narrowed)  # type: ignore[misc]
        if present is None:
            return None
        return core.bitwise(self.b, "or", absent, present)  # type: ignore[attr-defined]

    # -- powers ------------------------------------------------------------------------

    def _power(self, node: ast.BinOp) -> Value | None:
        """`a ** b` of ints: squaring and multiplying, each product checked, so a
        result past a word falls back to Python; a negative exponent, whose
        result is a float, too, unless the checker already typed it a float."""
        if not isinstance(node.op, ast.Pow):
            return None
        kinds = {self._plain_type(node.left), self._plain_type(node.right)}
        if not kinds <= {T.INT, T.BOOL}:
            return None
        if self._plain_type(node) == T.FLOAT:
            base = self._coerce(self._expr(node.left), "float")  # type: ignore[attr-defined]
            exponent = self._coerce(self._expr(node.right), "float")  # type: ignore[attr-defined]
            self._guard_float_power(base, exponent)
            self.frontend.module.require("math", 1)  # type: ignore[attr-defined]
            return math_dialect.call(self.b, "pow", base, exponent)  # type: ignore[attr-defined]
        base = self._coerce(self._expr(node.left), "int")  # type: ignore[attr-defined]
        exponent = self._coerce(self._expr(node.right), "int")  # type: ignore[attr-defined]
        return self._int_power(base, exponent, None)

    def _guard_float_power(self, base: Value, exponent: Value) -> None:
        """`0 ** -1` raises in Python where C gives an infinity."""
        zero = core.const(self.b, 0.0, F64)  # type: ignore[attr-defined]
        b = self.b  # type: ignore[attr-defined]
        fine = core.bitwise(
            b, "or", core.cmp(b, "ne", base, zero), core.cmp(b, "ge", exponent, zero)
        )
        self._guard(  # type: ignore[attr-defined]
            fine,
            "bounds",
            "0.0 cannot be raised to a negative power",
            raises="ZeroDivisionError: 0.0 cannot be raised to a negative power",
        )

    def _pow_call(self, node: ast.Call) -> Value | None:
        """`pow(a, b)` and `pow(a, b, m)` of ints."""
        if node.keywords or len(node.args) not in (2, 3):
            return None
        if not {self._plain_type(a) for a in node.args} <= {T.INT, T.BOOL}:
            if len(node.args) == 2 and {self._plain_type(a) for a in node.args} <= {
                T.INT,
                T.BOOL,
                T.FLOAT,
            }:
                base = self._coerce(self._expr(node.args[0]), "float")  # type: ignore[attr-defined]
                exponent = self._coerce(self._expr(node.args[1]), "float")  # type: ignore[attr-defined]
                self._guard_float_power(base, exponent)
                self.frontend.module.require("math", 1)  # type: ignore[attr-defined]
                return math_dialect.call(self.b, "pow", base, exponent)  # type: ignore[attr-defined]
            return None
        values = [self._coerce(self._expr(a), "int") for a in node.args]  # type: ignore[attr-defined]
        modulus = values[2] if len(values) == 3 else None
        if modulus is not None:
            self._guard(  # type: ignore[attr-defined]
                core.cmp(self.b, "ne", modulus, self._int_constant(0)),  # type: ignore[attr-defined]
                "bounds",
                "pow() 3rd argument cannot be 0",
                raises="ValueError: pow() 3rd argument cannot be 0",
            )
        return self._int_power(values[0], values[1], modulus)

    def _int_power(self, base: Value, exponent: Value, modulus: Value | None) -> Value:
        b = self.b  # type: ignore[attr-defined]
        zero = self._int_constant(0)  # type: ignore[attr-defined]
        one = self._int_constant(1)  # type: ignore[attr-defined]
        # A negative exponent gives a float, or with a modulus an inverse: Python's.
        core.guard(b, core.cmp(b, "ge", exponent, zero), "contract", "a negative exponent")

        def reduce(value: Value) -> Value:
            if modulus is None:
                return value
            return core.mod(b, value, modulus, overflow="wrap", rounding="floor")

        result = self._alloca(I64, "pow.result")  # type: ignore[attr-defined]
        square = self._alloca(I64, "pow.base")  # type: ignore[attr-defined]
        left = self._alloca(I64, "pow.exp")  # type: ignore[attr-defined]
        core.store(b, reduce(one), result)
        core.store(b, reduce(base), square)
        core.store(b, exponent, left)
        header = self._block("pow.head")  # type: ignore[attr-defined]
        body = self._block("pow.body")  # type: ignore[attr-defined]
        odd = self._block("pow.odd")  # type: ignore[attr-defined]
        after = self._block("pow.after")  # type: ignore[attr-defined]
        again = self._block("pow.square")  # type: ignore[attr-defined]
        done = self._block("pow.end")  # type: ignore[attr-defined]
        core.br(b, Successor(header))
        b.at_end(header)
        core.cond_br(
            b, core.cmp(b, "gt", core.load(b, left), zero), Successor(body), Successor(done)
        )
        b.at_end(body)
        bit = core.bitwise(b, "and", core.load(b, left), one)
        core.cond_br(b, core.cmp(b, "ne", bit, zero), Successor(odd), Successor(after))
        b.at_end(odd)
        product = self._checked_product(core.load(b, result), core.load(b, square))
        core.store(b, reduce(product), result)
        core.br(b, Successor(after))
        b.at_end(after)
        remaining = core.shift(b, "shr", core.load(b, left), one)
        core.store(b, remaining, left)
        # The base is squared only while a bit is left to use it: a square past
        # the word then means a result past it.
        core.cond_br(b, core.cmp(b, "gt", remaining, zero), Successor(again), Successor(done))
        b.at_end(again)
        current = core.load(b, square)
        core.store(b, reduce(self._checked_product(current, current)), square)
        core.br(b, Successor(header))
        b.at_end(done)
        return core.load(b, result)

    def _checked_product(self, left: Value, right: Value) -> Value:
        return self._checked_binary(left, right, "mul")  # type: ignore[attr-defined]

    # -- assignments ------------------------------------------------------------------

    def _chained_assign(self, node: ast.Assign) -> None:
        """`a = b = value`: the value once, bound to each target from left to right,
        the later ones reading it back from the first, a name."""
        first = node.targets[0]
        if not isinstance(first, ast.Name):
            if self._plain_type(node.value) not in (T.INT, T.FLOAT, T.BOOL):
                raise Unsupported("a chained assignment binds a name first")
            # A number, made once, then bound to each target in turn.
            read = self._once(node.value)
            for target in node.targets:
                each = ast.copy_location(ast.Assign([target], read), node)
                self.__dict__.setdefault("_made_nodes", []).append(each)
                self._assign(each)  # type: ignore[attr-defined]
            return
        head = ast.copy_location(ast.Assign([first], node.value), node)
        self._assign(head)  # type: ignore[attr-defined]
        for target in node.targets[1:]:
            read = self._typed(ast.Name(first.id, ast.Load()), self._type_of(first), first)  # type: ignore[attr-defined]
            following = ast.copy_location(ast.Assign([target], read), node)
            self.__dict__.setdefault("_made_nodes", []).append(following)
            self._assign(following)  # type: ignore[attr-defined]

    def _unpack_into_places(self, target: ast.expr, node: ast.Assign) -> bool:
        """`node.left, node.right = node.right, node.left`: a tuple display
        unpacked into fields or elements. Every value is made first, each into
        a hidden local, and then bound to its target from left to right, as
        Python evaluates the right side before it assigns."""
        value = node.value
        if not (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
            and not any(isinstance(e, ast.Starred) for e in [*target.elts, *value.elts])
            and any(isinstance(e, (ast.Attribute, ast.Subscript)) for e in target.elts)
        ):
            return False
        held: list[ast.expr] = []
        for element in value.elts:
            t = self._type_of(element)  # type: ignore[attr-defined]
            self._hidden_count += 1
            name = f".u{self._hidden_count}"
            made = self._typed(ast.Name(name, ast.Store()), t, element)
            first = ast.copy_location(ast.Assign([made], element), node)
            self.__dict__.setdefault("_made_nodes", []).append(first)
            self._assign(first)  # type: ignore[attr-defined]
            held.append(self._typed(ast.Name(name, ast.Load()), t, element))
        for place, read in zip(target.elts, held, strict=True):
            each = ast.copy_location(ast.Assign([place], read), node)
            self.__dict__.setdefault("_made_nodes", []).append(each)
            self._assign(each)  # type: ignore[attr-defined]
        return True

    def _assign_choice(self, target: ast.expr, node: ast.Assign) -> bool:
        """`r, c = (a, b) if flag else (b, a)`: the choice as an `if` statement,
        each side made and unpacked where it is chosen."""
        value = node.value
        if not (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, ast.IfExp)
            and isinstance(T.strip_literal(self._type_of(value)), T.Tuple_)  # type: ignore[attr-defined]
        ):
            return False
        branches = [
            [ast.copy_location(ast.Assign([target], side), node)]
            for side in (value.body, value.orelse)
        ]
        choice = ast.copy_location(ast.If(value.test, branches[0], branches[1]), node)
        self.__dict__.setdefault("_made_nodes", []).append(choice)
        self._if(choice)  # type: ignore[attr-defined]
        return True

    # -- a caught exception's `args` ----------------------------------------------------

    def _args_of(self, node: ast.expr) -> Value | None:
        """The slot of the caught exception whose `args` `node` reads."""
        if isinstance(node, ast.Attribute) and node.attr == "args":
            return self._caught_of(node.value)  # type: ignore[attr-defined]
        return None

    def _args_flags(self, slot: Value, wanted: int) -> Value:
        """The exception's flags, where `args` is one of the shapes `wanted`
        names; any other shape falls back."""
        exception = core.load(self.b, slot)  # type: ignore[attr-defined]
        address = self._field_address(exception, 3)  # type: ignore[attr-defined]
        flags = self._read(address, Shape("int"))  # type: ignore[attr-defined]
        b = self.b  # type: ignore[attr-defined]
        shaped = core.bitwise(b, "and", flags, self._word(wanted))  # type: ignore[attr-defined]
        core.guard(
            b,
            core.cmp(b, "ne", shaped, self._word(0)),  # type: ignore[attr-defined]
            "contract",
            "an exception's `args` beyond one string",
        )
        return flags

    def _first_arg(self, node: ast.expr) -> Value | None:
        """`e.args[0]` of an exception raised with one string: that string, owned."""
        if not (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in (0, -1)
            and type(node.slice.value) is int
        ):
            return None
        slot = self._args_of(node.value)
        if slot is None:
            return None
        self._args_flags(slot, ARGS_ONE_TEXT)
        return self._rt("ppy_exc_str", (core.load(self.b, slot),), HANDLE)  # type: ignore[attr-defined]

    def _args_shown(self, node: ast.expr) -> Value | None:
        """`str(e.args)`: `()` or `('message',)`."""
        slot = self._args_of(node)
        if slot is None:
            return None
        flags = self._args_flags(slot, ARGS_ONE_TEXT | ARGS_NONE)
        b = self.b  # type: ignore[attr-defined]
        builder = self._rt("ppy_str_builder", (self._word(0),), HANDLE)  # type: ignore[attr-defined]
        none = core.cmp(
            b, "ne", core.bitwise(b, "and", flags, self._word(ARGS_NONE)), self._word(0)
        )  # type: ignore[attr-defined]
        empty = self._block("args.empty")  # type: ignore[attr-defined]
        one = self._block("args.one")  # type: ignore[attr-defined]
        done = self._block("args.done")  # type: ignore[attr-defined]
        core.cond_br(b, none, Successor(empty), Successor(one))
        b.at_end(empty)
        self._add_text(builder, "()")  # type: ignore[attr-defined]
        core.br(b, Successor(done))
        b.at_end(one)
        self._add_text(builder, "(")  # type: ignore[attr-defined]
        text = self._rt("ppy_exc_str", (core.load(b, slot),), HANDLE)  # type: ignore[attr-defined]
        self._rt("ppy_str_add_repr", (builder, text))  # type: ignore[attr-defined]
        self._release(text)  # type: ignore[attr-defined]
        self._add_text(builder, ",)")  # type: ignore[attr-defined]
        core.br(b, Successor(done))
        b.at_end(done)
        return self._rt("ppy_str_finish", (builder,), HANDLE)  # type: ignore[attr-defined]

    def _shown_text(self, node: ast.expr) -> Value | None:
        found = self._first_arg(node)
        if found is None:
            found = self._args_shown(node)
        if found is not None:
            return found
        return super()._shown_text(node)  # type: ignore[misc]

    def _string_call(self, node: ast.Call, discard: bool) -> Value | None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "str"
            and len(node.args) == 1
            and not node.keywords
        ):
            found = self._first_arg(node.args[0])
            if found is None:
                found = self._args_shown(node.args[0])
            if found is not None:
                return self._keep_or_drop(found, discard)  # type: ignore[attr-defined]
        return super()._string_call(node, discard)  # type: ignore[misc]

    def _add_formatted(self, builder: Value, node: ast.expr, conversion: int, spec: str) -> None:
        if conversion in {-1, ord("s")} and not spec:
            found = self._first_arg(node)
            if found is None:
                found = self._args_shown(node)
            if found is not None:
                self._rt("ppy_str_add", (builder, found), None)  # type: ignore[attr-defined]
                self._release(found)  # type: ignore[attr-defined]
                return
        super()._add_formatted(builder, node, conversion, spec)  # type: ignore[misc]

    def _object_length(self, node: ast.expr) -> Value | None:
        """`len(e.args)`: 0 or 1."""
        slot = self._args_of(node)
        if slot is None:
            return super()._object_length(node)  # type: ignore[misc]
        flags = self._args_flags(slot, ARGS_ONE_TEXT | ARGS_NONE)
        b = self.b  # type: ignore[attr-defined]
        none = core.cmp(
            b, "ne", core.bitwise(b, "and", flags, self._word(ARGS_NONE)), self._word(0)
        )  # type: ignore[attr-defined]
        return core.select(b, none, self._word(0), self._word(1))  # type: ignore[attr-defined]
