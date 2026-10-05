"""Numbers that may be `None`, natively: `int | None`, `float | None`, `bool | None`.

Such a value is the number and a flag that says whether there is one. A
local holds them in two stack slots (`_FunctionLowering.optionals`); a
parameter, a result, a field, and an element hold them packed, as the IR
tuple `(number, flag)`, which crosses the native ABI as the number's atom
and a byte, and lies in memory as two words (`Shape("optional")`). `None` is
the flag clear with the number 0, so two `None`s are the same words.

What CPython does with one is done here in those terms: `x is None` is the
flag, `if x:` is the flag and the number, `x == 3` is the flag and the
comparison, `x or d` picks by both, and `print(x)` prints `None` where the
flag is clear. Arithmetic, an order comparison, or a negation that finds
`None` raises the `TypeError` CPython raises, with its text. A read the
checker narrowed (`x` after `if x is not None:`) is the number, checked.
"""

from __future__ import annotations

import ast

from ..analysis import types as T
from ..backend.llvm.lowering import Unsupported, optional_scalar
from ..ir import BOOL, F64, I64, IRType, Successor, TupleType, Value
from ..ir.dialects import core
from .collections import HANDLE, OPTIONAL_STR
from .intness import gives_bool, gives_int

__all__ = ["OptionalLowering", "optional_ir", "optional_kind"]

_SCALARS: dict[str, IRType] = {"int": I64, "float": F64, "bool": BOOL}
#: The checker's type of each kind of number.
_TYPES = {"int": T.INT, "float": T.FLOAT, "bool": T.BOOL}
#: How CPython names an operand's type in a `TypeError`.
_NAMES = {T.INT: "int", T.FLOAT: "float", T.BOOL: "bool", T.STR: "str", T.NONE: "NoneType"}
#: Each operator as CPython spells it in a `TypeError`.
_SYMBOLS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
    ast.Mod: "%", ast.Pow: "** or pow()", ast.LShift: "<<", ast.RShift: ">>",
    ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^", ast.MatMult: "@",
}  # fmt: skip
_ORDERS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
_UNARY = {ast.USub: "unary -", ast.UAdd: "unary +", ast.Invert: "unary ~"}


def _plain(node: ast.expr) -> bool:
    """Whether evaluating `node` does nothing but read: a name or a constant."""
    return isinstance(node, (ast.Name, ast.Constant))


def optional_kind(t: T.Type) -> str | None:
    """ "int", "float", or "bool" for `int | None` and the like, else None."""
    return optional_scalar(t)


def optional_ir(kind: str) -> TupleType:
    """The packed IR value of a number of `kind` or `None`: the number, the flag."""
    return TupleType((_SCALARS[kind], BOOL))


def _text_or_none_type(t: T.Type) -> bool:
    """Whether `t` is `str | None`."""
    base = T.strip_literal(t)
    return (
        isinstance(base, T.Union_)
        and T.is_optional(base)
        and T.strip_literal(T.remove_none(base)) == T.STR
    )


def is_packed(t: IRType) -> bool:
    return (
        isinstance(t, TupleType)
        and len(t.items) == 2
        and t.items[1] == BOOL
        and t.items[0] in (I64, F64, BOOL)
    )


class OptionalLowering:  # pylint: disable=attribute-defined-outside-init
    """Mixed into `_FunctionLowering`."""

    # -- what an expression is ----------------------------------------------------

    def _optional_kind_of(self, node: ast.expr) -> str | None:
        """The kind of number `node` is or `None`, where the checker says it may
        be `None` there; None where it is a plain number (or anything else)."""
        if (
            isinstance(node, ast.Name)
            and node.id in self.optionals  # type: ignore[attr-defined]
            and T.strip_literal(self._type_of(node)) == T.NONE  # type: ignore[attr-defined]
        ):
            return self.optionals[node.id][2]  # type: ignore[attr-defined]
        return optional_kind(self._type_of(node))  # type: ignore[attr-defined]

    def _maybe_none(self, node: ast.expr) -> bool:
        """Whether `node` is a number that may be `None` here, or `None` itself.
        A field or an element the checker narrowed may be `None` all the same:
        a call between the test and the read may have set it to `None`."""
        if isinstance(node, ast.Constant) and node.value is None:
            return True
        if isinstance(node, ast.Name):
            # A local's narrowing holds: nothing but its own code binds it.
            return self._optional_kind_of(node) is not None
        return self._either_kind(node) is not None

    def _either_kind(self, node: ast.expr) -> str | None:
        """The kind of number `node` is or `None`, as the checker says it here
        or as what holds it holds it: a local, a field, an element. The
        checker's narrowing of a field outlives what can change the field (a
        call, a branch it was narrowed in), so what tolerates `None` (`is`,
        `==`, truth, printing) asks the flag whatever the checker says."""
        if isinstance(node, ast.Name) and node.id in self.optionals:  # type: ignore[attr-defined]
            return self.optionals[node.id][2]  # type: ignore[attr-defined]
        return self._optional_kind_of(node) or self._stored_kind(node)

    # -- packing ---------------------------------------------------------------------

    def _pack(self, present: Value, value: Value) -> Value:
        return core.tuple_make(self.b, value, present)  # type: ignore[attr-defined]

    def _unpack(self, packed: Value) -> tuple[Value, Value]:
        return (
            core.tuple_extract(self.b, packed, 1),  # type: ignore[attr-defined]
            core.tuple_extract(self.b, packed, 0),  # type: ignore[attr-defined]
        )

    def _zero(self, kind: str) -> Value:
        b = self.b  # type: ignore[attr-defined]
        if kind == "float":
            return core.const(b, 0.0, F64)
        if kind == "bool":
            return core.const(b, False, BOOL)
        return core.const(b, 0, I64)

    def _as_kind(self, value: Value, kind: str, node: ast.expr) -> Value:
        """A number given where `kind` or `None` is declared: CPython keeps an
        `int` an `int` in a `float | None`, and a `bool` a `bool` in an
        `int | None`, which a flag and a number of `kind` cannot."""
        held = {I64: "int", F64: "float", BOOL: "bool"}.get(value.type)
        if held == kind:
            return value
        if kind == "float" and (held != "float" or gives_int(self._type_of(node))):  # type: ignore[attr-defined]
            raise Unsupported(f"an `{held}` where `float | None` is declared, which CPython keeps")
        if kind == "int" and (held == "bool" or gives_bool(self._type_of(node))):  # type: ignore[attr-defined]
            raise Unsupported("a `bool` where `int | None` is declared, which CPython keeps")
        if held is None:
            raise Unsupported(f"`{ast.unparse(node)}` is not a number")
        return self._coerce(value, kind)  # type: ignore[attr-defined]

    # -- evaluating one ------------------------------------------------------------

    def _optional_pair(self, node: ast.expr, kind: str) -> tuple[Value, Value]:
        """`node` as a number of `kind` or `None`: whether it holds a number,
        and the number (0 where it does not)."""
        b = self.b  # type: ignore[attr-defined]
        if isinstance(node, ast.Constant) and node.value is None:
            return core.const(b, False, BOOL), self._zero(kind)
        if isinstance(node, ast.Name) and node.id in self.optionals:  # type: ignore[attr-defined]
            present_slot, slot, held = self.optionals[node.id]  # type: ignore[attr-defined]
            present = core.load(b, present_slot)
            value = core.load(b, slot)
            if held != kind:
                value = self._as_kind(value, kind, node)
            return present, value
        if isinstance(node, ast.IfExp):
            return self._optional_choice(node, kind)
        got = self._optional_get(node)  # type: ignore[attr-defined]
        if got is not None:
            return got[0], self._as_kind(got[1], kind, node)
        if self._either_kind(node) is not None:
            present, value = self._unpack(self._optional_read(node))
            return present, self._as_kind(value, kind, node)
        if T.strip_literal(self._type_of(node)) == T.NONE:  # type: ignore[attr-defined]
            raise Unsupported(f"`{ast.unparse(node)}` is `None` here, a value native code has not")
        value = self._expr(node)  # type: ignore[attr-defined]
        return core.const(b, True, BOOL), self._as_kind(value, kind, node)

    def _optional_packed(self, node: ast.expr, kind: str) -> Value:
        present, value = self._optional_pair(node, kind)
        return self._pack(present, value)

    def _optional_read(self, node: ast.expr) -> Value:
        """A call, a field, or an element that is a number or `None`, packed."""
        if isinstance(node, ast.Call):
            found = self._call(node)  # type: ignore[attr-defined]
        elif isinstance(node, ast.Attribute) and self._object_of(node.value) is not None:  # type: ignore[attr-defined]
            found = self._field_value(node)  # type: ignore[attr-defined]
        elif (
            isinstance(node, ast.Subscript)
            and not isinstance(node.slice, ast.Slice)
            and self._is_collection(node.value)  # type: ignore[attr-defined]
        ):
            found = self._item(node.value, node.slice)  # type: ignore[attr-defined]
        else:
            raise Unsupported(f"`{ast.unparse(node)}` may be `None`, which has no native number")
        if not is_packed(found.type):
            raise Unsupported(f"`{ast.unparse(node)}` may be `None`, which has no native number")
        return found

    def _optional_choice(self, node: ast.IfExp, kind: str) -> tuple[Value, Value]:
        """`a if c else b` of numbers or `None`: only the side `c` picks runs."""
        condition = self._test(node.test)  # type: ignore[attr-defined]
        then = self._block("optional.then")  # type: ignore[attr-defined]
        other = self._block("optional.else")  # type: ignore[attr-defined]
        done = self._block("optional.end")  # type: ignore[attr-defined]
        present = done.add_argument(BOOL, "present")
        value = done.add_argument(_SCALARS[kind], "value")
        core.cond_br(self.b, condition, Successor(then), Successor(other))  # type: ignore[attr-defined]
        for block, side in ((then, node.body), (other, node.orelse)):
            self.b.at_end(block)  # type: ignore[attr-defined]
            found = self._optional_pair(side, kind)
            core.br(self.b, Successor(done, list(found)))  # type: ignore[attr-defined]
        self.b.at_end(done)  # type: ignore[attr-defined]
        return present, value

    def _optional_number(self, node: ast.expr, operation: str = "") -> Value:
        """The number of a value the checker says holds one here (or, with an
        `operation`, one CPython would raise its `TypeError` for where it is
        `None`): checked, since a field may change under a narrowing."""
        kind = self._stored_kind(node)
        assert kind is not None
        present, value = self._unpack(self._optional_read(node))
        raises = operation or "TypeError: a value narrowed to a number is None"
        self._guard(present, "contract", "a number is None", raises=raises)  # type: ignore[attr-defined]
        return value

    def _stored_kind(self, node: ast.expr) -> str | None:
        """The kind of number a call, a field, or an element holds where it may
        also hold `None`, whatever the checker narrowed it to here."""
        if isinstance(node, ast.Attribute):
            shape = self._object_of(node.value)  # type: ignore[attr-defined]
            if shape is None:
                return None
            try:
                _offset, field = self._field(shape, node.attr)  # type: ignore[attr-defined]
            except Unsupported:
                return None
            return field.parts[0] if field.kind == "optional" and field != OPTIONAL_STR else None
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Slice):
            kind = self._kind_of(node.value)  # type: ignore[attr-defined]
            if kind is None or kind.value is None or kind.value == OPTIONAL_STR:
                return None
            return kind.value.parts[0] if kind.value.kind == "optional" else None
        if isinstance(node, ast.Call):
            return self._optional_kind_of(node)
        return None

    def _narrowed_read(self, node: ast.expr) -> Value | None:
        """`self.label` where the checker narrowed it to a number: the number,
        checked. A read that may still be `None` stays in Python. None where
        `node` reads nothing that may be `None`."""
        if not isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
            return None
        stored = self._stored_kind(node)
        if stored is None:
            return None
        if self._optional_kind_of(node) is not None:
            raise Unsupported(f"`{ast.unparse(node)}` may be `None`, which has no native number")
        return self._optional_number(node)

    def _optional_expr(self, node: ast.expr) -> Value | None:
        """An expression `_expr` meets that involves a number that may be
        `None`, as a plain value; None where it involves none."""
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Pow, ast.MatMult)):
                return None
            return self._optional_binary(node)
        if isinstance(node, ast.UnaryOp):
            return None if isinstance(node.op, ast.Not) else self._optional_unary(node)
        if isinstance(node, ast.BoolOp):
            return self._optional_or(node)
        return self._narrowed_read(node)

    def _optional_compare(self, node: ast.Compare) -> Value | None:
        """A comparison with a side that may be `None`."""
        found = self._optional_is(node)
        if found is None:
            found = self._optional_equal(node)
        if found is None:
            found = self._optional_order(node)
        return found

    # -- strings that may be `None` --------------------------------------------------

    def _text_or_none(self, node: ast.expr) -> object | None:
        """`OPTIONAL_STR` where the checker says `node` is a string or `None`."""
        return OPTIONAL_STR if _text_or_none_type(self._type_of(node)) else None  # type: ignore[attr-defined]

    def _local_text_or_none(self, name: str) -> bool:
        """Whether a local is a string at one binding and `None` at another."""
        analysis = self.frontend.analysis.functions.get(self.info.qualname)  # type: ignore[attr-defined]
        final = analysis.locals.get(name) if analysis is not None else None
        return final is not None and _text_or_none_type(final)

    def _maybe_text(self, node: ast.expr) -> bool:
        """Whether `node` is a string that may be `None`, as the checker says it
        here or as what holds it holds it (a local, a field, an element)."""
        if isinstance(node, ast.Name):
            held = self.collections.get(node.id)  # type: ignore[attr-defined]
            return held is not None and held.kind == OPTIONAL_STR
        if self._text_or_none(node) is not None:
            return True
        if isinstance(node, ast.Attribute):
            shape = self._object_of(node.value)  # type: ignore[attr-defined]
            if shape is None:
                return False
            try:
                _offset, field = self._field(shape, node.attr)  # type: ignore[attr-defined]
            except Unsupported:
                return False
            return field == OPTIONAL_STR
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Slice):
            kind = self._kind_of(node.value)  # type: ignore[attr-defined]
            return kind is not None and kind.value == OPTIONAL_STR
        return False

    def _raw_handle(self, node: ast.expr) -> tuple[Value, bool]:
        """A string that may be `None`, as its handle: null for `None`, whatever
        the checker narrowed it to."""
        self._reading_raw = True
        try:
            return self._handle(node)  # type: ignore[attr-defined,no-any-return]
        finally:
            self._reading_raw = False

    def _narrowed_text(self, node: ast.expr) -> tuple[Value, bool] | None:
        """`self.name` where the checker narrowed a field (or an element) that
        may be `None` to its string: the handle, checked, since a call between
        the test and the read may have set it to `None`."""
        if self.__dict__.get("_reading_raw") or not isinstance(
            node, (ast.Attribute, ast.Subscript)
        ):
            return None
        if T.strip_literal(self._type_of(node)) != T.STR or not self._maybe_text(node):  # type: ignore[attr-defined]
            return None
        handle, owned = self._raw_handle(node)
        self._guard(  # type: ignore[attr-defined]
            self._present(handle),  # type: ignore[attr-defined]
            "contract",
            "a string narrowed from `None` is None",
            raises="TypeError: a value narrowed to a string is None",
        )
        return handle, owned

    def _text_equal(self, node: ast.Compare) -> Value:
        """`s == t` where a side is a string that may be `None`: `None` equals
        only `None`, a string only an equal string, and nothing else either."""
        b = self.b  # type: ignore[attr-defined]
        op = node.ops[0]
        handles = []
        for side in (node.left, node.comparators[0]):
            if isinstance(side, ast.Constant) and side.value is None:
                handles.append((self._rt("ppy_coll_none", (), HANDLE), False))  # type: ignore[attr-defined]
            elif self._maybe_text(side) or self._string_of(side) is not None:  # type: ignore[attr-defined]
                handles.append(self._raw_handle(side))
            else:
                # A number is never a string, and never `None`.
                self._expr(side)  # type: ignore[attr-defined]
                handles.append(None)
        if any(found is None for found in handles):
            for found in handles:
                if found is not None:
                    self._done_with(*found)  # type: ignore[attr-defined]
            return core.const(b, isinstance(op, ast.NotEq), BOOL)
        (first, first_owned), (second, second_owned) = handles[0], handles[1]  # type: ignore[misc]
        same = self._rt("ppy_str_equal", (first, second))  # type: ignore[attr-defined]
        self._done_with(first, first_owned)  # type: ignore[attr-defined]
        self._done_with(second, second_owned)  # type: ignore[attr-defined]
        return core.cmp(b, "eq" if isinstance(op, ast.Eq) else "ne", same, self._word(1))  # type: ignore[attr-defined]

    def _text_truth(self, node: ast.expr, empty: bool) -> Value:
        """`if s:` of a string that may be `None`: there, and not empty."""
        b = self.b  # type: ignore[attr-defined]
        handle, owned = self._raw_handle(node)
        present = self._present(handle)  # type: ignore[attr-defined]
        there = self._block("text.there")  # type: ignore[attr-defined]
        done = self._block("text.truth")  # type: ignore[attr-defined]
        truth = done.add_argument(BOOL, "truth")
        core.cond_br(b, present, Successor(there), Successor(done, [present]))
        self.b.at_end(there)  # type: ignore[attr-defined]
        length = self._rt("ppy_str_bytes", (handle,))  # type: ignore[attr-defined]
        core.br(self.b, Successor(done, [core.cmp(self.b, "gt", length, self._word(0))]))  # type: ignore[attr-defined]
        self.b.at_end(done)  # type: ignore[attr-defined]
        self._done_with(handle, owned)  # type: ignore[attr-defined]
        if empty:
            return core.bitwise(self.b, "xor", truth, core.const(self.b, True, BOOL))  # type: ignore[attr-defined]
        return truth

    def _text_formatted(self, builder: Value, node: ast.expr, spec: str, conversion: int) -> bool:
        """`f"{s}"`, `str(s)`, `repr(s)` of a string that may be `None`."""
        if spec:
            raise Unsupported("a format spec over a value that may be `None` stays in Python")
        handle, owned = self._raw_handle(node)
        there = self._block("format.text")  # type: ignore[attr-defined]
        absent = self._block("format.none")  # type: ignore[attr-defined]
        done = self._block("format.done")  # type: ignore[attr-defined]
        core.cond_br(self.b, self._present(handle), Successor(there), Successor(absent))  # type: ignore[attr-defined]
        self.b.at_end(there)  # type: ignore[attr-defined]
        if conversion in (ord("r"), ord("a")):
            self._add_repr(builder, handle)  # type: ignore[attr-defined]
        else:
            self._rt("ppy_str_add", (builder, handle), None)  # type: ignore[attr-defined]
        core.br(self.b, Successor(done))  # type: ignore[attr-defined]
        self.b.at_end(absent)  # type: ignore[attr-defined]
        self._add_text(builder, "None")  # type: ignore[attr-defined]
        core.br(self.b, Successor(done))  # type: ignore[attr-defined]
        self.b.at_end(done)  # type: ignore[attr-defined]
        self._done_with(handle, owned)  # type: ignore[attr-defined]
        return True

    def _text_or(self, node: ast.BoolOp) -> Value | None:
        """`name or "default"` of strings, one of which may be `None`: the first
        that is there and not empty, else the last; an owned handle."""
        if not isinstance(node.op, ast.Or) or T.strip_literal(self._type_of(node)) != T.STR:  # type: ignore[attr-defined]
            return None
        if not all(
            self._maybe_text(v)
            or self._string_of(v) is not None  # type: ignore[attr-defined]
            or (isinstance(v, ast.Constant) and v.value is None)
            for v in node.values
        ):
            return None
        done = self._block("text.or")  # type: ignore[attr-defined]
        result = done.add_argument(HANDLE, "or")
        for index, value_node in enumerate(node.values):
            handle, owned = self._raw_handle(value_node)
            if not owned:
                self._retain(handle)  # type: ignore[attr-defined]
            if index == len(node.values) - 1:
                core.br(self.b, Successor(done, [handle]))  # type: ignore[attr-defined]
                break
            there = self._block("text.or.there")  # type: ignore[attr-defined]
            following = self._block("text.or.next")  # type: ignore[attr-defined]
            core.cond_br(self.b, self._present(handle), Successor(there), Successor(following))  # type: ignore[attr-defined]
            self.b.at_end(there)  # type: ignore[attr-defined]
            length = self._rt("ppy_str_bytes", (handle,))  # type: ignore[attr-defined]
            full = core.cmp(self.b, "gt", length, self._word(0))  # type: ignore[attr-defined]
            core.cond_br(self.b, full, Successor(done, [handle]), Successor(following))  # type: ignore[attr-defined]
            self.b.at_end(following)  # type: ignore[attr-defined]
            self._release(handle)  # type: ignore[attr-defined]
        self.b.at_end(done)  # type: ignore[attr-defined]
        return result

    def _formats_maybe_none(self, node: ast.expr) -> bool:
        """Whether an f-string field may be `None`, which the string builder
        writes (`_optional_formatted`), not a standalone print's scalar shims."""
        return self._either_kind(node) is not None or self._maybe_text(node)

    def _prints_maybe_none(self, argument: ast.expr) -> bool:
        """A local `print` is handed that may be `None` here."""
        if not isinstance(argument, ast.Name):
            return False
        t = T.strip_literal(self._type_of(argument))  # type: ignore[attr-defined]
        if argument.id in self.optionals:  # type: ignore[attr-defined]
            return t != _TYPES[self.optionals[argument.id][2]]  # type: ignore[attr-defined]
        held = self.collections.get(argument.id)  # type: ignore[attr-defined]
        return held is not None and held.kind == OPTIONAL_STR and t != T.STR

    # -- locals ----------------------------------------------------------------------

    def _optional_local(self, node: ast.expr, kind: str) -> ast.Name:
        """A name for `node`'s value: itself where it is such a local, else a
        hidden local holding it, which the name machinery then reads."""
        if isinstance(node, ast.Name) and node.id in self.optionals:  # type: ignore[attr-defined]
            return node
        count = self.__dict__.get("_optional_hidden", 0)
        self._optional_hidden = count + 1
        name = f".optional{count}"
        if kind == "str":
            handle, owned = self._raw_handle(node)
            self._bind(name, OPTIONAL_STR, handle, owned)  # type: ignore[attr-defined]
            made = ast.Name(name, ast.Load())
            return self._typed(made, T.union(T.STR, T.NONE), node)  # type: ignore[attr-defined,return-value]
        present, value = self._optional_pair(node, kind)
        present_slot = self._alloca(BOOL, f"{name}.present")  # type: ignore[attr-defined]
        slot = self._alloca(_SCALARS[kind], name)  # type: ignore[attr-defined]
        core.store(self.b, present, present_slot)  # type: ignore[attr-defined]
        core.store(self.b, value, slot)  # type: ignore[attr-defined]
        self.optionals[name] = (present_slot, slot, kind)  # type: ignore[attr-defined]
        made = ast.Name(name, ast.Load())
        return self._typed(made, T.union(_TYPES[kind], T.NONE), node)  # type: ignore[attr-defined,return-value]

    def _name_optional_arguments(self, call: ast.Call) -> ast.Call | None:
        """`print(f(x), node.label)`: the call with each argument that may be
        `None` and is not a local already made a hidden local, evaluated in
        order; None where there is none."""
        wanted = [
            i
            for i, argument in enumerate(call.args)
            if not isinstance(argument, ast.Name)
            and (self._either_kind(argument) is not None or self._maybe_text(argument))
        ]
        if not wanted:
            return None
        if any(isinstance(argument, ast.Starred) for argument in call.args):
            raise Unsupported("a value that may be `None` is printed with a starred argument")
        args = list(call.args)
        for i in range(wanted[-1] + 1):
            if i in wanted:
                kind = self._either_kind(args[i]) or "str"
                args[i] = self._optional_local(args[i], kind)
            elif not _plain(args[i]):
                # Evaluated here, in its turn, ahead of the ones after it.
                args[i] = self._scalar_local(args[i])
        made = ast.copy_location(ast.Call(func=call.func, args=args, keywords=call.keywords), call)
        return self._typed(made, T.NONE, call)  # type: ignore[attr-defined,return-value]

    def _scalar_local(self, node: ast.expr) -> ast.Name:
        """A hidden local holding what `node` gives, evaluated now: a number, or
        a string, a collection, or an object, held by handle."""
        t = T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]
        self._hidden_count += 1  # type: ignore[attr-defined]
        name = f".t{self._hidden_count}"  # type: ignore[attr-defined]
        if t not in (T.INT, T.FLOAT, T.BOOL):
            kind = self._reference_of(node)  # type: ignore[attr-defined]
            if kind is None:
                raise Unsupported(
                    "a value that may be `None` is printed after one with no native form"
                )
            handle, owned = self._handle(node)  # type: ignore[attr-defined]
            self._bind(name, kind, handle, owned)  # type: ignore[attr-defined]
            return self._typed(ast.Name(name, ast.Load()), t, node)  # type: ignore[attr-defined,return-value]
        value = self._expr(node)  # type: ignore[attr-defined]
        self._store(ast.Name(name, ast.Store()), value)  # type: ignore[attr-defined]
        return self._typed(ast.Name(name, ast.Load()), t, node)  # type: ignore[attr-defined,return-value]

    def _bind_optional_parameter(self, name: str, kind: str, argument: Value) -> None:
        """A parameter that is a number or `None`: its flag and its number, each
        in a slot of its own, as a local of its type holds them."""
        present_slot = self._alloca(BOOL, f"{name}.present")  # type: ignore[attr-defined]
        slot = self._alloca(_SCALARS[kind], name)  # type: ignore[attr-defined]
        present, value = self._unpack(argument)
        core.store(self.b, present, present_slot)  # type: ignore[attr-defined]
        core.store(self.b, value, slot)  # type: ignore[attr-defined]
        self.optionals[name] = (present_slot, slot, kind)  # type: ignore[attr-defined]

    def _bind_optional(self, target: ast.expr, kind: str, packed: Value) -> None:
        """A loop target bound to an element that is a number or `None`."""
        if not isinstance(target, ast.Name):
            raise Unsupported("an element that may be `None` is bound to a name")
        found = self.optionals.get(target.id)  # type: ignore[attr-defined]
        if found is None:
            if target.id in self.slots or target.id in self.collections:  # type: ignore[attr-defined]
                raise Unsupported(f"`{target.id}` is a number here and may be `None` there")
            present_slot = self._alloca(BOOL, f"{target.id}.present")  # type: ignore[attr-defined]
            slot = self._alloca(_SCALARS[kind], target.id)  # type: ignore[attr-defined]
            found = (present_slot, slot, kind)
            self.optionals[target.id] = found  # type: ignore[attr-defined]
        present_slot, slot, held = found
        if held != kind:
            raise Unsupported(f"`{target.id}` holds a `{held}` or `None`, not a `{kind}`")
        present, value = self._unpack(packed)
        core.store(self.b, present, present_slot)  # type: ignore[attr-defined]
        core.store(self.b, value, slot)  # type: ignore[attr-defined]

    def _augmented_optional(self, read: ast.expr, combined: ast.BinOp) -> None:
        """`node.label += 1`, `xs[i] += 1` where what is read may be `None`: the
        read typed so, and the sum a number, so that CPython's `TypeError`
        for `+=` is raised where it is `None`."""
        kind = self._stored_kind(read)
        if kind is None:
            return
        self._typed(read, T.union(_TYPES[kind], T.NONE), read)  # type: ignore[attr-defined]
        self._typed(combined, _TYPES[kind] if kind != "bool" else T.INT, combined)  # type: ignore[attr-defined]
        combined.ppy_augmented = True  # type: ignore[attr-defined]

    def _optional_formatted(
        self, builder: Value, node: ast.expr, spec: str, conversion: int = -1
    ) -> bool:
        """`f"{x}"` and `str(x)` of a number that may be `None`: `None`, or the
        number as it is written."""
        if self._maybe_text(node) and not (
            isinstance(node, ast.Name) and T.strip_literal(self._type_of(node)) == T.STR  # type: ignore[attr-defined]
        ):
            return self._text_formatted(builder, node, spec, conversion)
        kind = self._either_kind(node)
        if kind is None:
            return False
        if spec:
            raise Unsupported("a format spec over a value that may be `None` stays in Python")
        present, value = self._optional_pair(node, kind)
        number = self._block("format.number")  # type: ignore[attr-defined]
        absent = self._block("format.none")  # type: ignore[attr-defined]
        done = self._block("format.done")  # type: ignore[attr-defined]
        core.cond_br(self.b, present, Successor(number), Successor(absent))  # type: ignore[attr-defined]
        self.b.at_end(number)  # type: ignore[attr-defined]
        self._add_plain(builder, kind, value)  # type: ignore[attr-defined]
        core.br(self.b, Successor(done))  # type: ignore[attr-defined]
        self.b.at_end(absent)  # type: ignore[attr-defined]
        self._add_text(builder, "None")  # type: ignore[attr-defined]
        core.br(self.b, Successor(done))  # type: ignore[attr-defined]
        self.b.at_end(done)  # type: ignore[attr-defined]
        return True

    # -- results -------------------------------------------------------------------

    def _optional_result(self) -> str | None:
        """The kind of number the function returns where it may return `None`."""
        results = self.function.results  # type: ignore[attr-defined]
        if len(results) != 1 or not is_packed(results[0]):
            return None
        return optional_kind(
            T.substitute(self.info.ret, self.bindings) if self.bindings else self.info.ret
        )  # type: ignore[attr-defined]

    def _return_optional(self, value: ast.expr | None) -> bool:
        """`return x`, `return None`, or falling off the end of a function that
        returns a number or `None`."""
        kind = self._optional_result()
        if kind is None:
            return False
        if value is None:
            packed = self._pack(core.const(self.b, False, BOOL), self._zero(kind))  # type: ignore[attr-defined]
        else:
            packed = self._optional_packed(value, kind)
        self._check_thread_failures()  # type: ignore[attr-defined]
        self._leave_for_return()  # type: ignore[attr-defined]
        self._release_collections()  # type: ignore[attr-defined]
        core.ret(self.b, packed)  # type: ignore[attr-defined]
        return True

    # -- tests -----------------------------------------------------------------------

    def _optional_is(self, node: ast.Compare) -> Value | None:
        """`x is None`, `x is not None`, `x == None`, `x != None` of any
        expression that may be `None`: its flag."""
        op = node.ops[0]
        if not isinstance(op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
            return None
        left, right = node.left, node.comparators[0]
        if isinstance(left, ast.Constant) and left.value is None:
            left, right = right, left
        if not (isinstance(right, ast.Constant) and right.value is None):
            return None
        if self._maybe_text(left):
            handle, owned = self._raw_handle(left)
            present = self._present(handle)  # type: ignore[attr-defined]
            self._done_with(handle, owned)  # type: ignore[attr-defined]
            if isinstance(op, (ast.IsNot, ast.NotEq)):
                return present
            return core.bitwise(self.b, "xor", present, core.const(self.b, True, BOOL))  # type: ignore[attr-defined]
        kind = self._either_kind(left)
        if kind is None:
            return None
        present, _value = self._optional_pair(left, kind)
        if isinstance(op, (ast.IsNot, ast.NotEq)):
            return present
        return core.bitwise(self.b, "xor", present, core.const(self.b, True, BOOL))  # type: ignore[attr-defined]

    def _optional_equal(self, node: ast.Compare) -> Value | None:
        """`x == 3`, `x != y` where a side may be `None`: `None` equals only
        `None`, and two numbers compare as numbers."""
        op = node.ops[0]
        if not isinstance(op, (ast.Eq, ast.NotEq)):
            return None
        left, right = node.left, node.comparators[0]
        if self._maybe_text(left) or self._maybe_text(right):
            return self._text_equal(node)
        kinds = [self._either_kind(side) for side in (left, right)]
        if kinds == [None, None]:
            return None
        b = self.b  # type: ignore[attr-defined]
        pairs = []
        for side, kind in zip((left, right), kinds, strict=True):
            if kind is not None:
                pairs.append(self._optional_pair(side, kind))
            else:
                pairs.append((core.const(b, True, BOOL), self._expr(side)))  # type: ignore[attr-defined]
        (lp, lv), (rp, rv) = pairs[0], pairs[1]
        common = self._unify(self._kind(lv), self._kind(rv))  # type: ignore[attr-defined]
        numbers = core.cmp(b, "eq", self._coerce(lv, common), self._coerce(rv, common))  # type: ignore[attr-defined]
        both = core.bitwise(b, "and", lp, rp)
        neither = core.bitwise(b, "xor", core.bitwise(b, "or", lp, rp), core.const(b, True, BOOL))
        equal = core.bitwise(b, "or", core.bitwise(b, "and", both, numbers), neither)
        if isinstance(op, ast.NotEq):
            return core.bitwise(b, "xor", equal, core.const(b, True, BOOL))
        return equal

    def _optional_in_display(
        self, item: ast.expr, elements: list[ast.expr], like: ast.Compare
    ) -> Value | None:
        """`x in (True, None)` where `x` may be `None`, or a display holds
        `None`: `x == e` for each `e` in turn, `None` equal only to `None`."""
        kind = self._either_kind(item)
        nones = [isinstance(e, ast.Constant) and e.value is None for e in elements]
        if kind is None and not any(nones):
            return None
        if kind is None or kind == "float":
            # A plain number is never `None`; a float's NaN is Python's to find.
            return None
        if not all(
            none or (isinstance(e, ast.Constant) and type(e.value) in (int, bool))
            for e, none in zip(elements, nones, strict=True)
        ):
            return None
        left = self._optional_local(item, kind)
        tests = [
            (lambda e=e: self._pair(left, ast.Eq(), e, like))  # type: ignore[attr-defined,misc]
            for e in elements
        ]
        return self._any_of(tests, stop_on=True)  # type: ignore[attr-defined]

    def _optional_isinstance(
        self, subject: ast.expr, wanted: list[str], possible: set[str]
    ) -> Value | None:
        """`isinstance(x, int)` where `x` is a number or `None`: one answer for
        the number, whatever class it is, and one for `None`; the flag picks."""
        kind = self._either_kind(subject)
        if kind is None or "NoneType" not in possible:
            return None
        answers = {
            any(name in T.BUILTIN_MRO.get(runtime, (runtime,)) for name in wanted)
            for runtime in possible - {"NoneType"}
        }
        if len(answers) != 1:
            return None
        present, _value = self._optional_pair(subject, kind)
        number = core.const(self.b, answers.pop(), BOOL)  # type: ignore[attr-defined]
        mro = T.BUILTIN_MRO.get("NoneType", ("NoneType", "object"))
        none = core.const(self.b, any(name in mro for name in wanted), BOOL)  # type: ignore[attr-defined]
        return core.select(self.b, present, number, none)  # type: ignore[attr-defined]

    def _kind(self, value: Value) -> str:
        return {I64: "int", F64: "float", BOOL: "bool"}.get(value.type, "int")

    def _optional_test(self, node: ast.expr, *, empty: bool = False) -> Value | None:
        """`if x:` of a value that may be `None`: it holds a number, and the
        number is not zero (`not x`, with `empty`)."""
        if self._maybe_text(node) and not (
            isinstance(node, ast.Name) and T.strip_literal(self._type_of(node)) == T.STR  # type: ignore[attr-defined]
        ):
            return self._text_truth(node, empty)
        kind = self._either_kind(node)
        if kind is None:
            return None
        present, value = self._optional_pair(node, kind)
        truth = core.bitwise(self.b, "and", present, self._truth(value))  # type: ignore[attr-defined]
        if empty:
            return core.bitwise(self.b, "xor", truth, core.const(self.b, True, BOOL))  # type: ignore[attr-defined]
        return truth

    # -- operations that raise on `None` ---------------------------------------------

    def _operand_name(self, node: ast.expr) -> str:
        t = T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]
        if self._maybe_none(node):
            kind = self._either_kind(node)
            t = _TYPES[kind] if kind is not None else T.NONE
        return _NAMES.get(t, "")

    def _optional_operands(
        self, left: ast.expr, right: ast.expr, message: str
    ) -> tuple[Value, Value] | None:
        """Both operands of a binary operation where one may be `None`: the
        numbers, past CPython's `TypeError` where either is `None`. `message`
        is the error with `{0}` and `{1}` for the operands' type names."""
        if not (self._maybe_none(left) or self._maybe_none(right)):
            return None
        b = self.b  # type: ignore[attr-defined]
        values = []
        flags = []
        for side in (left, right):
            kind = self._either_kind(side)
            if isinstance(side, ast.Constant) and side.value is None:
                values.append(None)
                flags.append(core.const(b, False, BOOL))
            elif kind is not None:
                present, value = self._optional_pair(side, kind)
                values.append(value)
                flags.append(present)
            else:
                values.append(self._expr(side))  # type: ignore[attr-defined]
                flags.append(None)
        names = [self._operand_name(side) for side in (left, right)]
        if not all(names):
            raise Unsupported("an operand that may be `None` meets one of another type")
        # Where both are there, the operation goes on; otherwise the message
        # names what each side turned out to be.
        cases = []
        lp, rp = flags[0], flags[1]
        true = core.const(b, True, BOOL)
        for left_none in (False, True):
            for right_none in (False, True):
                if not (left_none or right_none):
                    continue
                if (lp is None and left_none) or (rp is None and right_none):
                    continue
                if (values[0] is None and not left_none) or (values[1] is None and not right_none):
                    continue
                conditions = []
                if lp is not None:
                    conditions.append(core.bitwise(b, "xor", lp, true) if left_none else lp)
                if rp is not None:
                    conditions.append(core.bitwise(b, "xor", rp, true) if right_none else rp)
                hit = conditions[0]
                for more in conditions[1:]:
                    hit = core.bitwise(b, "and", hit, more)
                shown = [
                    "NoneType" if left_none else names[0],
                    "NoneType" if right_none else names[1],
                ]
                cases.append((hit, message.format(*shown)))
        for hit, raises in cases:
            fine = core.bitwise(b, "xor", hit, true)
            self._guard(fine, "contract", "an operand is None", raises=raises)  # type: ignore[attr-defined]
        if values[0] is None or values[1] is None:
            # `None + 1` always raises; nothing after it runs.
            raise Unsupported("an operation on `None` that always raises stays in Python")
        return values[0], values[1]

    def _optional_binary(self, node: ast.BinOp) -> Value | None:
        """`x + 1` where `x` may be `None`."""
        symbol = _SYMBOLS.get(type(node.op))
        if symbol is None:
            return None
        if getattr(node, "ppy_augmented", False):
            symbol += "="
        found = self._optional_operands(
            node.left,
            node.right,
            f"TypeError: unsupported operand type(s) for {symbol}: '{{0}}' and '{{1}}'",
        )
        if found is None:
            return None
        return self._binary(found[0], found[1], type(node.op))  # type: ignore[attr-defined]

    def _optional_order(self, node: ast.Compare) -> Value | None:
        """`x < 3` where `x` may be `None`."""
        symbol = _ORDERS.get(type(node.ops[0]))
        if symbol is None:
            return None
        left, right = node.left, node.comparators[0]
        found = self._optional_operands(
            left,
            right,
            f"TypeError: '{symbol}' not supported between instances of '{{0}}' and '{{1}}'",
        )
        if found is None:
            return None
        lv, rv = found
        common = self._unify(self._kind(lv), self._kind(rv))  # type: ignore[attr-defined]
        predicate = {ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}[type(node.ops[0])]
        return core.cmp(self.b, predicate, self._coerce(lv, common), self._coerce(rv, common))  # type: ignore[attr-defined]

    def _optional_unary(self, node: ast.UnaryOp) -> Value | None:
        """`-x` where `x` may be `None`."""
        spelled = _UNARY.get(type(node.op))
        if spelled is None:
            return None
        kind = self._either_kind(node.operand)
        if kind is None:
            return None
        present, value = self._optional_pair(node.operand, kind)
        self._guard(  # type: ignore[attr-defined]
            present,
            "contract",
            "an operand is None",
            raises=f"TypeError: bad operand type for {spelled}: 'NoneType'",
        )
        if isinstance(node.op, ast.USub):
            if value.type == F64:
                return core.neg(self.b, value)  # type: ignore[attr-defined]
            return self._checked_binary(self._int_constant(0), self._coerce(value, "int"), "sub")  # type: ignore[attr-defined]
        if isinstance(node.op, ast.UAdd):
            return self._coerce(value, "int") if value.type == BOOL else value  # type: ignore[attr-defined]
        promoted = self._coerce(value, "int")  # type: ignore[attr-defined]
        return core.bitwise(self.b, "xor", promoted, core.const(self.b, -1, I64))  # type: ignore[attr-defined]

    def _optional_or(self, node: ast.BoolOp) -> Value | None:
        """`x or d`, where `x` may be `None` and the whole is a number: the
        first operand that holds a true number, else the last."""
        if not isinstance(node.op, ast.Or):
            return None
        if not any(self._maybe_none(value) for value in node.values[:-1]):
            return None
        answer = T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]
        kind = {T.INT: "int", T.FLOAT: "float", T.BOOL: "bool"}.get(answer)
        if kind is None:
            return None
        done = self._block("optional.or")  # type: ignore[attr-defined]
        result = done.add_argument(_SCALARS[kind], "or")
        for index, value_node in enumerate(node.values):
            if index == len(node.values) - 1:
                value = self._as_kind(self._expr(value_node), kind, value_node)  # type: ignore[attr-defined]
                core.br(self.b, Successor(done, [value]))  # type: ignore[attr-defined]
                break
            inner = self._either_kind(value_node)
            if isinstance(value_node, ast.Constant) and value_node.value is None:
                continue
            if inner is not None:
                present, value = self._optional_pair(value_node, inner)
                value = self._as_kind(value, kind, value_node)
                truth = core.bitwise(self.b, "and", present, self._truth(value))  # type: ignore[attr-defined]
            else:
                value = self._as_kind(self._expr(value_node), kind, value_node)  # type: ignore[attr-defined]
                truth = self._truth(value)  # type: ignore[attr-defined]
            following = self._block("optional.or.next")  # type: ignore[attr-defined]
            core.cond_br(self.b, truth, Successor(done, [value]), Successor(following))  # type: ignore[attr-defined]
            self.b.at_end(following)  # type: ignore[attr-defined]
        self.b.at_end(done)  # type: ignore[attr-defined]
        return result
