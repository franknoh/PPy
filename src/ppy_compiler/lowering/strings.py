"""Native lowering of `str`: a string is a handle into the C runtime.

A string is a collection of family 4 (`ppy_runtime/strings.c`): counted,
immutable, UTF-8, with its length in code points kept. It is held and let
go the way a collection is -- a string local is a `Held` of the `str`
shape -- so everything that owns handles owns strings too: locals, fields,
collection elements and keys, parameters, and returns.

What is here is what a string does: literals, `+` and `*`, comparisons,
`in`, indexing and slicing, iteration, the methods, f-strings and their
format specs, and the conversions to and from numbers. A method whose
answer needs Python's Unicode tables (the case of a letter outside ASCII,
say) checks first and falls back, so the answer is always CPython's.

`list[str]` is the list the methods hand out (`split`, `splitlines`): a
sequence of string handles, indexed from either end as a list is.
"""

from __future__ import annotations

import ast
import re

from ..analysis import types as T
from ..backend.llvm.lowering import Unsupported
from ..ir import BOOL, F64, I64, U8, BufferType, PtrType, Successor, TupleType, Value
from ..ir.dialects import core
from .collections import HANDLE, STR, Kind, Shape

__all__ = ["StringLowering"]

#: `str.is...` methods, as `ppy_str_is` numbers them.
_TESTS = {
    "isdigit": 0,
    "isalpha": 1,
    "isalnum": 2,
    "isupper": 3,
    "islower": 4,
    "isspace": 5,
    "isdecimal": 6,
    "isnumeric": 6,
}

#: Case methods, as `ppy_str_case` numbers them.
_CASES = {"lower": 0, "upper": 1, "capitalize": 2, "title": 3, "swapcase": 4}

_STRIPS = {"strip": 0, "lstrip": 1, "rstrip": 2}
_PADS = {"ljust": 0, "rjust": 1, "center": 2}

#: What a format spec may be here, which is what `ppy_str_spec` reads.
_SPEC = re.compile(
    r"(?:(?P<fill>[\x20-\x7e])?(?P<align>[<>=^]))?(?P<sign>[-+ ])?(?P<z>z)?(?P<alt>#)?"
    r"(?P<zero>0)?(?P<width>\d{1,5})?(?P<group>[,_])?(?:\.(?P<precision>\d{1,3}))?"
    r"(?P<type>[a-zA-Z%])?"
)

_INT_TYPES = set("dbxXo")
_FLOAT_TYPES = set("eEfFgG%")

#: `!r`, `!s`, `!a` as `ast.FormattedValue.conversion` spells them.
_CONVERSIONS = {"str": 115, "repr": 114, "ascii": 97}


def _spec_ok(spec: str, kind: str) -> bool:
    """Whether the runtime writes `format(value, spec)` for a value of `kind`,
    and CPython accepts it: a spec either refuses keeps the call in Python."""
    found = _SPEC.fullmatch(spec)
    if found is None:
        return False
    sample: object = {"int": 7, "float": 7.5, "str": "s", "bool": 7}[kind]
    try:
        format(sample, spec)
    except (ValueError, TypeError):
        return False
    kind_type = found["type"] or ""
    if kind in {"int", "bool"}:
        if kind_type in _FLOAT_TYPES:
            kind = "float"
        elif (
            kind_type
            and kind_type not in _INT_TYPES
            or found["precision"] is not None
            or found["z"]
        ):
            return False
    if kind == "float":
        if kind_type and kind_type not in _FLOAT_TYPES:
            return False
        if found["alt"] or int(found["precision"] or 0) > 100:
            return False
    if kind == "str":
        if kind_type not in {"", "s"} or found["sign"] or found["alt"] or found["group"]:
            return False
        if found["z"] or found["zero"] or found["align"] == "=":
            return False
    return True


class StringLowering:
    """The string half of lowering one function; mixed into `_FunctionLowering`."""

    # -- what is a string ---------------------------------------------------

    def _string_of(self, node: ast.expr) -> Shape | None:
        """`STR` when `node` is a string, from the checker or from what holds it."""
        collections = self.collections  # type: ignore[attr-defined]
        if isinstance(node, ast.Name) and node.id in collections:
            return STR if collections[node.id].kind == STR else None
        found = T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]
        if found == T.STR:
            return STR
        if found != T.UNKNOWN:
            return None
        # A node the lowering made itself (`x += y` spelled out) has no type
        # from the checker; what it reads from does.
        if isinstance(node, ast.Attribute):
            owner = self._object_of(node.value)  # type: ignore[attr-defined]
            if owner is not None:
                _offset, shape = self._field(owner, node.attr)  # type: ignore[attr-defined]
                return STR if shape == STR else None
        if isinstance(node, ast.Subscript):
            kind = self._kind_of(node.value)  # type: ignore[attr-defined]
            if kind is not None and kind.value == STR:
                return STR
            return self._string_of(node.value)
        if isinstance(node, ast.BinOp):
            return self._string_of(node.left) or self._string_of(node.right)
        return None

    def _is_string_list(self, node: ast.expr) -> bool:
        kind = self._kind_of(node)  # type: ignore[attr-defined]
        return kind is not None and kind.name == "List"

    # -- literals ---------------------------------------------------------

    def _text_data(self, text: str) -> tuple[Value, Value]:
        """The UTF-8 bytes of a literal, as a pointer and a byte count."""
        try:
            data = text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise Unsupported("a string with a lone surrogate has no UTF-8 form") from error
        frontend = self.frontend  # type: ignore[attr-defined]
        module = frontend.module
        made: dict[str, object] = frontend.__dict__.setdefault("_string_globals", {})
        found = made.get(text)
        if found is None:
            found = module.add_global(
                f"ppy.str.{len(module.globals)}", BufferType(U8), text, visibility="private"
            )
            made[text] = found
        pointer = self.b.create(  # type: ignore[attr-defined]
            "core.call_intrinsic",
            (),
            (PtrType(U8),),
            {"intrinsic": "ppy.string_data", "symbol": found.symbol.name},  # type: ignore[attr-defined]
        ).result
        return pointer, self._word(len(data))  # type: ignore[attr-defined]

    def _string_literal(self, text: str) -> Value:
        self._use_collections()  # type: ignore[attr-defined]
        data, length = self._text_data(text)
        return self._rt("ppy_str_new", (data, length), HANDLE)  # type: ignore[attr-defined]

    # -- handles ------------------------------------------------------------

    def _string_handle(self, node: ast.expr) -> tuple[Value, bool] | None:
        """The string expressions a handle comes from here, or None for the
        ones the collections already answer (a name, a field, an element)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return self._string_literal(node.value), True
        if isinstance(node, ast.JoinedStr):
            return self._fstring(node), True
        if isinstance(node, ast.BinOp) and self._string_of(node) is not None:
            return self._string_binop(node), True
        if isinstance(node, ast.Subscript) and self._string_of(node.value) is not None:
            return self._string_item(node), True
        if isinstance(node, ast.IfExp) and self._string_of(node) is not None:
            return self._string_choice(node), True
        return None

    def _owned_string(self, node: ast.expr) -> Value:
        """A handle the caller owns: a borrowed one gets its own reference."""
        handle, owned = self._handle(node)  # type: ignore[attr-defined]
        if not owned:
            self._retain(handle)  # type: ignore[attr-defined]
        return handle

    def _string_choice(self, node: ast.IfExp) -> Value:
        """`a if c else b` of strings: each side owned, joined by a block argument."""
        condition = self._test(node.test)  # type: ignore[attr-defined]
        then_block = self._block("str.then")  # type: ignore[attr-defined]
        else_block = self._block("str.else")  # type: ignore[attr-defined]
        done = self._block("str.join")  # type: ignore[attr-defined]
        result = done.add_argument(HANDLE, "chosen")
        core.cond_br(self.b, condition, Successor(then_block), Successor(else_block))  # type: ignore[attr-defined]
        for block, side in ((then_block, node.body), (else_block, node.orelse)):
            self.b.at_end(block)  # type: ignore[attr-defined]
            value = self._owned_string(side)
            core.br(self.b, Successor(done, [value]))  # type: ignore[attr-defined]
        self.b.at_end(done)  # type: ignore[attr-defined]
        return result

    def _string_binop(self, node: ast.BinOp) -> Value:
        """`a + b + c` into one builder, `s * n`, `n * s`."""
        if isinstance(node.op, ast.Add):
            parts: list[ast.expr] = []

            def flatten(item: ast.expr) -> None:
                if (
                    isinstance(item, ast.BinOp)
                    and isinstance(item.op, ast.Add)
                    and self._string_of(item) is not None
                ):
                    flatten(item.left)
                    flatten(item.right)
                    return
                parts.append(item)

            flatten(node)
            return self._concat(parts)
        if isinstance(node.op, ast.Mult):
            text, count = node.left, node.right
            if self._string_of(text) is None:
                text, count = count, text
            if self._string_of(text) is None:
                raise Unsupported("`*` of two strings")
            handle, owned = self._handle(text)  # type: ignore[attr-defined]
            times = self._coerce(self._expr(count), "int")  # type: ignore[attr-defined]
            made = self._rt("ppy_str_repeat", (handle, times), HANDLE)  # type: ignore[attr-defined]
            self._done_with(handle, owned)  # type: ignore[attr-defined]
            return made
        raise Unsupported(f"`{type(node.op).__name__}` of strings has no native lowering")

    def _concat(self, parts: list[ast.expr]) -> Value:
        """Strings joined end to end, written once into one builder."""
        self._use_collections()  # type: ignore[attr-defined]
        builder = self._rt("ppy_str_builder", (self._word(0),), HANDLE)  # type: ignore[attr-defined]
        for part in parts:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                data, length = self._text_data(part.value)
                self._rt("ppy_str_add_bytes", (builder, data, length), None)  # type: ignore[attr-defined]
                continue
            if self._string_of(part) is None:
                raise Unsupported("`+` of a string and something else")
            handle, owned = self._handle(part)  # type: ignore[attr-defined]
            self._rt("ppy_str_add", (builder, handle), None)  # type: ignore[attr-defined]
            self._done_with(handle, owned)  # type: ignore[attr-defined]
        return self._rt("ppy_str_finish", (builder,), HANDLE)  # type: ignore[attr-defined]

    def _augment_string(self, name: str, node: ast.AugAssign) -> bool:
        """`s += t` and `s *= n` on a string local: the new string bound in its place."""
        held = self.collections.get(name)  # type: ignore[attr-defined]
        if held is None or held.kind != STR:
            return False
        read = ast.Name(id=name, ctx=ast.Load())
        ast.copy_location(read, node)
        if isinstance(node.op, ast.Add):
            if self._string_of(node.value) is None:
                raise Unsupported("`+=` of a string and something else")
            # The local's own reference goes to `ppy_str_extend`, which appends
            # in place when nothing else holds the string, and the result
            # takes the slot.
            part, owned = self._handle(node.value)  # type: ignore[attr-defined]
            current = core.load(self.b, held.slot)  # type: ignore[attr-defined]
            made = self._rt("ppy_str_extend", (current, part), HANDLE)  # type: ignore[attr-defined]
            core.store(self.b, made, held.slot)  # type: ignore[attr-defined]
            self._done_with(part, owned)  # type: ignore[attr-defined]
            return True
        if isinstance(node.op, ast.Mult):
            combined = ast.BinOp(left=read, op=node.op, right=node.value)
            ast.copy_location(combined, node)
            made = self._string_binop(combined)
        else:
            raise Unsupported(f"`{type(node.op).__name__}=` of a string has no native lowering")
        self._bind(name, STR, made, True)  # type: ignore[attr-defined]
        return True

    # -- indexing -----------------------------------------------------------

    def _index(self, handle: Value, index: ast.expr) -> Value:
        """A position counted from either end, checked to be inside."""
        b = self.b  # type: ignore[attr-defined]
        position = self._coerce(self._expr(index), "int")  # type: ignore[attr-defined]
        length = self._rt("ppy_str_len", (handle,))  # type: ignore[attr-defined]
        zero = self._word(0)  # type: ignore[attr-defined]
        negative = core.cmp(b, "lt", position, zero)
        wrapped = core.add(b, position, length, overflow="wrap")
        position = core.select(b, negative, wrapped, position)
        inside = core.bitwise(
            b, "and", core.cmp(b, "ge", position, zero), core.cmp(b, "lt", position, length)
        )
        self._require(inside, "string index out of range")  # type: ignore[attr-defined]
        return position

    def _string_item(self, node: ast.Subscript) -> Value:
        """`s[i]` and `s[a:b:c]`: new strings."""
        handle, owned = self._handle(node.value)  # type: ignore[attr-defined]
        if isinstance(node.slice, ast.Slice):
            made = self._slice(handle, node.slice)
        else:
            position = self._index(handle, node.slice)
            made = self._rt("ppy_str_at", (handle, position), HANDLE)  # type: ignore[attr-defined]
        self._done_with(handle, owned)  # type: ignore[attr-defined]
        return made

    def _slice(self, handle: Value, bounds: ast.Slice) -> Value:
        given = 0
        values = []
        for bit, part in ((1, bounds.lower), (2, bounds.upper)):
            if part is None or (isinstance(part, ast.Constant) and part.value is None):
                values.append(self._word(0))  # type: ignore[attr-defined]
                continue
            given |= bit
            values.append(self._coerce(self._expr(part), "int"))  # type: ignore[attr-defined]
        step = self._word(1)  # type: ignore[attr-defined]
        if bounds.step is not None and not (
            isinstance(bounds.step, ast.Constant) and bounds.step.value is None
        ):
            step = self._coerce(self._expr(bounds.step), "int")  # type: ignore[attr-defined]
            nonzero = core.cmp(self.b, "ne", step, self._word(0))  # type: ignore[attr-defined]
            self._require(nonzero, "slice step cannot be zero")  # type: ignore[attr-defined]
        arguments = (handle, values[0], values[1], step, self._word(given))  # type: ignore[attr-defined]
        return self._rt("ppy_str_slice", arguments, HANDLE)  # type: ignore[attr-defined]

    def _list_position(self, handle: Value, position: Value) -> Value:
        """A list index counted from either end, as `list` counts it."""
        b = self.b  # type: ignore[attr-defined]
        length = self._rt("ppy_coll_len", (handle,))  # type: ignore[attr-defined]
        negative = core.cmp(b, "lt", position, self._word(0))  # type: ignore[attr-defined]
        wrapped = core.add(b, position, length, overflow="wrap")
        return core.select(b, negative, wrapped, position)

    # -- comparisons ----------------------------------------------------------

    def _string_compare(self, node: ast.Compare) -> Value | None:
        """`a == b`, `a < b`, `sub in s`, `s in names`, and the rest over strings."""
        b = self.b  # type: ignore[attr-defined]
        operator = node.ops[0]
        left, right = node.left, node.comparators[0]
        if isinstance(operator, (ast.In, ast.NotIn)):
            found = self._membership(left, right)
            if found is None:
                return None
            if isinstance(operator, ast.NotIn):
                return core.bitwise(b, "xor", found, core.const(b, True, BOOL))
            return found
        if self._string_of(left) is None and self._string_of(right) is None:
            return None
        if self._string_of(left) is None or self._string_of(right) is None:
            if isinstance(operator, (ast.Eq, ast.NotEq)):
                # A string is never equal to a number.
                return core.const(b, isinstance(operator, ast.NotEq), BOOL)
            raise Unsupported("ordering a string against something else raises TypeError")
        predicate = {
            ast.Eq: "eq",
            ast.NotEq: "ne",
            ast.Lt: "lt",
            ast.LtE: "le",
            ast.Gt: "gt",
            ast.GtE: "ge",
        }.get(type(operator))
        if predicate is None:
            return None
        if predicate in {"eq", "ne"}:
            for mine, other in ((left, right), (right, left)):
                if isinstance(other, ast.Constant) and isinstance(other.value, str):
                    handle, owned = self._handle(mine)  # type: ignore[attr-defined]
                    data, length = self._text_data(other.value)
                    same = self._rt("ppy_str_equal_bytes", (handle, data, length))  # type: ignore[attr-defined]
                    self._done_with(handle, owned)  # type: ignore[attr-defined]
                    return core.cmp(b, predicate, same, self._word(1))  # type: ignore[attr-defined]
            first, first_owned = self._handle(left)  # type: ignore[attr-defined]
            second, second_owned = self._handle(right)  # type: ignore[attr-defined]
            same = self._rt("ppy_str_equal", (first, second))  # type: ignore[attr-defined]
            self._done_with(first, first_owned)  # type: ignore[attr-defined]
            self._done_with(second, second_owned)  # type: ignore[attr-defined]
            return core.cmp(b, predicate, same, self._word(1))  # type: ignore[attr-defined]
        first, first_owned = self._handle(left)  # type: ignore[attr-defined]
        second, second_owned = self._handle(right)  # type: ignore[attr-defined]
        order = self._rt("ppy_str_order", (first, second))  # type: ignore[attr-defined]
        self._done_with(first, first_owned)  # type: ignore[attr-defined]
        self._done_with(second, second_owned)  # type: ignore[attr-defined]
        return core.cmp(b, predicate, order, self._word(0))  # type: ignore[attr-defined]

    def _membership(self, item: ast.expr, container: ast.expr) -> Value | None:
        """`sub in s`, `s in list_of_strings`, `c in ("a", "b")`."""
        b = self.b  # type: ignore[attr-defined]
        if self._string_of(container) is not None:
            if self._string_of(item) is None:
                raise Unsupported("`in <str>` requires a string on the left")
            haystack, h_owned = self._handle(container)  # type: ignore[attr-defined]
            needle, n_owned = self._handle(item)  # type: ignore[attr-defined]
            found = self._rt("ppy_str_contains", (haystack, needle))  # type: ignore[attr-defined]
            self._done_with(haystack, h_owned)  # type: ignore[attr-defined]
            self._done_with(needle, n_owned)  # type: ignore[attr-defined]
            return core.cmp(b, "ne", found, self._word(0))  # type: ignore[attr-defined]
        if self._is_string_list(container) and self._string_of(item) is not None:
            listed, l_owned = self._handle(container)  # type: ignore[attr-defined]
            needle, n_owned = self._handle(item)  # type: ignore[attr-defined]
            found = self._rt("ppy_str_list_has", (listed, needle))  # type: ignore[attr-defined]
            self._done_with(listed, l_owned)  # type: ignore[attr-defined]
            self._done_with(needle, n_owned)  # type: ignore[attr-defined]
            return core.cmp(b, "ne", found, self._word(0))  # type: ignore[attr-defined]
        if (
            isinstance(container, (ast.Tuple, ast.List, ast.Set))
            and container.elts
            and self._string_of(item) is not None
            and all(self._string_of(e) is not None for e in container.elts)
        ):
            needle, n_owned = self._handle(item)  # type: ignore[attr-defined]
            found = core.const(b, False, BOOL)
            for element in container.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    data, length = self._text_data(element.value)
                    same = self._rt("ppy_str_equal_bytes", (needle, data, length))  # type: ignore[attr-defined]
                else:
                    other, o_owned = self._handle(element)  # type: ignore[attr-defined]
                    same = self._rt("ppy_str_equal", (needle, other))  # type: ignore[attr-defined]
                    self._done_with(other, o_owned)  # type: ignore[attr-defined]
                hit = core.cmp(b, "ne", same, self._word(0))  # type: ignore[attr-defined]
                found = core.bitwise(b, "or", found, hit)
            self._done_with(needle, n_owned)  # type: ignore[attr-defined]
            return found
        return None

    def _string_truth(self, node: ast.expr, empty: bool = False) -> Value | None:
        if self._string_of(node) is None:
            return None
        handle, owned = self._handle(node)  # type: ignore[attr-defined]
        length = self._rt("ppy_str_bytes", (handle,))  # type: ignore[attr-defined]
        self._done_with(handle, owned)  # type: ignore[attr-defined]
        return core.cmp(self.b, "eq" if empty else "gt", length, self._word(0))  # type: ignore[attr-defined]

    def _string_length(self, node: ast.expr) -> Value | None:
        if self._string_of(node) is None:
            return None
        handle, owned = self._handle(node)  # type: ignore[attr-defined]
        length = self._rt("ppy_str_len", (handle,))  # type: ignore[attr-defined]
        self._done_with(handle, owned)  # type: ignore[attr-defined]
        return length

    # -- loops ------------------------------------------------------------------

    def _for_string(self, node: ast.For) -> bool:
        """`for c in s`: each code point as a one-character string."""
        if self._string_of(node.iter) is None:
            return False
        if not isinstance(node.target, ast.Name):
            raise Unsupported("a character of a string is bound to a name")
        b = self.b  # type: ignore[attr-defined]
        handle, owned = self._handle(node.iter)  # type: ignore[attr-defined]
        if not owned:
            # The name the loop walks may be rebound inside it.
            self._retain(handle)  # type: ignore[attr-defined]
        keep = self._alloca(HANDLE, "walked")  # type: ignore[attr-defined]
        core.store(b, handle, keep)
        cursor = self._alloca(I64, "walk.at")  # type: ignore[attr-defined]
        core.store(b, self._word(0), cursor)  # type: ignore[attr-defined]
        header = self._block("walk.head")  # type: ignore[attr-defined]
        body = self._block("walk.body")  # type: ignore[attr-defined]
        latch = self._block("walk.latch")  # type: ignore[attr-defined]
        done = self._block("walk.end")  # type: ignore[attr-defined]
        core.br(b, Successor(header))
        b.at_end(header)
        walked = core.load(b, keep)
        at = core.load(b, cursor)
        more = core.cmp(b, "lt", at, self._rt("ppy_str_bytes", (walked,)))  # type: ignore[attr-defined]
        core.cond_br(b, more, Successor(body), Successor(done))
        b.at_end(body)
        width = self._rt("ppy_str_step", (walked, at))  # type: ignore[attr-defined]
        following = core.add(b, at, width, overflow="wrap")
        core.store(b, following, cursor)
        character = self._rt("ppy_str_span", (walked, at, following), HANDLE)  # type: ignore[attr-defined]
        self._bind(node.target.id, STR, character, True)  # type: ignore[attr-defined]
        self._loops.append((latch, done))  # type: ignore[attr-defined]
        self._body(node.body)  # type: ignore[attr-defined]
        self._loops.pop()  # type: ignore[attr-defined]
        if self._open():  # type: ignore[attr-defined]
            core.br(b, Successor(latch))
        b.at_end(latch)
        core.br(b, Successor(header))
        b.at_end(done)
        self._release(core.load(b, keep))  # type: ignore[attr-defined]
        return True

    # -- calls -------------------------------------------------------------------

    def _string_call(self, node: ast.Call, discard: bool) -> Value | None:
        """A builtin over strings, or a method of one; None where neither applies."""
        func = node.func
        if isinstance(func, ast.Attribute):
            if self._string_of(func.value) is not None:
                return self._keep_or_drop(self._string_method(node), discard)
            return None
        if not isinstance(func, ast.Name) or node.keywords:
            return None
        name = func.id
        arguments = node.args
        b = self.b  # type: ignore[attr-defined]
        if name == "len" and len(arguments) == 1:
            return self._string_length(arguments[0])
        if name == "ord" and len(arguments) == 1 and self._string_of(arguments[0]) is not None:
            handle, owned = self._handle(arguments[0])  # type: ignore[attr-defined]
            one = core.cmp(b, "eq", self._rt("ppy_str_len", (handle,)), self._word(1))  # type: ignore[attr-defined]
            self._require(one, "ord() expected a character")  # type: ignore[attr-defined]
            code = self._rt("ppy_str_ord", (handle,))  # type: ignore[attr-defined]
            self._done_with(handle, owned)  # type: ignore[attr-defined]
            return code
        if name == "chr" and len(arguments) == 1:
            code = self._coerce(self._expr(arguments[0]), "int")  # type: ignore[attr-defined]
            word = self._word  # type: ignore[attr-defined]
            valid = core.bitwise(
                b, "and", core.cmp(b, "ge", code, word(0)), core.cmp(b, "le", code, word(0x10FFFF))
            )
            surrogate = core.bitwise(
                b,
                "and",
                core.cmp(b, "ge", code, word(0xD800)),
                core.cmp(b, "le", code, word(0xDFFF)),
            )
            fine = core.bitwise(
                b, "and", valid, core.bitwise(b, "xor", surrogate, core.const(b, True, BOOL))
            )
            self._require(fine, "chr() arg not in range(0x110000)")  # type: ignore[attr-defined]
            self._use_collections()  # type: ignore[attr-defined]
            made = self._rt("ppy_str_chr", (code,), HANDLE)  # type: ignore[attr-defined]
            return self._keep_or_drop(made, discard)
        if name in _CONVERSIONS and len(arguments) <= 1:
            return self._keep_or_drop(self._to_string(arguments, name), discard)
        if name == "format" and 1 <= len(arguments) <= 2:
            spec = ""
            if len(arguments) == 2:
                given = arguments[1]
                if not isinstance(given, ast.Constant) or not isinstance(given.value, str):
                    raise Unsupported("`format` takes its spec as a string literal natively")
                spec = given.value
            self._use_collections()  # type: ignore[attr-defined]
            builder = self._rt("ppy_str_builder", (self._word(0),), HANDLE)  # type: ignore[attr-defined]
            self._add_formatted(builder, arguments[0], -1, spec)
            made = self._rt("ppy_str_finish", (builder,), HANDLE)  # type: ignore[attr-defined]
            return self._keep_or_drop(made, discard)
        if name in {"int", "float"} and arguments and self._string_of(arguments[0]) is not None:
            return self._parse_number(name, arguments)
        if name == "bool" and len(arguments) == 1:
            return self._string_truth(arguments[0])
        if (
            name in {"min", "max"}
            and len(arguments) >= 2
            and all(self._string_of(argument) is not None for argument in arguments)
        ):
            return self._keep_or_drop(self._string_extremum(name, arguments), discard)
        return None

    def _keep_or_drop(self, value: Value, discard: bool) -> Value:
        if discard and value.type == HANDLE:
            self._release(value)  # type: ignore[attr-defined]
            return self._word(0)  # type: ignore[attr-defined]
        return value

    def _string_extremum(self, name: str, arguments: list[ast.expr]) -> Value:
        """`min(a, b, ...)` of strings: the first of the least, as Python keeps it."""
        b = self.b  # type: ignore[attr-defined]
        best = self._owned_string(arguments[0])
        for argument in arguments[1:]:
            other = self._owned_string(argument)
            order = self._rt("ppy_str_order", (other, best))  # type: ignore[attr-defined]
            wins = core.cmp(b, "lt" if name == "min" else "gt", order, self._word(0))  # type: ignore[attr-defined]
            kept = core.select(b, wins, other, best)
            dropped = core.select(b, wins, best, other)
            self._release(dropped)  # type: ignore[attr-defined]
            best = kept
        return best

    def _to_string(self, arguments: list[ast.expr], name: str) -> Value:
        """`str(x)`, `repr(x)`, `ascii(x)` of a number, a bool, or a string."""
        self._use_collections()  # type: ignore[attr-defined]
        if not arguments:
            return self._string_literal("")
        builder = self._rt("ppy_str_builder", (self._word(0),), HANDLE)  # type: ignore[attr-defined]
        self._add_formatted(builder, arguments[0], _CONVERSIONS[name], "")
        return self._rt("ppy_str_finish", (builder,), HANDLE)  # type: ignore[attr-defined]

    def _parse_number(self, name: str, arguments: list[ast.expr]) -> Value:
        """`int(s)`, `int(s, base)`, `float(s)`: the text read as CPython reads it."""
        b = self.b  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        handle, owned = self._handle(arguments[0])  # type: ignore[attr-defined]
        if name == "int":
            base = word(10)
            if len(arguments) == 2:
                base = self._coerce(self._expr(arguments[1]), "int")  # type: ignore[attr-defined]
                ranged = core.bitwise(
                    b, "and", core.cmp(b, "ge", base, word(2)), core.cmp(b, "le", base, word(36))
                )
                valid = core.bitwise(b, "or", core.cmp(b, "eq", base, word(0)), ranged)
                self._require(valid, "int() base must be >= 2 and <= 36, or 0")  # type: ignore[attr-defined]
            slot = self._alloca(I64, "parsed")  # type: ignore[attr-defined]
            status = self._rt("ppy_str_to_int", (handle, base, slot))  # type: ignore[attr-defined]
            self._require(core.cmp(b, "eq", status, word(0)), "invalid literal for int()")  # type: ignore[attr-defined]
        else:
            if len(arguments) != 1:
                raise Unsupported("`float` takes one argument")
            slot = self._alloca(F64, "parsed")  # type: ignore[attr-defined]
            status = self._rt("ppy_str_to_float", (handle, slot))  # type: ignore[attr-defined]
            converted = core.cmp(b, "ne", status, word(0))
            self._require(converted, "could not convert string to float")  # type: ignore[attr-defined]
        value = core.load(b, slot)
        self._done_with(handle, owned)  # type: ignore[attr-defined]
        return value

    # -- methods ---------------------------------------------------------------

    def _optional_ints(self, arguments: list[ast.expr]) -> tuple[Value, Value, Value]:
        """`start` and `end` as a method takes them: two words and which were given."""
        given = 0
        values = []
        for bit, position in ((1, 0), (2, 1)):
            argument = arguments[position] if position < len(arguments) else None
            if argument is not None and not (
                isinstance(argument, ast.Constant) and argument.value is None
            ):
                given |= bit
                values.append(self._coerce(self._expr(argument), "int"))  # type: ignore[attr-defined]
            else:
                values.append(self._word(0))  # type: ignore[attr-defined]
        return values[0], values[1], self._word(given)  # type: ignore[attr-defined]

    def _string_argument(self, node: ast.expr, method: str) -> tuple[Value, bool]:
        if self._string_of(node) is None:
            raise Unsupported(f"`str.{method}` takes a string natively")
        return self._handle(node)  # type: ignore[attr-defined]

    def _string_method(self, node: ast.Call) -> Value:
        """`s.method(...)`: a handle the caller owns, a number, or a truth."""
        assert isinstance(node.func, ast.Attribute)
        attr = node.func.attr
        if node.keywords and not {k.arg for k in node.keywords} <= {"sep", "maxsplit", "keepends"}:
            raise Unsupported(f"`str.{attr}` keywords have no native lowering")
        self._use_collections()  # type: ignore[attr-defined]
        b = self.b  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        rt = self._rt  # type: ignore[attr-defined]
        require = self._require  # type: ignore[attr-defined]
        arguments = list(node.args)
        named = {k.arg: k.value for k in node.keywords if k.arg is not None}
        receiver, owned = self._handle(node.func.value)  # type: ignore[attr-defined]
        taken: list[tuple[Value, bool]] = []

        def text(position: int) -> Value:
            value, value_owned = self._string_argument(arguments[position], attr)
            taken.append((value, value_owned))
            return value

        result: Value
        if attr in _CASES and not arguments:
            result = rt("ppy_str_case", (receiver, word(_CASES[attr])), HANDLE)
            present = core.cmp(b, "ne", core.cast(b, result, I64), word(0))
            require(present, f"`str.{attr}` of text outside ASCII")
        elif attr in _TESTS and not arguments:
            answer = rt("ppy_str_is", (receiver, word(_TESTS[attr])))
            require(core.cmp(b, "ge", answer, word(0)), f"`str.{attr}` outside ASCII")
            result = core.cmp(b, "ne", answer, word(0))
        elif attr in _STRIPS and len(arguments) <= 1:
            chars = self._null()
            if arguments and not (
                isinstance(arguments[0], ast.Constant) and arguments[0].value is None
            ):
                chars = text(0)
            result = rt("ppy_str_strip", (receiver, chars, word(_STRIPS[attr])), HANDLE)
        elif attr in {"find", "rfind", "index", "rindex"} and 1 <= len(arguments) <= 3:
            sub = text(0)
            start, end, given = self._optional_ints(arguments[1:])
            right = word(int(attr.startswith("r")))
            result = rt("ppy_str_find", (receiver, sub, start, end, given, right))
            if attr.endswith("index"):
                require(self._found(result), "substring not found")  # type: ignore[attr-defined]
        elif attr == "count" and 1 <= len(arguments) <= 3:
            sub = text(0)
            start, end, given = self._optional_ints(arguments[1:])
            result = rt("ppy_str_count", (receiver, sub, start, end, given))
        elif attr in {"startswith", "endswith"} and 1 <= len(arguments) <= 3:
            if isinstance(arguments[0], ast.Tuple):
                raise Unsupported(f"`str.{attr}` of a tuple has no native lowering")
            affix = text(0)
            start, end, given = self._optional_ints(arguments[1:])
            tail = word(int(attr == "endswith"))
            found = rt("ppy_str_affix", (receiver, affix, start, end, given, tail))
            result = core.cmp(b, "ne", found, word(0))
        elif attr == "replace" and 2 <= len(arguments) <= 3:
            old, new = text(0), text(1)
            count = word(-1)
            if len(arguments) == 3:
                count = self._coerce(self._expr(arguments[2]), "int")  # type: ignore[attr-defined]
            result = rt("ppy_str_replace", (receiver, old, new, count), HANDLE)
        elif attr in {"split", "rsplit"} and len(arguments) <= 2:
            sep_node = arguments[0] if arguments else named.get("sep")
            limit_node = arguments[1] if len(arguments) > 1 else named.get("maxsplit")
            sep = self._null()
            if sep_node is not None and not (
                isinstance(sep_node, ast.Constant) and sep_node.value is None
            ):
                value, value_owned = self._string_argument(sep_node, attr)
                taken.append((value, value_owned))
                sep = value
                require(core.cmp(b, "gt", rt("ppy_str_bytes", (sep,)), word(0)), "empty separator")
            limit = word(-1)
            if limit_node is not None:
                limit = self._coerce(self._expr(limit_node), "int")  # type: ignore[attr-defined]
            result = rt(f"ppy_str_{attr}", (receiver, sep, limit), HANDLE)
        elif attr == "splitlines" and len(arguments) <= 1:
            keep_node = arguments[0] if arguments else named.get("keepends")
            keep = word(0)
            if keep_node is not None:
                keep = self._coerce(self._expr(keep_node), "int")  # type: ignore[attr-defined]
            result = rt("ppy_str_splitlines", (receiver, keep), HANDLE)
        elif attr == "join" and len(arguments) == 1:
            result = self._join(receiver, arguments[0])
        elif attr in {"removeprefix", "removesuffix"} and len(arguments) == 1:
            affix = text(0)
            tail = word(int(attr == "removesuffix"))
            result = rt("ppy_str_remove_affix", (receiver, affix, tail), HANDLE)
        elif attr == "zfill" and len(arguments) == 1:
            width = self._coerce(self._expr(arguments[0]), "int")  # type: ignore[attr-defined]
            result = rt("ppy_str_zfill", (receiver, width), HANDLE)
        elif attr in _PADS and 1 <= len(arguments) <= 2:
            width = self._coerce(self._expr(arguments[0]), "int")  # type: ignore[attr-defined]
            fill = word(ord(" "))
            if len(arguments) == 2:
                filler = text(1)
                one = core.cmp(b, "eq", rt("ppy_str_len", (filler,)), word(1))
                require(one, "the fill character must be exactly one character long")
                fill = rt("ppy_str_ord", (filler,))
            result = rt("ppy_str_pad", (receiver, width, fill, word(_PADS[attr])), HANDLE)
        elif attr == "format":
            raise Unsupported("`str.format` has no native lowering; an f-string does")
        else:
            raise Unsupported(f"`str.{attr}` has no native lowering")
        for value, value_owned in taken:
            self._done_with(value, value_owned)  # type: ignore[attr-defined]
        self._done_with(receiver, owned)  # type: ignore[attr-defined]
        return result

    def _null(self) -> Value:
        return self._rt("ppy_coll_none", (), HANDLE)  # type: ignore[attr-defined]

    def _join(self, separator: Value, iterable: ast.expr) -> Value:
        """`sep.join(xs)`: a list of strings, a `Vec` of them, or a literal list or tuple."""
        if isinstance(iterable, (ast.List, ast.Tuple)):
            listed = self._string_list(iterable.elts)
            made = self._rt("ppy_str_join", (separator, listed), HANDLE)  # type: ignore[attr-defined]
            self._release(listed)  # type: ignore[attr-defined]
            return made
        kind = self._kind_of(iterable)  # type: ignore[attr-defined]
        if kind is None or kind.value != STR or kind.name not in {"List", "Vec"}:
            raise Unsupported("`str.join` takes a list or a `Vec` of strings natively")
        handle, owned = self._handle(iterable)  # type: ignore[attr-defined]
        made = self._rt("ppy_str_join", (separator, handle), HANDLE)  # type: ignore[attr-defined]
        self._done_with(handle, owned)  # type: ignore[attr-defined]
        return made

    def _string_list(self, elements: list[ast.expr]) -> Value:
        """`[a, b, c]` of strings: a new list holding a reference to each."""
        self._use_collections()  # type: ignore[attr-defined]
        listed = self._rt("ppy_str_list", (), HANDLE)  # type: ignore[attr-defined]
        for element in elements:
            if isinstance(element, ast.Starred) or self._string_of(element) is None:
                raise Unsupported("a list of strings holds strings")
            self._rt("ppy_str_list_take", (listed, self._owned_string(element)), None)  # type: ignore[attr-defined]
        return listed

    def _string_list_handle(self, node: ast.expr) -> tuple[Value, bool] | None:
        """A literal list of strings, where the checker calls it `list[str]`."""
        if not isinstance(node, ast.List):
            return None
        kind = self._kind_of(node)  # type: ignore[attr-defined]
        if kind is None or kind.name != "List":
            return None
        return self._string_list(node.elts), True

    def _string_list_method(
        self, kind: Kind, handle: Value, attr: str, arguments: list[ast.expr]
    ) -> Value | None:
        """`list[str]`: `append`, `pop`, `insert`, `sort`, `reverse`, `clear`, `index`, `count`."""
        del kind
        b = self.b  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        rt = self._rt  # type: ignore[attr-defined]
        if attr == "append" and len(arguments) == 1:
            rt("ppy_str_list_take", (handle, self._owned_string(arguments[0])), None)
            return word(0)
        if attr == "pop" and len(arguments) <= 1:
            length = rt("ppy_coll_len", (handle,))
            self._require(core.cmp(b, "gt", length, word(0)), "pop from empty list")  # type: ignore[attr-defined]
            if not arguments:
                return self._read(rt("ppy_seq_pop_back", (handle,), HANDLE), STR)  # type: ignore[attr-defined]
            given = self._coerce(self._expr(arguments[0]), "int")  # type: ignore[attr-defined]
            position = self._list_position(handle, given)
            inside = core.bitwise(
                b, "and", core.cmp(b, "ge", position, word(0)), core.cmp(b, "lt", position, length)
            )
            self._require(inside, "pop index out of range")  # type: ignore[attr-defined]
            return rt("ppy_str_list_remove", (handle, position), HANDLE)
        if attr == "insert" and len(arguments) == 2:
            position = self._coerce(self._expr(arguments[0]), "int")  # type: ignore[attr-defined]
            rt("ppy_str_list_insert", (handle, position, self._owned_string(arguments[1])), None)
            return word(0)
        if attr == "sort" and not arguments:
            return rt("ppy_seq_sort", (handle,), None)
        if attr == "reverse" and not arguments:
            return rt("ppy_seq_reverse", (handle,), None)
        if attr == "clear" and not arguments:
            return rt("ppy_seq_clear", (handle,), None)
        if attr in {"index", "count"} and len(arguments) == 1:
            needle, owned = self._string_argument(arguments[0], attr)
            symbol = "ppy_str_list_index" if attr == "index" else "ppy_str_list_count"
            found = rt(symbol, (handle, needle))
            self._done_with(needle, owned)  # type: ignore[attr-defined]
            if attr == "index":
                self._require(self._found(found), "the string is not in the list")  # type: ignore[attr-defined]
            return found
        return None

    # -- f-strings and formatting ---------------------------------------------------

    def _fstring(self, node: ast.JoinedStr) -> Value:
        self._use_collections()  # type: ignore[attr-defined]
        builder = self._rt("ppy_str_builder", (self._word(0),), HANDLE)  # type: ignore[attr-defined]
        for item in node.values:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                data, length = self._text_data(item.value)
                self._rt("ppy_str_add_bytes", (builder, data, length), None)  # type: ignore[attr-defined]
                continue
            if not isinstance(item, ast.FormattedValue):
                raise Unsupported("this f-string part has no native lowering")
            spec = ""
            if item.format_spec is not None:
                parts = (
                    item.format_spec.values if isinstance(item.format_spec, ast.JoinedStr) else []
                )
                texts = [p.value for p in parts if isinstance(p, ast.Constant)]
                if len(texts) != len(parts) or not all(isinstance(t, str) for t in texts):
                    raise Unsupported("a format spec with fields has no native lowering")
                spec = "".join(texts)
            self._add_formatted(builder, item.value, item.conversion, spec)
        return self._rt("ppy_str_finish", (builder,), HANDLE)  # type: ignore[attr-defined]

    def _add_formatted(self, builder: Value, node: ast.expr, conversion: int, spec: str) -> None:
        """One field, `{value!conversion:spec}`, written onto the builder."""
        rt = self._rt  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        if self._is_string_list(node):
            if spec:
                raise Unsupported("a format spec over a list has no native lowering")
            listed, owned = self._handle(node)  # type: ignore[attr-defined]
            done = rt("ppy_str_add_list_repr", (builder, listed))
            self._require(  # type: ignore[attr-defined]
                core.cmp(self.b, "ne", done, word(0)),
                "`repr` of text outside ASCII",  # type: ignore[attr-defined]
            )
            self._done_with(listed, owned)  # type: ignore[attr-defined]
            return
        if self._string_of(node) is not None:
            handle, owned = self._handle(node)  # type: ignore[attr-defined]
            if conversion in {_CONVERSIONS["repr"], _CONVERSIONS["ascii"]}:
                if spec:
                    shown = rt("ppy_str_builder", (word(0),), HANDLE)
                    self._add_repr(shown, handle)
                    shown = rt("ppy_str_finish", (shown,), HANDLE)
                    self._add_spec(builder, "str", shown, spec)
                    self._release(shown)  # type: ignore[attr-defined]
                else:
                    self._add_repr(builder, handle)
            elif spec:
                self._add_spec(builder, "str", handle, spec)
            else:
                rt("ppy_str_add", (builder, handle), None)
            self._done_with(handle, owned)  # type: ignore[attr-defined]
            return
        value = self._expr(node)  # type: ignore[attr-defined]
        kind = {I64: "int", F64: "float", BOOL: "bool"}.get(value.type)
        if kind is None:
            raise Unsupported("an f-string field natively is a number, a bool, or a string")
        if spec and conversion in _CONVERSIONS.values():
            # `{x!r:>8}`: the text first, then the spec over it.
            shown = rt("ppy_str_builder", (word(0),), HANDLE)
            self._add_plain(shown, kind, value)
            shown = rt("ppy_str_finish", (shown,), HANDLE)
            self._add_spec(builder, "str", shown, spec)
            self._release(shown)  # type: ignore[attr-defined]
            return
        if spec:
            if kind == "bool":
                value, kind = core.cast(self.b, value, I64), "int"  # type: ignore[attr-defined]
            self._add_spec(builder, kind, value, spec)
            return
        self._add_plain(builder, kind, value)

    def _add_plain(self, builder: Value, kind: str, value: Value) -> None:
        if kind == "bool":
            value = core.cast(self.b, value, I64)  # type: ignore[attr-defined]
        self._rt(f"ppy_str_add_{kind}", (builder, value), None)  # type: ignore[attr-defined]

    def _add_repr(self, builder: Value, handle: Value) -> None:
        done = self._rt("ppy_str_add_repr", (builder, handle))  # type: ignore[attr-defined]
        ascii_only = core.cmp(self.b, "ne", done, self._word(0))  # type: ignore[attr-defined]
        self._require(ascii_only, "`repr` of text outside ASCII")  # type: ignore[attr-defined]

    def _add_spec(self, builder: Value, kind: str, value: Value, spec: str) -> None:
        if not _spec_ok(spec, kind):
            raise Unsupported(f"the format spec `{spec}` has no native lowering")
        data, length = self._text_data(spec)
        symbol = {"int": "ppy_str_format_int", "float": "ppy_str_format_float"}.get(
            kind, "ppy_str_format_str"
        )
        done = self._rt(symbol, (builder, value, data, length))  # type: ignore[attr-defined]
        written = core.cmp(self.b, "ne", done, self._word(0))  # type: ignore[attr-defined]
        self._require(written, f"the format spec `{spec}`")  # type: ignore[attr-defined]

    # -- printing and reading --------------------------------------------------------

    def _printed_list(self, node: ast.expr) -> tuple[Value, bool] | None:
        """`print(words)`: the list's repr, as a string to print and let go."""
        if not self._is_string_list(node):
            return None
        builder = self._rt("ppy_str_builder", (self._word(0),), HANDLE)  # type: ignore[attr-defined]
        self._add_formatted(builder, node, -1, "")
        return self._rt("ppy_str_finish", (builder,), HANDLE), True  # type: ignore[attr-defined]

    def _print_string(self, handle: Value) -> None:
        data = self._rt("ppy_str_data", (handle,), HANDLE)  # type: ignore[attr-defined]
        length = self._rt("ppy_str_bytes", (handle,))  # type: ignore[attr-defined]
        core.call_extern(self.b, "ppy_rt_print_str", (data, length), ())  # type: ignore[attr-defined]

    def _read_string(self, node: ast.Call) -> Value | None:
        """`ppy.input[str]()`, `ppy.scan[str]()`, `input()` in a standalone build."""
        target = ast.unparse(node.func)
        symbol = {
            "ppy.input[str]": "ppy_rt_input_str",
            "ppy.scan[str]": "ppy_rt_scan_str",
            "input": "ppy_rt_input_str",
        }.get(target)
        if symbol is None or node.keywords or len(node.args) > 1:
            return None
        if target != "input" and node.args:
            return None
        self._use_collections()  # type: ignore[attr-defined]
        if node.args:
            prompt, owned = self._handle(node.args[0])  # type: ignore[attr-defined]
            self._print_string(prompt)
            self._done_with(prompt, owned)  # type: ignore[attr-defined]
            core.call_extern(self.b, "ppy_rt_flush_stdout", (), ())  # type: ignore[attr-defined]
        return core.call_extern(self.b, symbol, (), (HANDLE,)).results[0]  # type: ignore[attr-defined]

    # -- unpacking --------------------------------------------------------------------

    def _unpack_strings(self, target: ast.expr, value: ast.expr) -> bool:
        """`a, b = s.split(",")` and `head, sep, tail = s.partition(",")`."""
        if not isinstance(target, (ast.Tuple, ast.List)):
            return False
        names = [name for name in target.elts if isinstance(name, ast.Name)]
        if len(names) != len(target.elts):
            return False
        b = self.b  # type: ignore[attr-defined]
        rt = self._rt  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        if (
            isinstance(value, ast.Tuple)
            and len(value.elts) == len(names)
            and any(self._string_of(item) is not None for item in value.elts)
        ):
            # `best, most = path, count`: every value first, as Python takes
            # them, and then each name bound.
            taken: list[tuple[bool, Value]] = []
            for item in value.elts:
                if self._string_of(item) is not None:
                    taken.append((True, self._owned_string(item)))
                else:
                    taken.append((False, self._expr(item)))  # type: ignore[attr-defined]
            for name, (text, item) in zip(names, taken, strict=True):
                if text:
                    self._bind(name.id, STR, item, True)  # type: ignore[attr-defined]
                else:
                    self._store(name, item)  # type: ignore[attr-defined]
            return True
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr in {"partition", "rpartition"}
            and self._string_of(value.func.value) is not None
            and len(value.args) == 1
            and len(names) == 3
        ):
            receiver, owned = self._handle(value.func.value)  # type: ignore[attr-defined]
            sep, sep_owned = self._string_argument(value.args[0], value.func.attr)
            nonempty = core.cmp(b, "gt", rt("ppy_str_bytes", (sep,)), word(0))
            self._require(nonempty, "empty separator")  # type: ignore[attr-defined]
            parts = self._alloca(TupleType((HANDLE, HANDLE, HANDLE)), "parts")  # type: ignore[attr-defined]
            address = core.cast(b, parts, PtrType(HANDLE, "stack"))
            right = word(int(value.func.attr == "rpartition"))
            rt("ppy_str_partition", (receiver, sep, right, address), None)
            self._done_with(sep, sep_owned)  # type: ignore[attr-defined]
            self._done_with(receiver, owned)  # type: ignore[attr-defined]
            for index, name in enumerate(names):
                slot = core.ptr_offset(b, address, word(index))
                self._bind(name.id, STR, core.load(b, slot), True)  # type: ignore[attr-defined]
            return True
        if not self._is_string_list(value):
            return False
        handle, owned = self._handle(value)  # type: ignore[attr-defined]
        length = rt("ppy_coll_len", (handle,))
        matches = core.cmp(b, "eq", length, word(len(names)))
        self._require(matches, "wrong number of values to unpack")  # type: ignore[attr-defined]
        items = [
            self._read(rt("ppy_seq_at", (handle, word(i)), HANDLE), STR)  # type: ignore[attr-defined]
            for i in range(len(names))
        ]
        for name, item in zip(names, items, strict=True):
            self._bind(name.id, STR, item, False)  # type: ignore[attr-defined]
        self._done_with(handle, owned)  # type: ignore[attr-defined]
        return True
