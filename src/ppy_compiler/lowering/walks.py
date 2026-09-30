"""More of what a `for` loop walks natively: `range` (any step, and reversed),
a string's characters, a list lent as a buffer, and a tuple, each alone or
under `enumerate`, `zip`, and `reversed`.

These join the collections' walks (`collection_api`): a source here is one
more kind of `_Source`, with its own start, test, current element, and step,
so `zip(range(n), row)` or `enumerate(word)` walks the way `enumerate(row)` does.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..analysis import types as T
from ..backend.llvm.lowering import Unsupported
from ..ir import BOOL, I64, BufferType, Value
from ..ir.dialects import core
from .collection_api import _called, _Cursor, _Source
from .collections import HANDLE, STR, Kind, Shape, shape_of


@dataclass(slots=True)
class _Counted(_Source):
    """A walk by position: over a range, a string, a buffer, or a tuple."""

    #: "range", "text", "buffer", "array", or "frame" (a generator's).
    what: str = ""
    #: A range's first value, step, and length; a tuple's elements' slots.
    first: Value | None = None
    step: Value | None = None
    count: Value | None = None
    #: The buffer's name, for a buffer.
    name: str = ""
    #: A tuple's elements, in a stack array, and how many.
    items: Value | None = None
    size: int = 0


def _constant_int(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) is int
    ):
        return -node.operand.value
    return None


class WalkLowering:  # pylint: disable=attribute-defined-outside-init
    """Mixed into `_FunctionLowering` ahead of the collections' walks."""

    # -- what is walked by position ------------------------------------------------------

    def _counted_kind(self, node: ast.expr) -> str | None:
        if (
            isinstance(node, ast.Call)
            and _called(node, "range")
            and 1 <= len(node.args) <= 3
            and not node.keywords
        ):
            t = T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]
            if isinstance(t, T.Instance) and t.name == "range":
                return "range"
        if isinstance(node, ast.Name) and node.id in getattr(self, "buffers", {}):
            return "buffer"
        if isinstance(node, ast.Tuple) and node.elts:
            if any(isinstance(e, ast.Starred) for e in node.elts):
                return None
            return "array"
        if isinstance(node, ast.Name) and node.id in getattr(self, "tuples", {}):
            return "array"
        if self._string_of(node) is not None:  # type: ignore[attr-defined]
            return "text"
        return None

    def _is_walk(self, node: ast.expr) -> bool:
        if (
            isinstance(node, ast.Call)
            and node.args
            and not any(k.arg != "start" for k in node.keywords)
        ):
            if _called(node, "enumerate") and len(node.args) <= 2:
                inner = node.args[0]
                return self._counted_kind(inner) is not None or self._is_walk(inner)
            if _called(node, "reversed") and len(node.args) == 1 and not node.keywords:
                kind = self._counted_kind(node.args[0])
                if kind in {"range", "buffer", "array"}:
                    return True
            if _called(node, "zip") and len(node.args) >= 2 and not node.keywords:
                return all(self._counted_kind(a) is not None or self._is_walk(a) for a in node.args)
        kind = self._counted_kind(node)
        if kind in {"array", "frame"}:
            return True
        if kind == "range" and len(node.args) == 3 and _constant_int(node.args[2]) is None:  # type: ignore[attr-defined]
            # A step only known when the loop runs: the plain `range` loop wants
            # a constant one.
            return True
        return super()._is_walk(node)  # type: ignore[misc]

    def _source(self, node: ast.expr):  # type: ignore[no-untyped-def]
        backwards = False
        inner = node
        if (
            isinstance(node, ast.Call)
            and _called(node, "reversed")
            and len(node.args) == 1
            and not node.keywords
            and self._counted_kind(node.args[0]) in {"range", "buffer", "array"}
        ):
            backwards = True
            inner = node.args[0]
        what = self._counted_kind(inner)
        if what is None:
            return super()._source(node)  # type: ignore[misc]
        if what == "range":
            assert isinstance(inner, ast.Call)
            return self._range_source(inner, backwards)
        if what == "frame":
            return self._frame_source(inner)  # type: ignore[attr-defined]
        if what == "text":
            handle, owned = self._handle(inner)  # type: ignore[attr-defined]
            slot = self._hold(STR, handle, owned)  # type: ignore[attr-defined]
            return _Counted(Kind("Vec", STR), slot, what="text")
        if what == "buffer":
            assert isinstance(inner, ast.Name)
            buffer = self.buffers[inner.id]  # type: ignore[attr-defined]
            shape = Shape(_buffer_kind(buffer))
            return _Counted(
                Kind("Vec", shape), None, backwards=backwards, what="buffer", name=inner.id
            )
        return self._array_source(inner, backwards)

    def _range_source(self, node: ast.Call, backwards: bool) -> _Counted:
        """A range's first value, its step, and how many values it has, from
        its bounds evaluated once; reversed, it starts at its last value."""
        start, stop, step, constant = self._range_bounds(node)  # type: ignore[attr-defined]
        b = self.b  # type: ignore[attr-defined]
        zero = self._word(0)  # type: ignore[attr-defined]
        one = self._word(1)  # type: ignore[attr-defined]
        upward = (
            core.const(b, constant > 0, BOOL)
            if constant is not None
            else core.cmp(b, "gt", step, zero)
        )
        low = core.select(b, upward, start, stop)
        high = core.select(b, upward, stop, start)
        span, spilled = core.checked(b, "sub", high, low)
        size, negated = core.checked(b, "sub", zero, step)
        magnitude = core.select(b, upward, step, size)
        spilled = core.bitwise(
            b, "or", spilled, core.bitwise(b, "and", negated, self._negated(upward))
        )  # type: ignore[attr-defined]
        core.guard(b, self._negated(spilled), "overflow", "a range past the word")  # type: ignore[attr-defined]
        nonempty = core.cmp(b, "lt", low, high)
        steps = core.div(b, core.sub(b, span, one, overflow="wrap"), magnitude, overflow="wrap")
        count = core.select(b, nonempty, core.add(b, steps, one, overflow="wrap"), zero)
        first = start
        if backwards:
            last_index = core.select(b, nonempty, steps, zero)
            offset = core.mul(b, last_index, step, overflow="wrap")
            first = core.add(b, start, offset, overflow="wrap")
            step = core.sub(b, zero, step, overflow="wrap")
        return _Counted(
            Kind("Vec", Shape("int")), None, what="range", first=first, step=step, count=count
        )

    def _array_source(self, node: ast.expr, backwards: bool) -> _Counted:
        """A tuple's elements, made (a display) or read (a tuple local) once, in
        a stack array the walk indexes."""
        t = T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]
        if isinstance(node, ast.Tuple):
            item_types = [self._type_of(e) for e in node.elts]  # type: ignore[attr-defined]
        elif isinstance(t, T.Tuple_) and not t.homogeneous:
            item_types = list(t.items)
        else:
            raise Unsupported("a tuple is walked when its items are known")
        records = self._records()  # type: ignore[attr-defined]
        shapes = [shape_of(item, records) for item in item_types]
        first = shapes[0]
        if first is None or first.kind not in {"int", "float", "bool", "tuple"}:
            raise Unsupported("a tuple of numbers, or of tuples of numbers, is walked natively")
        if any(shape != first for shape in shapes):
            raise Unsupported("a tuple whose items differ in type is walked in Python")
        if isinstance(node, ast.Tuple):
            values = [self._array_item(element, first) for element in node.elts]
        else:
            assert isinstance(node, ast.Name)
            packed = core.load(self.b, self.tuples[node.id])  # type: ignore[attr-defined]
            values = [core.tuple_extract(self.b, packed, i) for i in range(len(shapes))]  # type: ignore[attr-defined]
        element_type = values[0].type
        items = core.alloca(self._entry_builder(), element_type, len(values), name="walk.items")  # type: ignore[attr-defined]
        for index, value in enumerate(values):
            core.store(self.b, value, core.ptr_offset(self.b, items, self._word(index)))  # type: ignore[attr-defined]
        return _Counted(
            Kind("Vec", first),
            None,
            backwards=backwards,
            what="array",
            items=items,
            size=len(values),
        )

    def _array_item(self, node: ast.expr, shape: Shape) -> Value:
        if shape.kind != "tuple":
            return self._coerce(self._expr(node), shape.kind)  # type: ignore[attr-defined]
        if isinstance(node, ast.Tuple) and len(node.elts) == len(shape.parts):
            parts = [
                self._coerce(self._expr(e), kind)  # type: ignore[attr-defined]
                for e, kind in zip(node.elts, shape.parts, strict=True)
            ]
            return core.tuple_make(self.b, *parts)  # type: ignore[attr-defined]
        if isinstance(node, ast.Name) and node.id in self.tuples:  # type: ignore[attr-defined]
            return core.load(self.b, self.tuples[node.id])  # type: ignore[attr-defined]
        raise Unsupported("a tuple's items are tuple displays or tuple locals")

    # -- the walk --------------------------------------------------------------------------

    def _start(self, source):  # type: ignore[no-untyped-def]
        if not isinstance(source, _Counted):
            return super()._start(source)  # type: ignore[misc]
        b = self.b  # type: ignore[attr-defined]
        at = self._alloca(I64, "walk.at")  # type: ignore[attr-defined]
        value = None
        if source.what == "range":
            value = self._alloca(I64, "walk.value")  # type: ignore[attr-defined]
            assert source.first is not None
            core.store(b, source.first, value)
            core.store(b, self._word(0), at)  # type: ignore[attr-defined]
        elif source.what == "frame":
            core.store(b, self._word(0), at)  # type: ignore[attr-defined]
        elif source.backwards:
            core.store(b, core.sub(b, self._length_of(source), self._word(1), overflow="wrap"), at)  # type: ignore[attr-defined]
        else:
            core.store(b, self._word(0), at)  # type: ignore[attr-defined]
        return _Cursor(source, at, None, value)

    def _length_of(self, source: _Counted) -> Value:
        if source.what == "buffer":
            buffer = self.buffers[source.name]  # type: ignore[attr-defined]
            return core.cast(self.b, core.buffer_len(self.b, buffer), I64)  # type: ignore[attr-defined]
        if source.what == "text":
            return self._rt("ppy_str_bytes", (core.load(self.b, source.slot),))  # type: ignore[attr-defined]
        if source.what == "range":
            assert source.count is not None
            return source.count
        return self._word(source.size)  # type: ignore[attr-defined]

    def _more(self, cursor):  # type: ignore[no-untyped-def]
        source = cursor.source
        if not isinstance(source, _Counted):
            return super()._more(cursor)  # type: ignore[misc]
        b = self.b  # type: ignore[attr-defined]
        if source.what == "frame":
            return self._frame_step(core.load(b, source.slot))  # type: ignore[attr-defined]
        at = core.load(b, cursor.at)
        inside = core.cmp(b, "lt", at, self._length_of(source))
        if source.backwards:
            return core.bitwise(b, "and", inside, core.cmp(b, "ge", at, self._word(0)))  # type: ignore[attr-defined]
        return inside

    def _current(self, cursor):  # type: ignore[no-untyped-def]
        source = cursor.source
        if not isinstance(source, _Counted):
            return super()._current(cursor)  # type: ignore[misc]
        b = self.b  # type: ignore[attr-defined]
        shape = source.kind.value
        if source.what == "range":
            return [(shape, core.load(b, cursor.key))]
        if source.what == "frame":
            value = self._frame_value_read(core.load(b, source.slot), shape)  # type: ignore[attr-defined]
            self._take_yielded(source, shape, value)  # type: ignore[attr-defined]
            return [(shape, value)]
        at = core.load(b, cursor.at)
        if source.what == "buffer":
            return [(shape, self._buffer_element(source.name, at))]  # type: ignore[attr-defined]
        if source.what == "array":
            return [(shape, core.load(b, core.ptr_offset(b, source.items, at)))]
        # A character, a new reference the walk holds until the next one.
        none = self._rt("ppy_coll_none", (), HANDLE)  # type: ignore[attr-defined]
        character = self._rt("ppy_str_next", (core.load(b, source.slot), cursor.at, none), HANDLE)  # type: ignore[attr-defined]
        names = self.__dict__.setdefault("_characters", {})
        name = names.get(id(source))
        if name is None:
            self._walks += 1  # type: ignore[attr-defined]
            name = names[id(source)] = f".char{self._walks}"  # type: ignore[attr-defined]
        self._bind(name, STR, character, owned=True)  # type: ignore[attr-defined]
        return [(STR, character)]

    def _advance(self, cursor):  # type: ignore[no-untyped-def]
        source = cursor.source
        if not isinstance(source, _Counted):
            super()._advance(cursor)  # type: ignore[misc]
            return
        b = self.b  # type: ignore[attr-defined]
        if source.what in {"text", "frame"}:
            return  # reading the character, or stepping the generator, moved past it
        at = core.load(b, cursor.at)
        delta = self._word(-1 if source.backwards else 1)  # type: ignore[attr-defined]
        core.store(b, core.add(b, at, delta, overflow="wrap"), cursor.at)
        if source.what == "range":
            assert source.step is not None
            value = core.load(b, cursor.key)
            core.store(b, core.add(b, value, source.step, overflow="wrap"), cursor.key)

    def _let_go(self, slot):  # type: ignore[no-untyped-def]
        if slot is None:
            return
        super()._let_go(slot)  # type: ignore[misc]


def _buffer_kind(buffer: Value) -> str:
    from .ast_to_ir import _kind, _read_as  # pylint: disable=import-outside-toplevel

    assert isinstance(buffer.type, BufferType)
    return _read_as(_kind(buffer.type.element))
