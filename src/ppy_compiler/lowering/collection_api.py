"""The collections' wider API in native code: walks, whole-collection operations,
and the methods beyond pushing and popping.

`CollectionLowering` reads and writes one element at a time. What is here
works on whole collections and on walks over them:

* A walk is a cursor over one collection, its elements, keys, values, or
  items, forwards or backwards or between two keys. A `for` loop drives one
  or several (`zip`), with a count (`enumerate`); `extend`, a constructor
  filled from an iterable, and `min`, `max`, `sum` drive one without a body.
* A collection a walk reads is held for the walk: a name the loop body
  rebinds, or a temporary, stays alive until the walk ends, and a `return`
  from inside the loop lets go of it like any other local.
* Equality, search, copies, slices, `+`, and the set operations are calls
  into the runtime, which compares words as Python compares values. A
  comparison that meets a NaN is left to Python: CPython counts the same NaN
  object as equal to itself, and native memory has no objects to tell apart.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass

from ..analysis import types as T
from ..backend.llvm.lowering import Unsupported
from ..ir import BOOL, F64, I64, PtrType, Successor, TupleType, Value
from ..ir.dialects import core
from .collections import HANDLE, CollectionLowering, Kind, Shape, _pointer, shape_of

__all__ = ["CollectionApiLowering"]

#: The set operations, as `ppy_set_combine` numbers them.
_COMBINE = {"union": 0, "intersection": 1, "difference": 2, "symmetric_difference": 3}
_OPERATORS = {ast.BitOr: 0, ast.BitAnd: 1, ast.Sub: 2, ast.BitXor: 3}
#: The questions sets answer about each other, as `ppy_set_relation` numbers them.
_RELATIONS = {"issubset": 0, "issuperset": 1, "isdisjoint": 2}
#: The methods whose collection result is a new reference the caller owns.
_OWNED = frozenset({"copy", "to_sorted", "get", "pushpop", "replace", "pop", *_COMBINE})
#: The views a map walks.
_VIEWS = frozenset({"keys", "values", "items"})


@dataclass(slots=True)
class _Source:
    """What a walk goes over: a collection held in a slot, and which of its parts."""

    kind: Kind
    #: The slot holding the handle for the length of the walk.
    slot: Value
    #: "elements" (a sequence's elements, a map's or a set's keys), "values", or "items".
    mode: str = "elements"
    backwards: bool = False
    #: `between(low, high)`: the key buffers of the bounds.
    low: Value | None = None
    high: Value | None = None


@dataclass(slots=True)
class _Cursor:
    source: _Source
    #: The current position: an index, a node, or an entry; -1 past the end.
    at: Value
    #: A map's or a tree's version when the walk began.
    version: Value | None
    #: A tree walk's current key, kept for finding the next one.
    key: Value | None


@dataclass(slots=True)
class _Plan:
    """A `for` loop's iterable: the walks it drives in step, and its count."""

    sources: list[_Source]
    counter: Value | None = None


def _called(node: ast.expr, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def _equatable(shape: Shape) -> bool:
    """Whether native memory compares it as Python's `==` does: not an object,
    whose `==` is its class's, at any depth."""
    if shape.kind == "object":
        return False
    if shape.kind == "collection":
        kind = shape.collection
        assert kind is not None
        return all(_equatable(part) for part in (kind.key, kind.value) if part is not None)
    return True


class CollectionApiLowering(CollectionLowering):
    """The walks and whole-collection operations; mixed into `_FunctionLowering`."""

    #: How many hidden locals the walks and sort keys have named.
    _walks: int = 0

    # -- holding a collection for a walk ------------------------------------------

    def _hold(self, kind: Kind, handle: Value, owned: bool) -> Value:
        """A slot keeping `handle` alive until `_let_go`: a local of its own,
        which a `return` lets go of with the others."""
        self._walks += 1
        name = f".walk{self._walks}"
        self._bind(name, kind, handle, owned)
        return self.collections[name].slot

    def _let_go(self, slot: Value) -> None:
        old = core.load(self.b, slot)
        core.store(self.b, self._rt("ppy_coll_none", (), HANDLE), slot)
        self._release(old)

    # -- what a walk goes over ---------------------------------------------------

    def _source(self, node: ast.expr) -> _Source | None:
        """The walk an iterable is: a collection, a map's view, `between`, `reversed`, `sorted`."""
        if isinstance(node, ast.Call) and _called(node, "reversed"):
            if len(node.args) != 1 or node.keywords:
                return None
            inner = self._source(node.args[0])
            if inner is None or inner.backwards or inner.low is not None:
                return None
            if inner.kind.name in {"Heap", "MaxHeap"}:
                raise Unsupported(f"a {inner.kind.name} has no order to reverse")
            inner.backwards = True
            return inner
        if isinstance(node, ast.Call) and _called(node, "sorted"):
            return self._sorted_source(node)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and self._is_collection(node.func.value)
            and node.func.attr in {*_VIEWS, "between"}
        ):
            kind, handle, owned = self._receiver(node.func.value)
            if node.func.attr == "between":
                if kind.family != "tree" or len(node.args) != 2:
                    return None
                low = self._key(kind, node.args[0])
                high = self._key(kind, node.args[1])
                return _Source(kind, self._hold(kind, handle, owned), low=low, high=high)
            if kind.value is None or kind.family not in {"map", "tree"}:
                return None
            mode = "elements" if node.func.attr == "keys" else node.func.attr
            return _Source(kind, self._hold(kind, handle, owned), mode)
        if not self._is_collection(node):
            return None
        kind, handle, owned = self._receiver(node)
        if kind.name in {"Heap", "MaxHeap"}:
            raise Unsupported(f"a {kind.name} is read by `peek` and `pop`, not iterated")
        return _Source(kind, self._hold(kind, handle, owned))

    def _sorted_source(self, node: ast.Call) -> _Source:
        """`sorted(c)`: the elements (or keys) copied into a new `Vec` and sorted there."""
        if len(node.args) != 1 or any(k.arg != "reverse" for k in node.keywords):
            raise Unsupported("`sorted` takes one collection and `reverse=` natively")
        inner = self._source(node.args[0])
        if inner is None:
            raise Unsupported("`sorted` sorts a collection natively")
        shape = self._part(inner, "values" if inner.mode == "values" else "keys")
        if inner.mode == "items" or not self._orders(shape):
            raise Unsupported("these elements have no order to sort by")
        kind = Kind("Vec", shape)
        made = self._new(kind)
        self._walk(inner, lambda items: self._add_value(kind, made, items[0][1]))
        descending = self._word(0)
        for keyword in node.keywords:
            descending = core.cast(self.b, self._test(keyword.value), I64)  # type: ignore[attr-defined]
        self._rt("ppy_seq_sort_by", (made, self._word(shape.words), descending), None)
        return _Source(kind, self._hold(kind, made, owned=True))

    def _part(self, source: _Source, which: str) -> Shape:
        """The shape of a walk's elements (a map's or a set's keys) or of its values."""
        kind = source.kind
        shape = kind.value
        if kind.family in {"map", "tree"} and which != "values":
            shape = kind.key
        assert shape is not None
        return shape

    # -- cursors ---------------------------------------------------------------------

    def _start(self, source: _Source) -> _Cursor:
        handle = core.load(self.b, source.slot)
        family = source.kind.family
        at = self._alloca(I64, "walk.at")  # type: ignore[attr-defined]
        version = None
        if family in {"map", "tree"}:
            version = self._rt("ppy_coll_field", (handle, self._word(6)))
        key = None
        if family == "tree":
            assert source.kind.key is not None
            key = self._alloca(source.kind.key.ir_type(), "walk.key")  # type: ignore[attr-defined]
        if family == "seq":
            length = self._rt("ppy_coll_len", (handle,))
            start = (
                core.sub(self.b, length, self._word(1), overflow="wrap")
                if source.backwards
                else self._word(0)
            )
        elif family == "list":
            start = self._rt("ppy_coll_field", (handle, self._word(4 if source.backwards else 3)))
        elif family == "map":
            if source.backwards:
                used = self._rt("ppy_coll_field", (handle, self._word(3)))
                start = self._rt("ppy_map_back", (handle, used))
            else:
                start = self._rt("ppy_coll_step", (handle, self._word(-1)))
        elif source.low is not None:
            start = self._rt("ppy_tree_bound", (handle, source.low, self._word(1)))
        else:
            start = self._rt("ppy_tree_end", (handle, self._word(int(source.backwards))))
        core.store(self.b, start, at)
        return _Cursor(source, at, version, key)

    def _more(self, cursor: _Cursor) -> Value:
        """Whether the walk has a current element, after checking nothing was added
        or removed since it began."""
        source = cursor.source
        handle = core.load(self.b, source.slot)
        if cursor.version is not None:
            now = self._rt("ppy_coll_field", (handle, self._word(6)))
            self._require(
                core.cmp(self.b, "eq", now, cursor.version),
                f"{source.kind.name} changed during iteration",
                f"RuntimeError: {source.kind.name} changed during iteration",
            )
        at = core.load(self.b, cursor.at)
        if source.kind.family == "seq":
            inside = core.cmp(self.b, "lt", at, self._rt("ppy_coll_len", (handle,)))
            if not source.backwards:
                return inside
            return core.bitwise(self.b, "and", inside, self._found(at))
        if source.high is not None:
            below = self._rt("ppy_tree_below", (handle, at, source.high))
            return core.cmp(self.b, "ne", below, self._word(0))
        return self._found(at)

    def _current(self, cursor: _Cursor) -> list[tuple[Shape, Value]]:
        """What the walk is at: one element, key, or value, or a key and its value."""
        source = cursor.source
        kind = source.kind
        handle = core.load(self.b, source.slot)
        at = core.load(self.b, cursor.at)
        family = kind.family
        if family in {"seq", "list"}:
            symbol = "ppy_seq_at" if family == "seq" else "ppy_list_at"
            assert kind.value is not None
            return [(kind.value, self._read(self._rt(symbol, (handle, at), HANDLE), kind.value))]
        key_shape = kind.key
        assert key_shape is not None
        key_address = self._rt(f"ppy_{family}_key_at", (handle, at), HANDLE)
        key = self._read(key_address, key_shape)
        if cursor.key is not None:
            core.store(self.b, key, cursor.key)
        if source.mode == "elements":
            return [(key_shape, key)]
        value_shape = kind.value
        assert value_shape is not None
        value_address = self._rt(f"ppy_{family}_value_at", (handle, at), HANDLE)
        value = self._read(value_address, value_shape)
        if source.mode == "values":
            return [(value_shape, value)]
        return [(key_shape, key), (value_shape, value)]

    def _advance(self, cursor: _Cursor) -> None:
        source = cursor.source
        handle = core.load(self.b, source.slot)
        at = core.load(self.b, cursor.at)
        family = source.kind.family
        if family == "seq":
            step = self._word(-1 if source.backwards else 1)
            following = core.add(self.b, at, step, overflow="wrap")
        elif family == "list":
            symbol = "ppy_list_before" if source.backwards else "ppy_list_after"
            following = self._rt(symbol, (handle, at))
        elif family == "map":
            symbol = "ppy_map_back" if source.backwards else "ppy_coll_step"
            following = self._rt(symbol, (handle, at))
        else:
            assert cursor.key is not None
            buffer = core.cast(self.b, cursor.key, _pointer(cursor.key, HANDLE.pointee))
            mode = self._word(2 if source.backwards else 3)
            following = self._rt("ppy_tree_bound", (handle, buffer, mode))
        core.store(self.b, following, cursor.at)

    def _walk(self, source: _Source, visit: Callable[[list[tuple[Shape, Value]]], None]) -> None:
        """A walk with no body of its own: `visit` sees each step; the collection
        is let go of at the end."""
        cursor = self._start(source)
        header = self._block("walk.head")  # type: ignore[attr-defined]
        body = self._block("walk.body")  # type: ignore[attr-defined]
        done = self._block("walk.end")  # type: ignore[attr-defined]
        core.br(self.b, Successor(header))
        self.b.at_end(header)  # type: ignore[attr-defined]
        core.cond_br(self.b, self._more(cursor), Successor(body), Successor(done))
        self.b.at_end(body)  # type: ignore[attr-defined]
        visit(self._current(cursor))
        self._advance(cursor)
        core.br(self.b, Successor(header))
        self.b.at_end(done)  # type: ignore[attr-defined]
        self._let_go(source.slot)

    # -- loops ------------------------------------------------------------------------

    def _plan(self, node: ast.expr) -> _Plan | None:
        """What a `for` loop walks: one source, several in step (`zip`), and a count."""
        if isinstance(node, ast.Call) and _called(node, "enumerate"):
            arguments = list(node.args)
            for keyword in node.keywords:
                if keyword.arg != "start":
                    return None
                arguments.append(keyword.value)
            if not 1 <= len(arguments) <= 2:
                return None
            inner = self._plan(arguments[0])
            if inner is None or inner.counter is not None or len(inner.sources) != 1:
                return None
            start = self._word(0)
            if len(arguments) == 2:
                start = self._coerce(self._expr(arguments[1]), "int")  # type: ignore[attr-defined]
            counter = self._alloca(I64, "walk.count")  # type: ignore[attr-defined]
            core.store(self.b, start, counter)
            return _Plan(inner.sources, counter)
        if isinstance(node, ast.Call) and _called(node, "zip"):
            if node.keywords or len(node.args) < 2:
                return None
            sources = [self._source(argument) for argument in node.args]
            if any(source is None for source in sources):
                return None
            return _Plan([source for source in sources if source is not None])
        source = self._source(node)
        return _Plan([source]) if source is not None else None

    def _is_walk(self, node: ast.expr) -> bool:
        """Whether a `for` loop's iterable is one this lowering walks, before any code is made."""
        if isinstance(node, ast.Call) and any(
            _called(node, name) for name in ("enumerate", "reversed", "sorted")
        ):
            return bool(node.args) and self._is_walk(node.args[0])
        if isinstance(node, ast.Call) and _called(node, "zip"):
            return len(node.args) >= 2 and all(self._is_walk(a) for a in node.args)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {*_VIEWS, "between"}
        ):
            return self._is_collection(node.func.value)
        return self._is_collection(node)

    def _for_collection(self, node: ast.For) -> None:
        """`for x in c`, and over `c.items()`, `enumerate(c)`, `zip(a, b)`,
        `reversed(c)`, `sorted(c)`, `t.between(lo, hi)`, walking as the reference walks.

        A `Vec` or `Deque` reads its length at every step, as a list does.
        A linked list follows `next` from each node after its body ran. A
        map or a set checks that nothing was added or removed since the
        loop began, as `dict` does, and CPython's `RuntimeError` is what the
        check falls back to.
        """
        if node.orelse:
            raise Unsupported("a `for` loop's `else` has no native lowering")
        plan = self._plan(node.iter)
        if plan is None:
            raise Unsupported(f"`{ast.unparse(node.iter)}` is not walked natively")
        cursors = [self._start(source) for source in plan.sources]
        header = self._block("each.head")  # type: ignore[attr-defined]
        body = self._block("each.body")  # type: ignore[attr-defined]
        latch = self._block("each.latch")  # type: ignore[attr-defined]
        done = self._block("each.end")  # type: ignore[attr-defined]
        core.br(self.b, Successor(header))
        self.b.at_end(header)  # type: ignore[attr-defined]
        # `zip` asks each walk in turn and stops at the first that has ended,
        # without asking the ones after it.
        for index, cursor in enumerate(cursors):
            last = index == len(cursors) - 1
            following = body if last else self._block("each.next")  # type: ignore[attr-defined]
            core.cond_br(self.b, self._more(cursor), Successor(following), Successor(done))
            self.b.at_end(following)  # type: ignore[attr-defined]
        items = [self._current(cursor) for cursor in cursors]
        target = node.target
        if plan.counter is not None:
            if not isinstance(target, (ast.Tuple, ast.List)) or len(target.elts) != 2:
                raise Unsupported("`enumerate` is unpacked into a count and an element")
            self._store(target.elts[0], core.load(self.b, plan.counter))  # type: ignore[attr-defined]
            target = target.elts[1]
        if len(cursors) > 1:
            if not isinstance(target, (ast.Tuple, ast.List)) or len(target.elts) != len(cursors):
                raise Unsupported("`zip` is unpacked into one name per collection")
            for inner, parts in zip(target.elts, items, strict=True):
                self._bind_item(inner, parts)
        else:
            self._bind_item(target, items[0])
        self._loops.append((latch, done))  # type: ignore[attr-defined]
        self._body(node.body)  # type: ignore[attr-defined]
        self._loops.pop()  # type: ignore[attr-defined]
        if self._open():  # type: ignore[attr-defined]
            core.br(self.b, Successor(latch))
        self.b.at_end(latch)  # type: ignore[attr-defined]
        for cursor in cursors:
            self._advance(cursor)
        if plan.counter is not None:
            count = core.load(self.b, plan.counter)
            core.store(
                self.b, core.add(self.b, count, self._word(1), overflow="wrap"), plan.counter
            )
        core.br(self.b, Successor(header))
        self.b.at_end(done)  # type: ignore[attr-defined]
        for source in plan.sources:
            self._let_go(source.slot)
        # `between`'s bounds, made before the loop, are read until it ends.
        self._keys_done()

    def _bind_item(self, target: ast.expr, parts: list[tuple[Shape, Value]]) -> None:
        """A loop target bound to one step: an element, or a key and its value."""
        if len(parts) == 1:
            shape, value = parts[0]
            self._bind_element(target, shape, value)
            return
        if isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == len(parts):
            for inner, (shape, value) in zip(target.elts, parts, strict=True):
                self._bind_element(inner, shape, value)
            return
        if all(shape.kind in {"int", "float", "bool"} for shape, _ in parts):
            self._store_tuple(target, [value for _, value in parts])  # type: ignore[attr-defined]
            return
        raise Unsupported("an item of a map is unpacked into a key and a value")

    # -- storing ------------------------------------------------------------------------

    def _store_value(self, address: Value, shape: Shape, value: Value, owned: bool) -> None:
        """A value already in hand written into a fresh slot: a collection takes a reference."""
        if shape.reference and not owned:
            self._retain(value)
        self._write(address, shape, value)

    def _room(self, kind: Kind, handle: Value, front: bool = False) -> Value:
        """A new element's slot at the back (or the front) of a sequence or a list."""
        if kind.family == "seq":
            symbol = "ppy_seq_push_front" if front else "ppy_seq_push_back"
            return self._rt(symbol, (handle,), HANDLE)
        end = self._rt("ppy_coll_field", (handle, self._word(3 if front else 4)))
        links = (self._word(-1), end) if front else (end, self._word(-1))
        node = self._rt("ppy_list_node", (handle, *links))
        return self._rt("ppy_list_at", (handle, node), HANDLE)

    def _add_value(self, kind: Kind, handle: Value, value: Value, front: bool = False) -> None:
        """One value, read from somewhere else, added: pushed, linked, or put as a key."""
        if kind.family in {"map", "tree"}:
            key = kind.key
            assert key is not None
            buffer = self._alloca(key.ir_type(), "key")  # type: ignore[attr-defined]
            address = core.cast(self.b, buffer, _pointer(buffer, HANDLE.pointee))
            self._write(address, key, value)
            self._rt("ppy_coll_put_key", (handle, address))
            return
        assert kind.value is not None
        self._store_value(self._room(kind, handle, front), kind.value, value, owned=False)

    def _add_node(self, kind: Kind, handle: Value, node: ast.expr, front: bool = False) -> None:
        if kind.family in {"map", "tree"}:
            self._rt("ppy_coll_put_key", (handle, self._key(kind, node)))
            self._keys_done()
            return
        assert kind.value is not None
        self._store_into(self._room(kind, handle, front), kind.value, node, fresh=True)

    def _fill(self, kind: Kind, handle: Value, node: ast.expr, front: bool = False) -> None:
        """Every element of an iterable added to a collection: a display, a
        `range`, or another collection walked."""
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for element in node.elts:
                if isinstance(element, ast.Starred):
                    raise Unsupported("a starred element has no native lowering")
                self._add_node(kind, handle, element, front)
            return
        if isinstance(node, ast.Call) and _called(node, "range"):
            self._fill_range(kind, handle, node, front)
            return
        other = self._kind_of(node)
        if (
            other is not None
            and kind.family == "seq"
            and not front
            and other.family in {"seq", "list"}
            and other.value == kind.value
            and other.name not in {"Heap", "MaxHeap"}
        ):
            # One call copies the words, and handles `v.extend(v)`.
            source, owned = self._handle(node)
            self._rt("ppy_seq_extend", (handle, source), None)
            self._done_with(source, owned)
            return
        source = self._source(node)
        if source is None:
            raise Unsupported(f"`{ast.unparse(node)}` is not filled from natively")
        if source.mode == "items":
            raise Unsupported("a map's items are not filled from natively")
        if source.kind == kind:
            # The same collection may be both ends: walk a copy, as Python
            # walks the list it made of the argument first.
            held = core.load(self.b, source.slot)
            copy = self._rt("ppy_coll_copy", (held,), HANDLE)
            core.store(self.b, copy, source.slot)
            self._release(held)
        self._walk(source, lambda items: self._add_value(kind, handle, items[0][1], front))

    def _fill_range(self, kind: Kind, handle: Value, node: ast.Call, front: bool) -> None:
        """`range(...)`: its integers, counted up front the way `len(range(...))` counts them."""
        if node.keywords or not 1 <= len(node.args) <= 3:
            raise Unsupported("`range` takes one to three arguments")
        bounds = [self._coerce(self._expr(a), "int") for a in node.args]  # type: ignore[attr-defined]
        start, stop, step = self._word(0), bounds[0], self._word(1)
        if len(bounds) >= 2:
            start, stop = bounds[0], bounds[1]
        if len(bounds) == 3:
            step = bounds[2]
        self._require(
            core.cmp(self.b, "ne", step, self._word(0)),
            "range() arg 3 must not be zero",
            "ValueError: range() arg 3 must not be zero",
        )
        rising = core.cmp(self.b, "gt", step, self._word(0))
        low = core.select(self.b, rising, start, stop)
        high = core.select(self.b, rising, stop, start)
        width = core.select(
            self.b, rising, step, core.sub(self.b, self._word(0), step, overflow="wrap")
        )
        span = self._span(high, low)
        steps = core.div(
            self.b, core.sub(self.b, span, self._word(1), overflow="wrap"), width, overflow="wrap"
        )
        count = core.select(
            self.b,
            core.cmp(self.b, "lt", low, high),
            core.add(self.b, steps, self._word(1), overflow="wrap"),
            self._word(0),
        )
        index = self._alloca(I64, "range.i")  # type: ignore[attr-defined]
        core.store(self.b, self._word(0), index)
        header = self._block("range.head")  # type: ignore[attr-defined]
        body = self._block("range.body")  # type: ignore[attr-defined]
        done = self._block("range.end")  # type: ignore[attr-defined]
        core.br(self.b, Successor(header))
        self.b.at_end(header)  # type: ignore[attr-defined]
        at = core.load(self.b, index)
        core.cond_br(self.b, core.cmp(self.b, "lt", at, count), Successor(body), Successor(done))
        self.b.at_end(body)  # type: ignore[attr-defined]
        at = core.load(self.b, index)
        offset = core.mul(self.b, at, step, overflow="wrap")
        self._add_value(kind, handle, core.add(self.b, start, offset, overflow="wrap"), front)
        core.store(self.b, core.add(self.b, at, self._word(1), overflow="wrap"), index)
        core.br(self.b, Successor(header))
        self.b.at_end(done)  # type: ignore[attr-defined]

    def _span(self, high: Value, low: Value) -> Value:
        """`high - low`, which leaves the word only for ranges wider than it: those
        are left to Python."""
        difference = core.sub(self.b, high, low, overflow="wrap")
        empty = core.cmp(self.b, "ge", low, high)
        fits = core.cmp(self.b, "gt", difference, self._word(0))
        # CPython would build it and run out of memory first.
        self._require(
            core.bitwise(self.b, "or", empty, fits),
            "a range wider than a machine word",
            "MemoryError",
        )
        return difference

    # -- handles: new collections from expressions -------------------------------------

    def _handle(self, node: ast.expr) -> tuple[Value, bool]:
        if isinstance(node, ast.BinOp) and self._is_collection(node):
            return self._combined(node), True
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Slice)
            and self._is_collection(node.value)
        ):
            return self._slice(node.value, node.slice), True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript) and node.args:
            kind = self._constructed(node)
            if kind is not None and not self._counts(node.args[0]):
                return self._filled(kind, node), True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and self._is_collection(node.func.value)
            and node.func.attr in _OWNED
        ):
            found = self._collection_method(node.func.value, node.func.attr, node)
            return found, True
        return super()._handle(node)

    def _counts(self, node: ast.expr) -> bool:
        """Whether a `Vec`'s argument is how many zeros it starts with."""
        return T.strip_literal(self._type_of(node)) in (T.INT, T.BOOL)

    def _filled(self, kind: Kind, node: ast.Call) -> Value:
        """`Vec[int](items)`, `Heap[int](items)`, `HashSet[int](items)`: made, then filled."""
        if len(node.args) != 1 or node.keywords:
            raise Unsupported(f"`{kind.name}` takes one iterable")
        heap = {"Heap": 0, "MaxHeap": 1}.get(kind.name)
        if heap is not None:
            assert kind.value is not None
            if not self._orders(kind.value):
                raise Unsupported(f"a {kind.name} orders its elements, and these have no order")
        made = self._new(kind)
        self._fill(kind, made, node.args[0])
        if heap is not None:
            self._rt("ppy_heap_heapify", (made, self._word(heap)), None)
        return made

    def _combined(self, node: ast.BinOp) -> Value:
        """`v + w` of two sequences, and `a | b`, `a & b`, `a - b`, `a ^ b` of two sets."""
        kind = self._kind_of(node.left)
        other = self._kind_of(node.right)
        if kind is None or other != kind:
            raise Unsupported("a collection combines with another of its own type")
        left, left_owned = self._handle(node.left)
        right, right_owned = self._handle(node.right)
        if isinstance(node.op, ast.Add) and kind.name in {"Vec", "Deque"}:
            made = self._rt("ppy_seq_concat", (left, right), HANDLE)
        elif type(node.op) in _OPERATORS and kind.name in {"HashSet", "TreeSet"}:
            operation = self._word(_OPERATORS[type(node.op)])
            made = self._rt("ppy_set_combine", (left, right, operation), HANDLE)
        else:
            raise Unsupported(f"`{ast.unparse(node)}` has no native lowering")
        self._done_with(left, left_owned)
        self._done_with(right, right_owned)
        return made

    def _slice(self, container: ast.expr, bounds: ast.Slice) -> Value:
        """`v[a:b:c]`: a list's slice, as a new `Vec`."""
        kind, handle, owned = self._receiver(container)
        if kind.name != "Vec":
            raise Unsupported(f"a {kind.name} is not sliced")
        given = 0
        parts: list[Value] = []
        for bit, part in enumerate((bounds.lower, bounds.upper)):
            if part is None:
                parts.append(self._word(0))
                continue
            given |= 1 << bit
            parts.append(self._coerce(self._expr(part), "int"))  # type: ignore[attr-defined]
        step = self._word(1)
        if bounds.step is not None:
            step = self._coerce(self._expr(bounds.step), "int")  # type: ignore[attr-defined]
            self._require(
                core.cmp(self.b, "ne", step, self._word(0)),
                "slice step cannot be zero",
                "ValueError: slice step cannot be zero",
            )
        made = self._rt(
            "ppy_seq_slice", (handle, parts[0], parts[1], step, self._word(given)), HANDLE
        )
        self._done_with(handle, owned)
        return made

    # -- comparisons and search ------------------------------------------------------

    def _decided(self, answer: Value, undecided: int) -> Value:
        """A runtime answer that may be undecided (a NaN): left to Python where it is."""
        # Under `ppy run` Python answers; a standalone binary, with no NaN
        # objects to tell apart, says so and stops.
        self._require(
            core.cmp(self.b, "ne", answer, self._word(undecided)),
            "a NaN compared: CPython tells NaN objects apart, native memory does not",
            "ValueError: a NaN was compared, and a native binary cannot tell one NaN from another",
        )
        return answer

    def _collection_equality(self, node: ast.Compare) -> Value | None:
        """`a == b`, `a != b` of two collections of one type: by what they hold."""
        if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
            return None
        left, right = node.left, node.comparators[0]
        kind = self._kind_of(left)
        if kind is None or not self._is_collection(right):
            return None
        other = self._kind_of(right)
        if other != kind:
            # Two kinds of collection are never equal, as a list is not a deque.
            if other is not None and other.name != kind.name:
                return core.const(self.b, isinstance(node.ops[0], ast.NotEq), BOOL)
            raise Unsupported("collections of different element types compare in Python")
        if kind.name in {"Heap", "MaxHeap"} or not _equatable(Shape("collection", collection=kind)):
            raise Unsupported("these collections compare in Python")
        a, a_owned = self._handle(left)
        b, b_owned = self._handle(right)
        answer = self._decided(self._rt("ppy_coll_equal", (a, b)), -1)
        self._done_with(a, a_owned)
        self._done_with(b, b_owned)
        predicate = "eq" if isinstance(node.ops[0], ast.Eq) else "ne"
        return core.cmp(self.b, predicate, answer, self._word(1))

    def _value_buffer(self, shape: Shape, node: ast.expr) -> tuple[Value, Value | None]:
        """A value's words in a stack buffer, for a search; and the handle to let go
        of afterwards, where the value made one."""
        if not _equatable(shape):
            raise Unsupported(
                "objects compare by their class's `==`, which native code does not call"
            )
        buffer = self._alloca(shape.ir_type(), "value")  # type: ignore[attr-defined]
        address = core.cast(self.b, buffer, _pointer(buffer, HANDLE.pointee))
        value, owned = self._value(node, shape)
        self._write(address, shape, value)
        return address, value if owned else None

    def _contains(self, container: ast.expr, key: ast.expr) -> Value:
        """`x in c`: a key of a map or a set, or an element of a sequence or a list."""
        kind = self._kind_of(container)
        if kind is None or kind.family in {"map", "tree"}:
            return super()._contains(container, key)
        if kind.name in {"Heap", "MaxHeap"}:
            raise Unsupported(f"a {kind.name} is read by `peek` and `pop`")
        _, handle, owned = self._receiver(container)
        assert kind.value is not None
        address, made = self._value_buffer(kind.value, key)
        found = self._rt("ppy_coll_find_value", (handle, address, self._word(0)))
        found = self._decided(found, -2)
        if made is not None:
            self._release(made)
        self._done_with(handle, owned)
        return self._found(found)

    # -- methods ------------------------------------------------------------------------

    def _collection_method(self, receiver: ast.expr, attr: str, node: ast.Call) -> Value:
        """A method of a collection: the ones here first, then one element at a time."""
        kind, handle, owned = self._receiver(receiver)
        if kind.name == "List":
            # `list[str]`, the list string methods hand out, has its own methods.
            if node.keywords:
                raise Unsupported(f"`List.{attr}` takes no keyword arguments")
            found = self._string_list_method(kind, handle, attr, node.args)  # type: ignore[attr-defined]
            self._keys_done()
            if found is None:
                raise Unsupported(f"`List.{attr}` has no native lowering")
            self._done_with(handle, owned)
            return found
        found = self._whole_method(kind, handle, attr, node)
        if found is None and not node.keywords:
            method = {
                "seq": self._sequence_method,
                "list": self._list_method,
                "map": self._keyed_method,
                "tree": self._keyed_method,
            }[kind.family]
            found = method(kind, handle, attr, node.args)
        # The keys made for the calls just emitted go now, in the block that made them.
        self._keys_done()
        if found is None:
            raise Unsupported(f"`{kind.name}.{attr}` has no native lowering")
        self._done_with(handle, owned)
        return found

    def _whole_method(self, kind: Kind, handle: Value, attr: str, node: ast.Call) -> Value | None:
        """The methods beyond one element at a time."""
        arguments = node.args
        shape = kind.value
        if attr == "sort" and kind.name == "Vec":
            return self._sort(kind, handle, node)
        if node.keywords:
            return None
        if attr == "copy" and kind.name != "LinkedList":
            return self._rt("ppy_coll_copy", (handle,), HANDLE)
        if attr == "extend" and kind.name in {"Vec", "Deque", "LinkedList"}:
            self._fill(kind, handle, arguments[0])
            return self._word(0)
        if attr == "extendleft" and kind.name == "Deque":
            self._fill(kind, handle, arguments[0], front=True)
            return self._word(0)
        if kind.name in {"Vec", "Deque"}:
            found = self._sequence_extra(kind, handle, attr, arguments)
            if found is not None:
                return found
        heap = {"Heap": 0, "MaxHeap": 1}.get(kind.name)
        if heap is not None:
            assert shape is not None
            if attr in {"pushpop", "replace"}:
                if attr == "replace":
                    length = self._rt("ppy_coll_len", (handle,))
                    self._require(
                        core.cmp(self.b, "gt", length, self._word(0)),
                        f"replace in an empty {kind.name}",
                        f"IndexError: replace in an empty {kind.name}",
                    )
                scratch = self._rt("ppy_coll_scratch", (handle,), HANDLE)
                self._store_into(scratch, shape, arguments[0], fresh=True)
                replace = self._word(int(attr == "replace"))
                out = self._rt("ppy_heap_exchange", (handle, self._word(heap), replace), HANDLE)
                return self._read(out, shape)
            if attr == "to_sorted":
                made = self._rt("ppy_coll_copy", (handle,), HANDLE)
                self._rt("ppy_seq_sort_by", (made, self._word(shape.words), self._word(heap)), None)
                return made
        if kind.family in {"map", "tree"}:
            return self._keyed_extra(kind, handle, attr, arguments)
        return None

    def _sequence_extra(
        self, kind: Kind, handle: Value, attr: str, arguments: list[ast.expr]
    ) -> Value | None:
        """`insert`, `pop(i)`, `remove`, `index`, `count`, `rotate` of a `Vec` or a `Deque`."""
        shape = kind.value
        assert shape is not None
        if attr == "insert":
            length = self._rt("ppy_coll_len", (handle,))
            position = self._coerce(self._expr(arguments[0]), "int")  # type: ignore[attr-defined]
            inside = core.bitwise(
                self.b,
                "and",
                core.cmp(self.b, "ge", position, self._word(0)),
                core.cmp(self.b, "le", position, length),
            )
            self._require(
                inside,
                "insert index out of range",
                "IndexError: insert at {0} is out of range for length {1}",
                (position, length),
            )
            address = self._rt("ppy_seq_insert", (handle, position), HANDLE)
            self._store_into(address, shape, arguments[1], fresh=True)
            return self._word(0)
        if attr == "pop" and arguments and kind.name == "Vec":
            self._nonempty(kind, handle, "pop")
            length = self._rt("ppy_coll_len", (handle,))
            position = self._coerce(self._expr(arguments[0]), "int")  # type: ignore[attr-defined]
            inside = core.bitwise(
                self.b,
                "and",
                core.cmp(self.b, "ge", position, self._word(0)),
                core.cmp(self.b, "lt", position, length),
            )
            self._require(
                inside,
                "pop index out of range",
                "IndexError: index {0} is out of range for length {1}",
                (position, length),
            )
            return self._read(self._rt("ppy_seq_erase", (handle, position), HANDLE), shape)
        if attr in {"remove", "index", "count"}:
            address, made = self._value_buffer(shape, arguments[0])
            if attr == "count":
                found = self._decided(self._rt("ppy_coll_count_value", (handle, address)), -1)
            else:
                found = self._rt("ppy_coll_find_value", (handle, address, self._word(0)))
                found = self._decided(found, -2)
                missing = f"ValueError: the value is not in the {kind.name}"
                reported: tuple[Value, ...] = ()
                if shape.kind == "int":
                    missing = f"ValueError: {{0}} is not in the {kind.name}"
                    reported = (self._read_word(address, 0, "int"),)
                self._require(
                    self._found(found),
                    f"{kind.name}.{attr}(x): x not in {kind.name}",
                    missing,
                    reported,
                )
            if made is not None:
                self._release(made)
            if attr != "remove":
                return found
            taken = self._rt("ppy_seq_erase", (handle, found), HANDLE)
            if shape.reference:
                self._rt("ppy_coll_release_words", (handle, taken), None)
            return self._word(0)
        if attr == "rotate" and kind.name == "Deque":
            steps = self._word(1)
            if arguments:
                steps = self._coerce(self._expr(arguments[0]), "int")  # type: ignore[attr-defined]
            self._rt("ppy_seq_rotate", (handle, steps), None)
            return self._word(0)
        return None

    def _keyed_extra(
        self, kind: Kind, handle: Value, attr: str, arguments: list[ast.expr]
    ) -> Value | None:
        """`get`, `setdefault`, `pop(key, default)`, `update`, the set operations, and
        a tree's `pop_min` and `pop_max`."""
        family = kind.family
        shape = kind.value
        key_shape = kind.key
        assert key_shape is not None
        if attr in _COMBINE or attr in _RELATIONS or attr == "update":
            if self._kind_of(arguments[0]) != kind:
                raise Unsupported(f"`{attr}` takes a collection of the same type")
            other, other_owned = self._handle(arguments[0])
            if attr == "update":
                self._rt("ppy_coll_update", (handle, other), None)
                found = self._word(0)
            elif attr in _COMBINE:
                operation = self._word(_COMBINE[attr])
                found = self._rt("ppy_set_combine", (handle, other, operation), HANDLE)
            else:
                relation = self._word(_RELATIONS[attr])
                answer = self._rt("ppy_set_relation", (handle, other, relation))
                found = core.cmp(self.b, "ne", answer, self._word(0))
            self._done_with(other, other_owned)
            return found
        if attr in {"pop_min", "pop_max"} and family == "tree":
            length = self._rt("ppy_coll_len", (handle,))
            self._require(
                core.cmp(self.b, "gt", length, self._word(0)),
                f"{attr} of an empty {kind.name}",
                f"IndexError: pop from an empty {kind.name}",
            )
            node = self._rt("ppy_tree_end", (handle, self._word(int(attr == "pop_max"))))
            key_address = self._rt("ppy_tree_key_at", (handle, node), HANDLE)
            key = self._read(key_address, key_shape)
            value = None
            if shape is not None:
                if shape.reference or key_shape.kind not in {"int", "float", "bool"}:
                    raise Unsupported(f"`{attr}` of this map has no native tuple form")
                value = self._read(self._rt("ppy_tree_value_at", (handle, node), HANDLE), shape)
            if key_shape.reference:
                # The tree lets go of the key it removes; what is handed out is
                # the caller's, as a popped element is.
                self._retain(key)
            self._rt("ppy_tree_remove", (handle, key_address))
            return key if value is None else core.tuple_make(self.b, key, value)
        if shape is None:
            return None
        numbers = shape.kind in {"int", "float", "bool"}
        defaulted = (attr == "get" and not numbers) or attr == "setdefault"
        if defaulted or (attr == "pop" and len(arguments) == 2):
            return self._defaulted(kind, handle, attr, arguments, shape)
        return None

    def _defaulted(
        self, kind: Kind, handle: Value, attr: str, arguments: list[ast.expr], shape: Shape
    ) -> Value:
        """`get`, `setdefault`, and `pop` with a default: the value where the key is,
        the default where it is not. A collection handed out is the caller's."""
        family = kind.family
        default, default_owned = self._value(arguments[1], shape)
        key = self._key(kind, arguments[0])
        entry = self._rt(f"ppy_{family}_find", (handle, key))
        present = self._found(entry)
        result = self._alloca(shape.ir_type(), f"{attr}.result")  # type: ignore[attr-defined]
        hit = self._block(f"{attr}.hit")  # type: ignore[attr-defined]
        miss = self._block(f"{attr}.miss")  # type: ignore[attr-defined]
        done = self._block(f"{attr}.end")  # type: ignore[attr-defined]
        core.cond_br(self.b, present, Successor(hit), Successor(miss))
        self.b.at_end(hit)  # type: ignore[attr-defined]
        if attr == "pop":
            self._rt(f"ppy_{family}_remove", (handle, key))
        found = self._read(self._rt(f"ppy_{family}_value_at", (handle, entry), HANDLE), shape)
        # `get` and `pop` hand their caller a reference; `setdefault` lends the
        # value the map keeps.
        if shape.reference and attr == "get":
            self._retain(found)
        core.store(self.b, found, result)
        if default_owned:
            self._release(default)
        core.br(self.b, Successor(done))
        self.b.at_end(miss)  # type: ignore[attr-defined]
        if attr == "setdefault":
            made = self._rt(f"ppy_{family}_put", (handle, key))
            address = self._rt(f"ppy_{family}_value_at", (handle, made), HANDLE)
            self._store_value(address, shape, default, default_owned)
        elif shape.reference and not default_owned:
            self._retain(default)
        core.store(self.b, self._converted(default, shape), result)
        core.br(self.b, Successor(done))
        self.b.at_end(done)  # type: ignore[attr-defined]
        return core.load(self.b, result)

    def _converted(self, value: Value, shape: Shape) -> Value:
        """A value as the shape stores it: `3` is `3.0` in a map of floats."""
        if shape.kind in {"int", "float", "bool"}:
            return self._coerce(value, shape.kind)  # type: ignore[attr-defined]
        if shape.kind == "tuple" and isinstance(value.type, TupleType):
            items = [core.tuple_extract(self.b, value, i) for i in range(shape.words)]
            coerced = [
                self._coerce(item, part)  # type: ignore[attr-defined]
                for item, part in zip(items, shape.parts, strict=True)
            ]
            return core.tuple_make(self.b, *coerced)
        return value

    # -- sorting -----------------------------------------------------------------------

    def _sort(self, kind: Kind, handle: Value, node: ast.Call) -> Value:
        """`v.sort()`, with `reverse=` and `key=`: a stable sort, on the elements or on
        the keys a function gives them."""
        shape = kind.value
        assert shape is not None
        if node.args:
            raise Unsupported("`sort` takes keyword arguments")
        options = {keyword.arg: keyword.value for keyword in node.keywords}
        if set(options) - {"key", "reverse"}:
            raise Unsupported("`sort` takes `key=` and `reverse=`")
        descending = self._word(0)
        if "reverse" in options:
            descending = core.cast(self.b, self._test(options["reverse"]), I64)  # type: ignore[attr-defined]
        key = options.get("key")
        if key is None or (isinstance(key, ast.Constant) and key.value is None):
            if not self._orders(shape):
                raise Unsupported("these elements have no order to sort by")
            self._rt("ppy_seq_sort_by", (handle, self._word(shape.words), descending), None)
            return self._word(0)
        evaluate, key_shape = self._key_function(key, shape)
        if key_shape.reference or not key_shape.comparable:
            raise Unsupported("a sort key is a number, a tuple of numbers, or an ordered dataclass")
        # Each element's key and its position, sorted by key; the elements are
        # then put in the order the positions came out.
        pairs = Kind("Vec", Shape("tuple", (*_word_kinds(key_shape), "int")))
        made = self._rt(
            "ppy_seq_new",
            (
                self._word(0),
                self._word(key_shape.words + 1),
                self._word(key_shape.floats),
                self._word(0),
            ),
            HANDLE,
        )
        slot = self._hold(pairs, made, owned=True)
        index = self._alloca(I64, "sort.i")  # type: ignore[attr-defined]
        core.store(self.b, self._word(0), index)
        source = _Source(kind, self._hold(kind, handle, owned=False))

        def keyed(items: list[tuple[Shape, Value]]) -> None:
            value = evaluate(items[0][1])
            pair = self._rt("ppy_seq_push_back", (core.load(self.b, slot),), HANDLE)
            self._write(pair, key_shape, value)
            position = core.load(self.b, index)
            words = core.cast(self.b, pair, PtrType(I64))
            last = core.ptr_offset(self.b, words, self._word(key_shape.words))
            core.store(self.b, position, last)
            core.store(self.b, core.add(self.b, position, self._word(1), overflow="wrap"), index)

        self._walk(source, keyed)
        ordered = core.load(self.b, slot)
        self._rt("ppy_seq_sort_by", (ordered, self._word(key_shape.words), descending), None)
        self._rt("ppy_seq_permute", (handle, ordered, self._word(key_shape.words)), None)
        self._let_go(slot)
        return self._word(0)

    def _key_function(
        self, key: ast.expr, element: Shape
    ) -> tuple[Callable[[Value], Value], Shape]:
        """A sort key as native code: a lambda's body with its parameter bound to the
        element, or a call to a native function of one argument."""
        if isinstance(key, ast.Lambda):
            arguments = key.args
            if len(arguments.args) != 1 or arguments.vararg or arguments.kwonlyargs:
                raise Unsupported("a sort key takes one argument")
            name = arguments.args[0].arg
            body: ast.expr = key.body
            found = self._type_of(body)
        elif isinstance(key, ast.Name):
            self._walks += 1
            name = f".key{self._walks}"
            body = ast.Call(func=key, args=[ast.Name(id=name, ctx=ast.Load())], keywords=[])
            ast.copy_location(body, key)
            ast.fix_missing_locations(body)
            called = T.strip_literal(self._type_of(key))
            if not isinstance(called, T.Callable_):
                raise Unsupported(f"`{key.id}` is not a function native code can call")
            found = called.ret
        else:
            raise Unsupported("a sort key is a lambda or a function's name")
        if self._is_local(name):
            raise Unsupported(f"the sort key's `{name}` would stand for a local")
        key_shape = shape_of(found, self._records())
        if key_shape is None:
            raise Unsupported(f"a sort key giving `{found}` has no native form")

        def evaluate(value: Value) -> Value:
            target = ast.Name(id=name, ctx=ast.Store())
            if element.kind == "record":
                self.objects[name] = value  # type: ignore[attr-defined]
            else:
                self._bind_element(target, element, value)
            result, owned = self._value(body, key_shape)
            if owned:
                self._release(result)
            return result

        return evaluate, key_shape

    def _is_local(self, name: str) -> bool:
        tables = ("slots", "collections", "tuples", "objects", "buffers")
        return any(name in getattr(self, table, {}) for table in tables)

    # -- reductions ----------------------------------------------------------------------

    def _reduction(self, operation: str, node: ast.Call) -> Value | None:
        """`sum(c)`, `min(c)`, `max(c)` of a collection or one of its walks."""
        if len(node.args) != 1 or node.keywords or not self._is_walk(node.args[0]):
            return None
        source = self._source(node.args[0])
        if source is None:
            return None
        if source.mode == "items":
            raise Unsupported(f"`{operation}` of a map's items has no native lowering")
        shape = self._part(source, "values" if source.mode == "values" else "keys")
        if operation == "sum":
            return self._sum(source, shape)
        if shape.reference or not shape.comparable:
            raise Unsupported(f"`{operation}` compares numbers, tuples, and ordered dataclasses")
        return self._extremum_of(source, shape, operation)

    def _sum(self, source: _Source, shape: Shape) -> Value:
        """`sum(c)`: integers add with the overflow guard, and floats as CPython adds
        them, with Neumaier's compensation carried and added at the end."""
        if shape.kind not in {"int", "float", "bool"}:
            raise Unsupported("`sum` adds numbers")
        if shape.kind != "float":
            total = self._alloca(I64, "sum.acc")  # type: ignore[attr-defined]
            core.store(self.b, self._word(0), total)

            def add(items: list[tuple[Shape, Value]]) -> None:
                value = self._coerce(items[0][1], "int")  # type: ignore[attr-defined]
                added = self._binary(core.load(self.b, total), value, ast.Add)  # type: ignore[attr-defined]
                core.store(self.b, added, total)

            self._walk(source, add)
            return core.load(self.b, total)
        seen = self._alloca(BOOL, "sum.seen")  # type: ignore[attr-defined]
        core.store(self.b, core.const(self.b, False, BOOL), seen)
        running = self._alloca(F64, "sum.f")  # type: ignore[attr-defined]
        carried = self._alloca(F64, "sum.c")  # type: ignore[attr-defined]
        zero = core.const(self.b, 0.0, F64)
        core.store(self.b, zero, running)
        core.store(self.b, zero, carried)

        def add_float(items: list[tuple[Shape, Value]]) -> None:
            x = items[0][1]
            f = core.load(self.b, running)
            t = core.add(self.b, f, x)
            bigger = core.cmp(self.b, "ge", _absolute(self, f), _absolute(self, x))
            first = core.add(self.b, core.sub(self.b, f, t), x)
            second = core.add(self.b, core.sub(self.b, x, t), f)
            step = core.select(self.b, bigger, first, second)
            core.store(self.b, core.add(self.b, core.load(self.b, carried), step), carried)
            core.store(self.b, t, running)
            core.store(self.b, core.const(self.b, True, BOOL), seen)

        self._walk(source, add_float)
        # An empty sum is the int 0, which a float result cannot be.
        # Not an error in Python, but an answer native code has no float for:
        # under `ppy run` Python gives it, and a standalone binary says so.
        self._require(
            core.load(self.b, seen),
            "sum of no floats is the int 0",
            "TypeError: sum() of no floats is the int 0, which a native float cannot hold",
        )
        f = core.load(self.b, running)
        c = core.load(self.b, carried)
        finite = core.cmp(self.b, "eq", core.sub(self.b, c, c), zero)
        nonzero = core.cmp(self.b, "ne", c, zero)
        compensated = core.bitwise(self.b, "and", finite, nonzero)
        return core.select(self.b, compensated, core.add(self.b, f, c), f)

    def _extremum_of(self, source: _Source, shape: Shape, operation: str) -> Value:
        """`min(c)` and `max(c)`: the first element no later one beats, as CPython keeps it."""
        best = self._alloca(shape.ir_type(), f"{operation}.best")  # type: ignore[attr-defined]
        seen = self._alloca(BOOL, f"{operation}.seen")  # type: ignore[attr-defined]
        core.store(self.b, core.const(self.b, False, BOOL), seen)
        best_address = core.cast(self.b, best, _pointer(best, HANDLE.pointee))
        candidate = self._alloca(shape.ir_type(), f"{operation}.at")  # type: ignore[attr-defined]
        candidate_address = core.cast(self.b, candidate, _pointer(candidate, HANDLE.pointee))

        def keep(items: list[tuple[Shape, Value]]) -> None:
            self._write(candidate_address, shape, items[0][1])
            ordered = (
                (candidate_address, best_address)
                if operation == "min"
                else (best_address, candidate_address)
            )
            widths = (
                self._word(shape.words),
                self._word(shape.floats),
                self._word(shape.handles),
            )
            beats = self._rt("ppy_coll_before", (*ordered, *widths))
            first = core.bitwise(
                self.b, "xor", core.load(self.b, seen), core.const(self.b, True, BOOL)
            )
            take = core.bitwise(self.b, "or", first, core.cmp(self.b, "ne", beats, self._word(0)))
            chosen = self._block(f"{operation}.take")  # type: ignore[attr-defined]
            after = self._block(f"{operation}.next")  # type: ignore[attr-defined]
            core.cond_br(self.b, take, Successor(chosen), Successor(after))
            self.b.at_end(chosen)  # type: ignore[attr-defined]
            self._write(best_address, shape, items[0][1])
            core.store(self.b, core.const(self.b, True, BOOL), seen)
            core.br(self.b, Successor(after))
            self.b.at_end(after)  # type: ignore[attr-defined]

        self._walk(source, keep)
        self._require(
            core.load(self.b, seen),
            f"{operation}() arg is an empty sequence",
            f"ValueError: {operation}() iterable argument is empty",
        )
        return self._read(best_address, shape)


def _word_kinds(shape: Shape) -> tuple[str, ...]:
    """A shape's words, each `int` or `float`: a record's fields or a tuple's items."""
    parts = shape.parts if shape.kind in {"tuple", "record"} else (shape.kind,)
    return tuple("float" if part == "float" else "int" for part in parts)


def _absolute(lowering: CollectionApiLowering, value: Value) -> Value:
    zero = core.const(lowering.b, 0.0, F64)
    negative = core.cmp(lowering.b, "lt", value, zero)
    return core.select(lowering.b, negative, core.sub(lowering.b, zero, value), value)
