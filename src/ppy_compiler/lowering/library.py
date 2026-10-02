"""`collections`' containers natively: `defaultdict`, `Counter`, `OrderedDict`,
and how a `deque` is shown.

Each mapping is a dict over the collections runtime (`Kind("Dict", ...)`)
with a flavor saying which (`Kind.flavor`), so a subscript, a loop, `len`,
`in`, and the dict methods are a dict's. What is here is where they are not:

- a `defaultdict`'s missing key gets its factory's value, put in the map,
  before the subscript reads it. A class (`int`, `list`) makes the value its
  value type says; a function (`lambda: -1`) is a closure the map holds in
  its header (word 25), called for each missing key;
- a `Counter`'s missing key reads 0 and is not added. `most_common` ranks the
  counts as `sorted(..., reverse=True)` does, ties in insertion order, which
  is also the order `repr` shows them in. `update` and `subtract` count an
  iterable or add a mapping's counts; `+`, `-`, `|`, `&` are `Counter`'s;
- an `OrderedDict` is a dict in insertion order already: `move_to_end` and
  `popitem(last=...)` move and take entries at either end.

`repr` is CPython's: `defaultdict(<class 'int'>, {...})`, `Counter({...})`
in `most_common` order, `OrderedDict({...})` (before 3.12, `OrderedDict([(k,
v), ...])`), `deque([...])`. A `defaultdict` whose factory is a function is
shown by Python, which names the function by its address.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass

from ..analysis import types as T
from ..analysis.lexical import LexicalBindings
from ..backend.llvm.lowering import Unsupported
from ..ir import BOOL, F64, I64, Successor, Value
from ..ir.dialects import core
from ..ir.raising import said
from .collection_api import _Cursor, _Source
from .collections import HANDLE, Kind, Shape, _pointer

__all__ = ["LibraryLowering"]

_DEFAULTDICT = "collections.defaultdict"
_COUNTER = "collections.Counter"
_ORDERED = "collections.OrderedDict"
_DEQUE = "collections.deque"
_MAPPINGS = frozenset({_DEFAULTDICT, _COUNTER, _ORDERED})

#: What each class a `defaultdict` may take as its factory makes, as the
#: shape it is natively: the kind of the shape, and a collection's name and
#: flavor.
_FACTORIES: dict[str, tuple[str, str, str]] = {
    "builtins.int": ("int", "", ""),
    "builtins.float": ("float", "", ""),
    "builtins.bool": ("bool", "", ""),
    "builtins.str": ("str", "", ""),
    "builtins.list": ("collection", "List", ""),
    "builtins.set": ("collection", "Set", ""),
    "builtins.dict": ("collection", "Dict", ""),
    _COUNTER: ("collection", "Dict", _COUNTER),
    _DEQUE: ("collection", "Deque", _DEQUE),
}

#: How `repr` names a class factory: `<class 'int'>`.
_CLASS_NAMES = {
    ("int", "", ""): "int",
    ("float", "", ""): "float",
    ("bool", "", ""): "bool",
    ("str", "", ""): "str",
    ("collection", "List", ""): "list",
    ("collection", "Set", ""): "set",
    ("collection", "Dict", ""): "dict",
    ("collection", "Dict", _COUNTER): "collections.Counter",
    ("collection", "Deque", _DEQUE): "collections.deque",
}

#: `Counter`'s operators, as `ppy_counter_combine` numbers them.
_COUNTER_OPERATORS = {ast.Add: 0, ast.Sub: 1, ast.BitOr: 2, ast.BitAnd: 3}


def _signature_of(shape: Shape) -> tuple[str, str, str]:
    collection = shape.collection
    if shape.kind == "collection" and collection is not None:
        return ("collection", collection.name, collection.flavor)
    return (shape.kind, "", "")


@dataclass(slots=True)
class _Ranked(_Source):
    """A `Counter`'s entries in `most_common` order: a copy of the counter held
    in `slot` (CPython's `most_common` is a list made when it is called), the
    entry numbers in `order`, and how many to walk."""

    order: Value | None = None
    limit: Value | None = None


class LibraryLowering:
    """`collections`' mappings, and how they and a `deque` are shown; mixed
    into `_FunctionLowering` through `StdlibLowering`."""

    # -- what an expression is ----------------------------------------------------------

    def _library_call(self, node: ast.expr) -> str | None:
        """`defaultdict(...)`, `Counter(...)`, `OrderedDict(...)`: which."""
        if not isinstance(node, ast.Call):
            return None
        lexical = self.frontend.analysis.symbols.lexical  # type: ignore[attr-defined]
        if not isinstance(lexical, LexicalBindings):
            return None
        found = lexical.targets_at(node.func)
        if len(found) != 1:
            return None
        target = next(iter(found))
        return target if target in _MAPPINGS else None

    def _flavored(self, node: ast.expr) -> Kind | None:
        kind = self._kind_of(node)  # type: ignore[attr-defined]
        return kind if kind is not None and kind.flavor else None

    def _counter_call(self, node: ast.expr, *names: str) -> ast.Call | None:
        """`c.most_common(...)` or another of `names` called on a `Counter`."""
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in names
        ):
            kind = self._flavored(node.func.value)
            if kind is not None and kind.flavor == _COUNTER:
                return node
        return None

    def _kind_of(self, node: ast.expr) -> Kind | None:
        elements = self._counter_call(node, "elements")
        if elements is not None:
            counter = self._flavored(elements.func.value)  # type: ignore[attr-defined]
            assert counter is not None and counter.key is not None
            return Kind("List", counter.key)
        return super()._kind_of(node)  # type: ignore[misc,no-any-return]

    # -- making -------------------------------------------------------------------------

    def _made(self, kind: Kind, node: ast.expr) -> Value | None:
        target = self._library_call(node)
        if target is not None and kind.flavor == target:
            assert isinstance(node, ast.Call)
            return self._library_mapping(kind, node)
        if kind.flavor in _MAPPINGS and isinstance(node, ast.BinOp):
            return self._counter_combined(kind, node)
        return super()._made(kind, node)  # type: ignore[misc,no-any-return]

    def _handle(self, node: ast.expr) -> tuple[Value, bool]:
        made = self._counter_call(node, "most_common", "elements")
        if made is not None:
            # A new list, the caller's.
            return self._collection_method(made.func.value, made.func.attr, made), True  # type: ignore[attr-defined]
        if isinstance(node, ast.BinOp):
            kind = self._flavored(node)
            if kind is not None and kind.flavor == _COUNTER:
                return self._counter_combined(kind, node), True
        return super()._handle(node)  # type: ignore[misc,no-any-return]

    def _library_mapping(self, kind: Kind, node: ast.Call) -> Value:
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise Unsupported(f"`{ast.unparse(node.func)}` with keywords has no native lowering")
        if kind.flavor == _DEFAULTDICT:
            return self._defaultdict(kind, node)
        if len(node.args) > 1:
            raise Unsupported(f"`{ast.unparse(node.func)}` takes one argument natively")
        made = self._new(kind)  # type: ignore[attr-defined]
        if not node.args:
            return made
        source = node.args[0]
        other = self._builtin_of(source)  # type: ignore[attr-defined]
        if other is not None and other.name == "Dict":
            # From a mapping: its entries in its order (a `Counter`'s counts
            # added to none, which is `dict.update`).
            if other.key != kind.key or other.value != kind.value:
                raise Unsupported("a mapping is copied from one of its own key and value types")
            handle, owned = self._handle(source)  # type: ignore[attr-defined]
            self._rt("ppy_coll_update", (made, handle), None)  # type: ignore[attr-defined]
            self._done_with(handle, owned)  # type: ignore[attr-defined]
            return made
        if kind.flavor == _ORDERED:
            self._ordered_pairs(kind, made, source)
            return made
        self._count_into(kind, made, source, 1)
        return made

    def _ordered_pairs(self, kind: Kind, made: Value, source: ast.expr) -> None:
        """`OrderedDict([(k, v), ...])`: each pair put in turn."""
        if not isinstance(source, (ast.List, ast.Tuple)) or not all(
            isinstance(e, ast.Tuple) and len(e.elts) == 2 for e in source.elts
        ):
            raise Unsupported("an `OrderedDict` is made from a dict or a display of pairs")
        for element in source.elts:
            assert isinstance(element, ast.Tuple)
            self._put(kind, made, element.elts[0], element.elts[1])  # type: ignore[attr-defined]

    def _count_into(self, kind: Kind, made: Value, source: ast.expr, sign: int) -> None:
        """Each element of an iterable counted once (`Counter(xs)`,
        `c.update(xs)`, `c.subtract(xs)`), in the order it walks."""
        key_shape = kind.key
        assert key_shape is not None
        walked = self._source(source)  # type: ignore[attr-defined]
        if walked is None:
            raise Unsupported(f"`{ast.unparse(source)}` is not counted natively")
        if walked.mode == "items":
            raise Unsupported("a map's items are not counted natively")
        buffer = self._alloca(key_shape.ir_type(), "count.key")  # type: ignore[attr-defined]
        address = core.cast(self.b, buffer, _pointer(buffer, HANDLE.pointee))  # type: ignore[attr-defined]

        def count(items: list[tuple[Shape, Value]]) -> None:
            shape, value = items[0]
            if shape != key_shape:
                raise Unsupported("a `Counter` counts elements of its key type")
            self._write(address, key_shape, value)  # type: ignore[attr-defined]
            if key_shape.floats and key_shape.kind != "record":
                self._nan_key(address, key_shape)  # type: ignore[attr-defined]
            self._rt("ppy_counter_add", (made, address, self._word(sign)), None)  # type: ignore[attr-defined]

        self._walk(walked, count)  # type: ignore[attr-defined]

    def _defaultdict(self, kind: Kind, node: ast.Call) -> Value:
        """`defaultdict(int)`, `defaultdict(list)`, `defaultdict(lambda: -1)`,
        and a mapping to start from after the factory."""
        if not 1 <= len(node.args) <= 2:
            raise Unsupported("`defaultdict` takes a factory and a mapping natively")
        value = kind.value
        assert value is not None
        factory = node.args[0]
        made_by = T.strip_literal(self._type_of(factory))  # type: ignore[attr-defined]
        closure: tuple[Value, bool] | None = None
        if isinstance(made_by, T.ClassObject):
            name = made_by.name if "." in made_by.name else f"builtins.{made_by.name}"
            wanted = _FACTORIES.get(name)
            if wanted is None or wanted != _signature_of(value):
                raise Unsupported(
                    f"`defaultdict({ast.unparse(factory)})` makes what its value type is not"
                )
        else:
            closure = self._handle(factory)  # type: ignore[attr-defined]
        made = self._new(kind)  # type: ignore[attr-defined]
        if closure is not None:
            handle, owned = closure
            if not owned:
                self._retain(handle)  # type: ignore[attr-defined]
            self._rt("ppy_map_set_factory", (made, handle), None)  # type: ignore[attr-defined]
        if len(node.args) == 2:
            other = self._builtin_of(node.args[1])  # type: ignore[attr-defined]
            if other is None or other.name != "Dict" or other.key != kind.key:
                raise Unsupported("`defaultdict` starts from a dict natively")
            if other.value != kind.value:
                raise Unsupported("`defaultdict` starts from a dict of its own value type")
            handle, owned = self._handle(node.args[1])  # type: ignore[attr-defined]
            self._rt("ppy_coll_update", (made, handle), None)  # type: ignore[attr-defined]
            self._done_with(handle, owned)  # type: ignore[attr-defined]
        return made

    # -- a missing key -------------------------------------------------------------------

    def _element_address(self, kind: Kind, handle: Value, index: ast.expr, *, write: bool) -> Value:
        if write or kind.flavor not in {_DEFAULTDICT, _COUNTER}:
            return super()._element_address(kind, handle, index, write=write)  # type: ignore[misc,no-any-return]
        b = self.b  # type: ignore[attr-defined]
        rt = self._rt  # type: ignore[attr-defined]
        shape = kind.value
        assert shape is not None
        alias = Kind("HashMap", kind.value, kind.key, kind.flavor)
        key = self._key(alias, index)  # type: ignore[attr-defined]
        entry = rt("ppy_map_find", (handle, key))
        slot = self._alloca(HANDLE, "item")  # type: ignore[attr-defined]
        found = self._block("item.found")  # type: ignore[attr-defined]
        missing = self._block("item.missing")  # type: ignore[attr-defined]
        done = self._block("item.done")  # type: ignore[attr-defined]
        core.cond_br(b, self._found(entry), Successor(found), Successor(missing))  # type: ignore[attr-defined]
        b.at_end(found)
        core.store(b, rt("ppy_map_value_at", (handle, entry), HANDLE), slot)
        core.br(b, Successor(done))
        b.at_end(missing)
        if kind.flavor == _COUNTER:
            # `Counter.__missing__`: 0, and the key is not added.
            core.store(b, rt("ppy_counter_zero", (), HANDLE), slot)
        else:
            # `defaultdict.__missing__`: the factory's value, then the entry.
            value = self._default_value(shape, handle)
            made = rt("ppy_map_put", (handle, key))
            address = rt("ppy_map_value_at", (handle, made), HANDLE)
            self._store_value(address, shape, value, owned=True)  # type: ignore[attr-defined]
            core.store(b, address, slot)
        core.br(b, Successor(done))
        b.at_end(done)
        self._keys_done()  # type: ignore[attr-defined]
        return core.load(b, slot)

    def _default_value(self, shape: Shape, handle: Value) -> Value:
        """What a `defaultdict`'s factory gives, owned: its closure called, or
        the value its class makes."""
        b = self.b  # type: ignore[attr-defined]
        factory = self._rt("ppy_map_factory", (handle,), HANDLE)  # type: ignore[attr-defined]
        held = self._alloca(shape.ir_type(), "default")  # type: ignore[attr-defined]
        called = self._block("default.call")  # type: ignore[attr-defined]
        classed = self._block("default.class")  # type: ignore[attr-defined]
        done = self._block("default.done")  # type: ignore[attr-defined]
        core.cond_br(b, self._present(factory), Successor(called), Successor(classed))  # type: ignore[attr-defined]
        b.at_end(called)
        results = (HANDLE,) if shape.reference else (shape.ir_type(),)
        record = core.cast(b, self._field_address(factory, 0), _pointer(factory, I64))  # type: ignore[attr-defined]
        code = core.load(b, record)
        values = self._call_indirect(code, (factory,), results)  # type: ignore[attr-defined]
        core.store(b, values[0], held)
        core.br(b, Successor(done))
        b.at_end(classed)
        core.store(b, self._class_default(shape), held)
        core.br(b, Successor(done))
        b.at_end(done)
        return core.load(b, held)

    def _class_default(self, shape: Shape) -> Value:
        """`int()`, `list()`, `Counter()`: the value a class factory makes, owned."""
        b = self.b  # type: ignore[attr-defined]
        if shape.kind in {"int", "bool"}:
            return core.const(b, 0, I64) if shape.kind == "int" else core.const(b, False, BOOL)
        if shape.kind == "float":
            return core.const(b, 0.0, F64)
        if shape.kind == "str":
            empty = self._string_literal("")  # type: ignore[attr-defined]
            self._retain(empty)  # type: ignore[attr-defined]
            return empty
        if shape.kind == "collection" and shape.collection is not None:
            return self._new(shape.collection)  # type: ignore[attr-defined,no-any-return]
        if shape.reference:
            # A value no class makes: the map was made with a function.
            return self._rt("ppy_coll_none", (), HANDLE)  # type: ignore[attr-defined,no-any-return]
        if shape.kind == "tuple":
            parts = [
                core.const(b, 0.0, F64) if part == "float" else core.const(b, 0, I64)
                for part in shape.parts
            ]
            return core.tuple_make(b, *parts)
        raise Unsupported(f"a `defaultdict` of `{shape.kind}` values has no native lowering")

    # -- methods --------------------------------------------------------------------------

    def _collection_method(self, receiver: ast.expr, attr: str, node: ast.Call) -> Value:
        kind = self._flavored(receiver)
        if kind is not None and kind.flavor == _COUNTER:
            done = self._counter_method(kind, receiver, attr, node)
            if done is not None:
                return done
        if kind is not None and kind.flavor == _ORDERED and attr in {"move_to_end", "popitem"}:
            return self._ordered_method(kind, receiver, attr, node)
        if kind is not None and kind.flavor in _MAPPINGS and attr == "fromkeys":
            raise Unsupported(f"`{kind.flavor}.fromkeys` has no native lowering")
        return super()._collection_method(receiver, attr, node)  # type: ignore[misc,no-any-return]

    def _counter_method(
        self, kind: Kind, receiver: ast.expr, attr: str, node: ast.Call
    ) -> Value | None:
        rt = self._rt  # type: ignore[attr-defined]
        if attr in {"update", "subtract"}:
            if node.keywords or len(node.args) > 1:
                raise Unsupported(f"`Counter.{attr}` takes one iterable or mapping natively")
            _, handle, owned = self._receiver(receiver)  # type: ignore[attr-defined]
            sign = 1 if attr == "update" else -1
            if node.args:
                source = node.args[0]
                other = self._builtin_of(source)  # type: ignore[attr-defined]
                if other is not None and other.name == "Dict":
                    if other.key != kind.key or other.value != kind.value:
                        raise Unsupported("a `Counter` adds the counts of its own key type")
                    taken, taken_owned = self._handle(source)  # type: ignore[attr-defined]
                    rt("ppy_counter_merge", (handle, taken, self._word(sign)), None)  # type: ignore[attr-defined]
                    self._done_with(taken, taken_owned)  # type: ignore[attr-defined]
                    self._counted_in_word(f"Counter.{attr}")
                else:
                    self._count_into(kind, handle, source, sign)
            self._done_with(handle, owned)  # type: ignore[attr-defined]
            return self._word(0)  # type: ignore[attr-defined,no-any-return]
        if attr == "total":
            _, handle, owned = self._receiver(receiver)  # type: ignore[attr-defined]
            made = rt("ppy_counter_total", (handle,))
            self._counted_in_word("Counter.total")
            self._done_with(handle, owned)  # type: ignore[attr-defined]
            return made  # type: ignore[no-any-return]
        if attr == "elements":
            if node.args or node.keywords:
                raise Unsupported("`Counter.elements` takes no arguments")
            listed = self._kind_of(node)
            assert listed is not None
            _, handle, owned = self._receiver(receiver)  # type: ignore[attr-defined]
            made = self._new(listed)  # type: ignore[attr-defined]
            rt("ppy_counter_elements", (made, handle), None)
            self._done_with(handle, owned)  # type: ignore[attr-defined]
            return made  # type: ignore[no-any-return]
        if attr == "most_common":
            listed = self._kind_of(node)
            if listed is None or listed.value is None:
                raise Unsupported(
                    "`most_common` of string keys is walked by a loop natively, not kept"
                )
            source = self._ranked(node)
            made = self._new(listed)  # type: ignore[attr-defined]
            self._walk(  # type: ignore[attr-defined]
                source,
                lambda items: self._add_value(  # type: ignore[attr-defined]
                    listed, made, core.tuple_make(self.b, *self._flat(items))  # type: ignore[attr-defined]
                ),
            )
            return made  # type: ignore[no-any-return]
        return None

    def _flat(self, items: list[tuple[Shape, Value]]) -> list[Value]:
        """A key and its count as the items of one tuple of numbers."""
        flat: list[Value] = []
        for shape, value in items:
            if shape.kind == "tuple":
                flat.extend(core.tuple_extract(self.b, value, i) for i in range(shape.words))  # type: ignore[attr-defined]
            else:
                flat.append(value)
        return flat

    def _counted_in_word(self, what: str) -> None:
        """A count past a word: what Python counts with its own ints."""
        fault = self._rt("ppy_math_fault", ())  # type: ignore[attr-defined]
        self._guard(  # type: ignore[attr-defined]
            core.cmp(self.b, "eq", fault, self._word(0)),  # type: ignore[attr-defined]
            "range",
            f"`{what}` past a word",
            raises="OverflowError: the result does not fit in a 64-bit integer",
        )

    def _counter_combined(self, kind: Kind, node: ast.BinOp) -> Value:
        """`a + b`, `a - b`, `a | b`, `a & b` of two `Counter`s: a new one."""
        operation = _COUNTER_OPERATORS.get(type(node.op))
        if operation is None or self._flavored(node.left) != kind:
            raise Unsupported(f"`{ast.unparse(node)}` has no native lowering")
        if self._flavored(node.right) != kind:
            raise Unsupported("a `Counter` combines with another of its own key type")
        left, left_owned = self._handle(node.left)
        right, right_owned = self._handle(node.right)
        made = self._new(kind)  # type: ignore[attr-defined]
        self._rt("ppy_counter_combine", (made, left, right, self._word(operation)), None)  # type: ignore[attr-defined]
        self._counted_in_word("Counter" + {0: " +", 1: " -", 2: " |", 3: " &"}[operation])
        self._done_with(left, left_owned)  # type: ignore[attr-defined]
        self._done_with(right, right_owned)  # type: ignore[attr-defined]
        return made  # type: ignore[no-any-return]

    def _ordered_method(self, kind: Kind, receiver: ast.expr, attr: str, node: ast.Call) -> Value:
        b = self.b  # type: ignore[attr-defined]
        rt = self._rt  # type: ignore[attr-defined]
        keywords = {k.arg: k.value for k in node.keywords}
        if None in keywords or set(keywords) - {"last"}:
            raise Unsupported(f"`OrderedDict.{attr}` takes `last=` natively")
        given = list(node.args)
        if attr == "move_to_end":
            if not 1 <= len(given) <= 2 or (len(given) == 2 and "last" in keywords):
                raise Unsupported("`move_to_end` takes a key and `last`")
            _, handle, owned = self._receiver(receiver)  # type: ignore[attr-defined]
            alias = Kind("HashMap", kind.value, kind.key, kind.flavor)
            key = self._key(alias, given[0])  # type: ignore[attr-defined]
            last = given[1] if len(given) == 2 else keywords.get("last")
            flag = (
                core.cast(b, self._test(last), I64) if last is not None else self._word(1)  # type: ignore[attr-defined]
            )
            moved = rt("ppy_map_move", (handle, key, flag))
            spelled, words = self._key_report(alias, key)  # type: ignore[attr-defined]
            self._require(  # type: ignore[attr-defined]
                core.cmp(b, "ne", moved, self._word(0)),  # type: ignore[attr-defined]
                "key not found",
                f"KeyError: {spelled}" if spelled else "KeyError",
                words,
            )
            self._keys_done()  # type: ignore[attr-defined]
            self._done_with(handle, owned)  # type: ignore[attr-defined]
            return self._word(0)  # type: ignore[attr-defined,no-any-return]
        # popitem
        if len(given) > 1 or (given and "last" in keywords):
            raise Unsupported("`popitem` takes `last`")
        key_shape, value_shape = kind.key, kind.value
        assert key_shape is not None and value_shape is not None
        if key_shape.reference or value_shape.reference:
            raise Unsupported("`popitem` of strings or collections hands Python its pair")
        if key_shape.kind == "tuple" or value_shape.kind == "tuple":
            raise Unsupported("`popitem` hands out a pair of numbers natively")
        last = given[0] if given else keywords.get("last")
        flag = core.cast(b, self._test(last), I64) if last is not None else self._word(1)  # type: ignore[attr-defined]
        _, handle, owned = self._receiver(receiver)  # type: ignore[attr-defined]
        entry = rt("ppy_map_end_entry", (handle, flag))
        self._require(  # type: ignore[attr-defined]
            self._found(entry),  # type: ignore[attr-defined]
            "popitem of an empty OrderedDict",
            _text("popitem"),
        )
        key = self._read(rt("ppy_map_key_at", (handle, entry), HANDLE), key_shape)  # type: ignore[attr-defined]
        value = self._read(rt("ppy_map_value_at", (handle, entry), HANDLE), value_shape)  # type: ignore[attr-defined]
        buffer = self._alloca(key_shape.ir_type(), "pop.key")  # type: ignore[attr-defined]
        address = core.cast(b, buffer, _pointer(buffer, HANDLE.pointee))
        self._write(address, key_shape, key)  # type: ignore[attr-defined]
        rt("ppy_map_remove", (handle, address))
        self._done_with(handle, owned)  # type: ignore[attr-defined]
        return core.tuple_make(b, key, value)

    # -- most_common, walked ---------------------------------------------------------------

    def _ranked(self, node: ast.Call) -> _Ranked:
        """`c.most_common(n)` as a walk over a copy of the counter."""
        if node.keywords or len(node.args) > 1:
            raise Unsupported("`most_common` takes how many natively")
        kind = self._flavored(node.func.value)  # type: ignore[attr-defined]
        assert kind is not None
        limit = None
        if node.args and not (
            isinstance(node.args[0], ast.Constant) and node.args[0].value is None
        ):
            limit = self._coerce(self._expr(node.args[0]), "int")  # type: ignore[attr-defined]
        handle, owned = self._handle(node.func.value)  # type: ignore[attr-defined,attr-defined]
        copy = self._rt("ppy_coll_copy", (handle,), HANDLE)  # type: ignore[attr-defined]
        self._done_with(handle, owned)  # type: ignore[attr-defined]
        slot = self._hold(kind, copy, owned=True)  # type: ignore[attr-defined]
        order = self._rt("ppy_counter_ranked", (copy,), HANDLE)  # type: ignore[attr-defined]
        held = self._hold(Kind("List", Shape("int")), order, owned=True)  # type: ignore[attr-defined]
        length = self._rt("ppy_coll_len", (order,))  # type: ignore[attr-defined]
        if limit is not None:
            b = self.b  # type: ignore[attr-defined]
            positive = core.select(b, core.cmp(b, "gt", limit, self._word(0)), limit, self._word(0))  # type: ignore[attr-defined]
            length = core.select(b, core.cmp(b, "lt", positive, length), positive, length)
        return _Ranked(kind, slot, "items", order=held, limit=length)

    def _is_walk(self, node: ast.expr) -> bool:
        if self._counter_call(node, "most_common") is not None:
            return True
        return super()._is_walk(node)  # type: ignore[misc,no-any-return]

    def _source(self, node: ast.expr):  # type: ignore[no-untyped-def]
        ranked = self._counter_call(node, "most_common")
        if ranked is not None:
            return self._ranked(ranked)
        return super()._source(node)  # type: ignore[misc]

    def _start(self, source):  # type: ignore[no-untyped-def]
        if not isinstance(source, _Ranked):
            return super()._start(source)  # type: ignore[misc]
        at = self._alloca(I64, "ranked.at")  # type: ignore[attr-defined]
        core.store(self.b, self._word(0), at)  # type: ignore[attr-defined]
        return _Cursor(source, at, None, None)

    def _more(self, cursor):  # type: ignore[no-untyped-def]
        source = cursor.source
        if not isinstance(source, _Ranked):
            return super()._more(cursor)  # type: ignore[misc]
        assert source.limit is not None
        return core.cmp(self.b, "lt", core.load(self.b, cursor.at), source.limit)  # type: ignore[attr-defined]

    def _current(self, cursor):  # type: ignore[no-untyped-def]
        source = cursor.source
        if not isinstance(source, _Ranked):
            return super()._current(cursor)  # type: ignore[misc]
        b = self.b  # type: ignore[attr-defined]
        rt = self._rt  # type: ignore[attr-defined]
        assert source.order is not None
        order = core.load(b, source.order)
        counter = core.load(b, source.slot)
        entry = self._read(rt("ppy_seq_at", (order, core.load(b, cursor.at)), HANDLE), Shape("int"))  # type: ignore[attr-defined]
        key_shape = source.kind.key
        assert key_shape is not None
        key = self._read(rt("ppy_map_key_at", (counter, entry), HANDLE), key_shape)  # type: ignore[attr-defined]
        count = self._read(rt("ppy_map_value_at", (counter, entry), HANDLE), Shape("int"))  # type: ignore[attr-defined]
        return [(key_shape, key), (Shape("int"), count)]

    def _advance(self, cursor):  # type: ignore[no-untyped-def]
        if not isinstance(cursor.source, _Ranked):
            super()._advance(cursor)  # type: ignore[misc]
            return
        at = core.load(self.b, cursor.at)  # type: ignore[attr-defined]
        core.store(self.b, core.add(self.b, at, self._word(1), overflow="wrap"), cursor.at)  # type: ignore[attr-defined]

    def _walk(self, source, visit) -> None:  # type: ignore[no-untyped-def]
        super()._walk(source, visit)  # type: ignore[misc]
        if isinstance(source, _Ranked) and source.order is not None:
            self._let_go(source.order)  # type: ignore[attr-defined]

    # -- equality -------------------------------------------------------------------------

    def _collection_equality(self, node: ast.Compare) -> Value | None:
        for side in (node.left, *node.comparators):
            kind = self._flavored(side)
            if kind is not None and kind.flavor in {_COUNTER, _ORDERED}:
                # A `Counter` ignores zero counts, two `OrderedDict`s their order.
                raise Unsupported(f"`==` of a `{kind.flavor}` has no native lowering")
        return super()._collection_equality(node)  # type: ignore[misc,no-any-return]

    # -- repr -----------------------------------------------------------------------------

    def _printed_list(self, node: ast.expr) -> tuple[Value, bool] | None:
        kind = self._flavored(node)
        common = self._counter_call(node, "most_common")
        if kind is None and common is None:
            return super()._printed_list(node)  # type: ignore[misc,no-any-return]
        builder = self._rt("ppy_str_builder", (self._word(0),), HANDLE)  # type: ignore[attr-defined]
        self._add_formatted(builder, node, -1, "")
        return self._rt("ppy_str_finish", (builder,), HANDLE), True  # type: ignore[attr-defined]

    def _shown_text(self, node: ast.expr) -> Value | None:
        if self._flavored(node) is None and self._counter_call(node, "most_common") is None:
            return super()._shown_text(node)  # type: ignore[misc,no-any-return]
        builder = self._rt("ppy_str_builder", (self._word(0),), HANDLE)  # type: ignore[attr-defined]
        self._add_formatted(builder, node, -1, "")
        return self._rt("ppy_str_finish", (builder,), HANDLE)  # type: ignore[attr-defined,no-any-return]

    def _empty_library(self, node: ast.expr) -> str | None:
        """`deque()`, `Counter()`, `OrderedDict()` shown where they are made,
        holding nothing yet: their text."""
        if not isinstance(node, ast.Call) or node.args or node.keywords:
            return None
        if self._kind_of(node) is not None:  # type: ignore[attr-defined]
            return None
        lexical = self.frontend.analysis.symbols.lexical  # type: ignore[attr-defined]
        found = lexical.targets_at(node.func) if isinstance(lexical, LexicalBindings) else set()
        texts = {_DEQUE: "deque([])", _COUNTER: "Counter()", _ORDERED: "OrderedDict()"}
        return texts.get(next(iter(found))) if len(found) == 1 else None

    def _add_formatted(self, builder: Value, node: ast.expr, conversion: int, spec: str) -> None:
        empty = self._empty_library(node)
        if empty is not None and not spec:
            self._add_text(builder, empty)  # type: ignore[attr-defined]
            return
        common = self._counter_call(node, "most_common")
        if common is not None:
            if spec:
                raise Unsupported("a format spec over a list has no native lowering")
            self._add_ranked_repr(builder, self._ranked(common))
            return
        kind = self._flavored(node)
        if kind is None:
            super()._add_formatted(builder, node, conversion, spec)  # type: ignore[misc]
            return
        if spec:
            raise Unsupported("a format spec over a container has no native lowering")
        handle, owned = self._handle(node)
        self._add_library_repr(builder, kind, handle)
        self._done_with(handle, owned)  # type: ignore[attr-defined]

    def _add_item_repr(self, builder: Value, shape: Shape, value: Value) -> None:
        collection = shape.collection
        if shape.kind == "collection" and collection is not None and collection.flavor:
            self._add_library_repr(builder, collection, value)
            return
        super()._add_item_repr(builder, shape, value)  # type: ignore[misc]

    def _add_library_repr(self, builder: Value, kind: Kind, handle: Value) -> None:
        """`deque([1, 2])`, `defaultdict(<class 'int'>, {...})`, `Counter({...})`,
        `OrderedDict({...})`, as CPython writes them."""
        b = self.b  # type: ignore[attr-defined]
        add = self._add_text  # type: ignore[attr-defined]
        plain = Kind("List" if kind.name == "Deque" else kind.name, kind.value, kind.key)
        if kind.flavor == _DEQUE:
            add(builder, "deque(")
            self._add_elements_repr(builder, plain, handle)  # type: ignore[attr-defined]
            add(builder, ")")
            return
        if kind.flavor == _DEFAULTDICT:
            value = kind.value
            assert value is not None
            named = _CLASS_NAMES.get(_signature_of(value))
            factory = self._rt("ppy_map_factory", (handle,), HANDLE)  # type: ignore[attr-defined]
            # A function factory is shown by its address, which Python has.
            core.guard(
                b,
                core.bitwise(b, "xor", self._present(factory), core.const(b, True, BOOL)),  # type: ignore[attr-defined]
                "contract",
                "a defaultdict's function factory",
            )
            add(builder, f"defaultdict(<class '{named}'>, " if named else "defaultdict(None, ")
            self._add_elements_repr(builder, plain, handle)  # type: ignore[attr-defined]
            add(builder, ")")
            return
        empty = self._block("repr.empty")  # type: ignore[attr-defined]
        full = self._block("repr.full")  # type: ignore[attr-defined]
        shown = self._block("repr.shown")  # type: ignore[attr-defined]
        length = self._rt("ppy_coll_len", (handle,))  # type: ignore[attr-defined]
        core.cond_br(b, core.cmp(b, "eq", length, self._word(0)), Successor(empty), Successor(full))  # type: ignore[attr-defined]
        b.at_end(empty)
        name = "Counter" if kind.flavor == _COUNTER else "OrderedDict"
        add(builder, f"{name}()")
        core.br(b, Successor(shown))
        b.at_end(full)
        if kind.flavor == _COUNTER:
            add(builder, "Counter({")
            copy = self._rt("ppy_coll_copy", (handle,), HANDLE)  # type: ignore[attr-defined]
            slot = self._hold(kind, copy, owned=True)  # type: ignore[attr-defined]
            order = self._rt("ppy_counter_ranked", (copy,), HANDLE)  # type: ignore[attr-defined]
            held = self._hold(Kind("List", Shape("int")), order, owned=True)  # type: ignore[attr-defined]
            length = self._rt("ppy_coll_len", (order,))  # type: ignore[attr-defined]
            self._add_pairs(builder, _Ranked(kind, slot, "items", order=held, limit=length), ": ")
            add(builder, "})")
        elif sys.version_info >= (3, 12):
            add(builder, "OrderedDict(")
            self._add_elements_repr(builder, plain, handle)  # type: ignore[attr-defined]
            add(builder, ")")
        else:
            add(builder, "OrderedDict([")
            slot = self._hold(kind, handle, owned=False)  # type: ignore[attr-defined]
            self._add_pairs(builder, _Source(kind, slot, "items"), ", ", "(", ")")
            add(builder, "])")
        core.br(b, Successor(shown))
        b.at_end(shown)

    def _add_ranked_repr(self, builder: Value, source: _Ranked) -> None:
        """`most_common()`'s list: `[('a', 2), ('b', 1)]`."""
        self._add_text(builder, "[")  # type: ignore[attr-defined]
        self._add_pairs(builder, source, ", ", "(", ")")
        self._add_text(builder, "]")  # type: ignore[attr-defined]

    def _add_pairs(
        self, builder: Value, source: _Source, between: str, opening: str = "", closing: str = ""
    ) -> None:
        """Each key and value of a walk, `key: value` or `(key, value)`, comma-separated."""
        b = self.b  # type: ignore[attr-defined]
        first = self._alloca(BOOL, "repr.first")  # type: ignore[attr-defined]
        core.store(b, core.const(b, True, BOOL), first)

        def visit(items: list[tuple[Shape, Value]]) -> None:
            later = self._block("repr.comma")  # type: ignore[attr-defined]
            joined = self._block("repr.item")  # type: ignore[attr-defined]
            core.cond_br(b, core.load(b, first), Successor(joined), Successor(later))
            b.at_end(later)
            self._add_text(builder, ", ")  # type: ignore[attr-defined]
            core.br(b, Successor(joined))
            b.at_end(joined)
            core.store(b, core.const(b, False, BOOL), first)
            (key_shape, key), (value_shape, value) = items[0], items[1]
            self._add_text(builder, opening)  # type: ignore[attr-defined]
            self._add_item_repr(builder, key_shape, key)
            self._add_text(builder, between)  # type: ignore[attr-defined]
            self._add_item_repr(builder, value_shape, value)
            self._add_text(builder, closing)  # type: ignore[attr-defined]

        self._walk(source, visit)  # type: ignore[attr-defined]


def _raising() -> dict[str, object]:
    import collections  # pylint: disable=import-outside-toplevel

    return {"popitem": lambda: collections.OrderedDict().popitem()}


def _text(key: str) -> str:
    action = _raising()[key]
    return said(action)  # type: ignore[arg-type]

