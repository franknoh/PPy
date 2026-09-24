"""Native lowering of the `ppy` collections: shapes, typed access, and ownership.

A collection is a handle into the C runtime (`ppy_runtime/collections.c`),
which works on words and knows nothing of types. What a word means is
decided here, once per element type, the way a template is instantiated:
an element's `Shape` says how many words it takes, which are doubles and
which are handles, and how to read and write it as an IR value.

Ownership is counted. A handle local holds one reference, taken when it is
bound and dropped when it is rebound or the function returns; a collection
stored in another holds one; the runtime drops what a collection holds when
the collection goes. An expression that makes a collection or takes one out
of another hands its caller a reference (`owned`); one that reads a
collection in place lends it (`borrowed`), and the use that keeps it takes
its own.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..analysis import types as T
from ..backend.llvm.lowering import Unsupported
from ..ir import (
    BOOL,
    F64,
    I8,
    I64,
    Builder,
    IRType,
    PtrType,
    StructType,
    Successor,
    TupleType,
    Value,
)
from ..ir.dialects import core

__all__ = ["HANDLE", "CollectionLowering", "Kind", "Shape", "kind_of", "shape_of"]

#: A collection's runtime handle: the address of its header.
HANDLE = PtrType(I8)

_SCALARS = {"int": I64, "float": F64, "bool": BOOL}

#: Which runtime family each collection lives in.
FAMILY = {
    "Vec": "seq",
    "Deque": "seq",
    "Heap": "seq",
    "MaxHeap": "seq",
    "LinkedList": "list",
    "HashMap": "map",
    "HashSet": "map",
    "TreeMap": "tree",
    "TreeSet": "tree",
}

#: `floor`, `ceiling`, `lower`, `higher`, as `ppy_tree_bound` numbers them.
_BOUNDS = {"floor": 0, "ceiling": 1, "lower": 2, "higher": 3}


@dataclass(frozen=True, slots=True)
class Shape:
    """What one element or value is, word by word."""

    #: "int", "float", "bool", "tuple", "record", or "collection".
    kind: str
    #: A tuple's items, or a record's fields: scalar kinds.
    parts: tuple[str, ...] = ()
    #: A record's field names.
    names: tuple[str, ...] = ()
    #: A record's class.
    record: str = ""
    #: A record whose instances compare, field by field (`order=True`).
    ordered: bool = False
    collection: Kind | None = None

    @property
    def words(self) -> int:
        return len(self.parts) if self.kind in {"tuple", "record"} else 1

    @property
    def floats(self) -> int:
        kinds = self.parts if self.kind in {"tuple", "record"} else (self.kind,)
        return sum(1 << i for i, kind in enumerate(kinds) if kind == "float")

    @property
    def handles(self) -> int:
        return 1 if self.kind == "collection" else 0

    @property
    def comparable(self) -> bool:
        """Whether `<` orders it the way the runtime compares words."""
        return self.kind in {"int", "float", "bool", "tuple"} or (
            self.kind == "record" and self.ordered
        )

    @property
    def integral(self) -> bool:
        """Whether it can be a key: an `int`, or a tuple of them."""
        kinds = self.parts if self.kind == "tuple" else (self.kind,)
        return all(kind == "int" for kind in kinds)

    def ir_type(self) -> IRType:
        if self.kind in _SCALARS:
            return _SCALARS[self.kind]
        if self.kind == "tuple":
            return TupleType(tuple(_SCALARS[part] for part in self.parts))
        if self.kind == "record":
            fields = tuple(zip(self.names, (_SCALARS[p] for p in self.parts), strict=True))
            return StructType(self.record.replace(".", "_"), fields)
        return HANDLE


@dataclass(frozen=True, slots=True)
class Kind:
    """A collection type: which collection, and what it holds."""

    name: str
    value: Shape | None
    key: Shape | None = None

    @property
    def family(self) -> str:
        return FAMILY[self.name]

    @property
    def words(self) -> int:
        return self.value.words if self.value is not None else 0

    @property
    def spelled(self) -> str:
        return spelled(self)


def spelled(kind: Kind) -> str:
    """The type as `analysis.collections.spelled` writes it, for matching a parameter."""

    def shape(item: Shape) -> str:
        if item.kind == "tuple":
            return f"tuple[{', '.join(item.parts)}]"
        if item.kind == "record":
            return item.record
        if item.kind == "collection":
            assert item.collection is not None
            return spelled(item.collection)
        return item.kind

    parts = [shape(part) for part in (kind.key, kind.value) if part is not None]
    return f"ppy.{kind.name}[{', '.join(parts)}]"


#: A class's layout for the lowering: its fields, and whether it is ordered.
Records = dict[str, tuple[tuple[tuple[str, str], ...], bool]]


def shape_of(t: T.Type, records: Records) -> Shape | None:
    """The shape of a value of type `t`, or None where it has none."""
    base = T.strip_literal(t)
    if base in (T.INT, T.FLOAT, T.BOOL):
        return Shape(str(base))
    if isinstance(base, T.Tuple_) and not base.homogeneous and base.items:
        parts = [T.strip_literal(item) for item in base.items]
        if all(part in (T.INT, T.FLOAT, T.BOOL) for part in parts):
            return Shape("tuple", tuple(str(part) for part in parts))
        return None
    if not isinstance(base, T.Instance):
        return None
    if base.name.startswith("ppy."):
        kind = kind_of(base, records)
        return Shape("collection", collection=kind) if kind is not None else None
    found = records.get(base.name)
    if found is None:
        return None
    fields, ordered = found
    return Shape(
        "record",
        tuple(kind for _, kind in fields),
        tuple(name for name, _ in fields),
        base.name,
        ordered,
    )


def kind_of(t: T.Type, records: Records) -> Kind | None:
    """The collection type `t` is, or None."""
    base = T.strip_literal(t)
    if not isinstance(base, T.Instance) or not base.name.startswith("ppy."):
        return None
    name = base.name.removeprefix("ppy.")
    if name not in FAMILY or not base.args:
        return None
    shapes = [shape_of(argument, records) for argument in base.args]
    if any(shape is None for shape in shapes):
        return None
    if name in {"HashMap", "TreeMap"}:
        if len(shapes) != 2:
            return None
        return Kind(name, shapes[1], shapes[0])
    if name in {"HashSet", "TreeSet"}:
        return Kind(name, None, shapes[0])
    return Kind(name, shapes[0])


@dataclass(slots=True)
class Held:
    """A collection local: its type and the slot holding its handle."""

    kind: Kind
    slot: Value


class CollectionLowering:
    """The collection half of lowering one function; mixed into `_FunctionLowering`."""

    # Supplied by the class this is mixed into.
    b: Builder
    frontend: object
    collections: dict[str, Held]

    # -- types ------------------------------------------------------------

    def _records(self) -> Records:
        cached = getattr(self.frontend, "_collection_records", None)
        if cached is not None:
            return cached
        records: Records = {}
        classes = self.frontend.analysis.symbols.classes  # type: ignore[attr-defined]
        for qualname, fields in self.frontend.layouts.items():  # type: ignore[attr-defined]
            info = classes.get(qualname) or classes.get(qualname.rpartition(".")[2])
            records[qualname] = (tuple(fields), _ordered(info.node) if info is not None else False)
        self.frontend._collection_records = records  # type: ignore[attr-defined]
        return records

    def _type_of(self, node: ast.expr) -> T.Type:
        """What the checker said of `node`, with a generic instance's arguments in."""
        found: T.Type = self.frontend.analysis.type_of(node)  # type: ignore[attr-defined]
        bindings = getattr(self, "bindings", None)
        return T.substitute(found, bindings) if bindings else found

    def _kind_of(self, node: ast.expr) -> Kind | None:
        """The collection an expression denotes, from what the checker said of it."""
        if isinstance(node, ast.Name) and node.id in self.collections:
            return self.collections[node.id].kind
        return kind_of(self._type_of(node), self._records())

    def _is_collection(self, node: ast.expr) -> bool:
        return self._kind_of(node) is not None

    # -- value classes ---------------------------------------------------

    def _record_construction(self, node: ast.Call) -> Value | None:
        """`Point(1, 2.0)` for a value class: its fields as one struct value."""
        called = T.strip_literal(self._type_of(node.func))
        made = T.strip_literal(self._type_of(node))
        if not isinstance(called, T.ClassObject) or not isinstance(made, T.Instance):
            return None
        found = self._records().get(made.name)
        if found is None:
            return None
        fields, _ = found
        given: dict[str, ast.expr] = {}
        for (name, _kind), argument in zip(fields, node.args, strict=False):
            given[name] = argument
        if len(node.args) > len(fields):
            raise Unsupported(f"`{made.name}` takes {len(fields)} fields")
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg in given:
                raise Unsupported(f"`{made.name}` is built from each field once")
            given[keyword.arg] = keyword.value
        missing = [name for name, _ in fields if name not in given]
        if missing:
            raise Unsupported(f"`{made.name}` is built natively from every field: {missing[0]}")
        values = [
            self._coerce(self._expr(given[name]), kind)  # type: ignore[attr-defined]
            for name, kind in fields
        ]
        shape = shape_of(made, self._records())
        assert shape is not None
        ir_type = shape.ir_type()
        assert isinstance(ir_type, StructType)
        return core.struct_make(self.b, ir_type, *values)

    def _record_value(self, node: ast.expr) -> Value | None:
        """A value-class expression's struct, where it has one natively."""
        made = T.strip_literal(self._type_of(node))
        if not isinstance(made, T.Instance) or made.name not in self._records():
            return None
        value = self._expr(node)  # type: ignore[attr-defined]
        return value if isinstance(value.type, StructType) else None

    # -- the runtime ------------------------------------------------------

    def _rt(self, symbol: str, arguments: tuple[Value, ...], result: IRType | None = I64) -> Value:
        """One call into the collections runtime."""
        results = (result,) if result is not None else ()
        found = core.call_extern(self.b, symbol, arguments, results)
        return found.results[0] if results else core.const(self.b, 0, I64)

    def _word(self, value: int) -> Value:
        return core.const(self.b, value, I64)

    def _use_collections(self) -> None:
        """Under `ppy run` the runtime is a shared library loaded beside the code.

        Building it takes a C compiler; without one the function stays in
        Python rather than calling functions nothing defines.
        """
        from ppy_runtime.aio import compiler  # pylint: disable=import-outside-toplevel

        if not self.frontend.standalone and compiler() is None:  # type: ignore[attr-defined]
            raise Unsupported("native collections need a C compiler to build their runtime")
        module = self.frontend.module  # type: ignore[attr-defined]
        known = module.attributes.get("ppy.libraries", ())
        if "ppy_collections" not in known:
            module.attributes["ppy.libraries"] = (*known, "ppy_collections")

    def _retain(self, handle: Value) -> None:
        self._rt("ppy_coll_retain", (handle,), None)

    def _release(self, handle: Value) -> None:
        self._rt("ppy_coll_release", (handle,), None)

    def _require(self, condition: Value, message: str) -> None:
        """A check CPython would raise for: native code falls back, a binary stops."""
        core.guard(self.b, condition, "bounds", message)

    def _found(self, index: Value) -> Value:
        return core.cmp(self.b, "ge", index, self._word(0))

    # -- making ------------------------------------------------------------

    def _new(self, kind: Kind, count: Value | None = None) -> Value:
        """A new, empty collection (a `Vec` of `count` zeros): an owned handle."""
        self._use_collections()
        value = kind.value
        words = self._word(kind.words)
        floats = self._word(value.floats if value is not None else 0)
        handles = self._word(value.handles if value is not None else 0)
        keys = self._word(kind.key.words if kind.key is not None else 0)
        if kind.family == "seq":
            made = self._rt("ppy_seq_new", (count or self._word(0), words, floats, handles), HANDLE)
            if count is not None and value is not None and value.kind == "collection":
                self._fill_new(made, count, value)
            return made
        if kind.family == "list":
            return self._rt("ppy_list_new", (words, floats, handles), HANDLE)
        symbol = "ppy_map_new" if kind.family == "map" else "ppy_tree_new"
        return self._rt(symbol, (keys, words, floats, handles), HANDLE)

    def _fill_new(self, made: Value, count: Value, value: Shape) -> None:
        """`Vec[Vec[int]](n)`: each of the `n` slots its own new, empty collection."""
        assert value.collection is not None
        index = self._alloca(I64, "fill.i")  # type: ignore[attr-defined]
        core.store(self.b, self._word(0), index)
        header = self._block("fill.head")  # type: ignore[attr-defined]
        body = self._block("fill.body")  # type: ignore[attr-defined]
        done = self._block("fill.end")  # type: ignore[attr-defined]
        core.br(self.b, Successor(header))
        self.b.at_end(header)  # type: ignore[attr-defined]
        at = core.load(self.b, index)
        core.cond_br(self.b, core.cmp(self.b, "lt", at, count), Successor(body), Successor(done))
        self.b.at_end(body)  # type: ignore[attr-defined]
        inner = self._new(value.collection)
        address = self._rt("ppy_seq_at", (made, core.load(self.b, index)), HANDLE)
        core.store(self.b, inner, core.cast(self.b, address, PtrType(HANDLE)))
        step = core.add(self.b, core.load(self.b, index), self._word(1), overflow="wrap")
        core.store(self.b, step, index)
        core.br(self.b, Successor(header))
        self.b.at_end(done)  # type: ignore[attr-defined]

    def _constructed(self, node: ast.expr) -> Kind | None:
        """The collection `node` makes, if it is `ppy.Vec[...](...)` or one like it."""
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Subscript):
            return None
        kind = self._kind_of(node)
        spelled_name = ast.unparse(node.func.value).rpartition(".")[2]
        if kind is None or kind.name != spelled_name:
            return None
        return kind

    def _make_collection(self, name: str, value: ast.expr) -> bool:
        """`q = ppy.Deque[int]()`, or any binding of a name to a collection."""
        kind = self._kind_of(value) if not isinstance(value, ast.Name) else None
        if isinstance(value, ast.Name) and value.id in self.collections:
            kind = self.collections[value.id].kind
        if kind is None:
            return False
        handle, owned = self._handle(value)
        self._bind(name, kind, handle, owned)
        return True

    def _bind(self, name: str, kind: Kind, handle: Value, owned: bool) -> None:
        """`name` now holds `handle`: one reference taken, the old one let go."""
        held = self.collections.get(name)
        if held is not None and held.kind != kind:
            raise Unsupported(f"`{name}` keeps one collection type")
        if held is None:
            slot = self._alloca(HANDLE, name)  # type: ignore[attr-defined]
            entry = self._entry_builder()  # type: ignore[attr-defined]
            empty = core.call_extern(entry, "ppy_coll_none", (), (HANDLE,)).results[0]
            core.store(entry, empty, slot)
            held = Held(kind, slot)
            self.collections[name] = held
            self.slots.pop(name, None)  # type: ignore[attr-defined]
        if not owned:
            self._retain(handle)
        old = core.load(self.b, held.slot)
        core.store(self.b, handle, held.slot)
        self._release(old)

    def _bind_parameter(self, name: str, kind: Kind, argument: Value) -> None:
        """A collection parameter: the caller's, held for the call like any local."""
        slot = self._alloca(HANDLE, name)  # type: ignore[attr-defined]
        core.store(self.b, argument, slot)
        self._retain(argument)
        self.collections[name] = Held(kind, slot)

    def _release_collections(self) -> None:
        """Let go of every collection local before the function returns."""
        for held in self.collections.values():
            self._release(core.load(self.b, held.slot))

    # -- handles in expressions ---------------------------------------------

    def _handle(self, node: ast.expr) -> tuple[Value, bool]:
        """A collection-valued expression: its handle, and whether it is owned.

        An owned handle is a reference the expression made (a new collection,
        one taken out of another, one a callee returned); the use that ends
        it keeps it or lets it go. A borrowed handle is read in place.
        """
        if isinstance(node, ast.Name):
            held = self.collections.get(node.id)
            if held is None:
                raise Unsupported(f"`{node.id}` is not a native collection")
            return core.load(self.b, held.slot), False
        kind = self._constructed(node)
        if kind is not None:
            assert isinstance(node, ast.Call)
            count = None
            if node.args:
                count = self._coerce(self._expr(node.args[0]), "int")  # type: ignore[attr-defined]
                self._require(core.cmp(self.b, "ge", count, self._word(0)), "a negative size")
            return self._new(kind, count), True
        if isinstance(node, ast.Subscript):
            value = self._item(node.value, node.slice)
            return value, False
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and self._is_collection(node.func.value):
                found = self._method(node.func.value, node.func.attr, node)
                return found, _OWNED_RESULTS.get(node.func.attr, False)
            return self._expr(node), True  # type: ignore[attr-defined]
        raise Unsupported(f"`{ast.unparse(node)}` has no native collection form")

    def _receiver(self, node: ast.expr) -> tuple[Kind, Value, bool]:
        """The collection a method, an index, a loop, or `len` works on."""
        kind = self._kind_of(node)
        if kind is None:
            raise Unsupported(f"`{ast.unparse(node)}` is not a native collection")
        handle, owned = self._handle(node)
        return kind, handle, owned

    def _done_with(self, handle: Value, owned: bool) -> None:
        """A temporary collection's last use: let it go."""
        if owned:
            self._release(handle)

    def _discard(self, node: ast.Call) -> None:
        """A statement call whose collection result nobody keeps."""
        handle, owned = self._handle(node)
        self._done_with(handle, owned)

    # -- elements ------------------------------------------------------------

    def _read(self, address: Value, shape: Shape) -> Value:
        """The element at `address`, as the IR value of its shape."""
        if shape.kind == "collection":
            return core.load(self.b, core.cast(self.b, address, _pointer(address, HANDLE)))
        items = [self._read_word(address, i, kind) for i, kind in enumerate(_kinds(shape))]
        if shape.kind == "tuple":
            return core.tuple_make(self.b, *items)
        if shape.kind == "record":
            ir_type = shape.ir_type()
            assert isinstance(ir_type, StructType)
            return core.struct_make(self.b, ir_type, *items)
        return items[0]

    def _read_word(self, address: Value, index: int, kind: str) -> Value:
        stored = F64 if kind == "float" else I64
        pointer = core.cast(self.b, address, _pointer(address, stored))
        if index:
            pointer = core.ptr_offset(self.b, pointer, self._word(index))
        word = core.load(self.b, pointer)
        if kind == "bool":
            return core.cmp(self.b, "ne", word, self._word(0))
        return word

    def _write(self, address: Value, shape: Shape, value: Value) -> None:
        """`value` into the element at `address`, word by word."""
        if shape.kind == "collection":
            core.store(self.b, value, core.cast(self.b, address, _pointer(address, HANDLE)))
            return
        if shape.kind == "tuple":
            items = [core.tuple_extract(self.b, value, i) for i in range(shape.words)]
        elif shape.kind == "record":
            items = [core.struct_extract(self.b, value, name) for name in shape.names]
        else:
            items = [value]
        for i, (kind, item) in enumerate(zip(_kinds(shape), items, strict=True)):
            stored = F64 if kind == "float" else I64
            pointer = core.cast(self.b, address, _pointer(address, stored))
            if i:
                pointer = core.ptr_offset(self.b, pointer, self._word(i))
            word = self._coerce(item, "float" if kind == "float" else "int")  # type: ignore[attr-defined]
            core.store(self.b, word, pointer)

    def _value(self, node: ast.expr, shape: Shape) -> tuple[Value, bool]:
        """An expression as an element of `shape`: its IR value, and whether it is
        an owned handle (a collection element only)."""
        if shape.kind == "collection":
            return self._handle(node)
        if shape.kind == "tuple":
            items = self._tuple_expr(node)  # type: ignore[attr-defined]
            if items is None:
                found = self._expr(node)  # type: ignore[attr-defined]
                if not isinstance(found.type, TupleType):
                    raise Unsupported(f"`{ast.unparse(node)}` is not a tuple")
                items = [core.tuple_extract(self.b, found, i) for i in range(shape.words)]
            if len(items) != shape.words:
                raise Unsupported("the tuple does not match the element's width")
            coerced = [
                self._coerce(item, part)  # type: ignore[attr-defined]
                for item, part in zip(items, shape.parts, strict=True)
            ]
            return core.tuple_make(self.b, *coerced), False
        if shape.kind == "record":
            found = self._expr(node)  # type: ignore[attr-defined]
            if found.type != shape.ir_type():
                raise Unsupported(f"`{ast.unparse(node)}` is not a `{shape.record}`")
            return found, False
        return self._coerce(self._expr(node), shape.kind), False  # type: ignore[attr-defined]

    def _store_into(self, address: Value, shape: Shape, node: ast.expr, fresh: bool) -> None:
        """An element written where one may already be: a collection stored takes a
        reference, and the one it replaces (unless the slot is `fresh`) is let go."""
        value, owned = self._value(node, shape)
        if shape.kind != "collection":
            self._write(address, shape, value)
            return
        if not owned:
            self._retain(value)
        old = None if fresh else self._read(address, shape)
        self._write(address, shape, value)
        if old is not None:
            self._release(old)

    def _key(self, kind: Kind, node: ast.expr) -> Value:
        """A key's words in a stack buffer, and the buffer's address."""
        key = kind.key
        assert key is not None
        buffer = self._alloca(key.ir_type(), "key")  # type: ignore[attr-defined]
        address = core.cast(self.b, buffer, _pointer(buffer, I8))
        value, _ = self._value(node, key)
        self._write(address, key, value)
        return address

    # -- operations ------------------------------------------------------------

    def _length(self, node: ast.expr) -> Value:
        _, handle, owned = self._receiver(node)
        length = self._rt("ppy_coll_len", (handle,))
        self._done_with(handle, owned)
        return length

    def _truth_of(self, node: ast.expr, empty: bool = False) -> Value:
        """A collection as a condition: whether it holds anything (or, `empty`, nothing)."""
        length = self._length(node)
        return core.cmp(self.b, "eq" if empty else "gt", length, self._word(0))

    def _contains(self, container: ast.expr, key: ast.expr) -> Value:
        """`key in s`: whether a map or a set holds `key`."""
        kind, handle, owned = self._receiver(container)
        if kind.family not in {"map", "tree"}:
            raise Unsupported(f"`in` asks a map or a set, not a {kind.name}")
        found = self._rt(f"ppy_{kind.family}_find", (handle, self._key(kind, key)))
        self._done_with(handle, owned)
        return self._found(found)

    def _item(self, container: ast.expr, index: ast.expr, value: ast.expr | None = None) -> Value:
        """`v[i]` (0 to `len - 1`) or a map's `m[key]`, read; or with `value`, written."""
        kind, handle, owned = self._receiver(container)
        shape = kind.value
        if isinstance(index, ast.Slice) or shape is None:
            raise Unsupported(f"a {kind.name} has no index")
        if kind.name in {"Vec", "Deque"}:
            position = self._coerce(self._expr(index), "int")  # type: ignore[attr-defined]
            length = self._rt("ppy_coll_len", (handle,))
            inside = core.bitwise(
                self.b,
                "and",
                core.cmp(self.b, "ge", position, self._word(0)),
                core.cmp(self.b, "lt", position, length),
            )
            self._require(inside, "index out of range")
            address = self._rt("ppy_seq_at", (handle, position), HANDLE)
        elif kind.name in {"HashMap", "TreeMap"}:
            key = self._key(kind, index)
            if value is not None:
                entry = self._rt(f"ppy_{kind.family}_put", (handle, key))
            else:
                entry = self._rt(f"ppy_{kind.family}_find", (handle, key))
                self._require(self._found(entry), "key not found")
            address = self._rt(f"ppy_{kind.family}_value_at", (handle, entry), HANDLE)
        else:
            raise Unsupported(f"a {kind.name} has no index")
        if value is not None:
            self._store_into(address, shape, value, fresh=False)
            self._done_with(handle, owned)
            return self._word(0)
        found = self._read(address, shape)
        if owned:
            if shape.kind == "collection":
                raise Unsupported("an element read from a temporary collection outlives it")
            self._release(handle)
        return found

    def _method(self, receiver: ast.expr, attr: str, node: ast.Call) -> Value:
        """A method of a collection, as calls into the runtime."""
        kind, handle, owned = self._receiver(receiver)
        if node.keywords:
            raise Unsupported(f"`{kind.name}.{attr}` takes no keyword arguments")
        method = {
            "seq": self._sequence_method,
            "list": self._list_method,
            "map": self._keyed_method,
            "tree": self._keyed_method,
        }[kind.family]
        found = method(kind, handle, attr, node.args)
        if found is None:
            raise Unsupported(f"`{kind.name}.{attr}` has no native lowering")
        self._done_with(handle, owned)
        return found

    def _nonempty(self, kind: Kind, handle: Value, what: str) -> None:
        length = self._rt("ppy_coll_len", (handle,))
        self._require(
            core.cmp(self.b, "gt", length, self._word(0)), f"{what} of an empty {kind.name}"
        )

    def _sequence_method(
        self, kind: Kind, handle: Value, attr: str, arguments: list[ast.expr]
    ) -> Value | None:
        shape = kind.value
        assert shape is not None
        heap = {"Heap": 0, "MaxHeap": 1}.get(kind.name)
        pushes = {
            ("Vec", "push"): "ppy_seq_push_back",
            ("Deque", "push_back"): "ppy_seq_push_back",
            ("Deque", "push_front"): "ppy_seq_push_front",
        }
        if (kind.name, attr) in pushes:
            address = self._rt(pushes[(kind.name, attr)], (handle,), HANDLE)
            self._store_into(address, shape, arguments[0], fresh=True)
            return self._word(0)
        if heap is not None and attr == "push":
            if not shape.comparable:
                raise Unsupported(f"a {kind.name} orders its elements, and these have no order")
            self._store_into(
                self._rt("ppy_coll_scratch", (handle,), HANDLE), shape, arguments[0], fresh=True
            )
            return self._rt("ppy_heap_push", (handle, self._word(heap)), None)
        if attr == "clear":
            return self._rt("ppy_seq_clear", (handle,), None)
        if attr == "sort" and kind.name == "Vec":
            if not shape.comparable:
                raise Unsupported("these elements have no order to sort by")
            return self._rt("ppy_seq_sort", (handle,), None)
        if attr == "reverse" and kind.name == "Vec":
            return self._rt("ppy_seq_reverse", (handle,), None)
        removals = {
            ("Vec", "pop"): "ppy_seq_pop_back",
            ("Deque", "pop_back"): "ppy_seq_pop_back",
            ("Deque", "pop_front"): "ppy_seq_pop_front",
        }
        if (kind.name, attr) in removals:
            self._nonempty(kind, handle, attr)
            return self._read(self._rt(removals[(kind.name, attr)], (handle,), HANDLE), shape)
        if heap is not None and attr == "pop":
            self._nonempty(kind, handle, attr)
            return self._read(self._rt("ppy_heap_pop", (handle, self._word(heap)), HANDLE), shape)
        reads = {("Vec", "last"): True, ("Deque", "front"): False, ("Deque", "back"): True}
        if heap is not None:
            reads[(kind.name, "peek")] = False
        if (kind.name, attr) in reads:
            self._nonempty(kind, handle, attr)
            position = self._word(0)
            if reads[(kind.name, attr)]:
                length = self._rt("ppy_coll_len", (handle,))
                position = core.sub(self.b, length, self._word(1), overflow="wrap")
            return self._read(self._rt("ppy_seq_at", (handle, position), HANDLE), shape)
        return None

    def _list_method(
        self, kind: Kind, handle: Value, attr: str, arguments: list[ast.expr]
    ) -> Value | None:
        """`LinkedList`: nodes named by id, each checked to be in the list before use."""
        shape = kind.value
        assert shape is not None
        if attr in {"push_back", "push_front"}:
            end = self._rt("ppy_coll_field", (handle, self._word(4 if attr == "push_back" else 3)))
            links = (end, self._word(-1)) if attr == "push_back" else (self._word(-1), end)
            node = self._rt("ppy_list_node", (handle, *links))
            self._store_into(
                self._rt("ppy_list_at", (handle, node), HANDLE), shape, arguments[0], fresh=True
            )
            return node
        if attr in {"head", "tail"}:
            return self._rt("ppy_coll_field", (handle, self._word(3 if attr == "head" else 4)))
        if attr == "clear":
            return self._rt("ppy_list_clear", (handle,), None)
        if attr in {"pop_front", "pop_back", "front", "back"}:
            self._nonempty(kind, handle, attr)
            end = self._word(3 if attr in {"pop_front", "front"} else 4)
            node = self._rt("ppy_coll_field", (handle, end))
            if attr.startswith("pop"):
                self._rt("ppy_list_unlink", (handle, node), None)
            return self._read(self._rt("ppy_list_at", (handle, node), HANDLE), shape)
        if not arguments:
            return None
        node = self._coerce(self._expr(arguments[0]), "int")  # type: ignore[attr-defined]
        valid = self._rt("ppy_list_valid", (handle, node))
        self._require(core.cmp(self.b, "ne", valid, self._word(0)), "a node not in the list")
        if attr in {"insert_after", "insert_before"}:
            other = self._rt(
                "ppy_list_step", (handle, node, self._word(1 if attr == "insert_after" else 0))
            )
            links = (node, other) if attr == "insert_after" else (other, node)
            made = self._rt("ppy_list_node", (handle, *links))
            self._store_into(
                self._rt("ppy_list_at", (handle, made), HANDLE), shape, arguments[1], fresh=True
            )
            return made
        if attr in {"next", "prev"}:
            return self._rt("ppy_list_step", (handle, node, self._word(1 if attr == "next" else 0)))
        if attr == "remove":
            self._rt("ppy_list_unlink", (handle, node), None)
            return self._read(self._rt("ppy_list_at", (handle, node), HANDLE), shape)
        if attr == "value":
            return self._read(self._rt("ppy_list_at", (handle, node), HANDLE), shape)
        if attr == "set":
            self._store_into(
                self._rt("ppy_list_at", (handle, node), HANDLE), shape, arguments[1], fresh=False
            )
            return self._word(0)
        return None

    def _keyed_method(
        self, kind: Kind, handle: Value, attr: str, arguments: list[ast.expr]
    ) -> Value | None:
        """`HashMap`, `HashSet`, `TreeMap`, `TreeSet`."""
        family = kind.family
        key_shape = kind.key
        assert key_shape is not None
        if attr == "clear":
            return self._rt(f"ppy_{family}_clear", (handle,), None)
        if attr in {"min", "max"} and family == "tree":
            self._nonempty(kind, handle, attr)
            node = self._rt("ppy_tree_end", (handle, self._word(int(attr == "max"))))
            return self._read(self._rt("ppy_tree_key_at", (handle, node), HANDLE), key_shape)
        if not arguments:
            return None
        key = self._key(kind, arguments[0])
        if attr in _BOUNDS and family == "tree":
            node = self._rt("ppy_tree_bound", (handle, key, self._word(_BOUNDS[attr])))
            self._require(self._found(node), f"no key for `{attr}`")
            return self._read(self._rt("ppy_tree_key_at", (handle, node), HANDLE), key_shape)
        if attr == "add":
            self._rt(f"ppy_{family}_put", (handle, key))
            return self._word(0)
        shape = kind.value
        if attr == "get" and shape is not None:
            if shape.kind not in {"int", "float", "bool"}:
                raise Unsupported("`get` with a default takes a map of numbers natively")
            default = self._coerce(self._expr(arguments[1]), shape.kind)  # type: ignore[attr-defined]
            found = self._rt(f"ppy_{family}_find", (handle, key))
            present = self._found(found)
            # A miss reads the first record, which always exists, and discards it.
            safe = core.select(self.b, present, found, self._word(0))
            value = self._read(self._rt(f"ppy_{family}_value_at", (handle, safe), HANDLE), shape)
            return core.select(self.b, present, value, default)
        if attr in {"pop", "remove"}:
            removed = self._rt(f"ppy_{family}_remove", (handle, key))
            self._require(self._found(removed), "key not found")
            if attr == "pop" and shape is not None:
                return self._read(
                    self._rt(f"ppy_{family}_value_at", (handle, removed), HANDLE), shape
                )
            return self._word(0)
        if attr == "discard":
            return self._rt(f"ppy_{family}_remove", (handle, key))
        return None

    # -- loops -------------------------------------------------------------------

    def _for_collection(self, node: ast.For) -> None:
        """`for x in c`, walking as the reference walks.

        A `Vec` or `Deque` reads its length at every step, as a list does.
        A linked list follows `next` from each node after its body ran. A
        map or a set checks that nothing was added or removed since the
        loop began, as `dict` does, and CPython's `RuntimeError` is what the
        check falls back to.
        """
        kind, handle, owned = self._receiver(node.iter)
        if kind.name in {"Heap", "MaxHeap"}:
            raise Unsupported(f"a {kind.name} is read by `peek` and `pop`, not iterated")
        keep = self._alloca(HANDLE, "iterated")  # type: ignore[attr-defined]
        core.store(self.b, handle, keep)
        family = kind.family
        item_shape = kind.key if family in {"map", "tree"} else kind.value
        assert item_shape is not None
        cursor = self._alloca(I64, "each.at")  # type: ignore[attr-defined]
        if family == "list":
            start = self._rt("ppy_coll_field", (handle, self._word(3)))
        elif family == "tree":
            start = self._rt("ppy_tree_end", (handle, self._word(0)))
        else:
            start = self._word(0)
        version = None
        if family in {"map", "tree"}:
            version = self._rt("ppy_coll_field", (handle, self._word(6)))
        core.store(self.b, start, cursor)
        header = self._block("each.head")  # type: ignore[attr-defined]
        body = self._block("each.body")  # type: ignore[attr-defined]
        latch = self._block("each.latch")  # type: ignore[attr-defined]
        done = self._block("each.end")  # type: ignore[attr-defined]
        core.br(self.b, Successor(header))
        self.b.at_end(header)  # type: ignore[attr-defined]
        current = core.load(self.b, keep)
        if version is not None:
            now = self._rt("ppy_coll_field", (current, self._word(6)))
            self._require(
                core.cmp(self.b, "eq", now, version), f"{kind.name} changed during iteration"
            )
        at = core.load(self.b, cursor)
        if family == "seq":
            more = core.cmp(self.b, "lt", at, self._rt("ppy_coll_len", (current,)))
        elif family == "map":
            used = self._rt("ppy_coll_field", (current, self._word(3)))
            more = core.cmp(self.b, "lt", at, used)
        else:
            more = self._found(at)
        core.cond_br(self.b, more, Successor(body), Successor(done))
        self.b.at_end(body)  # type: ignore[attr-defined]
        if family == "seq":
            address = self._rt("ppy_seq_at", (current, at), HANDLE)
        elif family == "list":
            address = self._rt("ppy_list_at", (current, at), HANDLE)
        elif family == "tree":
            address = self._rt("ppy_tree_key_at", (current, at), HANDLE)
        else:
            alive = self._rt("ppy_map_alive", (current, at))
            present = self._block("each.present")  # type: ignore[attr-defined]
            core.cond_br(
                self.b,
                core.cmp(self.b, "ne", alive, self._word(0)),
                Successor(present),
                Successor(latch),
            )
            self.b.at_end(present)  # type: ignore[attr-defined]
            address = self._rt("ppy_map_key_at", (current, at), HANDLE)
        key_copy = None
        if family == "tree":
            key_copy = self._alloca(item_shape.ir_type(), "each.key")  # type: ignore[attr-defined]
            core.store(self.b, self._read(address, item_shape), key_copy)
        self._bind_element(node.target, item_shape, self._read(address, item_shape))
        self._loops.append((latch, done))  # type: ignore[attr-defined]
        self._body(node.body)  # type: ignore[attr-defined]
        self._loops.pop()  # type: ignore[attr-defined]
        if self._open():  # type: ignore[attr-defined]
            core.br(self.b, Successor(latch))
        self.b.at_end(latch)  # type: ignore[attr-defined]
        current = core.load(self.b, keep)
        at = core.load(self.b, cursor)
        if family == "list":
            following = self._rt("ppy_list_after", (current, at))
        elif family == "tree":
            assert key_copy is not None
            buffer = core.cast(self.b, key_copy, _pointer(key_copy, I8))
            following = self._rt("ppy_tree_bound", (current, buffer, self._word(3)))
        else:
            following = core.add(self.b, at, self._word(1), overflow="wrap")
        core.store(self.b, following, cursor)
        core.br(self.b, Successor(header))
        self.b.at_end(done)  # type: ignore[attr-defined]
        self._done_with(core.load(self.b, keep), owned)

    def _bind_element(self, target: ast.expr, shape: Shape, value: Value) -> None:
        """A loop target, or a name, bound to an element read in place."""
        if shape.kind == "collection":
            if not isinstance(target, ast.Name):
                raise Unsupported("a collection element is bound to a name")
            assert shape.collection is not None
            self._bind(target.id, shape.collection, value, owned=False)
            return
        if shape.kind == "tuple":
            items = [core.tuple_extract(self.b, value, i) for i in range(shape.words)]
            self._store_tuple(target, items)  # type: ignore[attr-defined]
            return
        self._store(target, value)  # type: ignore[attr-defined]


#: The methods whose collection result the caller owns: taken out, not read in place.
_OWNED_RESULTS = {"pop": True, "pop_front": True, "pop_back": True, "remove": True}


def _kinds(shape: Shape) -> tuple[str, ...]:
    return shape.parts if shape.kind in {"tuple", "record"} else (shape.kind,)


def _ordered(node: ast.ClassDef) -> bool:
    """`@dataclass(order=True)`: instances compare field by field, as tuples do."""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                if keyword.arg == "order" and isinstance(keyword.value, ast.Constant):
                    return keyword.value.value is True
    return False


def _pointer(address: Value, pointee: IRType) -> PtrType:
    """A pointer to `pointee` in the same memory as `address`: a stack buffer stays one."""
    assert isinstance(address.type, PtrType)
    return PtrType(pointee, address.type.address_space)
