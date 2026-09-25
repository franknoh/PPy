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
import hashlib
from dataclasses import dataclass

from ..analysis import types as T
from ..analysis.symbols import ClassInfo, dataclass_keyword
from ..backend.llvm.lowering import Unsupported
from ..driver.ir_pipeline import object_chain
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

__all__ = ["HANDLE", "STR", "CollectionLowering", "Kind", "Shape", "kind_of", "shape_of"]

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
    # `list[str]`, the list the string methods hand out.
    "List": "seq",
}

#: `floor`, `ceiling`, `lower`, `higher`, as `ppy_tree_bound` numbers them.
_BOUNDS = {"floor": 0, "ceiling": 1, "lower": 2, "higher": 3}

#: How the reference says a bound is missing: `KeyError('no key at most 5')`.
_BOUND_WORDS = {"floor": "at most", "ceiling": "at least", "lower": "below", "higher": "above"}


@dataclass(frozen=True, slots=True)
class Shape:
    """What one element or value is, word by word."""

    #: "int", "float", "bool", "str", "tuple", "record", "collection", or "object".
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
    #: An object's class arguments: `(int,)` for a `Stack[int]`.
    class_args: tuple[T.Type, ...] = ()

    @property
    def words(self) -> int:
        return len(self.parts) if self.kind in {"tuple", "record"} else 1

    @property
    def floats(self) -> int:
        kinds = self.parts if self.kind in {"tuple", "record"} else (self.kind,)
        return sum(1 << i for i, kind in enumerate(kinds) if kind == "float")

    @property
    def handles(self) -> int:
        return 1 if self.kind in {"collection", "object", "str"} else 0

    @property
    def reference(self) -> bool:
        """Held by handle: a collection, a string, or an instance of an object class."""
        return self.kind in {"collection", "object", "str"}

    @property
    def comparable(self) -> bool:
        """Whether `<` orders it the way the runtime compares words."""
        return self.kind in {"int", "float", "bool", "tuple", "str"} or (
            self.kind == "record" and self.ordered
        )

    @property
    def text(self) -> int:
        """The words that are strings, as a key's mask."""
        return 1 if self.kind == "str" else 0

    @property
    def leaves(self) -> int:
        """The handle words that are strings, which hold no handles themselves."""
        return self.text

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

    @property
    def spelled(self) -> str:
        """An object's type written out, as a parameter's element spells it."""
        if self.kind == "str":
            return "str"
        return str(T.Instance(self.record, self.class_args, (self.record, "object")))


#: A string: one handle word.
STR = Shape("str")


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
        if item.kind in {"record", "object"}:
            return item.spelled if item.kind == "object" else item.record
        if item.kind == "collection":
            assert item.collection is not None
            return spelled(item.collection)
        return item.kind

    parts = [shape(part) for part in (kind.key, kind.value) if part is not None]
    if kind.name == "List":
        return f"list[{', '.join(parts)}]"
    return f"ppy.{kind.name}[{', '.join(parts)}]"


#: A class's layout for the lowering: its fields, and whether it is ordered.
Records = dict[str, tuple[tuple[tuple[str, str], ...], bool]]


def shape_of(t: T.Type, records: Records) -> Shape | None:
    """The shape of a value of type `t`, or None where it has none.

    `Node | None` is a `Node`: an object's handle may be null, which is `None`.
    """
    base = T.strip_literal(t)
    if isinstance(base, T.Union_):
        members = [T.strip_literal(m) for m in base.members if m != T.NONE]
        if len(members) != 1 or len(members) == len(base.members):
            return None
        found = shape_of(members[0], records)
        return found if found is not None and found.kind == "object" else None
    if base in (T.INT, T.FLOAT, T.BOOL):
        return Shape(str(base))
    if base == T.STR:
        return STR
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
    if not fields:
        return Shape("object", record=base.name, class_args=base.args)
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
    if isinstance(base, T.Instance) and base.name == "list" and len(base.args) == 1:
        # A list of strings is the one Python list native code holds.
        return Kind("List", STR) if T.strip_literal(base.args[0]) == T.STR else None
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
    """A collection or object local: its type and the slot holding its handle."""

    kind: Kind | Shape
    slot: Value


@dataclass(frozen=True, slots=True)
class Layout:
    """An object class's fields, in words: where each starts and what it is."""

    fields: dict[str, tuple[int, Shape]]
    words: int
    floats: int
    handles: int


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
            held = self.collections[node.id].kind
            return held if isinstance(held, Kind) else None
        return kind_of(self._type_of(node), self._records())

    def _is_collection(self, node: ast.expr) -> bool:
        return self._kind_of(node) is not None

    def _object_of(self, node: ast.expr) -> Shape | None:
        """The object class an expression's value is an instance of, if any.

        A name is what the checker says it is at that point, which narrows it
        (`isinstance(shape, Square)`); a name the checker says nothing of is
        what it was bound as.
        """
        found = shape_of(self._type_of(node), self._records())
        if found is not None and found.kind == "object":
            if isinstance(node, ast.Name) and node.id in self.collections:
                held = self.collections[node.id].kind
                if not isinstance(held, Shape) or held.kind != "object":
                    return None
            return found
        if isinstance(node, ast.Name) and node.id in self.collections:
            held = self.collections[node.id].kind
            return held if isinstance(held, Shape) and held.kind == "object" else None
        return None

    def _accepts(self, parameter: object, kind: Kind | Shape) -> bool:
        """Whether a handle parameter takes `kind`: the same type, or an object of a
        class deriving from the parameter's."""
        element = parameter.element  # type: ignore[attr-defined]
        if kind.spelled == element:
            return True
        if not isinstance(kind, Shape) or kind.kind != "object":
            return False
        wanted = parameter.class_name  # type: ignore[attr-defined]
        return bool(wanted) and self._subclass_of(kind.record, wanted) and not kind.class_args

    def _reference_of(self, node: ast.expr) -> Kind | Shape | None:
        """A collection, a string, or an object: what native code holds `node` by handle as."""
        return self._kind_of(node) or self._object_of(node) or self._string_of(node)  # type: ignore[attr-defined]

    def _reference_of_type(self, t: T.Type) -> Kind | Shape | None:
        found = shape_of(t, self._records())
        if found is None or not found.reference:
            return None
        return found.collection if found.kind == "collection" else found

    # -- objects ------------------------------------------------------------

    def _class_info(self, shape: Shape) -> ClassInfo:
        return self._class_named(shape.record)

    def _class_named(self, qualname: str) -> ClassInfo:
        info = self._module_classes().get(qualname)
        if info is None:
            raise Unsupported(f"`{qualname}` is defined in another module")
        return info

    def _module_classes(self) -> dict[str, ClassInfo]:
        """This module's classes, by qualified name."""
        classes = self.frontend.analysis.symbols.classes  # type: ignore[attr-defined]
        return {info.qualname: info for info in classes.values()}

    def _chain(self, shape: Shape) -> list[ClassInfo]:
        """An object class and its bases, the root first: the order of its fields."""
        info = self._class_info(shape)
        chain = object_chain(info, self._module_classes())
        if chain is None:
            raise Unsupported(f"`{info.name}` derives from a class native code cannot hold")
        return chain

    def _subclasses(self, qualname: str) -> list[ClassInfo]:
        """The classes of this module an instance of `qualname` may be: it and those
        deriving from it."""
        return [
            info
            for info in self._module_classes().values()
            if qualname in info.mro and object_chain(info, self._module_classes()) is not None
        ]

    def _resolve(self, info: ClassInfo, attr: str) -> ClassInfo | None:
        """The class whose `attr` an instance of `info` runs: the first along its bases."""
        chain = object_chain(info, self._module_classes()) or [info]
        for entry in reversed(chain):
            if attr in entry.methods:
                return entry
        return None

    def _layout(self, shape: Shape) -> Layout:
        """Where each field of an object class sits, with its class arguments in."""
        cache: dict[tuple[str, tuple[T.Type, ...]], Layout] = self.frontend.__dict__.setdefault(
            "_object_layouts", {}
        )
        key = (shape.record, shape.class_args)
        found = cache.get(key)
        if found is not None:
            return found
        info = self._class_info(shape)
        bindings = dict(zip(info.type_params, shape.class_args, strict=False))
        fields: dict[str, tuple[int, Shape]] = {}
        offset = floats = handles = 0
        for owner in self._chain(shape):
            for name, declared in owner.fields.items():
                if name in owner.class_vars or name in fields:
                    continue
                field_shape = shape_of(T.substitute(declared, bindings), self._records())
                if field_shape is None:
                    raise Unsupported(f"field `{name}` of `{owner.name}` has no native form")
                fields[name] = (offset, field_shape)
                floats |= field_shape.floats << offset
                handles |= field_shape.handles << offset
                offset += field_shape.words
        if offset > 64:
            raise Unsupported(f"`{info.name}` has more than 64 words of fields")
        found = Layout(fields, max(offset, 1), floats, handles)
        cache[key] = found
        return found

    def _field(self, shape: Shape, name: str) -> tuple[int, Shape]:
        found = self._layout(shape).fields.get(name)
        if found is None:
            raise Unsupported(f"`{shape.record}` has no native field `{name}`")
        return found

    def _field_address(self, handle: Value, offset: int) -> Value:
        """Where a field starts: the object's one record, `offset` words in."""
        record = self._rt("ppy_seq_at", (handle, self._word(0)), HANDLE)
        if not offset:
            return record
        words = core.ptr_offset(self.b, core.cast(self.b, record, PtrType(I64)), self._word(offset))
        return core.cast(self.b, words, HANDLE)

    def _present(self, handle: Value) -> Value:
        """Whether a handle is an object: `None` is the null handle."""
        address = core.cast(self.b, handle, I64)
        return core.cmp(self.b, "ne", address, self._word(0))

    def _instance(self, shape: Shape, node: ast.Call) -> Value:
        """`Node(5)`, `Stack[int]()`: a new object, its fields set, an owned handle."""
        self._use_collections()
        layout = self._layout(shape)
        # String fields are leaves, as `_new` says of string elements.
        leaves = sum(field.leaves << offset for offset, field in layout.fields.values())
        made = self._rt(
            "ppy_seq_new",
            (
                self._word(1),
                self._word(layout.words),
                self._word(layout.floats),
                self._word(layout.handles | (leaves << 32)),
            ),
            HANDLE,
        )
        self._set_tag(made, shape.record)
        info = self._class_info(shape)
        if self._resolve(info, "__init__") is not None:
            self._method_call(shape, "__init__", made, node.args, node.keywords, discard=True)
            return made
        if not info.is_dataclass:
            if node.args or node.keywords:
                raise Unsupported(f"`{info.name}` takes no arguments without an `__init__`")
            return made
        given: dict[str, ast.expr] = {}
        names = list(layout.fields)
        if len(node.args) > len(names):
            raise Unsupported(f"`{info.name}` takes {len(names)} fields")
        given.update(zip(names, node.args, strict=False))
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg in given:
                raise Unsupported(f"`{info.name}` is built from each field once")
            given[keyword.arg] = keyword.value
        defaults: dict[str, ast.expr] = {}
        for owner in self._chain(shape):
            defaults.update(_field_defaults(owner.node))
        factories: dict[str, ast.expr] = {}
        for owner in self._chain(shape):
            factories.update(_field_factories(owner.node))
        for name, (offset, field_shape) in layout.fields.items():
            value = given.get(name, defaults.get(name))
            address = self._field_address(made, offset)
            if value is None and name in factories:
                self._write(address, field_shape, self._made_by(factories[name], field_shape))
                continue
            if value is None:
                raise Unsupported(f"`{info.name}` needs `{name}`, whose default is not a constant")
            self._store_into(address, field_shape, value, fresh=True)
        return made

    def _made_by(self, factory: ast.expr, shape: Shape) -> Value:
        """What `field(default_factory=...)` makes for one instance: a new
        collection, or a new object of a class that takes no arguments."""
        if shape.kind == "collection":
            assert shape.collection is not None
            spelled_name = ast.unparse(
                factory.value if isinstance(factory, ast.Subscript) else factory
            ).rpartition(".")[2]
            if spelled_name != shape.collection.name:
                raise Unsupported(f"`{ast.unparse(factory)}` does not make the field's collection")
            return self._new(shape.collection)
        if shape.kind == "object":
            named = ast.unparse(factory)
            if named != shape.record.rpartition(".")[2]:
                raise Unsupported(f"`{named}` does not make the field's class")
            call = ast.Call(func=factory, args=[], keywords=[])
            ast.copy_location(call, factory)
            return self._instance(shape, call)
        raise Unsupported(f"`{ast.unparse(factory)}` has no native form as a default")

    def _method(self, shape: Shape, attr: str) -> tuple[object, object, str]:
        """The native function behind a method, instantiated for a generic class."""
        owner = self._resolve(self._class_info(shape), attr)
        if owner is None:
            raise Unsupported(f"`{shape.record}` has no method `{attr}`")
        info = owner
        method = info.methods[attr]
        frontend = self.frontend
        if info.type_params:
            bindings = dict(zip(info.type_params, shape.class_args, strict=True))
            found = frontend.instantiate(method.qualname, (), bindings)  # type: ignore[attr-defined]
        else:
            found = frontend.declared.get(method.qualname)  # type: ignore[attr-defined]
        if found is None:
            raise Unsupported(f"`{info.name}.{attr}` has no native lowering")
        function, signature = found
        return function, signature, method.qualname

    def _method_call(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        shape: Shape,
        attr: str,
        receiver: Value,
        arguments: list[ast.expr],
        keywords: list[ast.keyword],
        *,
        discard: bool = False,
        exact: bool = False,
    ) -> Value:
        """`obj.method(args)`: the method's native function, called with the handle first.

        Where a subclass overrides the method, the call goes by the class the
        object was made as, which its tag says.
        """
        if keywords:
            raise Unsupported(f"`{attr}` takes positional arguments natively")
        function, signature, qualname = self._method(shape, attr)
        rest = _Parameters(tuple(signature.parameters[1:]))  # type: ignore[attr-defined]
        waiting = len(self._temporaries)  # type: ignore[attr-defined]
        values = self._call_arguments(rest, arguments, qualname)  # type: ignore[attr-defined]
        temporaries = self._temporaries[waiting:]  # type: ignore[attr-defined]
        del self._temporaries[waiting:]  # type: ignore[attr-defined]
        results = function.results  # type: ignore[attr-defined]
        overrides = [] if exact else self._overrides(shape, attr)
        if overrides:
            called = self._dispatch(shape, attr, receiver, values, overrides, function)
        else:
            called = core.call(self.b, function.name, (receiver, *values), results)  # type: ignore[attr-defined]
        for handle in temporaries:
            self._release(handle)
        if not results:
            if not discard:
                raise Unsupported(f"`{attr}` returns nothing a caller can use")
            return self._word(0)
        result = called.results[0]
        if discard and result.type == HANDLE:
            self._release(result)
        return result

    def _overrides(self, shape: Shape, attr: str) -> list[tuple[ClassInfo, list[int]]]:
        """Where `attr` runs differently for some class an instance of `shape` may
        be: each implementation other than the static one, with the tags of the
        classes that run it. Empty when every such class runs the same one."""
        if attr == "__init__":
            return []
        static = self._resolve(self._class_info(shape), attr)
        grouped: dict[str, tuple[ClassInfo, list[int]]] = {}
        for info in self._subclasses(shape.record):
            owner = self._resolve(info, attr)
            if owner is None or owner is static:
                continue
            grouped.setdefault(owner.qualname, (owner, []))[1].append(class_tag(info.qualname))
        return list(grouped.values())

    def _dispatch(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        shape: Shape,
        attr: str,
        receiver: Value,
        values: list[Value],
        overrides: list[tuple[ClassInfo, list[int]]],
        function: object,
    ) -> object:
        """A call that goes by the object's tag: each override where it matches,
        the static method otherwise."""
        results = function.results  # type: ignore[attr-defined]
        slot = self._alloca(results[0], f"{attr}.result") if results else None  # type: ignore[attr-defined]
        tag = self._tag(receiver)
        done = self._block(f"{attr}.done")  # type: ignore[attr-defined]
        for owner, tags in overrides:
            other = Shape("object", record=owner.qualname, class_args=shape.class_args)
            implementation, _signature, _ = self._method(other, attr)
            if (
                [t for _, t in implementation.params[1:]]
                != [  # type: ignore[attr-defined]
                    t
                    for _, t in function.params[1:]  # type: ignore[attr-defined]
                ]
                or implementation.results != results
            ):  # type: ignore[attr-defined]
                raise Unsupported(f"`{owner.name}.{attr}` overrides with another signature")
            matched = core.cmp(self.b, "eq", tag, self._word(tags[0]))
            for more in tags[1:]:
                matched = core.bitwise(
                    self.b, "or", matched, core.cmp(self.b, "eq", tag, self._word(more))
                )
            here = self._block(f"{attr}.{owner.name}")  # type: ignore[attr-defined]
            after = self._block(f"{attr}.next")  # type: ignore[attr-defined]
            core.cond_br(self.b, matched, Successor(here), Successor(after))
            self.b.at_end(here)  # type: ignore[attr-defined]
            made = core.call(self.b, implementation.name, (receiver, *values), results)  # type: ignore[attr-defined]
            if slot is not None:
                core.store(self.b, made.results[0], slot)
            core.br(self.b, Successor(done))
            self.b.at_end(after)  # type: ignore[attr-defined]
        made = core.call(self.b, function.name, (receiver, *values), results)  # type: ignore[attr-defined]
        if slot is not None:
            core.store(self.b, made.results[0], slot)
        core.br(self.b, Successor(done))
        self.b.at_end(done)  # type: ignore[attr-defined]
        return _Called((core.load(self.b, slot),) if slot is not None else ())

    def _set_tag(self, handle: Value, qualname: str) -> None:
        """The class an object was made as, in its header's spare word."""
        words = core.cast(self.b, handle, PtrType(I64))
        address = core.ptr_offset(self.b, words, self._word(_TAG))
        core.store(self.b, self._word(class_tag(qualname)), address)

    def _tag(self, handle: Value) -> Value:
        words = core.cast(self.b, handle, PtrType(I64))
        return core.load(self.b, core.ptr_offset(self.b, words, self._word(_TAG)))

    def _is_instance(self, node: ast.Call) -> Value | None:
        """`isinstance(obj, Cls)`: the object is there and was made as `Cls` or a
        class deriving from it."""
        if len(node.args) != 2 or node.keywords:
            return None
        shape = self._object_of(node.args[0])
        if shape is None:
            return None
        classes = node.args[1].elts if isinstance(node.args[1], ast.Tuple) else [node.args[1]]
        wanted: list[str] = []
        for spelled_class in classes:
            called = T.strip_literal(self._type_of(spelled_class))
            if not isinstance(called, T.ClassObject):
                raise Unsupported("`isinstance` natively takes classes of this module")
            wanted.append(called.name)
        handle, owned = self._handle(node.args[0])
        present = self._present(handle)
        if any(self._subclass_of(shape.record, name) for name in wanted):
            truth = present
        else:
            tags = sorted(
                {class_tag(info.qualname) for name in wanted for info in self._subclasses(name)}
            )
            if not tags:
                self._done_with(handle, owned)
                return core.const(self.b, False, BOOL)
            # A null handle has no tag to read: read it only where there is one.
            slot = self._alloca(BOOL, "isinstance")  # type: ignore[attr-defined]
            core.store(self.b, core.const(self.b, False, BOOL), slot)
            there = self._block("isinstance.there")  # type: ignore[attr-defined]
            done = self._block("isinstance.done")  # type: ignore[attr-defined]
            core.cond_br(self.b, present, Successor(there), Successor(done))
            self.b.at_end(there)  # type: ignore[attr-defined]
            tag = self._tag(handle)
            matched = core.cmp(self.b, "eq", tag, self._word(tags[0]))
            for more in tags[1:]:
                matched = core.bitwise(
                    self.b, "or", matched, core.cmp(self.b, "eq", tag, self._word(more))
                )
            core.store(self.b, matched, slot)
            core.br(self.b, Successor(done))
            self.b.at_end(done)  # type: ignore[attr-defined]
            truth = core.load(self.b, slot)
        self._done_with(handle, owned)
        return truth

    def _subclass_of(self, qualname: str, base: str) -> bool:
        info = self._module_classes().get(qualname)
        return info is not None and base in info.mro

    def _super_call(self, node: ast.Call, discard: bool) -> Value | None:
        """`super().method(args)` in a method: the next class's method along the
        bases, called directly on `self`."""
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name)
            and func.value.func.id == "super"
            and not func.value.args
        ):
            return None
        info = getattr(self, "info", None)
        owner_name = info.owner if info is not None else None
        receiver_name = info.params[0].name if info is not None and info.params else None
        if not owner_name or receiver_name is None or receiver_name not in self.collections:
            raise Unsupported("`super()` natively calls from a method of an object class")
        held = self.collections[receiver_name].kind
        assert isinstance(held, Shape)
        current = self._class_named(owner_name)
        chain = object_chain(current, self._module_classes())
        if chain is None or len(chain) < 2:
            raise Unsupported(f"`{current.name}` has no base for `super()` to call")
        found = None
        for entry in reversed(chain[:-1]):
            if func.attr in entry.methods:
                found = entry
                break
        handle = core.load(self.b, self.collections[receiver_name].slot)
        if found is None:
            if func.attr == "__init__" and not node.args and not node.keywords:
                return self._word(0)
            raise Unsupported(f"no base of `{current.name}` has `{func.attr}`")
        base = Shape("object", record=found.qualname, class_args=held.class_args)
        return self._method_call(
            base, func.attr, handle, node.args, node.keywords, discard=discard, exact=True
        )

    # -- operators on objects ------------------------------------------------

    def _has_method(self, shape: Shape, attr: str) -> bool:
        return self._resolve(self._class_info(shape), attr) is not None

    def _dunder(
        self, shape: Shape, attr: str, receiver: ast.expr, arguments: list[ast.expr]
    ) -> Value:
        """`receiver.attr(arguments)` for an operator: the object must be there."""
        handle, owned = self._handle(receiver)
        self._require(
            self._present(handle),
            f"`None` has no `{attr}`",
            _NONE_OPERAND.get(attr, f"AttributeError: 'NoneType' object has no attribute '{attr}'"),
        )
        found = self._method_call(shape, attr, handle, arguments, [])
        self._done_with(handle, owned)
        return found

    def _object_compare(self, node: ast.Compare) -> Value | None:
        """`a == b`, `a < b`, ... where `a` or `b` is an object whose class says
        what they mean: the method, or the reflected one, as Python tries them."""
        operator = type(node.ops[0])
        left, right = node.left, node.comparators[0]
        left_shape, right_shape = self._object_of(left), self._object_of(right)
        if left_shape is None and right_shape is None:
            return None
        if operator in (ast.Is, ast.IsNot, ast.In, ast.NotIn):
            return None
        forward, reflected = _COMPARE_DUNDERS[operator]
        if left_shape is not None and self._has_method(left_shape, forward):
            return self._dunder(left_shape, forward, left, [right])
        if right_shape is not None and self._has_method(right_shape, reflected):
            return self._dunder(right_shape, reflected, right, [left])
        if operator in (ast.Eq, ast.NotEq):
            if left_shape is not None and self._has_method(left_shape, "__eq__"):
                found = self._dunder(left_shape, "__eq__", left, [right])
                truth = self._truth(found)  # type: ignore[attr-defined]
                return core.bitwise(self.b, "xor", truth, core.const(self.b, True, BOOL))
            if self._dataclass_equality(left_shape or right_shape):
                raise Unsupported("a dataclass's generated `==` compares fields, natively not yet")
            # An object with no `__eq__` of its own is equal only to itself.
            sides = []
            for side in (left, right):
                handle, owned = self._handle(side)
                sides.append(core.cast(self.b, handle, I64))
                self._done_with(handle, owned)
            predicate = "eq" if operator is ast.Eq else "ne"
            return core.cmp(self.b, predicate, sides[0], sides[1])
        raise Unsupported(f"`{ast.unparse(node)}` compares objects whose class does not say how")

    def _dataclass_equality(self, shape: Shape | None) -> bool:
        """A dataclass object compares its fields: `eq=True`, the default."""
        if shape is None:
            return False
        return any(owner.is_dataclass for owner in self._chain(shape))

    def _object_binary(self, node: ast.BinOp) -> Value | None:
        """`a + b` and the other arithmetic operators through the class's methods."""
        names = _BINARY_DUNDERS.get(type(node.op))
        if names is None:
            return None
        forward, reflected = names
        left_shape, right_shape = self._object_of(node.left), self._object_of(node.right)
        if left_shape is not None and self._has_method(left_shape, forward):
            return self._dunder(left_shape, forward, node.left, [node.right])
        if right_shape is not None and self._has_method(right_shape, reflected):
            return self._dunder(right_shape, reflected, node.right, [node.left])
        if left_shape is not None or right_shape is not None:
            raise Unsupported(f"`{ast.unparse(node)}`: the class has no `{forward}`")
        return None

    def _object_unary(self, node: ast.UnaryOp) -> Value | None:
        attr = {ast.USub: "__neg__", ast.UAdd: "__pos__", ast.Invert: "__invert__"}.get(
            type(node.op)
        )
        shape = self._object_of(node.operand)
        if attr is None or shape is None:
            return None
        if not self._has_method(shape, attr):
            raise Unsupported(f"`{ast.unparse(node)}`: the class has no `{attr}`")
        return self._dunder(shape, attr, node.operand, [])

    def _object_item(self, node: ast.Subscript) -> Value | None:
        """`obj[key]`: the class's `__getitem__`."""
        shape = self._object_of(node.value)
        if shape is None:
            return None
        if isinstance(node.slice, ast.Slice) or not self._has_method(shape, "__getitem__"):
            raise Unsupported(f"`{ast.unparse(node)}` has no native lowering")
        return self._dunder(shape, "__getitem__", node.value, [node.slice])

    def _object_store_item(self, target: ast.Subscript, value: ast.expr) -> bool:
        """`obj[key] = value`: the class's `__setitem__`."""
        shape = self._object_of(target.value)
        if shape is None:
            return False
        if isinstance(target.slice, ast.Slice) or not self._has_method(shape, "__setitem__"):
            raise Unsupported(f"`{ast.unparse(target)} = ...` has no native lowering")
        handle, owned = self._handle(target.value)
        self._require(
            self._present(handle),
            "`None` has no `__setitem__`",
            "TypeError: 'NoneType' object does not support item assignment",
        )
        self._method_call(shape, "__setitem__", handle, [target.slice, value], [], discard=True)
        self._done_with(handle, owned)
        return True

    def _object_contains(self, container: ast.expr, item: ast.expr) -> Value | None:
        """`item in obj`: the class's `__contains__`."""
        shape = self._object_of(container)
        if shape is None:
            return None
        if not self._has_method(shape, "__contains__"):
            raise Unsupported(f"`in` asks `{shape.record}`, which has no `__contains__`")
        found = self._dunder(shape, "__contains__", container, [item])
        return self._truth(found)  # type: ignore[attr-defined]

    def _object_method(self, node: ast.Call, discard: bool) -> Value:
        """A method called on an object expression."""
        assert isinstance(node.func, ast.Attribute)
        shape = self._object_of(node.func.value)
        assert shape is not None
        handle, owned = self._handle(node.func.value)
        self._require(
            self._present(handle),
            f"`None` has no attribute `{node.func.attr}`",
            f"AttributeError: 'NoneType' object has no attribute '{node.func.attr}'",
        )
        result = self._method_call(
            shape, node.func.attr, handle, node.args, node.keywords, discard=discard
        )
        self._done_with(handle, owned)
        return result

    def _field_value(self, node: ast.Attribute) -> Value:
        """`obj.field`, read in place: a field holding a reference lends it."""
        shape = self._object_of(node.value)
        assert shape is not None
        offset, field_shape = self._field(shape, node.attr)
        handle, owned = self._handle(node.value)
        self._require(
            self._present(handle),
            f"`None` has no attribute `{node.attr}`",
            f"AttributeError: 'NoneType' object has no attribute '{node.attr}'",
        )
        value = self._read(self._field_address(handle, offset), field_shape)
        if owned:
            if field_shape.reference:
                raise Unsupported("a field read from a temporary object outlives it")
            self._release(handle)
        return value

    def _field_handle(self, node: ast.Attribute) -> tuple[Value, bool]:
        """A reference field as a handle: lent where its object is held, and
        taken (owned) where its object is a temporary that goes now."""
        shape = self._object_of(node.value)
        assert shape is not None
        offset, field_shape = self._field(shape, node.attr)
        handle, owned = self._handle(node.value)
        self._require(
            self._present(handle),
            f"`None` has no attribute `{node.attr}`",
            f"AttributeError: 'NoneType' object has no attribute '{node.attr}'",
        )
        value = self._read(self._field_address(handle, offset), field_shape)
        if owned:
            self._retain(value)
            self._release(handle)
        return value, owned

    def _element_handle(self, node: ast.Subscript) -> tuple[Value, bool]:
        """A reference element as a handle, taken where its collection is a temporary."""
        kind, handle, owned = self._receiver(node.value)
        shape = kind.value
        if isinstance(node.slice, ast.Slice) or shape is None:
            raise Unsupported(f"a {kind.name} has no index")
        value = self._read(self._element_address(kind, handle, node.slice, write=False), shape)
        if owned:
            self._retain(value)
            self._release(handle)
        return value, owned

    def _field_store(self, target: ast.Attribute, value: ast.expr) -> None:
        """`obj.field = value`: a reference stored takes one, the old one is let go."""
        shape = self._object_of(target.value)
        assert shape is not None
        offset, field_shape = self._field(shape, target.attr)
        handle, owned = self._handle(target.value)
        self._require(
            self._present(handle),
            f"`None` has no attribute `{target.attr}`",
            f"AttributeError: 'NoneType' object has no attribute '{target.attr}'",
        )
        self._store_into(self._field_address(handle, offset), field_shape, value, fresh=False)
        self._done_with(handle, owned)

    def _identity(self, node: ast.Compare) -> Value | None:
        """`x is None`, `x is not y`: handles compared as addresses."""
        operator = node.ops[0]
        if not isinstance(operator, (ast.Is, ast.IsNot)):
            return None
        left, right = node.left, node.comparators[0]
        sides = (left, right)
        nones = [isinstance(s, ast.Constant) and s.value is None for s in sides]
        if not all(
            n or self._reference_of(s) is not None for s, n in zip(sides, nones, strict=True)
        ):
            return None
        addresses = []
        for side in sides:
            handle, owned = self._handle(side)
            addresses.append(core.cast(self.b, handle, I64))
            self._done_with(handle, owned)
        predicate = "eq" if isinstance(operator, ast.Is) else "ne"
        return core.cmp(self.b, predicate, addresses[0], addresses[1])

    def _object_truth(self, node: ast.expr, empty: bool = False) -> Value | None:
        """An object as a condition: `__bool__` or `__len__` where the class has one,
        and otherwise whether it is there at all."""
        shape = self._object_of(node)
        if shape is None:
            return None
        info = self._class_info(shape)
        handle, owned = self._handle(node)
        if "__bool__" in info.methods or "__len__" in info.methods:
            # `None` is false, as in Python; only an object is asked.
            attr = "__bool__" if "__bool__" in info.methods else "__len__"
            present = self._present(handle)
            asked = self._block("truth.asked")  # type: ignore[attr-defined]
            done = self._block("truth.end")  # type: ignore[attr-defined]
            truth = done.add_argument(BOOL, "truth")
            core.cond_br(self.b, present, Successor(asked), Successor(done, [present]))
            self.b.at_end(asked)  # type: ignore[attr-defined]
            found = self._method_call(shape, attr, handle, [], [])
            core.br(self.b, Successor(done, [self._truth(found)]))  # type: ignore[attr-defined]
            self.b.at_end(done)  # type: ignore[attr-defined]
        else:
            truth = self._present(handle)
        self._done_with(handle, owned)
        if empty:
            return core.bitwise(self.b, "xor", truth, core.const(self.b, True, BOOL))
        return truth

    def _object_length(self, node: ast.expr) -> Value | None:
        shape = self._object_of(node)
        if shape is None:
            return None
        handle, owned = self._handle(node)
        self._require(
            self._present(handle),
            "`None` has no length",
            "TypeError: object of type 'NoneType' has no len()",
        )
        found = self._method_call(shape, "__len__", handle, [], [])
        self._done_with(handle, owned)
        return found

    # -- value classes ---------------------------------------------------

    def _record_construction(self, node: ast.Call) -> Value | None:
        """`Point(1, 2.0)` for a value class: its fields as one struct value."""
        called = T.strip_literal(self._type_of(node.func))
        made = T.strip_literal(self._type_of(node))
        if not isinstance(called, T.ClassObject) or not isinstance(made, T.Instance):
            return None
        found = self._records().get(made.name)
        if found is None or not found[0]:
            return None
        info = self.frontend.analysis.symbols.classes.get(made.name.rpartition(".")[2])  # type: ignore[attr-defined]
        if info is None or not info.is_dataclass or "__init__" in info.methods:
            # A class with its own `__init__` takes that `__init__`'s arguments,
            # which native code does not run for a value class.
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
        if not isinstance(made, T.Instance) or not self._records().get(made.name, ((), False))[0]:
            return None
        value = self._expr(node)  # type: ignore[attr-defined]
        return value if isinstance(value.type, StructType) else None

    # -- the runtime ------------------------------------------------------

    def _rt(self, symbol: str, arguments: tuple[Value, ...], result: IRType | None = I64) -> Value:
        """One call into the collections runtime."""
        results = (result,) if result is not None else ()
        found = core.call_extern(self.b, symbol, arguments, results)
        if symbol.startswith(_CALLS_BACK) and self._calls_back():
            # A method the runtime called back may have failed a guard; the
            # runtime can only say so, and this is where the code falls back.
            ok = core.call_extern(self.b, "ppy_coll_callback_ok", (), (I64,)).results[0]
            core.guard(
                self.b,
                core.cmp(self.b, "ne", ok, self._word(0)),
                "contract",
                "a method the collection called back failed a guard",
            )
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

    def _require(
        self, condition: Value, message: str, raises: str = "", values: tuple[Value, ...] = ()
    ) -> None:
        """A check CPython would raise for: native code falls back, a binary stops.

        `raises` is what CPython's traceback ends with there, `{0}` and on
        standing for `values`; a standalone binary prints it.
        """
        if not self.frontend.standalone:  # type: ignore[attr-defined]
            values = ()
        core.guard(self.b, condition, "bounds", message, raises=raises, values=values)

    def _key_report(self, kind: Kind, address: Value) -> tuple[str, tuple[Value, ...]]:
        """A key as `str()` spells it, `5` or `(1, 2)`, and the words it reads."""
        key = kind.key
        assert key is not None
        count = len(_kinds(key))
        if key.kind in {"str", "object"}:
            # A string's text, or an object's repr, is not a word the message can carry.
            return "", ()
        values = tuple(self._read_word(address, i, "int") for i in range(count))
        if key.kind != "tuple":
            return "{0}", values
        if count == 1:
            return "({0},)", values
        return "(" + ", ".join(f"{{{i}}}" for i in range(count)) + ")", values

    def _found(self, index: Value) -> Value:
        return core.cmp(self.b, "ge", index, self._word(0))

    # -- making ------------------------------------------------------------

    def _new(self, kind: Kind, count: Value | None = None) -> Value:
        """A new, empty collection (a `Vec` of `count` zeros): an owned handle."""
        self._use_collections()
        value = kind.value
        words = self._word(kind.words)
        floats = self._word(value.floats if value is not None else 0)
        # The runtime takes the handle words, and above bit 32 which of them
        # are strings: those can be in no cycle, so the collector skips them.
        handles = self._word((value.handles | (value.leaves << 32)) if value is not None else 0)
        keys = self._word(kind.key.words if kind.key is not None else 0)
        if kind.family == "seq":
            made = self._rt("ppy_seq_new", (count or self._word(0), words, floats, handles), HANDLE)
            if count is not None and value is not None and value.kind in {"collection", "str"}:
                self._fill_new(made, count, value)
            return made
        if kind.family == "list":
            return self._rt("ppy_list_new", (words, floats, handles), HANDLE)
        symbol = "ppy_map_new" if kind.family == "map" else "ppy_tree_new"
        made = self._rt(symbol, (keys, words, floats, handles), HANDLE)
        assert kind.key is not None
        if kind.key.text:
            self._rt("ppy_coll_text_keys", (made, self._word(kind.key.text)), None)
        self._install_methods(made, kind)
        return made

    # -- the program's own order and hash ---------------------------------------

    def _install_methods(self, made: Value, kind: Kind) -> None:
        """A collection of objects orders, hashes, and compares them as their class
        says: `__lt__` for a sort, a heap, and a tree's keys; `__hash__` and
        `__eq__` for a hash map's keys. The runtime calls the compiled methods
        back; a class without them is ordered by nothing and hashed by identity."""
        keyed = kind.family in {"map", "tree"}
        element = kind.key if keyed else kind.value
        if kind.family == "map" and element is not None and element.kind == "record":
            # A value class is copied, so it has no identity to hash by: only a
            # dataclass that hashes its fields (`frozen=True`) is a key natively.
            node = self._class_named(element.record).node
            frozen = dataclass_keyword(node, "frozen") is True
            if not (frozen or dataclass_keyword(node, "unsafe_hash") is True):
                raise Unsupported(f"`{element.record}` is hashed by identity, which a value has not")
            if any(part == "float" for part in element.parts):
                # `0.0 == -0.0` and NaN: equal floats are not always equal words.
                raise Unsupported(f"`{element.record}` hashes a float, whose words do not decide `==`")
        if element is None or element.kind != "object":
            return
        if kind.family == "tree" or kind.name in {"Vec", "Heap", "MaxHeap"}:
            less = self._callback(element, "__lt__", 2)
            if less is not None:
                self._rt("ppy_coll_order_by", (made, less), None)
            elif kind.family == "tree" or kind.name != "Vec":
                raise Unsupported(f"a {kind.name} orders by `__lt__`, which `{element.record}` has not")
        if keyed:
            hashed = equal = None
            if kind.family == "map":
                # A `__hash__` without `__eq__` compares by identity; an `__eq__`
                # without `__hash__` is unhashable, which the checker refuses.
                hashed = self._callback(element, "__hash__", 1)
                equal = self._callback(element, "__eq__", 2)
                if equal is not None and hashed is None:
                    raise Unsupported(f"`{element.record}` defines `__eq__` and no `__hash__`")
            nothing = self._word(0)
            self._rt(
                "ppy_coll_hash_by",
                (made, hashed or nothing, equal or nothing, self._word(1)),
                None,
            )

    def _callback(self, shape: Shape, attr: str, arity: int) -> Value | None:
        """The address of `attr`'s compiled method, which the runtime calls with
        handles, or None where the class does not define it.

        A method a subclass overrides would need the object's class to choose
        it, which a callback cannot: that stays in Python."""
        info = self._class_info(shape)
        if self._resolve(info, attr) is None:
            return None
        if self._overrides(shape, attr):
            raise Unsupported(f"`{attr}` is overridden below `{shape.record}`")
        try:
            function, signature, _qualname = self._method(shape, attr)
        except Unsupported:
            # `def __eq__(self, other: object)`: `other` is one of the keys.
            owner = self._resolve(info, attr)
            method = owner.methods[attr]  # type: ignore[union-attr]
            if attr != "__eq__" or len(method.params) != 2 or shape.class_args:
                raise
            found = self.frontend.narrowed(  # type: ignore[attr-defined]
                method.qualname, method.params[1].name, T.instance(shape.record)
            )
            if found is None:
                raise
            function, signature = found
        parameters = signature.parameters  # type: ignore[attr-defined]
        returns = signature.returns  # type: ignore[attr-defined]
        if len(parameters) != arity or not all(p.is_handle for p in parameters):
            raise Unsupported(f"`{shape.record}.{attr}` takes its own class natively")
        if returns not in {("i8",), ("i64",)}:
            raise Unsupported(f"`{shape.record}.{attr}` answers a `bool` or an `int` natively")
        return core.callback(self.b, function.name)  # type: ignore[attr-defined]

    def _orders(self, shape: Shape) -> bool:
        """Whether `<` orders `shape` natively: the runtime's words, or an object
        whose class has `__lt__`, which the collection calls back."""
        if shape.comparable:
            return True
        return shape.kind == "object" and self._resolve(self._class_info(shape), "__lt__") is not None

    def _calls_back(self) -> bool:
        """Whether some class of the module orders, hashes, or compares its
        instances itself, which the runtime may call: then each call that may
        is followed by asking whether one failed."""
        cached = getattr(self.frontend, "_calls_back", None)
        if cached is None:
            classes = self.frontend.analysis.symbols.classes  # type: ignore[attr-defined]
            cached = any(
                fields == () and _COMPARES & set(info.methods)
                for qualname, fields in self.frontend.layouts.items()  # type: ignore[attr-defined]
                for info in [classes.get(qualname) or classes.get(qualname.rpartition(".")[2])]
                if info is not None
            )
            self.frontend._calls_back = cached  # type: ignore[attr-defined]
        return cached

    def _fill_new(self, made: Value, count: Value, value: Shape) -> None:
        """`Vec[Vec[int]](n)`: each of the `n` slots its own new, empty collection
        (or, for `Vec[str](n)`, its own empty string)."""
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
        if value.collection is not None:
            inner = self._new(value.collection)
        else:
            inner = self._rt("ppy_str_empty", (), HANDLE)
        address = self._rt("ppy_seq_at", (made, core.load(self.b, index)), HANDLE)
        core.store(self.b, inner, core.cast(self.b, address, PtrType(HANDLE)))
        step = core.add(self.b, core.load(self.b, index), self._word(1), overflow="wrap")
        core.store(self.b, step, index)
        core.br(self.b, Successor(header))
        self.b.at_end(done)  # type: ignore[attr-defined]

    def _constructed(self, node: ast.expr) -> Kind | None:
        """The collection `node` makes, if it is `ppy.Vec[...](...)` or one like it."""
        if not isinstance(node, ast.Call):
            return None
        if isinstance(node.func, ast.Subscript):
            spelled_name = ast.unparse(node.func.value).rpartition(".")[2]
        elif isinstance(node.func, (ast.Name, ast.Attribute)):
            # `Vec()`, its type taken from where it goes.
            spelled_name = ast.unparse(node.func).rpartition(".")[2]
        else:
            return None
        kind = self._kind_of(node)
        if kind is None or kind.name != spelled_name:
            return None
        return kind

    def _make_collection(self, name: str, value: ast.expr, declared: T.Type | None = None) -> bool:
        """`q = ppy.Deque[int]()`, `root = Node(1)`, `root = None`: any binding of a
        name to a collection or an object."""
        kind = self._reference_of(value) if not isinstance(value, ast.Name) else None
        if isinstance(value, ast.Name) and value.id in self.collections:
            kind = self.collections[value.id].kind
        if kind is None and declared is not None:
            kind = self._reference_of_type(declared)
        if kind is None and name in self.collections:
            kind = self.collections[name].kind
        if kind is None:
            return False
        handle, owned = self._handle(value)
        self._bind(name, kind, handle, owned)
        return True

    def _bind(self, name: str, kind: Kind | Shape, handle: Value, owned: bool) -> None:
        """`name` now holds `handle`: one reference taken, the old one let go."""
        held = self.collections.get(name)
        if held is not None and held.kind != kind and not _related(held.kind, kind):
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

    def _bind_parameter(self, name: str, kind: Kind | Shape, argument: Value) -> None:
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
        if isinstance(node, ast.Constant) and node.value is None:
            return self._rt("ppy_coll_none", (), HANDLE), True
        if isinstance(node, ast.Name):
            held = self.collections.get(node.id)
            if held is None:
                raise Unsupported(f"`{node.id}` is not a native collection")
            return core.load(self.b, held.slot), False
        if isinstance(node, ast.Attribute) and self._object_of(node.value) is not None:
            return self._field_handle(node)
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Subscript | ast.Attribute):
            made = self._object_of(node)
            called = T.strip_literal(self._type_of(node.func))
            if made is not None and isinstance(called, T.ClassObject):
                return self._instance(made, node), True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and isinstance(T.strip_literal(self._type_of(node.func)), T.ClassObject)
        ):
            made = self._object_of(node)
            if made is not None:
                return self._instance(made, node), True
        made_string = self._string_handle(node)  # type: ignore[attr-defined]
        if made_string is not None:
            return made_string
        listed = self._string_list_handle(node)  # type: ignore[attr-defined]
        if listed is not None:
            return listed
        kind = self._constructed(node)
        if kind is not None:
            assert isinstance(node, ast.Call)
            count = None
            if node.args:
                count = self._coerce(self._expr(node.args[0]), "int")  # type: ignore[attr-defined]
                self._require(
                    core.cmp(self.b, "ge", count, self._word(0)),
                    "a negative size",
                    "ValueError: a Vec cannot start with {0} elements",
                    (count,),
                )
            return self._new(kind, count), True
        if isinstance(node, ast.Subscript) and self._object_of(node.value) is not None:
            return self._expr(node), True  # type: ignore[attr-defined]
        if isinstance(node, ast.Subscript):
            return self._element_handle(node)
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.IfExp)):
            return self._expr(node), True  # type: ignore[attr-defined]
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and self._is_collection(node.func.value):
                found = self._collection_method(node.func.value, node.func.attr, node)
                return found, _OWNED_RESULTS.get(node.func.attr, False)
            return self._expr(node), True  # type: ignore[attr-defined]
        raise Unsupported(f"`{ast.unparse(node)}` has no native collection form")

    def _receiver(self, node: ast.expr) -> tuple[Kind, Value, bool]:
        """The collection a method, an index, a loop, or `len` works on."""
        kind = self._kind_of(node)
        if kind is None or not isinstance(kind, Kind):
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
        if shape.reference:
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
        if shape.reference:
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
        if shape.reference:
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
        if not shape.reference:
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
        value, owned = self._value(node, key)
        self._write(address, key, value)
        if owned:
            # A key made for the lookup: the collection takes its own
            # reference when it keeps one, so this one goes after the call.
            self.__dict__.setdefault("_keys_made", []).append(value)
        return address

    def _keys_done(self) -> None:
        """Let go of the keys made for the runtime calls just emitted."""
        for value in self.__dict__.pop("_keys_made", []):
            self._release(value)

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
        self._keys_done()
        self._done_with(handle, owned)
        return self._found(found)

    def _record_place(self, target: ast.Attribute) -> tuple[Shape, int] | None:
        """`points[i].x`, `table[k].x`, `obj.pos.x`: a field of a value-class
        element or field, where it is held. Its record's shape and the field's
        word, or None where `target` is not one."""
        holder = target.value
        shape: Shape | None
        if isinstance(holder, ast.Subscript) and self._is_collection(holder.value):
            kind = self._kind_of(holder.value)
            shape = kind.value if kind is not None else None
        elif isinstance(holder, ast.Attribute):
            owner = self._object_of(holder.value)
            if owner is None:
                return None
            shape = self._field(owner, holder.attr)[1]
        else:
            return None
        if shape is None or shape.kind != "record" or target.attr not in shape.names:
            return None
        return shape, shape.names.index(target.attr)

    def _record_address(self, holder: ast.expr) -> Value:
        """Where a value-class element or field is held."""
        if isinstance(holder, ast.Subscript):
            kind, handle, owned = self._receiver(holder.value)
            if owned:
                raise Unsupported("a field written in a temporary collection's element is lost")
            return self._element_address(kind, handle, holder.slice, write=False)
        assert isinstance(holder, ast.Attribute)
        shape = self._object_of(holder.value)
        assert shape is not None
        offset, _ = self._field(shape, holder.attr)
        handle, owned = self._handle(holder.value)
        if owned:
            raise Unsupported("a field written in a temporary object is lost")
        self._require(
            self._present(handle),
            f"`None` has no attribute `{holder.attr}`",
            f"AttributeError: 'NoneType' object has no attribute '{holder.attr}'",
        )
        return self._field_address(handle, offset)

    def _record_field_read(self, target: ast.Attribute) -> Value:
        found = self._record_place(target)
        assert found is not None
        shape, word = found
        return self._read_word(self._record_address(target.value), word, shape.parts[word])

    def _record_field_store(self, target: ast.Attribute, value: Value) -> None:
        """`points[i].x = v`: the one word of the element, written where it is held."""
        found = self._record_place(target)
        assert found is not None
        shape, word = found
        self._check_record_writes(shape.record)
        address = self._record_address(target.value)
        kind = shape.parts[word]
        stored = F64 if kind == "float" else I64
        pointer = core.cast(self.b, address, _pointer(address, stored))
        if word:
            pointer = core.ptr_offset(self.b, pointer, self._word(word))
        coerced = self._coerce(value, "float" if kind == "float" else "int")  # type: ignore[attr-defined]
        core.store(self.b, coerced, pointer)

    def _check_record_writes(self, record: str) -> None:
        """A value class is copied where CPython shares: a copy read out of a
        collection before a field of the element is written in place would
        miss the write. A function that does both, in that order or in one
        loop, stays in Python."""
        checked: set[str] = self.__dict__.setdefault("_record_writes_checked", set())
        if record in checked:
            return
        checked.add(record)
        info = getattr(self, "info", None)
        if info is None:
            return
        if _copies_before_writes(info.node, record, self._type_of):
            raise Unsupported(
                f"a `{record.rpartition('.')[2]}` is copied out and written in place in one "
                "function, which CPython would share"
            )

    def _element_address(self, kind: Kind, handle: Value, index: ast.expr, *, write: bool) -> Value:
        """Where `v[i]` or `m[key]` is held; with `write`, a map makes the entry."""
        if kind.name in {"Vec", "Deque", "List"}:
            position = self._coerce(self._expr(index), "int")  # type: ignore[attr-defined]
            if kind.name == "List":
                position = self._list_position(handle, position)  # type: ignore[attr-defined]
            length = self._rt("ppy_coll_len", (handle,))
            inside = core.bitwise(
                self.b,
                "and",
                core.cmp(self.b, "ge", position, self._word(0)),
                core.cmp(self.b, "lt", position, length),
            )
            self._require(
                inside,
                "index out of range",
                "IndexError: index {0} is out of range for length {1}",
                (position, length),
            )
            return self._rt("ppy_seq_at", (handle, position), HANDLE)
        if kind.name in {"HashMap", "TreeMap"}:
            key = self._key(kind, index)
            if write:
                entry = self._rt(f"ppy_{kind.family}_put", (handle, key))
            else:
                entry = self._rt(f"ppy_{kind.family}_find", (handle, key))
                spelled, words = self._key_report(kind, key)
                self._require(
                    self._found(entry),
                    "key not found",
                    f"KeyError: {spelled}" if spelled else "KeyError",
                    words,
                )
            self._keys_done()
            return self._rt(f"ppy_{kind.family}_value_at", (handle, entry), HANDLE)
        raise Unsupported(f"a {kind.name} has no index")

    def _item(self, container: ast.expr, index: ast.expr, value: ast.expr | None = None) -> Value:
        """`v[i]` (0 to `len - 1`) or a map's `m[key]`, read; or with `value`, written."""
        kind, handle, owned = self._receiver(container)
        shape = kind.value
        if isinstance(index, ast.Slice) or shape is None:
            raise Unsupported(f"a {kind.name} has no index")
        if kind.name in {"Vec", "Deque", "List"}:
            position = self._coerce(self._expr(index), "int")  # type: ignore[attr-defined]
            if kind.name == "List":
                position = self._list_position(handle, position)  # type: ignore[attr-defined]
            length = self._rt("ppy_coll_len", (handle,))
            inside = core.bitwise(
                self.b,
                "and",
                core.cmp(self.b, "ge", position, self._word(0)),
                core.cmp(self.b, "lt", position, length),
            )
            self._require(
                inside,
                "index out of range",
                "IndexError: index {0} is out of range for length {1}",
                (position, length),
            )
            address = self._rt("ppy_seq_at", (handle, position), HANDLE)
        elif kind.name in {"HashMap", "TreeMap"}:
            key = self._key(kind, index)
            if value is not None:
                entry = self._rt(f"ppy_{kind.family}_put", (handle, key))
            else:
                entry = self._rt(f"ppy_{kind.family}_find", (handle, key))
                spelled, words = self._key_report(kind, key)
                self._require(
                    self._found(entry),
                    "key not found",
                    f"KeyError: {spelled}" if spelled else "KeyError",
                    words,
                )
            self._keys_done()
            address = self._rt(f"ppy_{kind.family}_value_at", (handle, entry), HANDLE)
        else:
            raise Unsupported(f"a {kind.name} has no index")
        if value is not None:
            self._store_into(address, shape, value, fresh=False)
            self._done_with(handle, owned)
            return self._word(0)
        found = self._read(address, shape)
        if owned:
            if shape.reference:
                raise Unsupported("an element read from a temporary collection outlives it")
            self._release(handle)
        return found

    def _collection_method(self, receiver: ast.expr, attr: str, node: ast.Call) -> Value:
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
        if kind.name == "List":
            method = self._string_list_method  # type: ignore[attr-defined]
        found = method(kind, handle, attr, node.args)
        self._keys_done()
        if found is None:
            raise Unsupported(f"`{kind.name}.{attr}` has no native lowering")
        self._done_with(handle, owned)
        return found

    def _nonempty(self, kind: Kind, handle: Value, what: str) -> None:
        length = self._rt("ppy_coll_len", (handle,))
        # The reference's words: a removal is "from", a look is "of" or "at".
        joined = "from" if what.startswith("pop") else "at" if what == "peek" else "of"
        self._require(
            core.cmp(self.b, "gt", length, self._word(0)),
            f"{what} of an empty {kind.name}",
            f"IndexError: {what} {joined} an empty {kind.name}",
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
            if not self._orders(shape):
                raise Unsupported(f"a {kind.name} orders its elements, and these have no order")
            self._store_into(
                self._rt("ppy_coll_scratch", (handle,), HANDLE), shape, arguments[0], fresh=True
            )
            return self._rt("ppy_heap_push", (handle, self._word(heap)), None)
        if attr == "clear":
            return self._rt("ppy_seq_clear", (handle,), None)
        if attr == "sort" and kind.name == "Vec":
            if not self._orders(shape):
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
        self._require(
            core.cmp(self.b, "ne", valid, self._word(0)),
            "a node not in the list",
            "IndexError: node {0} is not in the list",
            (node,),
        )
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
            spelled, words = self._key_report(kind, key)
            self._require(
                self._found(node),
                f"no key for `{attr}`",
                f"KeyError: 'no key {_BOUND_WORDS[attr]} {spelled}'",
                words,
            )
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
            spelled, words = self._key_report(kind, key)
            self._require(
                self._found(removed),
                "key not found",
                f"KeyError: {spelled}" if spelled else "KeyError",
                words,
            )
            if attr == "pop" and shape is not None:
                return self._read(
                    self._rt(f"ppy_{family}_value_at", (handle, removed), HANDLE), shape
                )
            return self._word(0)
        if attr == "discard":
            return self._rt(f"ppy_{family}_remove", (handle, key))
        return None

    # -- loops -------------------------------------------------------------------

    def _bind_element(self, target: ast.expr, shape: Shape, value: Value) -> None:
        """A loop target, or a name, bound to an element read in place."""
        if shape.reference:
            if not isinstance(target, ast.Name):
                raise Unsupported("a collection element is bound to a name")
            held = shape.collection if shape.kind == "collection" else shape
            assert held is not None
            self._bind(target.id, held, value, owned=False)
            return
        if shape.kind == "tuple":
            items = [core.tuple_extract(self.b, value, i) for i in range(shape.words)]
            self._store_tuple(target, items)  # type: ignore[attr-defined]
            return
        self._store(target, value)  # type: ignore[attr-defined]


#: A comparison's method, and the one Python tries on the right operand when
#: the left has none: `a > b` is `b < a`.
_COMPARE_DUNDERS = {
    ast.Eq: ("__eq__", "__eq__"),
    ast.NotEq: ("__ne__", "__ne__"),
    ast.Lt: ("__lt__", "__gt__"),
    ast.Gt: ("__gt__", "__lt__"),
    ast.LtE: ("__le__", "__ge__"),
    ast.GtE: ("__ge__", "__le__"),
}

#: What CPython says when an operator's object is `None`, where it is not the
#: plain missing attribute.
_NONE_OPERAND = {
    "__getitem__": "TypeError: 'NoneType' object is not subscriptable",
    "__contains__": "TypeError: argument of type 'NoneType' is not iterable",
    "__len__": "TypeError: object of type 'NoneType' has no len()",
}

#: An arithmetic operator's method, and its reflected form.
_BINARY_DUNDERS = {
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

#: The methods whose collection result the caller owns: taken out, not read in place.
_OWNED_RESULTS = {
    "pop": True,
    "pop_front": True,
    "pop_back": True,
    "remove": True,
    "pop_min": True,
    "pop_max": True,
}


def _copies_before_writes(function: ast.AST, record: str, type_of) -> bool:  # type: ignore[no-untyped-def]
    """Does `function` hold a copy of a `record` value (a name bound to one that
    is not built on the spot, a loop target, a parameter) where a later
    in-place field write, or one in the same loop, could change what CPython's
    shared object says?"""
    copies: list[tuple[ast.AST, list[ast.AST]]] = []
    writes: list[tuple[ast.AST, list[ast.AST]]] = []

    def named(t: T.Type) -> bool:
        base = T.strip_literal(t)
        return isinstance(base, T.Instance) and base.name == record

    def element_write(target: ast.expr) -> bool:
        return (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, (ast.Subscript, ast.Attribute))
            and named(type_of(target.value))
        )

    def visit(node: ast.AST, loops: list[ast.AST]) -> None:
        inner = [*loops, node] if isinstance(node, (ast.For, ast.While)) else loops
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if isinstance(target, ast.Name) and named(type_of(target)):
                    built = isinstance(value, ast.Call) and not isinstance(
                        value.func, ast.Attribute
                    )
                    if not built:
                        copies.append((node, loops))
                if element_write(target):
                    writes.append((node, loops))
        if isinstance(node, ast.AugAssign) and element_write(node.target):
            writes.append((node, loops))
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and named(type_of(node.target))
        ):
            copies.append((node, inner))
        if isinstance(node, ast.arguments) and any(
            argument.annotation is not None and named(type_of(argument)) for argument in node.args
        ):
            copies.append((node, []))
        for child in ast.iter_child_nodes(node):
            visit(child, inner)

    visit(function, [])
    for copy, copy_loops in copies:
        for write, write_loops in writes:
            if (write.lineno, write.col_offset) > (  # type: ignore[attr-defined]
                getattr(copy, "lineno", 0),
                getattr(copy, "col_offset", 0),
            ):
                return True
            if any(loop in write_loops for loop in copy_loops):
                return True
    return False


def _related(held: Kind | Shape, kind: Kind | Shape) -> bool:
    """Two object classes: a name holds either by the same handle, and what the
    checker says of each use decides which class it is read as."""
    return (
        isinstance(held, Shape) and isinstance(kind, Shape) and held.kind == kind.kind == "object"
    )


def _kinds(shape: Shape) -> tuple[str, ...]:
    return shape.parts if shape.kind in {"tuple", "record"} else (shape.kind,)


#: The runtime calls that may call a class's `__lt__`, `__hash__`, or `__eq__`.
_CALLS_BACK = (
    "ppy_heap_",
    "ppy_tree_",
    "ppy_map_",
    "ppy_set_",
    "ppy_seq_sort",
    "ppy_coll_find_key",
    "ppy_coll_put_key",
    "ppy_coll_update",
    "ppy_coll_equal",
)

#: The methods of a class the runtime calls back.
_COMPARES = frozenset({"__lt__", "__hash__", "__eq__"})


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


@dataclass(frozen=True, slots=True)
class _Called:
    """What `_dispatch` hands back in place of a call: its results."""

    results: tuple[Value, ...]


#: The header word an object's class tag is kept in: family word b, which a
#: sequence (an object's record lives in a one-element sequence) leaves unused.
_TAG = 4


def class_tag(qualname: str) -> int:
    """A class's tag: the same number wherever the class is compiled."""
    digest = hashlib.sha256(qualname.encode()).digest()
    return int.from_bytes(digest[:7], "little") | 1


@dataclass(frozen=True, slots=True)
class _Parameters:
    """A signature's parameters after the receiver, for matching call arguments."""

    parameters: tuple[object, ...]


def _field_defaults(node: ast.ClassDef) -> dict[str, ast.expr]:
    """A dataclass's fields with a constant default (`None` included), written
    out or as `field(default=...)`: the rest have to be given, or made by a
    factory, when native code builds one."""
    found: dict[str, ast.expr] = {}
    for item in node.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
            continue
        if isinstance(item.value, ast.Constant):
            found[item.target.id] = item.value
        default = _field_keyword(item.value, "default")
        if isinstance(default, ast.Constant):
            found[item.target.id] = default
    return found


def _field_factories(node: ast.ClassDef) -> dict[str, ast.expr]:
    """A dataclass's fields made per instance: `field(default_factory=Vec[int])`."""
    found: dict[str, ast.expr] = {}
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            factory = _field_keyword(item.value, "default_factory")
            if factory is not None:
                found[item.target.id] = factory
    return found


def _field_keyword(value: ast.expr | None, keyword: str) -> ast.expr | None:
    """`field(keyword=...)`'s value, where `value` is a call of `dataclasses.field`."""
    if not isinstance(value, ast.Call) or value.args:
        return None
    if ast.unparse(value.func) not in {"field", "dataclasses.field"}:
        return None
    for item in value.keywords:
        if item.arg == keyword:
            return item.value
    return None
