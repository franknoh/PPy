"""Python's own `list`, `dict`, and `set`, lowered over the collections runtime.

A `list[T]` is a sequence of the runtime's words, a `dict[K, V]` a hash map
and a `set[K]` a hash set: the same memory, reference counting, and cycle
collection a `ppy.Vec`, `ppy.HashMap`, and `ppy.HashSet` have. What is here
is where Python's containers mean something else than those:

- a list is indexed and inserted into from either end, and says what
  CPython says when an index is out of range;
- displays (`[a, b]`, `{k: v}`, `{a, b}`), `[x] * n`, comprehensions, and
  the constructors `list(...)`, `dict(...)`, `set(...)` make them;
- a dict's and a set's methods are the hash map's and the hash set's;
- a set is walked in the order CPython's hash table decides, which native
  memory does not keep: a loop over a set, or anything that depends on the
  order it walks in, stays in Python. `sorted`, `min`, `max`, `len`, `in`,
  and the set algebra do not depend on it.

A container parameter or result crosses between native functions by handle.
"""

from __future__ import annotations

import ast

from ..analysis import types as T
from ..backend.llvm.lowering import Unsupported
from ..ir import BOOL, F64, I64, BufferType, PtrType, Successor, Value
from ..ir.dialects import core
from .collection_api import CollectionApiLowering, _called
from .collections import BUILTINS, HANDLE, Kind, Shape

__all__ = ["ContainerLowering"]

#: The runtime collection whose methods each of Python's containers shares.
_ALIASES = {"List": "Vec", "Dict": "HashMap", "Set": "HashSet"}

#: What `list(...)`, `dict(...)`, `set(...)` construct.
_CONSTRUCTORS = {"list": "List", "dict": "Dict", "set": "Set"}

#: The list methods whose element result the caller owns.
_TAKEN = frozenset({"pop"})


class ContainerLowering(CollectionApiLowering):
    """`list`, `dict`, and `set` over the runtime; mixed into `_FunctionLowering`."""

    # -- what an expression is ------------------------------------------------------

    def _builtin_of(self, node: ast.expr) -> Kind | None:
        kind = self._kind_of(node)
        return kind if kind is not None and kind.name in _ALIASES else None

    def _alias(self, kind: Kind) -> Kind:
        """The runtime collection a container shares its methods with."""
        return Kind(_ALIASES[kind.name], kind.value, kind.key)

    # -- making ------------------------------------------------------------------------

    def _make_collection(self, name: str, value: ast.expr, declared: T.Type | None = None) -> bool:
        from .ast_to_ir import (  # pylint: disable=import-outside-toplevel
            _ALLOCATIONS,
            _line_buffer_read,
        )

        if isinstance(value, ast.Call) and (
            _line_buffer_read(value) is not None or ast.unparse(value.func) in _ALLOCATIONS
        ):
            # `xs = ppy.input[list[int]]()` in a standalone build is a buffer.
            return False
        # `xs: list[int] = []`: an empty display says nothing of what it will
        # hold, and the name's type does.
        expected = (
            self._builtin_of(ast.Name(name, ast.Load())) if name in self.collections else None
        )
        if expected is None and declared is not None:
            found = self._reference_of_type(declared)
            expected = found if isinstance(found, Kind) and found.name in _ALIASES else None
        if expected is None:
            expected = self._builtin_of(value)
        if expected is not None:
            made = self._made(expected, value)
            if made is not None:
                self._bind(name, expected, made, owned=True)
                return True
        return super()._make_collection(name, value, declared)

    def _handle(self, node: ast.expr) -> tuple[Value, bool]:
        kind = self._builtin_of(node)
        if kind is not None:
            made = self._made(kind, node)
            if made is not None:
                return made, True
        return super()._handle(node)

    def _value(self, node: ast.expr, shape: Shape) -> tuple[Value, bool]:
        # `d[k] = []`, `xs.append({})`: the empty display takes the slot's type.
        collection = shape.collection if shape.kind == "collection" else None
        if collection is not None and collection.name in _ALIASES:
            made = self._made(collection, node)
            if made is not None:
                return made, True
        return super()._value(node, shape)

    def _made(self, kind: Kind, node: ast.expr) -> Value | None:
        """A new container from a display, a comprehension, `[x] * n`, or a
        constructor; None where `node` is not one of those."""
        if isinstance(node, (ast.List, ast.Set)) and kind.name in {"List", "Set"}:
            made = self._new(kind)
            self._fill(kind, made, node)
            return made
        if isinstance(node, ast.Dict) and kind.name == "Dict":
            made = self._new(kind)
            for key, value in zip(node.keys, node.values, strict=True):
                if key is None:
                    raise Unsupported("`**` in a dict display has no native lowering")
                self._put(kind, made, key, value)
            return made
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            return self._comprehension(kind, node)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and kind.name == "List":
            return self._repeated(kind, node)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and _CONSTRUCTORS.get(node.func.id) == kind.name
        ):
            return self._constructed_builtin(kind, node)
        if _called(node, "sorted") and kind.name == "List":
            assert isinstance(node, ast.Call)
            return self._take(self._sorted_source(node))
        return None

    def _take(self, source) -> Value:  # type: ignore[no-untyped-def]
        """A new collection a walk made (`sorted`), handed over owned."""
        handle = core.load(self.b, source.slot)
        # The walk's slot holds the only reference: take it rather than let it go.
        core.store(self.b, self._rt("ppy_coll_none", (), HANDLE), source.slot)
        return handle

    def _constructed_builtin(self, kind: Kind, node: ast.Call) -> Value:
        """`list()`, `list(xs)`, `set(xs)`, `dict()`, `dict(d)`."""
        if node.keywords or len(node.args) > 1:
            raise Unsupported(f"`{ast.unparse(node.func)}` takes one iterable natively")
        made = self._new(kind)
        if not node.args:
            return made
        source = node.args[0]
        if kind.name == "Dict":
            other = self._builtin_of(source)
            if other is None or other.name != "Dict":
                raise Unsupported("`dict(...)` copies another dict natively")
            handle, owned = self._handle(source)
            self._rt("ppy_coll_update", (made, handle), None)
            self._done_with(handle, owned)
            return made
        self._refuse_set_order(source, kind)
        self._fill(kind, made, source)
        return made

    def _repeated(self, kind: Kind, node: ast.BinOp) -> Value:
        """`[x] * n` and `n * [x]`: `n` copies of what the display holds, in order."""
        left_list = self._builtin_of(node.left) is not None
        display, count_node = (node.left, node.right) if left_list else (node.right, node.left)
        if not isinstance(display, ast.List):
            raise Unsupported("a list is repeated natively when it is a display")
        shape = kind.value
        assert shape is not None
        count = self._coerce(self._expr(count_node), "int")  # type: ignore[attr-defined]
        values = [self._value(element, shape) for element in display.elts]
        made = self._new(kind)
        index = self._alloca(I64, "repeat.i")  # type: ignore[attr-defined]
        core.store(self.b, self._word(0), index)
        header = self._block("repeat.head")  # type: ignore[attr-defined]
        body = self._block("repeat.body")  # type: ignore[attr-defined]
        done = self._block("repeat.end")  # type: ignore[attr-defined]
        core.br(self.b, Successor(header))
        self.b.at_end(header)  # type: ignore[attr-defined]
        current = core.load(self.b, index)
        core.cond_br(
            self.b, core.cmp(self.b, "lt", current, count), Successor(body), Successor(done)
        )
        self.b.at_end(body)  # type: ignore[attr-defined]
        for value, _owned in values:
            # Each slot shares the one object, as `[obj] * n` does in Python.
            self._store_value(self._room(kind, made), shape, value, owned=False)
        core.store(self.b, core.add(self.b, current, self._word(1), overflow="wrap"), index)
        core.br(self.b, Successor(header))
        self.b.at_end(done)  # type: ignore[attr-defined]
        for value, owned in values:
            if owned and shape.reference:
                self._release(value)
        return made

    def _put(self, kind: Kind, handle: Value, key: ast.expr, value: ast.expr) -> None:
        """`d[key] = value` into a dict being made or written."""
        shape = kind.value
        assert shape is not None
        address = self._element_address(self._alias(kind), handle, key, write=True)
        self._store_into(address, shape, value, fresh=False)

    # -- comprehensions ----------------------------------------------------------------

    def _comprehension(self, kind: Kind, node: ast.expr) -> Value:
        """`[e for x in xs if c]`, `{e for ...}`, `{k: v for ...}`: the loops
        Python runs, into a new container. The loop variables are the
        comprehension's own and do not outlive it."""
        assert isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp))
        made = self._new(kind)
        name = f"__ppy_comprehension{id(node)}"
        self._bind(name, kind, made, owned=True)
        if isinstance(node, ast.DictComp):
            target = ast.Subscript(
                value=ast.Name(name, ast.Load()), slice=node.key, ctx=ast.Store()
            )
            innermost: ast.stmt = ast.Assign(targets=[target], value=node.value)
        else:
            method = "append" if kind.name == "List" else "add"
            call = ast.Call(
                func=ast.Attribute(ast.Name(name, ast.Load()), method, ast.Load()),
                args=[node.elt],
                keywords=[],
            )
            innermost = ast.Expr(call)
        statement = innermost
        names: set[str] = set()
        for generator in reversed(node.generators):
            if generator.is_async:
                raise Unsupported("an async comprehension has no native lowering")
            self._refuse_set_order(generator.iter, None)
            for condition in reversed(generator.ifs):
                statement = ast.If(test=condition, body=[statement], orelse=[])
            statement = ast.For(
                target=generator.target, iter=generator.iter, body=[statement], orelse=[]
            )
            names |= {n.id for n in ast.walk(generator.target) if isinstance(n, ast.Name)}
        # Python keeps a comprehension's names apart from the function's: the
        # function's own are set aside while it runs and back after, untouched.
        slots = self.slots  # type: ignore[attr-defined]
        outer_slots = {n: slots.pop(n) for n in names if n in slots}
        outer_held = {n: self.collections.pop(n) for n in names if n in self.collections}
        for fresh in (statement, *ast.walk(statement)):
            ast.copy_location(fresh, node)
        self._statement(statement)  # type: ignore[attr-defined]
        for spelled in names:
            slots.pop(spelled, None)
            held = self.collections.pop(spelled, None)
            if held is not None:
                self._release(core.load(self.b, held.slot))
        slots.update(outer_slots)
        self.collections.update(outer_held)
        held = self.collections.pop(name)
        return core.load(self.b, held.slot)

    # -- order ---------------------------------------------------------------------------

    def _refuse_set_order(self, node: ast.expr, into: Kind | None) -> None:
        """A set walked where the order it walks in shows: that is CPython's hash
        table's, which native memory does not keep. Into another set it does not."""
        walked = self._builtin_of(node)
        if walked is not None and walked.name == "Set" and (into is None or into.name != "Set"):
            raise Unsupported("a set is walked in CPython's hash order, which stays in Python")

    def _for_collection(self, node: ast.For) -> None:
        iterables = [node.iter]
        if any(_called(node.iter, name) for name in ("enumerate", "zip", "reversed")):
            assert isinstance(node.iter, ast.Call)
            iterables = list(node.iter.args)
        for iterable in iterables:
            self._refuse_set_order(iterable, None)
        super()._for_collection(node)

    # -- repr --------------------------------------------------------------------------

    def _printed_list(self, node: ast.expr) -> tuple[Value, bool] | None:
        empty = _empty_display(node)
        if empty is not None:
            builder = self._rt("ppy_str_builder", (self._word(0),), HANDLE)
            self._add_text(builder, empty)
            return self._rt("ppy_str_finish", (builder,), HANDLE), True
        kind = self._builtin_of(node)
        if kind is None or kind.name == "Set":
            return super()._printed_list(node)  # type: ignore[misc]
        builder = self._rt("ppy_str_builder", (self._word(0),), HANDLE)
        handle, owned = self._handle(node)
        self._add_container_repr(builder, kind, handle)
        self._done_with(handle, owned)
        return self._rt("ppy_str_finish", (builder,), HANDLE), True

    def _add_formatted(self, builder: Value, node: ast.expr, conversion: int, spec: str) -> None:
        empty = _empty_display(node)
        if empty is not None and not spec:
            self._add_text(builder, empty)
            return None
        kind = self._builtin_of(node)
        if kind is None:
            return super()._add_formatted(builder, node, conversion, spec)  # type: ignore[misc]
        if spec:
            raise Unsupported("a format spec over a container has no native lowering")
        if kind.name == "Set":
            raise Unsupported("a set is shown in CPython's hash order, which stays in Python")
        handle, owned = self._handle(node)
        self._add_container_repr(builder, kind, handle)
        self._done_with(handle, owned)
        return None

    def _add_text(self, builder: Value, text: str) -> None:
        data, length = self._text_data(text)  # type: ignore[attr-defined]
        self._rt("ppy_str_add_bytes", (builder, data, length), None)

    def _add_container_repr(self, builder: Value, kind: Kind, handle: Value) -> None:
        """`[1, 'a']` and `{1: [2.5]}`, as `repr` writes them."""
        from .collection_api import _Source  # pylint: disable=import-outside-toplevel

        if kind.name == "Set":
            raise Unsupported("a set is shown in CPython's hash order, which stays in Python")
        opening, closing = ("[", "]") if kind.name == "List" else ("{", "}")
        self._add_text(builder, opening)
        first = self._alloca(BOOL, "repr.first")  # type: ignore[attr-defined]
        core.store(self.b, core.const(self.b, True, BOOL), first)
        slot = self._hold(kind, handle, owned=False)
        source = _Source(kind, slot, "items" if kind.name == "Dict" else "elements")

        def visit(items: list[tuple[Shape, Value]]) -> None:
            later = self._block("repr.comma")  # type: ignore[attr-defined]
            joined = self._block("repr.item")  # type: ignore[attr-defined]
            core.cond_br(self.b, core.load(self.b, first), Successor(joined), Successor(later))
            self.b.at_end(later)  # type: ignore[attr-defined]
            self._add_text(builder, ", ")
            core.br(self.b, Successor(joined))
            self.b.at_end(joined)  # type: ignore[attr-defined]
            core.store(self.b, core.const(self.b, False, BOOL), first)
            if kind.name == "Dict":
                (key_shape, key), (value_shape, value) = items[0], items[1]
                self._add_item_repr(builder, key_shape, key)
                self._add_text(builder, ": ")
                self._add_item_repr(builder, value_shape, value)
            else:
                shape, value = items[0]
                self._add_item_repr(builder, shape, value)

        self._walk(source, visit)
        self._add_text(builder, closing)

    def _add_item_repr(self, builder: Value, shape: Shape, value: Value) -> None:
        """One element's `repr`."""
        if shape.kind == "int":
            self._rt("ppy_str_add_int", (builder, value), None)
        elif shape.kind == "float":
            self._rt("ppy_str_add_float", (builder, value), None)
        elif shape.kind == "bool":
            self._rt("ppy_str_add_bool", (builder, core.cast(self.b, value, I64)), None)
        elif shape.kind == "str":
            self._add_repr(builder, value)  # type: ignore[attr-defined]
        elif shape.kind == "tuple":
            self._add_text(builder, "(")
            for index, part in enumerate(shape.parts):
                if index:
                    self._add_text(builder, ", ")
                item = core.tuple_extract(self.b, value, index)
                self._add_item_repr(builder, Shape(part), item)
            self._add_text(builder, ",)" if len(shape.parts) == 1 else ")")
        elif shape.kind == "collection" and shape.collection is not None:
            if shape.collection.name not in _ALIASES:
                raise Unsupported("a `ppy` collection inside a list is shown by Python")
            self._add_container_repr(builder, shape.collection, value)
        elif shape.kind == "record" and self._plain_dataclass(shape.record):
            name = shape.record.rpartition(".")[2]
            self._add_text(builder, f"{name}(")
            for index, (field, part) in enumerate(zip(shape.names, shape.parts, strict=True)):
                self._add_text(builder, (", " if index else "") + f"{field}=")
                item = core.struct_extract(self.b, value, field)
                self._add_item_repr(builder, Shape(part), item)
            self._add_text(builder, ")")
        else:
            raise Unsupported(f"a `{shape.kind}` element is shown by Python")

    def _plain_dataclass(self, qualname: str) -> bool:
        """A dataclass whose `repr` is the generated one: `Point(x=1, y=2)`."""
        from ..analysis.symbols import dataclass_keyword  # pylint: disable=import-outside-toplevel

        info = self._class_named(qualname)
        return (
            info.is_dataclass
            and "__repr__" not in info.methods
            and dataclass_keyword(info.node, "repr") is not False
        )

    # -- any and all ----------------------------------------------------------------

    def _any_all(self, name: str, node: ast.Call) -> Value | None:
        """`any(c)` and `all(c)` of a collection of numbers, bools, or strings:
        whether one element is true, or every one is. The answer does not depend
        on the order the walk goes in, so a set is walked for it too."""
        if len(node.args) != 1 or node.keywords or not self._is_walk(node.args[0]):
            return None
        source = self._source(node.args[0])
        if source is None or source.mode == "items":
            return None
        shape = self._part(source, "values" if source.mode == "values" else "keys")
        if shape.kind not in {"int", "float", "bool", "str"}:
            return None
        seeking = name == "any"
        answer = self._alloca(BOOL, f"{name}.answer")  # type: ignore[attr-defined]
        core.store(self.b, core.const(self.b, not seeking, BOOL), answer)

        def look(items: list[tuple[Shape, Value]]) -> None:
            value = items[0][1]
            if shape.kind == "str":
                true = core.cmp(self.b, "gt", self._rt("ppy_str_bytes", (value,)), self._word(0))
            else:
                true = self._truth(value)  # type: ignore[attr-defined]
            if seeking:
                found = core.bitwise(self.b, "or", core.load(self.b, answer), true)
            else:
                found = core.bitwise(self.b, "and", core.load(self.b, answer), true)
            core.store(self.b, found, answer)

        self._walk(source, look)
        return core.load(self.b, answer)

    # -- lending ------------------------------------------------------------------------

    def _list_view(self, node: ast.expr, element: str) -> Value | None:
        """A list of numbers lent to a callee that takes a buffer: its words, in
        place, for the length of the call. A list's elements are one run of
        words from its start, which is what a buffer is."""
        kind = self._builtin_of(node)
        if kind is None or kind.name != "List" or kind.value is None:
            return None
        if kind.value.kind != element:
            raise Unsupported(f"a list of `{kind.value.kind}` is lent where `{element}` is taken")
        handle, owned = self._handle(node)
        if owned:
            self._temporaries.append(handle)  # type: ignore[attr-defined]
        scalar = F64 if element == "float" else I64
        start = self._rt("ppy_seq_at", (handle, self._word(0)), HANDLE)
        data = core.cast(self.b, start, PtrType(scalar))
        length = self._rt("ppy_coll_len", (handle,))
        return self.b.create(
            "core.call_intrinsic",
            (data, length),
            (BufferType(scalar),),
            {"intrinsic": "ppy.buffer_from_parts"},
        ).results[0]

    # -- methods -------------------------------------------------------------------------

    def _collection_method(self, receiver: ast.expr, attr: str, node: ast.Call) -> Value:
        kind = self._builtin_of(receiver)
        if kind is None:
            return super()._collection_method(receiver, attr, node)
        kind, handle, owned = self._receiver(receiver)
        if kind.name == "List":
            found = self._list_method_of(kind, handle, attr, node)
        else:
            alias = self._alias(kind)
            if attr in {"union", "intersection", "difference", "symmetric_difference"}:
                for argument in node.args:
                    self._refuse_set_order(argument, kind)
            if attr == "update" and kind.name == "Set":
                for argument in node.args:
                    self._refuse_set_order(argument, kind)
            found = self._whole_method(alias, handle, attr, node)
            if found is None and not node.keywords:
                found = self._keyed_method(alias, handle, attr, node.args)
        self._keys_done()
        if found is None:
            raise Unsupported(f"`{BUILTIN_NAMES[kind.name]}.{attr}` has no native lowering")
        self._done_with(handle, owned)
        return found

    def _list_method_of(self, kind: Kind, handle: Value, attr: str, node: ast.Call) -> Value | None:
        """A list's methods, where a list counts from either end."""
        shape = kind.value
        assert shape is not None
        arguments = node.args
        b = self.b
        if node.keywords and attr != "sort":
            return None
        if attr == "append" and len(arguments) == 1:
            self._add_node(kind, handle, arguments[0])
            return self._word(0)
        if attr == "extend" and len(arguments) == 1:
            self._refuse_set_order(arguments[0], kind)
            self._fill(kind, handle, arguments[0])
            return self._word(0)
        if attr == "insert" and len(arguments) == 2:
            # `insert(i, x)` clamps: past either end is that end.
            length = self._rt("ppy_coll_len", (handle,))
            given = self._coerce(self._expr(arguments[0]), "int")  # type: ignore[attr-defined]
            counted = self._list_position(handle, given)  # type: ignore[attr-defined]
            low = core.select(b, core.cmp(b, "lt", counted, self._word(0)), self._word(0), counted)
            position = core.select(b, core.cmp(b, "gt", low, length), length, low)
            self._store_node(
                lambda: self._rt("ppy_seq_insert", (handle, position), HANDLE),
                shape,
                arguments[1],
                fresh=True,
            )
            return self._word(0)
        if attr == "pop" and len(arguments) <= 1:
            length = self._rt("ppy_coll_len", (handle,))
            self._require(
                core.cmp(b, "gt", length, self._word(0)),
                "pop from empty list",
                "IndexError: pop from empty list",
            )
            if not arguments:
                return self._read(self._rt("ppy_seq_pop_back", (handle,), HANDLE), shape)
            given = self._coerce(self._expr(arguments[0]), "int")  # type: ignore[attr-defined]
            position = self._list_position(handle, given)  # type: ignore[attr-defined]
            inside = core.bitwise(
                b,
                "and",
                core.cmp(b, "ge", position, self._word(0)),
                core.cmp(b, "lt", position, length),
            )
            self._require(inside, "pop index out of range", "IndexError: pop index out of range")
            return self._read(self._rt("ppy_seq_erase", (handle, position), HANDLE), shape)
        if attr in {"remove", "index", "count"} and len(arguments) == 1:
            return self._search(kind, handle, attr, arguments[0])
        if attr == "sort":
            return self._sort(self._alias(kind), handle, node)
        if attr in {"reverse", "clear"} and not arguments:
            return self._rt(f"ppy_seq_{attr}", (handle,), None)
        if attr == "copy" and not arguments:
            return self._rt("ppy_coll_copy", (handle,), HANDLE)
        return None

    def _search(self, kind: Kind, handle: Value, attr: str, needle: ast.expr) -> Value:
        """`index`, `count`, `remove` of a list, with CPython's messages."""
        shape = kind.value
        assert shape is not None
        address, made = self._value_buffer(shape, needle)
        if attr == "count":
            found = self._decided(self._rt("ppy_coll_count_value", (handle, address)), -1)
        else:
            found = self._rt("ppy_coll_find_value", (handle, address, self._word(0)))
            found = self._decided(found, -2)
            if attr == "remove":
                missing, reported = "ValueError: list.remove(x): x not in list", ()
            elif shape.kind == "int":
                missing = "ValueError: {0} is not in list"
                reported = (self._read_word(address, 0, "int"),)
            else:
                missing, reported = "ValueError: the value is not in list", ()
            self._require(self._found(found), f"list.{attr}(x): x not in list", missing, reported)
        if made is not None:
            self._release(made)
        if attr != "remove":
            return found
        taken = self._rt("ppy_seq_erase", (handle, found), HANDLE)
        if shape.reference:
            self._rt("ppy_coll_release_words", (handle, taken), None)
        return self._word(0)

    # -- items ---------------------------------------------------------------------------

    def _item(self, container: ast.expr, index: ast.expr, value: ast.expr | None = None) -> Value:
        kind = self._builtin_of(container)
        if kind is None or isinstance(index, ast.Slice):
            return super()._item(container, index, value)
        kind, handle, owned = self._receiver(container)
        shape = kind.value
        if kind.name == "Set" or shape is None:
            raise Unsupported("a set has no index")
        # `xs[i] = e`, `d[k] = e`: `e` first, as Python evaluates it.
        stored = self._value(value, shape) if value is not None else None
        if kind.name == "Dict":
            address = self._element_address(
                self._alias(kind), handle, index, write=value is not None
            )
        else:
            position = self._coerce(self._expr(index), "int")  # type: ignore[attr-defined]
            position = self._list_position(handle, position)  # type: ignore[attr-defined]
            length = self._rt("ppy_coll_len", (handle,))
            inside = core.bitwise(
                self.b,
                "and",
                core.cmp(self.b, "ge", position, self._word(0)),
                core.cmp(self.b, "lt", position, length),
            )
            what = "list assignment index" if value is not None else "list index"
            self._require(inside, f"{what} out of range", f"IndexError: {what} out of range")
            address = self._rt("ppy_seq_at", (handle, position), HANDLE)
        if stored is not None:
            self._put_value(address, shape, stored[0], stored[1], fresh=False)
            self._done_with(handle, owned)
            return self._word(0)
        found = self._read(address, shape)
        if owned:
            if shape.reference:
                raise Unsupported("an element read from a temporary collection outlives it")
            self._release(handle)
        return found

    def _element_address(self, kind: Kind, handle: Value, index: ast.expr, *, write: bool) -> Value:
        if kind.name == "Dict":
            return super()._element_address(self._alias(kind), handle, index, write=write)
        if kind.name != "List":
            return super()._element_address(kind, handle, index, write=write)
        position = self._coerce(self._expr(index), "int")  # type: ignore[attr-defined]
        position = self._list_position(handle, position)  # type: ignore[attr-defined]
        length = self._rt("ppy_coll_len", (handle,))
        inside = core.bitwise(
            self.b,
            "and",
            core.cmp(self.b, "ge", position, self._word(0)),
            core.cmp(self.b, "lt", position, length),
        )
        what = "list assignment index" if write else "list index"
        self._require(inside, f"{what} out of range", f"IndexError: {what} out of range")
        return self._rt("ppy_seq_at", (handle, position), HANDLE)

    def _delete(self, node: ast.Delete) -> None:
        """`del xs[i]` and `del d[k]`."""
        for target in node.targets:
            if not isinstance(target, ast.Subscript) or isinstance(target.slice, ast.Slice):
                raise Unsupported("`del` of this target has no native lowering")
            kind = self._builtin_of(target.value)
            if kind is None or kind.name == "Set":
                raise Unsupported("`del` of this target has no native lowering")
            kind, handle, owned = self._receiver(target.value)
            if kind.name == "Dict":
                self._keyed_method(self._alias(kind), handle, "pop", [target.slice])
                self._keys_done()
            else:
                shape = kind.value
                assert shape is not None
                length = self._rt("ppy_coll_len", (handle,))
                given = self._coerce(self._expr(target.slice), "int")  # type: ignore[attr-defined]
                position = self._list_position(handle, given)  # type: ignore[attr-defined]
                inside = core.bitwise(
                    self.b,
                    "and",
                    core.cmp(self.b, "ge", position, self._word(0)),
                    core.cmp(self.b, "lt", position, length),
                )
                self._require(
                    inside,
                    "list assignment index out of range",
                    "IndexError: list assignment index out of range",
                )
                taken = self._rt("ppy_seq_erase", (handle, position), HANDLE)
                if shape.reference:
                    self._rt("ppy_coll_release_words", (handle, taken), None)
            self._done_with(handle, owned)

    # -- operators -----------------------------------------------------------------------

    def _combined(self, node: ast.BinOp) -> Value:
        kind = self._builtin_of(node.left)
        if kind is None:
            return super()._combined(node)
        if self._kind_of(node.right) != kind:
            raise Unsupported("a container combines with another of its own type")
        if kind.name == "List" and isinstance(node.op, ast.Mult):
            raise Unsupported("a list is repeated natively when it is a display")
        if kind.name == "Dict":
            raise Unsupported(f"`{ast.unparse(node)}` has no native lowering")
        left, left_owned = self._handle(node.left)
        right, right_owned = self._handle(node.right)
        alias = self._alias(kind)
        if kind.name == "List" and isinstance(node.op, ast.Add):
            made = self._rt("ppy_seq_concat", (left, right), HANDLE)
        else:
            from .collection_api import _OPERATORS  # pylint: disable=import-outside-toplevel

            if type(node.op) not in _OPERATORS:
                raise Unsupported(f"`{ast.unparse(node)}` has no native lowering")
            operation = self._word(_OPERATORS[type(node.op)])
            made = self._rt("ppy_set_combine", (left, right, operation), HANDLE)
        del alias
        self._done_with(left, left_owned)
        self._done_with(right, right_owned)
        return made

    def _slice(self, container: ast.expr, bounds: ast.Slice) -> Value:
        kind = self._builtin_of(container)
        if kind is None or kind.name != "List":
            return super()._slice(container, bounds)
        # A list's slice is a `Vec`'s, clamped and counted from either end.
        return self._slice_words(container, bounds)

    def _slice_words(self, container: ast.expr, bounds: ast.Slice) -> Value:
        kind, handle, owned = self._receiver(container)
        del kind
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

    def _contains(self, container: ast.expr, key: ast.expr) -> Value:
        kind = self._builtin_of(container)
        if kind is None:
            return super()._contains(container, key)
        if kind.name != "List":
            return super()._contains(container, key)
        return self._search_in(kind, container, key)

    def _search_in(self, kind: Kind, container: ast.expr, needle: ast.expr) -> Value:
        """`x in xs` of a list: a search from the front."""
        shape = kind.value
        assert shape is not None
        kind, handle, owned = self._receiver(container)
        address, made = self._value_buffer(shape, needle)
        found = self._rt("ppy_coll_find_value", (handle, address, self._word(0)))
        found = self._decided(found, -2)
        if made is not None:
            self._release(made)
        self._done_with(handle, owned)
        return core.cmp(self.b, "ge", found, self._word(0))


def _empty_display(node: ast.expr) -> str | None:
    """`[]` and `{}` shown as they are: an empty display says nothing of a type."""
    if isinstance(node, ast.List) and not node.elts:
        return "[]"
    if isinstance(node, ast.Dict) and not node.keys:
        return "{}"
    return None


#: How Python spells each container, for messages.
BUILTIN_NAMES = {value: name for name, value in BUILTINS.items()}

# `Shape` and `BOOL` are what callers of this module compare against.
_ = (Shape, BOOL)
