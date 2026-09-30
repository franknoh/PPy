"""Generators with a frame of their own: returned, passed on, held and stepped.

A generator consumed where it is made is inlined into its consumer
(`generators`). One that outlives that (returned, passed to another
function, stepped by `next` in a loop or a branch, an `__iter__` method)
is lowered as a function of its own, whose `yield`s the `lower-generators`
pass turns into a starter, which makes the generator's frame and returns its
handle, and a resume function, which runs the body from where it left off to
its next `yield`. The frame is a runtime object: it counts its references,
the collector sees what it holds, and freeing it lets go of the references
its locals hold, however far the generator got.

Here are both halves: the body of such a generator (its `yield`, its
`return`, its locals let go of at its end), and the code that holds and
steps one (a `for` over it, `next`, `list` and the other consumers of a
walk, passing it on and returning it).
"""

from __future__ import annotations

import ast

from ..analysis import types as T
from ..backend.llvm.lowering import Unsupported
from ..ir import I64, PtrType, Value
from ..ir.dialects import core
from ..ir.transforms.lower_generators import GENERATOR_YIELD, HANDLE_WORDS, SLOT_VALUE
from .collection_api import _called, _Cursor
from .collections import HANDLE, Kind, Shape, generator_spelled, shape_of
from .walks import _Counted

__all__ = ["FrameLowering", "frame_shape", "frame_words"]


def frame_shape(info: object, records: dict) -> Shape:
    """What a generator function yields, as native code holds it; refuses one
    whose body a frame cannot run."""
    if info.is_async:  # type: ignore[attr-defined]
        raise Unsupported("an async generator has no native lowering")
    node = info.node  # type: ignore[attr-defined]
    yields = {id(n) for n in ast.walk(node) if isinstance(n, (ast.Yield, ast.YieldFrom))}
    statements = {
        id(n.value)
        for n in ast.walk(node)
        if isinstance(n, ast.Expr) and isinstance(n.value, (ast.Yield, ast.YieldFrom))
    }
    if yields - statements:
        raise Unsupported(f"`{info.name}` uses a `yield`'s value, which `send` gives")  # type: ignore[attr-defined]
    for inner in ast.walk(node):
        if inner is node:
            continue
        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            raise Unsupported(f"`{info.name}` holds a `{type(inner).__name__}` natively unlowered")  # type: ignore[attr-defined]
        if (
            isinstance(inner, ast.Return)
            and inner.value is not None
            and not (isinstance(inner.value, ast.Constant) and inner.value.value is None)
        ):
            raise Unsupported("a generator's return value has no native lowering")
    ret = T.strip_literal(info.ret)  # type: ignore[attr-defined]
    if generator_spelled(ret) is None:
        raise Unsupported("a generator function is annotated `Iterator[T]` or `Generator[T]`")
    assert isinstance(ret, T.Instance)
    shape = shape_of(ret.args[0], records)
    if shape is None or shape.kind in {"generator", "function"}:
        raise Unsupported(f"a generator of `{ret.args[0]}` has no native form")
    return shape


def check_frame(function: object, value_words: int) -> None:
    """Refuse a generator whose frame cannot hold what it keeps across a `yield`:
    a buffer or a vector, or more references than the frame's mask names."""
    from ..ir import BufferType, VectorType  # pylint: disable=import-outside-toplevel

    kept = [a.type for a in function.body.blocks[0].arguments]  # type: ignore[attr-defined]
    owning = sum(1 for t in kept if t == HANDLE)
    for op in function.operations():  # type: ignore[attr-defined]
        if op.name != "core.alloca":
            continue
        kept.append(op.results[0].type.pointee)
        owning += bool(op.attributes.get("ppy.owns"))
    if any(isinstance(t, (BufferType, VectorType)) for t in kept):
        raise Unsupported("a generator holds a buffer or a vector, which its frame cannot")
    if owning > HANDLE_WORDS - SLOT_VALUE - value_words:
        raise Unsupported("a generator holds more references than its frame names")


def frame_words(shape: Shape) -> int:
    """How many words of the frame the yielded value takes."""
    return shape.words


class FrameLowering:  # pylint: disable=attribute-defined-outside-init
    """Mixed into `_FunctionLowering` ahead of the inlined generators."""

    # -- the generator's own body -------------------------------------------------------

    def _frame(self) -> Shape | None:
        """What this function yields, where it is a generator with a frame."""
        return self.__dict__.get("_frame_shape")

    def _yield_statement(self, node: ast.Yield | ast.YieldFrom) -> None:
        shape = self._frame()
        if shape is None or self._inlines_stack():  # type: ignore[attr-defined]
            super()._yield_statement(node)  # type: ignore[misc]
            return
        if isinstance(node, ast.YieldFrom):
            source = node.value
            if not self._is_walk(source):  # type: ignore[attr-defined]
                raise Unsupported(f"`yield from {ast.unparse(source)}` has no native lowering")
            self._walk(  # type: ignore[attr-defined]
                self._source(source),  # type: ignore[attr-defined]
                lambda items: self._frame_yield(shape, items[0][1], owned=False),
            )
            return
        if node.value is None:
            raise Unsupported("a generator that yields None has no native form")
        value, owned = self._value(node.value, shape)  # type: ignore[attr-defined]
        self._frame_yield(shape, value, owned)

    def _frame_yield(self, shape: Shape, value: Value, owned: bool) -> None:
        """Hand the value to whoever steps the generator, and stop here until
        it is stepped again. A handle goes with a reference of its own."""
        if shape.reference and not owned:
            self._retain(value)  # type: ignore[attr-defined]
        if not shape.reference:
            value = self._coerce_type(value, shape.ir_type())  # type: ignore[attr-defined]
        core.call_intrinsic(self.b, GENERATOR_YIELD, (value,), ())  # type: ignore[attr-defined]

    def _generator_return(self, node: ast.Return) -> bool:
        if self._frame() is None or self._inlines_stack():  # type: ignore[attr-defined]
            return super()._generator_return(node)  # type: ignore[misc]
        if node.value is not None and not (
            isinstance(node.value, ast.Constant) and node.value.value is None
        ):
            raise Unsupported("a generator's return value has no native lowering")
        self._frame_end()
        return True

    def _frame_end(self) -> None:
        """The body is done: its locals go, and the generator is exhausted."""
        self._leave_for_return()  # type: ignore[attr-defined]
        self._release_collections()  # type: ignore[attr-defined]
        core.ret(self.b, self._rt("ppy_coll_none", (), HANDLE))  # type: ignore[attr-defined]

    def _release_collections(self) -> None:
        super()._release_collections()  # type: ignore[misc]
        done = {id(held.slot) for held in self.collections.values()}  # type: ignore[attr-defined]
        for scope in self.__dict__.get("_suspended", ()):
            # An inlined generator a `return` leaves in the middle of: its own
            # locals, not the ones a generator expression sees of its consumer.
            for held in scope.values():
                if id(held.slot) in done:
                    continue
                done.add(id(held.slot))
                self._release(core.load(self.b, held.slot))  # type: ignore[attr-defined]
                core.store(self.b, self._rt("ppy_coll_none", (), HANDLE), held.slot)  # type: ignore[attr-defined]
        if self._frame() is None:
            return
        # The frame lets go of what its slots hold when it is freed: a slot let
        # go of here holds `None` from now on.
        slots = [held.slot for held in self.collections.values()]  # type: ignore[attr-defined]
        slots += list(getattr(self, "_exception_slots", ()))
        for slot in slots:
            core.store(self.b, self._rt("ppy_coll_none", (), HANDLE), slot)  # type: ignore[attr-defined]

    def _owning(self, slot: Value) -> Value:
        """Mark a slot that owns the reference it holds: a frame's lets go of it."""
        slot.owner.attributes["ppy.owns"] = True  # type: ignore[union-attr]
        return slot

    # -- holding and stepping one ---------------------------------------------------------

    def _generator_function_info(self, node: ast.expr) -> object | None:
        """The generator function a call makes a generator of, if any."""
        if not isinstance(node, ast.Call):
            return None
        typed = T.strip_literal(self._type_of(node.func))  # type: ignore[attr-defined]
        if not isinstance(typed, T.Callable_) or not typed.is_generator:
            return None
        return typed

    def _is_generator(self, node: ast.expr) -> bool:
        """A call to a generator function is inlined where the function can be;
        one that cannot (a method, one with a `try`) has a frame."""
        if not super()._is_generator(node):  # type: ignore[misc]
            return False
        if not isinstance(node, ast.Call):
            return True
        from .generators import _check_generator  # pylint: disable=import-outside-toplevel

        found = self._generator_function(node)  # type: ignore[attr-defined]
        if found is None:
            return True
        try:
            _check_generator(found.info)
        except Unsupported:
            return self._frame_value(node) is None
        return True

    def _stepped_generator(self, statement: ast.stmt, rest: list[ast.stmt]) -> bool:
        """`it = gen(...)` stepped where the inlined form cannot follow it (in a
        loop, a branch, passed on) holds a frame instead."""
        from .generators import (  # pylint: disable=import-outside-toplevel
            _check_segment,
            _check_segment_use,
            _segments,
            _Site,
            _stepped_target,
        )

        found = _stepped_target(statement)
        if found is None:
            return False
        name, source = found
        if not self._is_generator(source):  # type: ignore[attr-defined]
            return False
        sites = [_Site.of(index, entry, name) for index, entry in enumerate(rest)]
        steps = [site for site in sites if site is not None]
        try:
            if not steps:
                raise Unsupported("the generator is not stepped")
            for index, entry in enumerate(rest):
                _check_segment_use(entry, name, sites[index] is not None)
            shape = self._element_shape(source)  # type: ignore[attr-defined]
            if shape.reference and any(site.kind == "next" for site in steps):
                raise Unsupported("`next` of strings or objects has no native lowering")
            for segment in _segments(rest, steps)[1:-1]:
                _check_segment(segment)
        except Unsupported:
            if self._frame_value(source) is not None and self._has_frame(source):
                return False
            raise
        return super()._stepped_generator(statement, rest)  # type: ignore[misc]

    def _reference_of(self, node: ast.expr):  # type: ignore[no-untyped-def]
        found = super()._reference_of(node)  # type: ignore[misc]
        if found is not None:
            return found
        return self._frame_value(node)

    def _handle(self, node: ast.expr) -> tuple[Value, bool]:
        """A call to a generator function, as a value: its starter makes the frame."""
        if (
            isinstance(node, ast.Call)
            and _called(node, "next")
            and node.args
            and self._counted_kind(node.args[0]) == "frame"
        ):
            # The yielded reference is the expression's own.
            self.__dict__["_next_owned"] = True
            try:
                value = self._generator_consumer("next", node)
            finally:
                self.__dict__["_next_owned"] = False
            assert value is not None
            return value, True
        if isinstance(node, ast.Call) and _called(node, "iter") and len(node.args) == 1:
            made = self._frame_of(node)
            if made is not None:
                return made[1], made[2]
        if isinstance(node, ast.Call) and self._frame_value(node) is not None:
            found = self._generator_function(node)  # type: ignore[attr-defined]
            if found is not None and self._has_frame(node):
                return self._call(node), True  # type: ignore[attr-defined]
        return super()._handle(node)  # type: ignore[misc]

    def _has_frame(self, node: ast.expr) -> bool:
        """Whether the generator function a call names is lowered with a frame."""
        found = self._generator_function(node)  # type: ignore[attr-defined]
        return found is not None and found.info.qualname in self.frontend.declared  # type: ignore[attr-defined]

    def _frame_value(self, node: ast.expr) -> Shape | None:
        """The generator shape of a value native code holds as a frame."""
        shape = shape_of(self._type_of(node), self._records())  # type: ignore[attr-defined]
        if shape is None or shape.kind != "generator":
            return None
        return shape

    def _iterated_object(self, node: ast.expr) -> Shape | None:
        """An object whose class's `__iter__` is a generator with a frame."""
        shape = self._object_of(node)  # type: ignore[attr-defined]
        if shape is None:
            return None
        info = self._resolve(self._class_info(shape), "__iter__")  # type: ignore[attr-defined]
        if info is None:
            return None
        method = info.methods["__iter__"]
        if not method.is_generator:
            return None
        return shape_of(method.ret, self._records())  # type: ignore[attr-defined]

    def _frame_of(self, node: ast.expr) -> tuple[Shape, Value, bool] | None:
        """A generator's frame an expression gives, with its shape and whether it
        is owned: a generator value, or an iterated object's `__iter__()`."""
        if isinstance(node, ast.Call) and _called(node, "iter") and len(node.args) == 1:
            node = node.args[0]
        shape = self._frame_value(node)
        if shape is not None and not self._is_generator(node):
            handle, owned = self._handle(node)  # type: ignore[attr-defined]
            return shape, handle, owned
        shape = self._iterated_object(node)
        if shape is not None and shape.kind == "generator":
            receiver = self._object_of(node)  # type: ignore[attr-defined]
            handle, owned = self._handle(node)  # type: ignore[attr-defined]
            made = self._method_call(receiver, "__iter__", handle, [], [])  # type: ignore[attr-defined]
            self._done_with(handle, owned)  # type: ignore[attr-defined]
            return shape, made, True
        return None

    def _counted_kind(self, node: ast.expr) -> str | None:
        found = super()._counted_kind(node)  # type: ignore[misc]
        if found is not None:
            return found
        if isinstance(node, ast.Call) and _called(node, "iter") and len(node.args) == 1:
            node = node.args[0]
        shape = self._frame_value(node)
        if shape is not None and not self._is_generator(node):
            return "frame"
        shape = self._iterated_object(node)
        if shape is not None and shape.kind == "generator":
            return "frame"
        return None

    def _frame_source(self, node: ast.expr) -> _Counted:
        found = self._frame_of(node)
        assert found is not None
        shape, handle, owned = found
        element = shape_of(shape.class_args[0], self._records())  # type: ignore[attr-defined]
        assert element is not None
        slot = self._hold(shape, handle, owned)  # type: ignore[attr-defined]
        return _Counted(Kind("Vec", element), slot, what="frame")

    def _frame_step(self, handle: Value) -> Value:
        """Run the generator to its next `yield`: whether it gave a value."""
        b = self.b  # type: ignore[attr-defined]
        frame = self._frame_words(handle)
        code = core.load(b, frame)
        status = self._call_indirect(code, (handle,), (I64,))[0]  # type: ignore[attr-defined]
        return core.cmp(b, "ne", status, self._word(0))  # type: ignore[attr-defined]

    def _frame_words(self, handle: Value) -> Value:
        record = self._rt("ppy_gen_frame", (handle,), HANDLE)  # type: ignore[attr-defined]
        return core.cast(self.b, record, PtrType(I64))  # type: ignore[attr-defined]

    def _frame_value_read(self, handle: Value, element: Shape) -> Value:
        """What the generator last yielded; a handle is the reader's to keep."""
        b = self.b  # type: ignore[attr-defined]
        words = self._frame_words(handle)
        at = core.ptr_offset(b, words, self._word(SLOT_VALUE))  # type: ignore[attr-defined]
        wanted = element.ir_type()
        if wanted != I64:
            at = core.cast(b, at, PtrType(wanted))
        return core.load(b, at)

    def _take_yielded(self, key: object, element: Shape, value: Value) -> None:
        """A yielded handle, held by a hidden local until the next one replaces it."""
        if not element.reference:
            return
        names = self.__dict__.setdefault("_yielded", {})
        name = names.get(id(key))
        if name is None:
            self._walks += 1  # type: ignore[attr-defined]
            name = names[id(key)] = f".yielded{self._walks}"  # type: ignore[attr-defined]
        held = element.collection if element.kind == "collection" else element
        self._bind(name, held, value, owned=True)  # type: ignore[attr-defined]

    def _generator_consumer(self, name: str, node: ast.Call) -> Value | None:
        """`next(it)` and `next(it, default)` of a generator with a frame."""
        if name != "next" or node.keywords or len(node.args) not in {1, 2}:
            return super()._generator_consumer(name, node)  # type: ignore[misc]
        if self._counted_kind(node.args[0]) != "frame":
            return super()._generator_consumer(name, node)  # type: ignore[misc]
        found = self._frame_of(node.args[0])
        assert found is not None
        shape, handle, owned = found
        element = shape_of(shape.class_args[0], self._records())  # type: ignore[attr-defined]
        assert element is not None
        b = self.b  # type: ignore[attr-defined]
        got = self._alloca(element.ir_type(), "next.value")  # type: ignore[attr-defined]
        gave = self._block("next.gave")  # type: ignore[attr-defined]
        empty = self._block("next.empty")  # type: ignore[attr-defined]
        done = self._block("next.end")  # type: ignore[attr-defined]
        core.cond_br(b, self._frame_step(handle), _succ(gave), _succ(empty))
        b.at_end(gave)
        core.store(b, self._frame_value_read(handle, element), got)
        core.br(b, _succ(done))
        b.at_end(empty)
        if len(node.args) == 2:
            default, default_owned = self._value(node.args[1], element)  # type: ignore[attr-defined]
            if element.reference and not default_owned:
                self._retain(default)  # type: ignore[attr-defined]
            core.store(b, default, got)
            core.br(b, _succ(done))
        else:
            from .generators import _stop_tag  # pylint: disable=import-outside-toplevel

            self._raise_made(  # type: ignore[attr-defined]
                "StopIteration",
                _stop_tag(),
                self._string_literal(""),  # type: ignore[attr-defined]
                self._word(1),  # type: ignore[attr-defined]
            )
        b.at_end(done)
        self._done_with(handle, owned)  # type: ignore[attr-defined]
        value = core.load(b, got)
        if not self.__dict__.get("_next_owned"):
            self._take_yielded(node, element, value)
        return value


def _succ(block):  # type: ignore[no-untyped-def]
    from ..ir import Successor  # pylint: disable=import-outside-toplevel

    return Successor(block)


def _frame_cursor(source: _Counted, at: Value) -> _Cursor:
    return _Cursor(source, at, None, None)


#: Every handle word a frame can name, past its fixed words.
FRAME_HANDLES = HANDLE_WORDS - SLOT_VALUE
