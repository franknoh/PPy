"""Exceptions in native code: `raise`, `try`, `assert`, and the checks CPython raises for.

An exception is an object of the collections runtime (`ppy_runtime/exceptions.c`):
its class's tag, its class's name, and `str()` of it. Raising one makes it
the thread's pending exception and goes to where the code catches it: the
innermost `try` of the function, or the function's exit, which lets go of
what the function held and returns the raised status (`STATUS_RAISED`).

A caller whose call sits in a `try`, or that holds anything to let go of,
asks for the call's status and goes the same way when it is the raised
one; the backends make every other call return the raised status on up.
Python sees that status at the boundary as it sees a failed guard, and
runs the call again as Python, which raises the exception itself; a
standalone program's `main` prints CPython's last line for it.

A check CPython raises for (an index out of range, a missing key, a
division by zero) is an exception here too, in a module that raises or
catches: native code can catch it, which is what `except IndexError:` asks.
A check that stands for something only native code cannot do (an integer
past 64 bits) is not one; it falls back, and Python answers.

Everything here is gated by the module's own use of exceptions, so a module
that neither raises nor catches lowers exactly as it did.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from ..analysis import types as T
from ..backend.llvm.lowering import Unsupported
from ..ir import BOOL, I64, Block, IRType, Successor, Value
from ..ir.dialects import core
from .collections import HANDLE, STR, Shape, class_tag

__all__ = ["ARGS_NONE", "ARGS_ONE_TEXT", "ExceptionLowering", "exception_tag", "uses_exceptions"]

#: Checks that stand for what only native code cannot do: never an exception
#: here, since CPython would answer where native code stops.
_NATIVE_LIMITS = (
    "OverflowError: the result does not fit in a 64-bit integer",
    "ValueError: a NaN was compared",
    "TypeError: sum() of no floats",
    "MemoryError",
)

#: Checks whose text is not all of what CPython says, which appends the
#: value it could not convert: the exception is right, `str()` of it falls back.
_INEXACT = (
    "ValueError: invalid literal for int() with base",
    "ValueError: could not convert string to float",
    "ValueError: the string is not in list",
    "ValueError: the value is not in the ",
)

_PLACE = re.compile(r"\{(\d+)\}")

#: The flags word of an exception object (its fourth) says, past whether `str()`
#: of it is CPython's (bit 0), what `args` is: `(str(e),)`, or `()`.
ARGS_ONE_TEXT = 2
ARGS_NONE = 4


def exception_tag(name: str) -> int:
    """A class's tag, `builtins.<name>` for a builtin exception."""
    return class_tag(name if "." in name else f"builtins.{name}")


def _builtin_exception(name: str) -> bool:
    return "Exception" in T.BUILTIN_MRO.get(name, ())


def uses_exceptions(nodes: list[ast.AST]) -> bool:
    """Whether any of these function bodies raises, catches, or asserts."""
    return any(
        isinstance(inner, (ast.Try, ast.Raise, ast.Assert))
        for node in nodes
        for inner in ast.walk(node)
    )


@dataclass(slots=True)
class _Frame:
    """An open `try` region: where an exception raised in it goes."""

    target: Block
    #: The `finally` body, run by whatever leaves the region some other way.
    final: list[ast.stmt]
    #: How many loops were open when the region opened.
    loops: int
    #: A handler's slot for the exception it handles, for a bare `raise`.
    handled: Value | None = None
    #: Names a handler bound with `as`, deleted when it ends.
    names: list[str] = field(default_factory=list)


class ExceptionLowering:  # pylint: disable=attribute-defined-outside-init
    """The exception half of lowering one function; mixed into `_FunctionLowering`."""

    # -- setup ------------------------------------------------------------------

    def _exception_setup(self) -> None:
        self._frames: list[_Frame] = []
        #: Handler names bound with `as`: name -> the slot holding the exception.
        self.caught: dict[str, Value] = {}
        #: Every slot that may hold an exception, let go of when the function ends.
        self._exception_slots: list[Value] = []
        #: Where an exception nothing in the function catches goes, once made.
        self._propagate: Block | None = None
        #: The blocks after a `try` that some path falls into.
        self._reached: set[int] = set()
        if self._exceptions_on():
            # A check hoisted out of a loop fails before the iterations that
            # ran first; caught, that would be a different answer.
            self.hoist = False  # type: ignore[attr-defined]

    def _exceptions_on(self) -> bool:
        return bool(getattr(self.frontend, "native_exceptions", False))  # type: ignore[attr-defined]

    def _use_exceptions(self) -> None:
        self._use_collections()  # type: ignore[attr-defined]

    def _finish_exceptions(self) -> None:
        """The function's exit for an exception: let go of what it held, and return
        the raised status."""
        if self._propagate is None:
            return
        self.b.at_end(self._propagate)  # type: ignore[attr-defined]
        self._release_collections()  # type: ignore[attr-defined]
        core.guard(self.b, core.const(self.b, False, BOOL), "contract", "raised", label="raised")  # type: ignore[attr-defined]
        core.unreachable(self.b)  # type: ignore[attr-defined]

    def _release_collections(self) -> None:
        super()._release_collections()  # type: ignore[misc]
        for slot in getattr(self, "_exception_slots", ()):
            self._release(core.load(self.b, slot))  # type: ignore[attr-defined]

    def _reached_block(self, block: Block) -> bool:
        """Whether some block branches to `block`. One nothing reaches must not
        branch on, or what it branches to would seem to escape the entry."""
        registry = self.frontend.registry  # type: ignore[attr-defined]
        return any(
            block in other.successors_for(registry)
            for other in self.function.body.blocks  # type: ignore[attr-defined]
            if other is not block
        )

    def _seal(self, block: Block) -> bool:
        """Close `block` where nothing reaches it; whether it was."""
        if self._reached_block(block):
            return False
        self.b.at_end(block)  # type: ignore[attr-defined]
        core.unreachable(self.b)  # type: ignore[attr-defined]
        return True

    def _raise_target(self) -> Block:
        if self._frames:
            return self._frames[-1].target
        if self._propagate is None:
            self._propagate = self._block("raised")  # type: ignore[attr-defined]
        return self._propagate

    def _exception_slot(self, name: str) -> Value:
        """A slot for an exception, empty from the entry on."""
        slot = self._alloca(HANDLE, name)  # type: ignore[attr-defined]
        slot.owner.attributes["ppy.owns"] = True  # type: ignore[union-attr]
        entry = self._entry_builder()  # type: ignore[attr-defined]
        empty = core.call_extern(entry, "ppy_coll_none", (), (HANDLE,)).results[0]
        core.store(entry, empty, slot)
        self._exception_slots.append(slot)
        return slot

    # -- raising ----------------------------------------------------------------

    def _go_raise(self) -> None:
        """The pending exception goes where the code catches it."""
        core.br(self.b, Successor(self._raise_target()))  # type: ignore[attr-defined]

    def _raise_made(self, name: str, tag: int, message: Value, known: Value) -> None:
        """Raise a new exception of class `name`, taking `message`."""
        self._use_exceptions()
        spelled = self._string_literal(name.rpartition(".")[2])  # type: ignore[attr-defined]
        made = self._rt(  # type: ignore[attr-defined]
            "ppy_exc_make",
            (self._word(tag), spelled, message, known),
            HANDLE,  # type: ignore[attr-defined]
        )
        self._rt("ppy_exc_raise", (made,), None)  # type: ignore[attr-defined]
        self._go_raise()

    def _raise_text(self, raises: str, values: tuple[Value, ...]) -> None:
        """Raise the exception a check's CPython line names: `IndexError: index {0} ...`."""
        name, _, rest = raises.partition(": ")
        builder = self._rt("ppy_str_builder", (self._word(0),), HANDLE)  # type: ignore[attr-defined]
        position = 0
        for found in _PLACE.finditer(rest):
            self._add_text(builder, rest[position : found.start()])
            value = values[int(found.group(1))]
            if value.type != I64:
                value = core.cast(self.b, value, I64)  # type: ignore[attr-defined]
            self._rt("ppy_str_add_int", (builder, value), None)  # type: ignore[attr-defined]
            position = found.end()
        self._add_text(builder, rest[position:])
        message = self._rt("ppy_str_finish", (builder,), HANDLE)  # type: ignore[attr-defined]
        exact = not raises.startswith(_INEXACT) and not (name == "KeyError" and not rest)
        # `args` is `(message,)` where the message is all of CPython's and not a key's repr.
        flags = int(exact) | (ARGS_ONE_TEXT if exact and name != "KeyError" and rest else 0)
        self._raise_made(name, exception_tag(name), message, self._word(flags))  # type: ignore[attr-defined]

    def _add_text(self, builder: Value, text: str) -> None:
        if text:
            data, length = self._text_data(text)  # type: ignore[attr-defined]
            self._rt("ppy_str_add_bytes", (builder, data, length), None)  # type: ignore[attr-defined]

    def _guard(
        self,
        condition: Value,
        kind: str,
        message: str = "",
        raises: str = "",
        values: tuple[Value, ...] = (),
    ) -> None:
        """A check: where it fails, the exception CPython raises there, in a module
        that catches; a fall back to Python, as ever, elsewhere and for what only
        native code cannot do."""
        name = raises.partition(":")[0]
        if (
            not self._exceptions_on()
            or not raises
            or raises.startswith(_NATIVE_LIMITS)
            or not _builtin_exception(name)
            or getattr(self, "device", False)
        ):
            if not self.frontend.standalone:  # type: ignore[attr-defined]
                values = ()  # only a standalone binary prints them
            core.guard(self.b, condition, kind, message, raises=raises, values=values)  # type: ignore[attr-defined]
            return
        kept = self._block(f"{kind}.ok")  # type: ignore[attr-defined]
        failed = self._block(f"{kind}.raise")  # type: ignore[attr-defined]
        core.cond_br(self.b, condition, Successor(kept), Successor(failed))  # type: ignore[attr-defined]
        self.b.at_end(failed)  # type: ignore[attr-defined]
        self._raise_text(raises, values)
        self.b.at_end(kept)  # type: ignore[attr-defined]

    # -- calls ------------------------------------------------------------------

    def _call_native(
        self, callee: str, arguments: tuple[Value, ...], results: tuple[IRType, ...]
    ) -> object:
        """A call to a native function; in a module that catches, one whose
        exception goes where the code catches it."""
        if not self._exceptions_on():
            return core.call(self.b, callee, arguments, results)  # type: ignore[attr-defined]
        made = core.call(self.b, callee, arguments, results, capture_status=True)  # type: ignore[attr-defined]
        *values, status = made.results
        answered = self._block("call.ok")  # type: ignore[attr-defined]
        failed = self._block("call.failed")  # type: ignore[attr-defined]
        zero = self._word(0)  # type: ignore[attr-defined]
        answers = core.cmp(self.b, "eq", status, zero)  # type: ignore[attr-defined]
        core.cond_br(self.b, answers, Successor(answered), Successor(failed))  # type: ignore[attr-defined]
        self.b.at_end(failed)  # type: ignore[attr-defined]
        raised = core.cmp(self.b, "eq", status, self._word(-1))  # type: ignore[attr-defined]
        # Anything but an exception is a failed guard: this call falls back too.
        core.guard(self.b, raised, "contract", "a call fell back")  # type: ignore[attr-defined]
        self._go_raise()
        self.b.at_end(answered)  # type: ignore[attr-defined]
        return _Called(tuple(values))

    # -- statements -------------------------------------------------------------

    def _raise(self, node: ast.Raise) -> None:
        self._use_exceptions()
        if node.cause is not None and not self._plain_cause(node.cause):
            raise Unsupported("`raise ... from` takes `None`, a name, or a new exception natively")
        if node.exc is None:
            handled = next((f.handled for f in reversed(self._frames) if f.handled), None)
            if handled is None:
                raise Unsupported("a bare `raise` outside a handler has no native lowering")
            self._reraise(core.load(self.b, handled))  # type: ignore[attr-defined]
            return
        if isinstance(node.exc, ast.Name) and node.exc.id in self.caught:
            self._reraise(core.load(self.b, self.caught[node.exc.id]))  # type: ignore[attr-defined]
            return
        held = self._object_of(node.exc)  # type: ignore[attr-defined]
        if held is not None and self._is_exception(held):  # type: ignore[attr-defined]
            # `raise err`, or `raise ParseError(line, text)` of a class with
            # fields or methods: the object is the exception.
            handle, owned = self._handle(node.exc)  # type: ignore[attr-defined]
            if not owned:
                self._retain(handle)  # type: ignore[attr-defined]
            self._rt("ppy_exc_raise", (handle,), None)  # type: ignore[attr-defined]
            self._go_raise()
            return
        call = node.exc if isinstance(node.exc, ast.Call) else None
        spelled = call.func if call is not None else node.exc
        name = self._exception_class(spelled)
        arguments = call.args if call is not None else []
        if call is not None and call.keywords:
            raise Unsupported("an exception takes positional arguments natively")
        if len(arguments) > 1:
            raise Unsupported("an exception with more than one argument has no native lowering")
        message, known = self._exception_message(name, arguments[0] if arguments else None)
        self._raise_made(name, exception_tag(name), message, known)

    def _plain_cause(self, cause: ast.expr) -> bool:
        """`from None`, `from err`, or `from SomeError(...)`: a cause only changes
        what a traceback prints above its last line, which is all native code
        prints, so it is not kept. A cause whose making could do anything else
        is not one of these."""
        if isinstance(cause, ast.Constant) and cause.value is None:
            return True
        if isinstance(cause, ast.Name):
            return True
        if isinstance(cause, ast.Call) and not cause.keywords:
            called = T.strip_literal(self._type_of(cause.func))  # type: ignore[attr-defined]
            simple = (ast.Constant, ast.Name)
            return isinstance(called, T.ClassObject) and all(
                isinstance(argument, simple) for argument in cause.args
            )
        return False

    def _reraise(self, exception: Value) -> None:
        self._retain(exception)  # type: ignore[attr-defined]
        self._rt("ppy_exc_raise", (exception,), None)  # type: ignore[attr-defined]
        self._go_raise()

    def _exception_class(self, node: ast.expr) -> str:
        """The class a `raise` names: a builtin exception, or a project class
        deriving from one with nothing of its own."""
        called = T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]
        if not isinstance(called, T.ClassObject):
            raise Unsupported(f"`raise {ast.unparse(node)}` names no class natively")
        name = called.name
        if _builtin_exception(name):
            return name
        info = self._project_exception(name)
        if info is None:
            raise Unsupported(f"`{name}` is not an exception class native code can raise")
        return info.qualname

    def _project_exception(self, name: str):  # type: ignore[no-untyped-def]
        """A project class deriving from a builtin exception that native code can
        make: its instances are objects whose record starts with the exception
        runtime's four words, with the fields of its classes after them."""
        classes = self.frontend.analysis.symbols.classes  # type: ignore[attr-defined]
        info = classes.get(name) or classes.get(name.rpartition(".")[2])
        if info is None or "Exception" not in info.mro:
            return None
        if not self._is_exception(Shape("object", record=info.qualname)):  # type: ignore[attr-defined]
            return None
        return info

    def _exception_header(self, shape: Shape, made: Value, node: ast.Call) -> None:
        """A new exception object's first four words: its class's tag and name, and
        `str()` of it as `BaseException` makes it from the constructor's
        arguments (an `__init__` that calls `super().__init__` sets it again)."""
        name = shape.record
        self._write_word(made, 0, self._word(exception_tag(name)))  # type: ignore[attr-defined]
        spelled = self._string_literal(name.rpartition(".")[2])  # type: ignore[attr-defined]
        self._write(self._field_address(made, 1), STR, spelled)  # type: ignore[attr-defined]
        init = self._resolve(self._class_info(shape), "__init__")  # type: ignore[attr-defined]
        if init is not None and _calls_super_init(init.methods["__init__"].node):
            message, known = self._string_literal(""), self._word(1)  # type: ignore[attr-defined]
        else:
            message, known = self._args_text(shape, list(node.args))
        self._write(self._field_address(made, 2), STR, message)  # type: ignore[attr-defined]
        self._write_word(made, 3, known)

    def _write_word(self, made: Value, index: int, value: Value) -> None:
        address = self._field_address(made, index)  # type: ignore[attr-defined]
        self._write(address, Shape("int"), value)  # type: ignore[attr-defined]

    def _args_text(self, shape: Shape, arguments: list[ast.expr]) -> tuple[Value, Value]:
        """`str()` of an exception made with these positional arguments: nothing,
        the one argument, or the tuple of them, as `BaseException.__str__` says."""
        info = self._class_info(shape)  # type: ignore[attr-defined]
        keyed = "KeyError" in info.mro
        if len(arguments) <= 1:
            return self._exception_message(
                "KeyError" if keyed else shape.record, arguments[0] if arguments else None
            )
        builder = self._rt("ppy_str_builder", (self._word(0),), HANDLE)  # type: ignore[attr-defined]
        known = self._word(1)  # type: ignore[attr-defined]
        self._add_text(builder, "(")
        for index, argument in enumerate(arguments):
            if index:
                self._add_text(builder, ", ")
            if self._string_of(argument) is not None:  # type: ignore[attr-defined]
                text = self._owned_string(argument)  # type: ignore[attr-defined]
                shown = self._rt("ppy_str_add_repr", (builder, text))  # type: ignore[attr-defined]
                known = core.bitwise(self.b, "and", known, shown)  # type: ignore[attr-defined]
                self._release(text)  # type: ignore[attr-defined]
                continue
            value = self._expr(argument)  # type: ignore[attr-defined]
            kind = {I64: "int", BOOL: "bool"}.get(value.type)
            if kind is None:
                raise Unsupported("an exception's arguments are strings, ints, or bools natively")
            self._rt(f"ppy_str_add_{kind}", (builder, value), None)  # type: ignore[attr-defined]
        self._add_text(builder, ")")
        return self._rt("ppy_str_finish", (builder,), HANDLE), known  # type: ignore[attr-defined]

    def _exception_init(self, handle: Value, shape: Shape, arguments: list[ast.expr]) -> None:
        """`super().__init__(...)` of a class deriving from a builtin exception: the
        arguments become `str()` of it, as `BaseException.__init__` sets `args`."""
        message, known = self._args_text(shape, arguments)
        address = self._field_address(handle, 2)  # type: ignore[attr-defined]
        old = self._read(address, STR)  # type: ignore[attr-defined]
        self._write(address, STR, message)  # type: ignore[attr-defined]
        self._release(old)  # type: ignore[attr-defined]
        self._write_word(handle, 3, known)

    def _exception_message(self, name: str, argument: ast.expr | None) -> tuple[Value, Value]:
        """`str()` of `name(argument)`, and whether it is CPython's."""
        one = self._word(1)  # type: ignore[attr-defined]
        if argument is None:
            return self._string_literal(""), self._word(1 | ARGS_NONE)  # type: ignore[attr-defined]
        builder = self._rt("ppy_str_builder", (self._word(0),), HANDLE)  # type: ignore[attr-defined]
        keyed = name == "KeyError"
        known = one
        if self._string_of(argument) is not None:  # type: ignore[attr-defined]
            text = self._owned_string(argument)  # type: ignore[attr-defined]
            if keyed:
                # `str()` of a `KeyError` is `repr()` of its key.
                known = self._rt("ppy_str_add_repr", (builder, text))  # type: ignore[attr-defined]
            else:
                self._rt("ppy_str_add", (builder, text), None)  # type: ignore[attr-defined]
                known = self._word(1 | ARGS_ONE_TEXT)  # type: ignore[attr-defined]
            self._release(text)  # type: ignore[attr-defined]
        else:
            value = self._expr(argument)  # type: ignore[attr-defined]
            kind = {I64: "int", BOOL: "bool"}.get(value.type)
            if kind is None:
                raise Unsupported("an exception's argument is a string, an int, or a bool natively")
            self._rt(f"ppy_str_add_{kind}", (builder, value), None)  # type: ignore[attr-defined]
        return self._rt("ppy_str_finish", (builder,), HANDLE), known  # type: ignore[attr-defined]

    def _assert(self, node: ast.Assert) -> None:
        self._use_exceptions()
        condition = self._test(node.test)  # type: ignore[attr-defined]
        kept = self._block("assert.ok")  # type: ignore[attr-defined]
        failed = self._block("assert.failed")  # type: ignore[attr-defined]
        core.cond_br(self.b, condition, Successor(kept), Successor(failed))  # type: ignore[attr-defined]
        self.b.at_end(failed)  # type: ignore[attr-defined]
        message, known = self._exception_message("AssertionError", node.msg)
        self._raise_made("AssertionError", exception_tag("AssertionError"), message, known)
        self.b.at_end(kept)  # type: ignore[attr-defined]

    def _try(self, node: ast.Try) -> None:
        """`try` with its handlers, `else`, and `finally`.

        The body's exceptions go to a dispatch that asks each handler in turn
        by tag; one that matches takes the exception, one that none matches,
        and one raised in a handler or in `else`, runs `finally` and goes on
        to the code around the `try`."""
        self._use_exceptions()
        if not self._exceptions_on():
            raise Unsupported("`try` is lowered where the module's exceptions are native")
        loops = len(self._loops)  # type: ignore[attr-defined]
        dispatch = self._block("try.dispatch")  # type: ignore[attr-defined]
        unwind = self._block("try.unwind")  # type: ignore[attr-defined]
        after = self._block("try.after")  # type: ignore[attr-defined]
        final = node.finalbody
        self._frames.append(_Frame(dispatch, final, loops))
        self._body(node.body)  # type: ignore[attr-defined]
        self._frames.pop()
        if self._open():  # type: ignore[attr-defined]
            self._frames.append(_Frame(unwind, final, loops))
            self._body(node.orelse)  # type: ignore[attr-defined]
            self._frames.pop()
            self._leave_normally(final, after)
        if self._seal(dispatch):
            # Nothing in the body raises: no handler runs.
            self._finish_try(unwind, final, after)
            return
        self.b.at_end(dispatch)  # type: ignore[attr-defined]
        tag = self._rt("ppy_exc_pending_tag", ())  # type: ignore[attr-defined]
        caught_all = False
        for handler in node.handlers:
            tags = self._handler_tags(handler.type)
            matched = self._block("except")  # type: ignore[attr-defined]
            following = self._block("except.next")  # type: ignore[attr-defined]
            if tags is None:
                core.br(self.b, Successor(matched))  # type: ignore[attr-defined]
                caught_all = True
            else:
                condition = core.const(self.b, False, BOOL)  # type: ignore[attr-defined]
                for wanted in tags:
                    condition = core.bitwise(
                        self.b,
                        "or",
                        condition,
                        core.cmp(self.b, "eq", tag, self._word(wanted)),  # type: ignore[attr-defined]
                    )
                core.cond_br(self.b, condition, Successor(matched), Successor(following))  # type: ignore[attr-defined]
            self.b.at_end(matched)  # type: ignore[attr-defined]
            self._handler(handler, unwind, final, loops, after)
            self.b.at_end(following)  # type: ignore[attr-defined]
            if caught_all:
                core.unreachable(self.b)  # type: ignore[attr-defined]
                break
        if not caught_all:
            core.br(self.b, Successor(unwind))  # type: ignore[attr-defined]
        self._finish_try(unwind, final, after)

    def _finish_try(self, unwind: Block, final: list[ast.stmt], after: Block) -> None:
        """What no handler took, or a handler or `else` raised: `finally`, then on;
        and the code after the `try`, where anything reaches it."""
        if not self._seal(unwind):
            self.b.at_end(unwind)  # type: ignore[attr-defined]
            saved = self._rt("ppy_exc_take", (), HANDLE)  # type: ignore[attr-defined]
            self._body(final)  # type: ignore[attr-defined]
            if self._open():  # type: ignore[attr-defined]
                self._rt("ppy_exc_raise", (saved,), None)  # type: ignore[attr-defined]
                self._go_raise()
        self.b.at_end(after)  # type: ignore[attr-defined]
        if id(after) not in self._reached:
            # Every way through the `try` returns or raises: nothing follows it.
            core.unreachable(self.b)  # type: ignore[attr-defined]

    def _handler(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        handler: ast.ExceptHandler,
        unwind: Block,
        final: list[ast.stmt],
        loops: int,
        after: Block,
    ) -> None:
        slot = self._exception_slot("handled")
        exception = self._rt("ppy_exc_take", (), HANDLE)  # type: ignore[attr-defined]
        core.store(self.b, exception, slot)  # type: ignore[attr-defined]
        frame = _Frame(unwind, final, loops, handled=slot)
        previous = None
        if handler.name:
            previous = self.caught.get(handler.name)
            self.caught[handler.name] = slot
            frame.names.append(handler.name)
        self._frames.append(frame)
        self._body(handler.body)  # type: ignore[attr-defined]
        self._frames.pop()
        if handler.name:
            if previous is None:
                self.caught.pop(handler.name, None)
            else:
                self.caught[handler.name] = previous
        if self._open():  # type: ignore[attr-defined]
            self._let_go_handled(slot)
            self._leave_normally(final, after)

    def _let_go_handled(self, slot: Value) -> None:
        """A handler's end: the exception it handled goes, as `as e` does."""
        self._release(core.load(self.b, slot))  # type: ignore[attr-defined]
        empty = self._rt("ppy_coll_none", (), HANDLE)  # type: ignore[attr-defined]
        core.store(self.b, empty, slot)  # type: ignore[attr-defined]

    def _leave_normally(self, final: list[ast.stmt], after: Block) -> None:
        self._body(final)  # type: ignore[attr-defined]
        if self._open():  # type: ignore[attr-defined]
            core.br(self.b, Successor(after))  # type: ignore[attr-defined]
            self._reached.add(id(after))

    def _handler_tags(self, spelled: ast.expr | None) -> list[int] | None:
        """The tags an `except` clause catches, or None for all of them."""
        if spelled is None:
            return None
        names = spelled.elts if isinstance(spelled, ast.Tuple) else [spelled]
        tags: set[int] = set()
        for one in names:
            called = T.strip_literal(self._type_of(one))  # type: ignore[attr-defined]
            if not isinstance(called, T.ClassObject):
                raise Unsupported(f"`except {ast.unparse(one)}` names no class natively")
            name = called.name
            if name in {"Exception", "BaseException"}:
                return None
            if _builtin_exception(name):
                tags.update(
                    exception_tag(other)
                    for other, mro in T.BUILTIN_MRO.items()
                    if name in mro and "Exception" in mro
                )
                wanted = name
            else:
                own = self._project_exception(name)
                if own is None:
                    raise Unsupported(f"`except {name}` catches a class native code does not raise")
                wanted = own.qualname
            classes = self.frontend.analysis.symbols.classes  # type: ignore[attr-defined]
            tags.update(
                exception_tag(info.qualname)
                for info in classes.values()
                if wanted in info.mro and self._project_exception(info.qualname) is not None
            )
        return sorted(tags)

    # -- leaving a region some other way ------------------------------------------

    def _leave_frames(self, down_to: int) -> None:
        """Run the `finally` of each region from the innermost out to `down_to`, and
        let go of the exceptions their handlers hold, as a `return`, `break`, or
        `continue` that leaves them does."""
        frames = self._frames
        try:
            for index in range(len(frames) - 1, down_to - 1, -1):
                frame = frames[index]
                self._frames = frames[:index]
                if frame.handled is not None:
                    self._let_go_handled(frame.handled)
                if frame.final:
                    self._body(frame.final)  # type: ignore[attr-defined]
                    if not self._open():  # type: ignore[attr-defined]
                        raise Unsupported(
                            "a `finally` that leaves the function has no native lowering"
                        )
        finally:
            self._frames = frames

    def _leave_for_loop(self) -> None:
        """A `break` or `continue`: the regions opened inside the loop it leaves."""
        depth = len(self._loops)  # type: ignore[attr-defined]
        inside = [i for i, frame in enumerate(self._frames) if frame.loops >= depth]
        if inside:
            self._leave_frames(inside[0])

    def _leave_for_return(self) -> None:
        if self._frames:
            self._leave_frames(0)

    # -- using a caught exception -------------------------------------------------

    def _handle(self, node: ast.expr) -> tuple[Value, bool]:
        """A name a handler bound with `as` lends the exception it holds."""
        slot = self._caught_of(node)
        if slot is not None:
            return core.load(self.b, slot), False  # type: ignore[attr-defined]
        return super()._handle(node)  # type: ignore[misc]

    def _caught_of(self, node: ast.expr) -> Value | None:
        if isinstance(node, ast.Name) and node.id in getattr(self, "caught", {}):
            return self.caught[node.id]
        return None

    def _caught_text(self, slot: Value) -> Value:
        """`str(e)`: an owned string; where it is not CPython's, the call falls back."""
        exception = core.load(self.b, slot)  # type: ignore[attr-defined]
        known = self._rt("ppy_exc_known", (exception,))  # type: ignore[attr-defined]
        told = core.cmp(self.b, "ne", known, self._word(0))  # type: ignore[attr-defined]
        core.guard(self.b, told, "contract", "str() of an exception")  # type: ignore[attr-defined]
        return self._rt("ppy_exc_str", (exception,), HANDLE)  # type: ignore[attr-defined]

    def _shown_text(self, node: ast.expr) -> Value | None:
        slot = self._caught_of(node)
        if slot is not None:
            return self._caught_text(slot)
        return super()._shown_text(node)  # type: ignore[misc]

    def _string_call(self, node: ast.Call, discard: bool) -> Value | None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "str"
            and len(node.args) == 1
            and not node.keywords
        ):
            slot = self._caught_of(node.args[0])
            if slot is not None:
                return self._keep_or_drop(self._caught_text(slot), discard)  # type: ignore[attr-defined]
        return super()._string_call(node, discard)  # type: ignore[misc]

    def _add_formatted(self, builder: Value, node: ast.expr, conversion: int, spec: str) -> None:
        slot = self._caught_of(node)
        if slot is not None and conversion in {-1, ord("s")} and not spec:
            text = self._caught_text(slot)
            self._rt("ppy_str_add", (builder, text), None)  # type: ignore[attr-defined]
            self._release(text)  # type: ignore[attr-defined]
            return
        super()._add_formatted(builder, node, conversion, spec)  # type: ignore[misc]

    def _is_instance(self, node: ast.Call) -> Value | None:
        slot = self._caught_of(node.args[0]) if len(node.args) == 2 else None
        if slot is None:
            return super()._is_instance(node)  # type: ignore[misc]
        tags = self._handler_tags(node.args[1])
        if tags is None:
            return core.const(self.b, True, BOOL)  # type: ignore[attr-defined]
        tag = self._rt("ppy_exc_tag", (core.load(self.b, slot),))  # type: ignore[attr-defined]
        found = core.const(self.b, False, BOOL)  # type: ignore[attr-defined]
        for wanted in tags:
            same = core.cmp(self.b, "eq", tag, self._word(wanted))  # type: ignore[attr-defined]
            found = core.bitwise(self.b, "or", found, same)  # type: ignore[attr-defined]
        return found


@dataclass(frozen=True, slots=True)
class _Called:
    """What a call answered, where its status was asked for."""

    results: tuple[Value, ...]


def _calls_super_init(node: ast.FunctionDef) -> bool:
    """Whether `__init__` calls `super().__init__(...)` on every path: a statement
    of its body, not one inside a branch or a loop."""
    for statement in node.body:
        call = statement.value if isinstance(statement, ast.Expr) else None
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "__init__"
            and isinstance(call.func.value, ast.Call)
            and isinstance(call.func.value.func, ast.Name)
            and call.func.value.func.id == "super"
        ):
            return True
    return False
