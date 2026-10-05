"""Effects in native code under `ppy run`: `print`, `input`, and calls into Python.

A native function that fails a guard runs again as Python from the start.
That is only sound while nothing the function did can be seen: a second
run would print its lines twice. So effects come in two sorts here.

- Output is held. `print` writes into the thread's buffer (`pyio.c`), and the
  boundary writes the buffer out through `sys.stdout` or `sys.stderr` when
  the call answers, or drops it when the call falls back and Python prints
  the same lines itself. A print costs no Python and no GIL, and a function
  that only prints keeps every guard it had.
- A barrier cannot be taken back: `input()`, `print(..., flush=True)`, and a
  call into Python. It writes out what is held first, so the order of
  output is CPython's, and from it on the call must never fall back.
  `check_effects` proves that over the IR: no guard that falls back, no call
  to a function that may fall back, and no exception the boundary could not
  raise as CPython would, is reachable after a barrier, in the function or in
  any caller after the call. A check CPython raises for is an exception
  there (the module is lowered in exception mode), so it raises natively
  with CPython's text; a check that only stands for a native limit (an
  integer past 64 bits) must have been proven away, or the function stays
  in Python.

A function with effects is native but holds the GIL at its boundary, and the
boundary's C wrapper, which knows nothing of held output, is not used for it.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field

from ..analysis import types as T
from ..analysis.closures import own_names
from ..analysis.decorators import reaches_body
from ..analysis.effects import Effect
from ..analysis.lexical import LexicalBindings
from ..analysis.stdlib import lookup as stdlib_lookup
from ..backend.llvm.lowering import Unsupported
from ..ir import (
    BOOL,
    F64,
    I64,
    IntType,
    IRFunction,
    IRModule,
    Operation,
    Successor,
    SymbolRef,
    Value,
)
from ..ir.dialects import core
from .collections import HANDLE, STR, class_tag

__all__ = ["EffectLowering", "EffectSummary", "check_effects", "wants_exceptions"]

#: The kinds of `pyio.c`.
_NONE, _INT, _FLOAT, _BOOL, _STR, _OBJECT, _KEYWORD = 0, 1, 2, 3, 4, 5, 6

#: A tuple result's kind: this bit, the count, and each item's kind (`pyio.c`).
_TUPLE = 8

#: A pure call's flag for a result the callee promises: one of another kind
#: raises `TypeError` rather than falling back (`pyio.c`).
_PROMISED = 2

#: Effects a callee may have and still be run twice unseen: it reads, allocates,
#: and may raise, and changes nothing.
_RERUNNABLE = frozenset(
    {Effect.ALLOC, Effect.READ_OBJECT, Effect.READ_MEMORY, Effect.READ_GLOBAL, Effect.MAY_RAISE}
)

#: Builtins that, given numbers, bools, strings, and `None`, run no code of the
#: program and change nothing: a call to one is run again unseen.
_PURE_BUILTINS = frozenset(
    {
        "abs", "all", "any", "ascii", "bin", "bool", "chr", "divmod", "float", "format",
        "hash", "hex", "int", "isinstance", "len", "max", "min", "oct", "ord", "pow",
        "repr", "round", "str", "sum",
    }
)  # fmt: skip

#: Builtins whose result is always of the type the checker gives it.
_CERTAIN_BUILTINS = frozenset({"input", "repr", "ascii", "str", "chr", "bin", "hex", "oct"})

#: `open`'s parameters in order, and what each is when not given.
_OPEN = ("file", "mode", "buffering", "encoding", "errors", "newline")

#: Streams, as `pyio.c` numbers them.
_STDOUT, _STDERR = 1, 2

#: Runtime calls that cannot be taken back.
BARRIERS = frozenset({"ppy_io_call", "ppy_io_flush_or_raise"})

#: Runtime calls that hold output.
HOLDS = frozenset({"ppy_io_print"})

#: Builtins whose call has no effect a later argument of `print` could show.
_QUIET = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "chr",
        "divmod",
        "float",
        "hex",
        "int",
        "len",
        "max",
        "min",
        "oct",
        "ord",
        "pow",
        "range",
        "repr",
        "round",
        "sorted",
        "str",
        "sum",
        "tuple",
        "list",
        "set",
        "dict",
        "isinstance",
        "enumerate",
        "zip",
        "reversed",
    }
)

#: Effects of a callee that a later argument of `print` must not have.
_LOUD = frozenset(
    {
        "io",
        "write_object",
        "write_global",
        "write_memory",
        "random",
        "python_callback",
        "python_dynamic",
        "external_unknown",
        "network",
        "process",
        "time",
        "thread",
        "sync",
    }
)


def wants_exceptions(nodes: list[ast.AST]) -> bool:
    """Whether a body reaches a barrier the lowering knows of before lowering it:
    a module that does is lowered in exception mode, so what may fail after
    the barrier raises natively instead of falling back."""
    for node in nodes:
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call) or not isinstance(inner.func, ast.Name):
                continue
            if inner.func.id in {"input", "open"}:
                return True
            if inner.func.id == "print" and any(k.arg == "flush" for k in inner.keywords):
                return True
    return False


def _result_kind(returned: T.Type) -> int:
    """The `pyio.c` kind a result of this type is taken back as, or -1."""
    found = {T.INT: _INT, T.FLOAT: _FLOAT, T.BOOL: _BOOL, T.STR: _STR}.get(returned)  # type: ignore[call-overload]
    if found is not None:
        return found  # type: ignore[no-any-return]
    if isinstance(returned, T.Tuple_) and not returned.homogeneous and 0 < len(returned.items) <= 8:
        kind = _TUPLE | len(returned.items) << 4
        for index, item in enumerate(returned.items):
            item_kind = {T.INT: _INT, T.FLOAT: _FLOAT, T.BOOL: _BOOL}.get(T.strip_literal(item))  # type: ignore[call-overload]
            if item_kind is None:
                return -1
            kind |= item_kind << (8 + 4 * index)
        return kind
    return -1


def _plain(t: T.Type) -> bool:
    """Whether `str()` of a value of this type runs none of the program's code."""
    t = T.strip_literal(t)
    if isinstance(t, T.Union_):
        return all(_plain(m) for m in t.members)
    if isinstance(t, T.Tuple_):
        return all(_plain(item) for item in t.items)
    if isinstance(t, T.Instance):
        return t.name in T.BUILTIN_MRO and all(_plain(a) for a in t.args)
    return False


class EffectLowering:  # pylint: disable=too-few-public-methods
    """`print` and `input` under `ppy run`; mixed into `_FunctionLowering`."""

    def _effects_on(self) -> bool:
        return bool(getattr(self.frontend, "effects", False))  # type: ignore[attr-defined]

    def _effect_call(self, node: ast.Call, discard: bool) -> Value | None:
        """`print(...)` and `input(...)` of the builtins; None for any other call."""
        if not self._effects_on():
            return None
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in self._effect_files()
        ):
            return self._effect_file_method(node, discard)
        if not isinstance(node.func, ast.Name):
            return None
        name = node.func.id
        if name not in {"print", "input"}:
            return None
        lexical = self.frontend.analysis.symbols.lexical  # type: ignore[attr-defined]
        if not isinstance(lexical, LexicalBindings) or lexical.targets_at(node.func) != {
            f"builtins.{name}"
        }:
            return None
        if name == "print":
            self._effect_print(node)
            return self._word(0)  # type: ignore[attr-defined,no-any-return]
        read = self._effect_input(node)
        if discard:
            self._release(read)  # type: ignore[attr-defined]
            return self._word(0)  # type: ignore[attr-defined,no-any-return]
        return read

    # -- print ------------------------------------------------------------------

    def _print_options(self, node: ast.Call) -> tuple[int, str, str, bool]:
        stream, separator, end, flush = _STDOUT, " ", "\n", False
        seen: set[str] = set()
        for keyword in node.keywords:
            name, value = keyword.arg, keyword.value
            if name is None or name in seen:
                raise Unsupported("`print(**options)` has no native lowering")
            seen.add(name)
            constant = value.value if isinstance(value, ast.Constant) else ...
            if name in {"sep", "end"}:
                if constant is None:
                    continue
                if not isinstance(constant, str):
                    raise Unsupported(f"`print({name}=...)` natively is a string literal or None")
                if name == "sep":
                    separator = constant
                else:
                    end = constant
            elif name == "flush":
                if not isinstance(constant, bool):
                    raise Unsupported("`print(flush=...)` natively is True or False")
                flush = constant
            elif name == "file":
                if constant is None:
                    continue
                stream = self._print_stream(value)
            else:
                raise Unsupported(f"`print` takes no keyword `{name}`")
        return stream, separator, end, flush

    def _print_stream(self, value: ast.expr) -> int:
        """`sys.stdout` or `sys.stderr`, looked up when the output is written."""
        if (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.attr in {"stdout", "stderr"}
        ):
            binding = self.frontend.analysis.symbols.imports.get(value.value.id)  # type: ignore[attr-defined]
            if binding is not None and binding.canonical == "sys":
                return _STDOUT if value.attr == "stdout" else _STDERR
        raise Unsupported("`print(file=...)` natively is `sys.stdout` or `sys.stderr`")

    def _effect_print(self, node: ast.Call) -> None:
        """`print(a, b, sep=..., end=..., file=..., flush=...)`.

        Python evaluates every argument before it writes any, then writes each
        one's `str()` in turn. Here each argument is written as it is
        evaluated, which is the same wherever the arguments after the first
        have no effect a later one could show; `str()` of a value of the
        program's own class runs its `__str__`, which may print, so the text
        before it is held first, as CPython would have written it."""
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise Unsupported("`print(*values)` has no native lowering")
        for argument in node.args[1:]:
            reason = self._loud(argument)
            if reason:
                raise Unsupported(
                    f"`print` whose later argument {reason}: Python evaluates every "
                    "argument before it prints any"
                )
        stream, separator, end, flush = self._print_options(node)
        rt = self._rt  # type: ignore[attr-defined]
        builder = rt("ppy_str_builder", (self._word(0),), HANDLE)  # type: ignore[attr-defined]
        for index, argument in enumerate(node.args):
            if index:
                self._add_text(builder, separator)  # type: ignore[attr-defined]
            if isinstance(argument, ast.Constant) and (
                argument.value is None or isinstance(argument.value, str)
            ):
                self._add_text(builder, str(argument.value))  # type: ignore[attr-defined]
                continue
            if not _plain(self._type_of(argument)):  # type: ignore[attr-defined]
                self._hold_text(stream, builder)
                builder = rt("ppy_str_builder", (self._word(0),), HANDLE)  # type: ignore[attr-defined]
            self._add_formatted(builder, argument, -1, "")  # type: ignore[attr-defined]
        self._add_text(builder, end)  # type: ignore[attr-defined]
        self._hold_text(stream, builder)
        if flush:
            status = rt("ppy_io_flush_or_raise", (self._word(stream),))  # type: ignore[attr-defined]
            self._after_barrier(status)

    def _hold_text(self, stream: int, builder: Value) -> None:
        rt = self._rt  # type: ignore[attr-defined]
        text = rt("ppy_str_finish", (builder,), HANDLE)
        rt("ppy_io_print", (self._word(stream), text), None)  # type: ignore[attr-defined]
        self._release(text)  # type: ignore[attr-defined]

    def _loud(self, node: ast.expr) -> str:
        """What in `node` may have an effect a later argument of `print` would
        show, or "" where nothing does."""
        functions = self.frontend.analysis.functions  # type: ignore[attr-defined]
        declared = self.frontend.declared  # type: ignore[attr-defined]
        for inner in ast.walk(node):
            if isinstance(inner, (ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)):
                return "suspends or assigns"
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            if isinstance(func, ast.Name):
                if func.id in _QUIET and func.id not in declared:
                    continue
                callee = next(
                    (a for q, a in functions.items() if q.rpartition(".")[2] == func.id), None
                )
                if callee is not None and not set(callee.effects.spelled()) & _LOUD:
                    continue
                return f"calls `{func.id}`"
            if isinstance(func, ast.Attribute):
                if self._string_of(func.value) is not None:  # type: ignore[attr-defined]
                    continue
                if func.attr in {"count", "index", "get", "keys", "values", "items", "copy"}:
                    continue
                return f"calls `.{func.attr}()`"
            return "makes a call"
        return ""

    # -- text files ---------------------------------------------------------------

    def _effect_files(self) -> dict[str, Value]:
        """Names bound by `with open(...) as f`: each its slot, holding the file's
        number at the boundary (`ppy_runtime/effects.py`)."""
        return self.__dict__.setdefault("_effect_file_slots", {})  # type: ignore[no-any-return]

    def _effect_with(self, node: ast.With) -> None:
        """`with open(path, ...) as f:` over a text file: Python's own `open`,
        so encodings and newlines are CPython's, and `f.close()` on every way
        out of the block, as the file's `__exit__` does."""
        if not self._effects_on() or len(node.items) != 1:
            raise Unsupported("`with` natively opens one text file: `with open(path) as f:`")
        item = node.items[0]
        call = item.context_expr
        lexical = self.frontend.analysis.symbols.lexical  # type: ignore[attr-defined]
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and isinstance(lexical, LexicalBindings)
            and lexical.targets_at(call.func) == {"builtins.open"}
        ):
            raise Unsupported("`with` natively opens one text file: `with open(path) as f:`")
        if item.optional_vars is not None and not isinstance(item.optional_vars, ast.Name):
            raise Unsupported("the file `with open(...)` opens is bound to a name")
        name = item.optional_vars.id if item.optional_vars is not None else ".file"
        arguments = self._open_arguments(call)
        opened = self._python_call("builtins:open", arguments, _OBJECT)
        assert opened is not None
        slot = self._alloca(I64, f"{name}.file")  # type: ignore[attr-defined]
        core.store(self.b, opened, slot)  # type: ignore[attr-defined]
        self._effect_files()[name] = slot
        closing = ast.Expr(
            ast.Call(ast.Attribute(ast.Name(name, ast.Load()), "close", ast.Load()), [], [])
        )
        guarded = ast.Try(body=node.body, handlers=[], orelse=[], finalbody=[closing])
        ast.copy_location(guarded, node)
        ast.fix_missing_locations(guarded)
        self.frontend.synthetic.append(guarded)  # type: ignore[attr-defined]
        self._try(guarded)  # type: ignore[attr-defined]

    def _open_arguments(self, call: ast.Call) -> list[tuple[int, Value, bool]]:
        """`open`'s arguments, each in its place: a text mode, and strings or None."""
        given: dict[str, ast.expr] = dict(zip(_OPEN, call.args, strict=False))
        if len(call.args) > len(_OPEN):
            raise Unsupported("`open` takes at most six arguments natively")
        for keyword in call.keywords:
            if keyword.arg not in _OPEN or keyword.arg in given:
                raise Unsupported(f"`open({keyword.arg or '**'}=...)` has no native lowering")
            given[keyword.arg] = keyword.value
        mode = given.get("mode")
        if mode is not None and not (
            isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" not in mode.value
        ):
            raise Unsupported("`open` natively takes a text mode as a string literal")
        arguments: list[tuple[int, Value, bool]] = []
        for parameter in _OPEN:
            value = given.get(parameter)
            if value is None:
                default = {"mode": "r", "buffering": -1}.get(parameter)
                if isinstance(default, str):
                    data = self._rt("ppy_str_interned", self._text_data(default), HANDLE)  # type: ignore[attr-defined]
                    arguments.append((_STR, data, False))
                elif default is not None:
                    arguments.append((_INT, self._word(default), False))  # type: ignore[attr-defined]
                else:
                    arguments.append((_NONE, self._word(0), False))  # type: ignore[attr-defined]
                continue
            if isinstance(value, ast.Constant) and value.value is None:
                arguments.append((_NONE, self._word(0), False))  # type: ignore[attr-defined]
            elif parameter == "buffering":
                arguments.append((_INT, self._coerce(self._expr(value), "int"), False))  # type: ignore[attr-defined]
            elif self._string_of(value) is not None:  # type: ignore[attr-defined]
                handle, owned = self._handle(value)  # type: ignore[attr-defined]
                arguments.append((_STR, handle, owned))
            else:
                raise Unsupported(f"`open({parameter}=...)` natively is a string or None")
        return arguments

    def _effect_file_method(self, node: ast.Call, discard: bool) -> Value:
        """`f.read()`, `f.readline()`, `f.readlines()`, `f.write(s)`, `f.close()`."""
        func = node.func
        assert isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
        method = func.attr
        if node.keywords:
            raise Unsupported(f"`{ast.unparse(func)}` takes no keywords natively")
        receiver = (_OBJECT, core.load(self.b, self._effect_files()[func.value.id]), False)  # type: ignore[attr-defined]
        if method in {"read", "readline"} and len(node.args) <= 1:
            arguments = [receiver]
            if node.args:
                size = self._coerce(self._expr(node.args[0]), "int")  # type: ignore[attr-defined]
                arguments.append((_INT, size, False))
            text = self._python_method(method, arguments, _STR)
            assert text is not None  # a string result is a value
            if discard:
                self._release(text)  # type: ignore[attr-defined]
                return self._word(0)  # type: ignore[attr-defined,no-any-return]
            return text
        if method == "readlines" and not node.args:
            return self._read_lines(receiver[1], discard)
        if method == "write" and len(node.args) == 1 and discard:
            if self._string_of(node.args[0]) is None:  # type: ignore[attr-defined]
                raise Unsupported("a text file is written a string")
            handle, owned = self._handle(node.args[0])  # type: ignore[attr-defined]
            self._python_method("write", [receiver, (_STR, handle, owned)], _NONE)
            return self._word(0)  # type: ignore[attr-defined,no-any-return]
        if method in {"close", "flush"} and not node.args:
            self._python_method(method, [receiver], _NONE)
            return self._word(0)  # type: ignore[attr-defined,no-any-return]
        raise Unsupported(f"`{ast.unparse(func)}(...)` of a text file has no native lowering")

    def _read_lines(self, number: Value, discard: bool) -> Value:
        """`f.readlines()`: `readline` until it gives the empty string, which is
        what `readlines` does for a text file."""
        listed = self._rt("ppy_str_list", (), HANDLE)  # type: ignore[attr-defined]
        self._each_line(number, lambda line: self._rt("ppy_str_list_take", (listed, line), None))  # type: ignore[attr-defined]
        if discard:
            self._release(listed)  # type: ignore[attr-defined]
            return self._word(0)  # type: ignore[attr-defined,no-any-return]
        return listed

    def _each_line(self, number: Value, take: object, body: list[ast.stmt] | None = None) -> None:
        """A loop over a file's lines: each one, a new string, handed to `take`,
        or bound to the loop's name with `body` run for it."""
        b = self.b  # type: ignore[attr-defined]
        header = self._block("lines.head")  # type: ignore[attr-defined]
        more = self._block("lines.body")  # type: ignore[attr-defined]
        latch = self._block("lines.latch")  # type: ignore[attr-defined]
        done = self._block("lines.end")  # type: ignore[attr-defined]
        core.br(b, Successor(header))
        b.at_end(header)
        line = self._python_method("readline", [(_OBJECT, number, False)], _STR)
        assert line is not None
        empty = core.cmp(b, "eq", self._rt("ppy_str_bytes", (line,)), self._word(0))  # type: ignore[attr-defined]
        ended = self._block("lines.last")  # type: ignore[attr-defined]
        core.cond_br(b, empty, Successor(ended), Successor(more))
        b.at_end(ended)
        self._release(line)  # type: ignore[attr-defined]
        core.br(b, Successor(done))
        b.at_end(more)
        if body is None:
            take(line)  # type: ignore[operator]
            core.br(b, Successor(latch))
        else:
            assert isinstance(take, str)
            self._bind(take, STR, line, True)  # type: ignore[attr-defined]
            self._loops.append((latch, done))  # type: ignore[attr-defined]
            self._body(body)  # type: ignore[attr-defined]
            self._loops.pop()  # type: ignore[attr-defined]
            if self._open():  # type: ignore[attr-defined]
                core.br(b, Successor(latch))
        b.at_end(latch)
        if not self._dead_latch(latch):  # type: ignore[attr-defined]
            core.br(b, Successor(header))
        b.at_end(done)

    def _effect_for(self, node: ast.For) -> bool:
        """`for line in f:` over a file `with open(...)` opened: `readline` until
        the empty string, which is what iterating a text file does."""
        if not (isinstance(node.iter, ast.Name) and node.iter.id in self._effect_files()):
            return False
        if node.orelse or not isinstance(node.target, ast.Name):
            raise Unsupported("a file's lines are bound to one name, with no `else`")
        number = core.load(self.b, self._effect_files()[node.iter.id])  # type: ignore[attr-defined]
        self._each_line(number, node.target.id, node.body)
        return True

    def _python_method(
        self, method: str, arguments: list[tuple[int, Value, bool]], kind: int
    ) -> Value | None:
        """`arguments[0].method(*arguments[1:])`, called through Python."""
        return self._python_call(method, arguments, kind, method=True)

    # -- input and calls into Python --------------------------------------------

    def _effect_input(self, node: ast.Call) -> Value:
        """`input()` and `input(prompt)`: Python's own, called with the GIL."""
        if node.keywords or len(node.args) > 1:
            raise Unsupported("`input` takes at most one argument")
        arguments: list[tuple[int, Value, bool]] = []
        if node.args:
            if self._string_of(node.args[0]) is None:  # type: ignore[attr-defined]
                raise Unsupported("`input(prompt)` natively takes a string prompt")
            handle, owned = self._handle(node.args[0])  # type: ignore[attr-defined]
            arguments.append((_STR, handle, owned))
        line = self._python_call("builtins:input", arguments, _STR)
        assert line is not None  # a string result is a value
        return line

    def _effect_python_call(self, node: ast.Call, discard: bool) -> Value | None:
        """A call to a function native code has no lowering for, made through
        Python: a function of this module that stayed in Python, one of a
        module imported, or a builtin. Its arguments are numbers, bools,
        strings, and `None`, boxed, by position or by keyword; its result one
        of those, a tuple of numbers, or nothing. None where the call is not
        one of these.

        A callee that changes nothing a second run could see (`_rerunnable`)
        is no barrier: its result is checked to be exactly what the checker
        said, and where it is not, the native call falls back and Python runs
        it all again. Any other callee is a barrier, after which nothing may
        fall back, so its result is taken only where it cannot be anything
        but what native code holds (`_certain`)."""
        if not self._effects_on():
            return None
        func = node.func
        probe = func
        while isinstance(probe, ast.Attribute):
            probe = probe.value
        if not isinstance(probe, ast.Name) or probe.id in self._effect_locals():
            return None
        symbols = self.frontend.analysis.symbols  # type: ignore[attr-defined]
        head = probe.id
        if isinstance(func, ast.Name):
            known = head in symbols.functions or head in symbols.imports
            if not known:
                lexical = symbols.lexical
                known = isinstance(lexical, LexicalBindings) and lexical.targets_at(func) == {
                    f"builtins.{head}"
                }
        else:
            known = head in symbols.imports
        if not known:
            return None
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise Unsupported("a call into Python with `*arguments` has no native lowering")
        if any(k.arg is None for k in node.keywords):
            raise Unsupported("a call into Python with `**keywords` has no native lowering")
        returned = T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]
        pure = self._rerunnable(node)
        kind = _NONE
        certain = True
        if not discard and returned != T.NONE:
            kind = _result_kind(returned)
            certain = kind >= 0 and self._certain(node, returned)
            if kind < 0 or (not pure and not certain):
                why = (
                    "which native code has no value for"
                    if kind < 0
                    else "which Python does not promise, and after the call native code "
                    "cannot fall back to take something else"
                )
                raise Unsupported(
                    f"`{ast.unparse(func)}` stays in Python and gives `{returned}`, {why}"
                )
        callee = ast.unparse(func)
        arguments = [self._crossing(argument, callee) for argument in node.args]
        for keyword in node.keywords:
            assert keyword.arg is not None
            name = self._string_literal(keyword.arg)  # type: ignore[attr-defined]
            arguments.append((_KEYWORD, name, False))
            arguments.append(self._crossing(keyword.value, callee))
        module = self.info.module  # type: ignore[attr-defined]
        made = self._python_call(
            f"{module}:{ast.unparse(func)}", arguments, kind, pure=pure, checked=not certain
        )
        return made if made is not None else self._word(0)  # type: ignore[attr-defined,no-any-return]

    def _module_file(self) -> Value | None:
        """`__file__`: the module's own, read through Python where it is called
        (a module may be imported from anywhere). Reading it changes nothing,
        so it is no barrier. None in a build without Python."""
        if not self._effects_on() or self.frontend.standalone:  # type: ignore[attr-defined]
            return None
        if "__file__" in self._effect_locals():
            return None
        module = self.info.module  # type: ignore[attr-defined]
        return self._python_call(f"{module}:__file__.__str__", [], _STR, pure=True)

    def _crossing(self, argument: ast.expr, callee: str) -> tuple[int, Value, bool]:
        """One argument of a call into Python: its kind, its value, and whether
        the caller owns it."""
        if isinstance(argument, ast.Constant) and argument.value is None:
            return (_NONE, self._word(0), False)  # type: ignore[attr-defined]
        if self._string_of(argument) is not None:  # type: ignore[attr-defined]
            handle, owned = self._handle(argument)  # type: ignore[attr-defined]
            return (_STR, handle, owned)
        refused = Unsupported(
            f"`{callee}` stays in Python, and `{ast.unparse(argument)}` crosses into Python "
            "only as a number, a bool, or a string"
        )
        given = T.strip_literal(self._type_of(argument))  # type: ignore[attr-defined]
        if given not in (T.INT, T.FLOAT, T.BOOL):
            raise refused
        value = self._expr(argument)  # type: ignore[attr-defined]
        argument_kind = {I64: _INT, F64: _FLOAT}.get(value.type)
        if argument_kind is None and value.type == BOOL:
            argument_kind = _BOOL
        if argument_kind is None:
            raise refused
        return (argument_kind, value, False)

    def _rerunnable(self, node: ast.Call) -> bool:
        """Whether a second run of the callee could not be told from the first:
        a builtin given numbers and strings, a `math` function, or a function
        of this module whose effects are only reading and allocating. Such a
        call is no barrier, and the native call may fall back after it."""
        func = node.func
        symbols = self.frontend.analysis.symbols  # type: ignore[attr-defined]
        if isinstance(func, ast.Name):
            info = symbols.functions.get(func.id)
            if info is not None:
                effects = info.effects
                return not info.dynamic and not set(effects.effects) - _RERUNNABLE
            return func.id in _PURE_BUILTINS and func.id not in symbols.imports
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            binding = symbols.imports.get(func.value.id)
            return binding is not None and binding.canonical == "math"
        return False

    def _certain(self, node: ast.Call, returned: T.Type) -> bool:
        """Whether a barrier callee's result can only be of the type the checker
        gave it, which native code holds: a builtin or a modelled library
        function's own result (CPython keeps those promises), or a function of
        this module every `return` of which gives exactly that type. An `int`
        never is, whatever its callee: Python's may not fit 64 bits."""
        func = node.func
        symbols = self.frontend.analysis.symbols  # type: ignore[attr-defined]
        if returned == T.INT:
            # A length, a code point, and a hash are words by CPython's own making.
            return (
                isinstance(func, ast.Name)
                and func.id in {"len", "ord", "hash"}
                and func.id not in symbols.functions
                and func.id not in symbols.imports
            )
        if returned not in (T.STR, T.BOOL, T.FLOAT):
            return False
        if isinstance(func, ast.Name):
            info = symbols.functions.get(func.id)
            if info is not None:
                # A wrapper the decorator made gives what it likes.
                return reaches_body(info.decorators) and self._returns_exactly(info.node, returned)
            binding = symbols.imports.get(func.id)
            if binding is not None:
                # `from timeit import timeit`: a modelled standard-library
                # function, whose result CPython makes what the model says.
                return (
                    binding.canonical.partition(".")[0] in sys.stdlib_module_names
                    and stdlib_lookup(binding.canonical) is not None
                )
            return func.id in _CERTAIN_BUILTINS
        root = func
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(func, ast.Attribute) and isinstance(root, ast.Name):
            binding = symbols.imports.get(root.id)
            if binding is None:
                return False
            return binding.canonical.partition(".")[0] in sys.stdlib_module_names
        return False

    def _returns_exactly(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, wanted: T.Type
    ) -> bool:
        """Whether every way out of `node` returns a value of exactly `wanted`."""
        if isinstance(node, ast.AsyncFunctionDef) or not node.body:
            return False
        if not isinstance(node.body[-1], (ast.Return, ast.Raise)):
            return False  # it may fall off the end and give `None`
        pending: list[ast.AST] = list(node.body)
        while pending:
            inner = pending.pop()
            if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            if isinstance(inner, (ast.Yield, ast.YieldFrom)):
                return False
            if isinstance(inner, ast.Return):
                if inner.value is None:
                    return False
                given = T.strip_literal(self._type_of(inner.value))  # type: ignore[attr-defined]
                if given != wanted:
                    return False
            pending.extend(ast.iter_child_nodes(inner))
        return True

    def _effect_locals(self) -> frozenset[str]:
        """Names the function binds itself: a call to one is not to a module's."""
        found = self.__dict__.get("_effect_bound")
        if found is None:
            node = self.info.node  # type: ignore[attr-defined]
            names = {a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg)}
            for inner in ast.walk(node):
                if isinstance(inner, ast.Name) and isinstance(inner.ctx, (ast.Store, ast.Del)):
                    names.add(inner.id)
                elif isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if inner is not node:
                        names.add(inner.name)
                elif isinstance(inner, (ast.Global, ast.Nonlocal)):
                    names.update(inner.names)
                elif isinstance(inner, ast.alias):
                    names.add((inner.asname or inner.name).partition(".")[0])
            # A nested function sees the names of the functions around it too.
            sources = getattr(self.frontend, "sources", {})  # type: ignore[attr-defined]
            outer = sources.get(self.info.enclosing or "")  # type: ignore[attr-defined]
            while outer is not None:
                names |= own_names(outer[0].node)
                outer = sources.get(outer[0].enclosing or "")
            found = self.__dict__["_effect_bound"] = frozenset(names)
        return found  # type: ignore[no-any-return]

    def _python_call(
        self,
        spelled: str,
        arguments: list[tuple[int, Value, bool]],
        kind: int,
        *,
        method: bool = False,
        pure: bool = False,
        checked: bool = True,
    ) -> Value | None:
        """Call the Python callable `module:name` with the arguments boxed, and
        unbox its result as `kind`; what it raises is raised natively. A `pure`
        call is no barrier; where its result is not `kind` it falls back, or,
        not `checked` (the result is promised), raises `TypeError`."""
        rt = self._rt  # type: ignore[attr-defined]
        for argument_kind, value, _owned in arguments:
            if argument_kind == _KEYWORD:
                rt("ppy_io_push", (self._word(_KEYWORD), core.cast(self.b, value, I64)), None)  # type: ignore[attr-defined]
            elif argument_kind == _NONE:
                rt("ppy_io_push", (self._word(_NONE), self._word(0)), None)  # type: ignore[attr-defined]
            elif argument_kind == _STR:
                rt("ppy_io_push_text", (value,), None)
            elif argument_kind == _FLOAT:
                rt("ppy_io_push_float", (value,), None)
            else:
                word = value if value.type == I64 else core.cast(self.b, value, I64)  # type: ignore[attr-defined]
                rt("ppy_io_push", (self._word(argument_kind), word), None)  # type: ignore[attr-defined]
        data, length = self._text_data(spelled)  # type: ignore[attr-defined]
        call = "ppy_io_call_pure" if pure else "ppy_io_call"
        flags = int(method) | (_PROMISED if pure and not checked else 0)
        status = rt(call, (data, length, self._word(kind), self._word(flags)))  # type: ignore[attr-defined]
        for argument_kind, value, owned in arguments:
            if argument_kind == _STR and owned:
                self._release(value)  # type: ignore[attr-defined]
        if pure:
            self._after_pure_call(status, checked)
        else:
            self._after_barrier(status)
        if kind & _TUPLE:
            items = []
            for index in range((kind >> 4) & 0xF):
                item_kind = (kind >> (8 + 4 * index)) & 0xF
                word = rt("ppy_io_result_at", (self._word(index),))  # type: ignore[attr-defined]
                if item_kind == _FLOAT:
                    items.append(core.cast(self.b, word, F64))  # type: ignore[attr-defined]
                elif item_kind == _BOOL:
                    items.append(core.cmp(self.b, "ne", word, self._word(0)))  # type: ignore[attr-defined]
                else:
                    items.append(word)
            return core.tuple_make(self.b, *items)  # type: ignore[attr-defined,no-any-return]
        if kind == _STR:
            return rt("ppy_io_result_text", (), HANDLE)  # type: ignore[no-any-return]
        if kind == _FLOAT:
            return rt("ppy_io_result_float", (), F64)  # type: ignore[no-any-return]
        if kind == _BOOL:
            word = rt("ppy_io_result", ())
            return core.cmp(self.b, "ne", word, self._word(0))  # type: ignore[attr-defined,no-any-return]
        if kind in {_INT, _OBJECT}:
            return rt("ppy_io_result", ())  # type: ignore[no-any-return]
        return None

    def _after_pure_call(self, status: Value, checked: bool) -> None:
        """A pure call's status: 0; 1 where its result was of another kind, and
        the native call falls back; or -1 with Python's exception pending. A
        promised result is never 1: a broken promise is a `TypeError`, so no
        check follows a barrier for it."""
        if checked:
            mismatched = core.cmp(self.b, "ne", status, self._word(1))  # type: ignore[attr-defined]
            core.guard(self.b, mismatched, "contract", "a call into Python gave another type")  # type: ignore[attr-defined]
        answered = core.cmp(self.b, "eq", status, self._word(0))  # type: ignore[attr-defined]
        if not self._exceptions_on():  # type: ignore[attr-defined]
            # Nothing was crossed: Python runs the call again and raises it.
            core.guard(self.b, answered, "contract", "Python raised")  # type: ignore[attr-defined]
            return
        kept = self._block("effect.ok")  # type: ignore[attr-defined]
        failed = self._block("effect.raised")  # type: ignore[attr-defined]
        core.cond_br(self.b, answered, Successor(kept), Successor(failed))  # type: ignore[attr-defined]
        self.b.at_end(failed)  # type: ignore[attr-defined]
        self._go_raise()  # type: ignore[attr-defined]
        self.b.at_end(kept)  # type: ignore[attr-defined]

    def _after_barrier(self, status: Value) -> None:
        """A barrier's status: 0, or -1 with Python's exception pending natively."""
        answered = core.cmp(self.b, "eq", status, self._word(0))  # type: ignore[attr-defined]
        if not self._exceptions_on():  # type: ignore[attr-defined]
            # Lowered again in exception mode (`lower_module_to_ir`); this pass
            # only finds that the module needs it.
            self.frontend.wants_exceptions = True  # type: ignore[attr-defined]
            core.guard(self.b, answered, "effect", "Python raised")  # type: ignore[attr-defined]
            return
        kept = self._block("effect.ok")  # type: ignore[attr-defined]
        failed = self._block("effect.raised")  # type: ignore[attr-defined]
        core.cond_br(self.b, answered, Successor(kept), Successor(failed))  # type: ignore[attr-defined]
        self.b.at_end(failed)  # type: ignore[attr-defined]
        self._go_raise()  # type: ignore[attr-defined]
        self.b.at_end(kept)  # type: ignore[attr-defined]


# -- the barrier rule over the IR ---------------------------------------------------------


@dataclass(slots=True)
class EffectSummary:
    """What a function may do, callees included."""

    #: It holds output (prints).
    holds: bool = False
    #: It reaches a barrier.
    barrier: bool = False
    #: It may fall back to Python.
    falls_back: bool = False
    #: It may raise an exception the boundary could not raise as CPython would.
    raises_inexactly: bool = False
    #: Why, for a report: the first of each found.
    why: dict[str, str] = field(default_factory=dict)

    def key(self) -> tuple[bool, bool, bool, bool]:
        return (self.holds, self.barrier, self.falls_back, self.raises_inexactly)


#: Dialects whose operations may fall back where the backend lowers them.
_FALLING_DIALECTS = frozenset({"concurrency", "aio", "parallel", "gpu"})


def _builtin_tags() -> frozenset[int]:
    return frozenset(
        class_tag(f"builtins.{name}")
        for name, mro in T.BUILTIN_MRO.items()
        if "BaseException" in mro
    )


def _owner(value: Value) -> object:
    try:
        return value.owner
    except NotImplementedError:
        return None


def _constant(value: Value) -> object:
    owner = _owner(value)
    if isinstance(owner, Operation) and owner.name == "core.const":
        return owner.attributes.get("value")
    return None


def _checked_arithmetic(op: Operation) -> bool:
    """An integer operation the backend checks for overflow, falling back
    (`from_ir._arith`, `_neg`, `_divmod`, `_shift`)."""
    name = op.name
    if name not in {
        "core.add",
        "core.sub",
        "core.mul",
        "core.neg",
        "core.div",
        "core.mod",
        "core.shl",
    }:
        return False
    if not op.results or not isinstance(op.results[0].type, IntType):
        return False
    default = "wrap" if name == "core.shl" else "python"
    overflow = op.attributes.get("overflow", default)
    unchecked = (
        {"wrap", "native"}
        if name in {"core.neg", "core.div", "core.mod", "core.shl"}
        else {
            "wrap",
            "native",
            "proven",
        }
    )
    return overflow not in unchecked


def _callee(op: Operation) -> str | None:
    found = op.attributes.get("callee")
    return found.name if isinstance(found, SymbolRef) else None


def _extern(op: Operation) -> str | None:
    if op.name != "core.call_extern":
        return None
    return str(op.attributes.get("callee"))


def _linked_call(op: Operation) -> Operation | None:
    """The call whose status a `core.guard` asks after (`_call_native`), if any."""
    if op.name != "core.guard" or not op.operands:
        return None
    condition = _owner(op.operands[0])
    if not isinstance(condition, Operation) or condition.name != "core.cmp":
        return None
    for operand in condition.operands:
        made = _owner(operand)
        if (
            isinstance(made, Operation)
            and made.name in {"core.call", "core.call_indirect"}
            and made.attributes.get("capture_status")
            and operand is made.results[-1]
        ):
            return made
    return None


class _Checker:
    """The barrier rule over one module (see the module's docstring)."""

    def __init__(self, module: IRModule, exceptions: bool, left: frozenset[str]) -> None:
        self.module = module
        self.exceptions = exceptions
        #: This module's functions that stayed in Python: a caller of one stays
        #: too, for that reason, whatever it would have done here.
        self.left = left
        self.summaries: dict[str, EffectSummary] = {}
        self.builtin_tags = _builtin_tags()

    def external(self, name: str) -> EffectSummary:
        """A callee this module only declares: assumed to fall back and raise;
        a barrier where its signature says it has effects."""
        target = self.module.functions.get(name)
        effects = bool(target is not None and target.attributes.get("ppy.effects"))
        return EffectSummary(holds=effects, barrier=effects, falls_back=True, raises_inexactly=True)

    def summary(self, name: str) -> EffectSummary:
        found = self.summaries.get(name)
        if found is not None:
            return found
        if name in self.left:
            return EffectSummary()
        target = self.module.functions.get(name)
        if target is None or target.is_declaration:
            return self.external(name)
        return EffectSummary()

    def bad_raise(self, op: Operation) -> bool:
        """`ppy_exc_raise` of anything but a builtin exception with CPython's text."""
        if _extern(op) != "ppy_exc_raise" or not op.operands or op.attributes.get("ppy.rethrow"):
            return False
        made = _owner(op.operands[0])
        if not isinstance(made, Operation) or _extern(made) != "ppy_exc_make":
            return True
        tag, flags = _constant(made.operands[0]), _constant(made.operands[3])
        return not (
            isinstance(tag, int)
            and tag in self.builtin_tags
            and isinstance(flags, int)
            and flags & 1
        )

    def classify(self, op: Operation) -> tuple[str, str, str]:
        """(falls back, raises inexactly, barrier) reasons for one operation; "" for none."""
        falls = inexact = barrier = ""
        name = op.name
        if name == "core.guard":
            label = str(op.attributes.get("label") or "")
            if label != "raised" and not label.startswith("sanitize:") and _linked_call(op) is None:
                message = str(op.attributes.get("message") or op.attributes.get("kind"))
                falls = f"a check that falls back ({message})"
        elif name == "core.call":
            callee = _callee(op) or ""
            summary = self.summary(callee)
            shown = self.spelled(callee)
            if summary.falls_back:
                falls = f"a call to `{shown}`, which may fall back,"
            if summary.raises_inexactly:
                inexact = f"a call to `{shown}`, which may raise what Python would say otherwise,"
            if (
                not self.exceptions
                and not op.attributes.get("capture_status")
                and (summary.raises_inexactly or summary.falls_back)
            ):
                falls = falls or f"a call to `{shown}`, whose exception falls back,"
            if summary.barrier:
                barrier = f"`{shown}()`"
        elif name == "core.call_indirect":
            falls = "a call through a function value"
            inexact = falls
        elif name == "core.call_extern":
            extern = _extern(op)
            if extern == "ppy_io_call":
                barrier = self.python_callee(op)
            elif extern in BARRIERS:
                barrier = "`print(flush=True)`"
            elif self.bad_raise(op):
                inexact = "an exception PPy cannot raise with Python's own text"
        elif op.dialect in _FALLING_DIALECTS:
            falls = f"`{name}`"
        elif _checked_arithmetic(op):
            falls = "integer arithmetic that may not fit 64 bits"
        return falls, inexact, barrier

    def python_callee(self, op: Operation) -> str:
        """`input()` or the like: what a call into Python calls."""
        made = _owner(op.operands[0]) if op.operands else None
        symbol = made.attributes.get("symbol") if isinstance(made, Operation) else None
        found = self.module.globals.get(str(symbol)) if symbol is not None else None
        spelled = str(found.value) if found is not None and found.value is not None else ""
        name = spelled.partition(":")[2]
        return f"`{name}()`" if name else "a call into Python"

    def spelled(self, name: str) -> str:
        """A function's qualname, for a report."""
        target = self.module.functions.get(name)
        return str(target.attributes.get("ppy.qualname", name)) if target is not None else name

    def summarize(self, function: IRFunction) -> EffectSummary:
        made = EffectSummary()
        for op in function.operations():
            falls, inexact, barrier = self.classify(op)
            if _extern(op) in HOLDS:
                made.holds = True
            if op.name == "core.call" and self.summary(_callee(op) or "").holds:
                made.holds = True
            if falls and not made.falls_back:
                made.falls_back = True
                made.why["falls_back"] = falls
            if inexact and not made.raises_inexactly:
                made.raises_inexactly = True
                made.why["raises_inexactly"] = inexact
            if barrier and not made.barrier:
                made.barrier = True
                made.why["barrier"] = barrier
        made.holds = made.holds or made.barrier
        return made

    def settle(self) -> None:
        """Every function's summary, to a fixpoint over calls (recursion included)."""
        bodies = [f for f in self.module.functions.values() if not f.is_declaration]
        for function in bodies:
            self.summaries[function.name] = EffectSummary()
        changed = True
        while changed:
            changed = False
            for function in bodies:
                made = self.summarize(function)
                if made.key() != self.summaries[function.name].key():
                    self.summaries[function.name] = made
                    changed = True

    def violation(self, function: IRFunction) -> str:
        """What in `function` may happen after a barrier that must not, or ""."""
        blocks = list(function.blocks())
        barrier_at: dict[int, tuple[int, str]] = {}
        for block in blocks:
            for index, op in enumerate(block):
                _falls, _inexact, barrier = self.classify(op)
                if barrier:
                    barrier_at[id(block)] = (index, barrier)
                    break
        if not barrier_at:
            return ""
        # Blocks entered after a barrier ran: anything a barrier's block reaches.
        after: set[int] = set()
        work = [s for b in blocks if id(b) in barrier_at for s in b.successors]
        while work:
            block = work.pop()
            if id(block) in after:
                continue
            after.add(id(block))
            work.extend(block.successors)
        first = next(iter(barrier_at.values()))[1]
        for block in blocks:
            start = 0
            if id(block) not in after:
                if id(block) not in barrier_at:
                    continue
                start = barrier_at[id(block)][0] + 1
            for op in list(block)[start:]:
                falls, inexact, _barrier = self.classify(op)
                if falls:
                    return f"{falls} can follow {first}"
                if inexact:
                    return f"{inexact} can follow {first}"
                if (
                    _extern(op) == "ppy_exc_raise"
                    and op.attributes.get("ppy.rethrow")
                    and self.summaries[function.name].raises_inexactly
                ):
                    # Raised again, caught from anywhere in the function, before
                    # the barrier as well: judged by what the function raises at all.
                    why = self.summaries[function.name].why["raises_inexactly"]
                    return f"{why}, raised again, can follow {first}"
        return ""

    def referenced(self) -> dict[str, set[str]]:
        """Functions named other than by a direct call: address taken, spawned, run
        in parallel. Their callers are not known, so neither is what follows."""
        found: dict[str, set[str]] = {}
        for function in self.module.functions.values():
            if function.is_declaration:
                continue
            for op in function.operations():
                if op.name in {"core.call", "async.create"}:
                    continue
                for value in op.attributes.values():
                    if isinstance(value, SymbolRef):
                        found.setdefault(value.name, set()).add(function.name)
        return found


def check_effects(
    module: IRModule, exceptions: bool, left: frozenset[str] = frozenset()
) -> tuple[dict[str, str], dict[str, EffectSummary]]:
    """The barrier rule over a module: the functions that break it, each with why
    (callers of one included), and every function's summary. `left` are the
    module's functions that stayed in Python, whose callers go for that."""
    checker = _Checker(module, exceptions, left)
    checker.settle()
    broken: dict[str, str] = {}
    referenced = checker.referenced()
    for function in module.functions.values():
        if function.is_declaration:
            continue
        summary = checker.summaries[function.name]
        if summary.holds and function.name in referenced:
            broken[function.name] = (
                "prints or reads, and runs where its caller is not known "
                "(a function value, a thread, or a parallel loop)"
            )
            continue
        why = checker.violation(function)
        if why:
            broken[function.name] = why
    # Whatever calls or names a function that broke the rule goes with it.
    changed = True
    while changed:
        changed = False
        for function in module.functions.values():
            if function.is_declaration or function.name in broken:
                continue
            for op in function.operations():
                named = {v.name for v in op.attributes.values() if isinstance(v, SymbolRef)}
                hit = named & broken.keys()
                if hit:
                    callee = min(hit)
                    broken[function.name] = (
                        f"calls `{checker.spelled(callee)}`, which stays in Python"
                    )
                    changed = True
                    break
    return broken, checker.summaries


def rule_of(summary: EffectSummary) -> str:
    """The rule a native function's effects run under, for `ppy explain`."""
    if summary.barrier:
        return (
            "effects: output is written at each barrier and when the call returns; "
            f"nothing falls back after the first barrier ({summary.why.get('barrier', '')})"
        )
    if summary.holds:
        return "effects: output is held until the call returns, and dropped if it falls back"
    return ""
