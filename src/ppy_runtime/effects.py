"""The boundary's side of effects in native code: output, input, and calls into Python.

`pyio.c` holds what a native call prints until the call ends, and reaches
Python through one hook this module installs. The boundary of a function
that has effects (`NativeSignature.effects`) goes through `Effects`:

- `enter` before the native call, `leave` after it;
- a call that answered writes out what it printed (`commit`);
- a call that fell back drops it, and Python runs the call again;
- a call that raised after a barrier (a flush, a call into Python) cannot
  run again, so its exception is raised here as it is: a builtin one made
  from its name and text, one Python raised inside the call as that very
  object.

The lowering makes sure a call never falls back after a barrier
(`ppy_compiler/lowering/effects.py`); if one did, that is an error here,
never a second run of what the call already did.
"""

from __future__ import annotations

import builtins
import ctypes
import hashlib
import sys
import threading
from typing import Any

from .abi import STATUS_OK, STATUS_RAISED

__all__ = ["EffectError", "Effects", "effects_for", "register_namespace"]

_HOOK = ctypes.CFUNCTYPE(
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
)

#: The kinds of `pyio.c`.
_NONE, _INT, _FLOAT, _BOOL, _STR, _OBJECT = range(6)

_I64_LOW = -(1 << 63)
_I64_HIGH = (1 << 63) - 1

_SIGNATURES: dict[str, tuple[Any, tuple[Any, ...]]] = {
    "ppy_io_set_hook": (None, (ctypes.c_int64,)),
    "ppy_io_enter": (ctypes.c_int64, ()),
    "ppy_io_leave": (ctypes.c_int64, (ctypes.c_int64,)),
    "ppy_io_commit": (ctypes.c_int64, ()),
    "ppy_io_discard": (None, ()),
    "ppy_io_state": (ctypes.POINTER(ctypes.c_int64), ()),
    "ppy_io_answer": (None, (ctypes.c_int64, ctypes.c_int64)),
    "ppy_io_answer_text": (None, (ctypes.c_char_p, ctypes.c_int64)),
    "ppy_io_pending": (
        None,
        (
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_char_p,
            ctypes.c_int64,
            ctypes.c_char_p,
            ctypes.c_int64,
        ),
    ),
    "ppy_exc_take": (ctypes.c_void_p, ()),
    "ppy_exc_record": (ctypes.POINTER(ctypes.c_int64), (ctypes.c_void_p,)),
    "ppy_str_data": (ctypes.c_void_p, (ctypes.c_void_p,)),
    "ppy_str_bytes": (ctypes.c_int64, (ctypes.c_void_p,)),
}

#: Module namespaces by the name the compiler knew each module by, for a
#: call into Python to find its callable where the program would.
_NAMESPACES: dict[str, dict[str, Any]] = {}


class EffectError(RuntimeError):
    """A native call fell back after an effect it cannot take back: a PPy bug."""


def register_namespace(qualname: str, namespace: dict[str, Any] | None) -> None:
    """Where the names a native function calls into Python are looked up: the
    globals of the function whose qualname this is, under every prefix of it
    that may be its module's name."""
    if namespace is None:
        return
    head = qualname
    while "." in head:
        head = head.rpartition(".")[0]
        _NAMESPACES.setdefault(head, namespace)


def _class_tag(qualname: str) -> int:
    """`lowering.collections.class_tag`, which the runtime cannot import."""
    digest = hashlib.sha256(qualname.encode()).digest()
    return int.from_bytes(digest[:7], "little") | 1


def _text(address: int, length: int) -> str:
    return ctypes.string_at(address, length).decode("utf-8", "surrogatepass") if length else ""


class Effects:
    """One runtime's hook and its boundary calls."""

    def __init__(self, library: ctypes.CDLL) -> None:
        self.lib = library
        for name, (result, arguments) in _SIGNATURES.items():
            function = getattr(library, name)
            function.restype = result
            function.argtypes = arguments
        self.local = threading.local()
        self._hook = _HOOK(self._dispatch)
        library.ppy_io_set_hook(ctypes.cast(self._hook, ctypes.c_void_p).value or 0)

    # -- the boundary ------------------------------------------------------------

    def enter(self) -> int:
        local = self.local
        local.depth = getattr(local, "depth", 0) + 1
        return int(self.lib.ppy_io_enter())

    def leave(self, status: int, outer: int, qualname: str, let_go: Any) -> bool | BaseException:
        """After the native call: True where it answered, and the caller then
        converts its result and calls `commit`; False where it fell back, with
        what it printed dropped; or what a call that crossed a barrier raised,
        for the caller to raise."""
        crossed = self.lib.ppy_io_leave(outer)
        if status == STATUS_OK:
            return True
        try:
            if not crossed:
                self.lib.ppy_io_discard()
                if status == STATUS_RAISED:
                    let_go()
                return False
            if status != STATUS_RAISED:
                self.lib.ppy_io_discard()
                return EffectError(
                    f"PPy: native `{qualname}` fell back after an effect; "
                    "please report this as a bug"
                )
            error = self._native_exception()
            let_go()
            if self.lib.ppy_io_commit() != 0:
                # A write failed first, as it did first in the program.
                return self._raised().pop()
            return error
        finally:
            self._done()

    def abandon(self, outer: int) -> None:
        """After a native call that stopped on a sanitizer check: its output dropped."""
        self.lib.ppy_io_leave(outer)
        self.lib.ppy_io_discard()
        self._done()

    def commit(self) -> None:
        """Write out what an answered call printed; raises what a write raised."""
        try:
            self._commit()
        finally:
            self._done()

    def _commit(self) -> None:
        if self.lib.ppy_io_commit() != 0:
            raise self._raised().pop()

    def _done(self) -> None:
        local = self.local
        local.depth -= 1
        if local.depth == 0:
            local.__dict__.pop("objects", None)
            local.__dict__.pop("raised", None)

    def _native_exception(self) -> BaseException:
        """The exception pending in native code, as Python's."""
        handle = self.lib.ppy_exc_take()
        if not handle:
            return EffectError("PPy: a native call raised with no exception pending")
        record = self.lib.ppy_exc_record(handle)
        name = self._string(record[1])
        text = self._string(record[2])
        flags = record[3]
        if flags & 2:
            raised = self._raised()
            number = flags >> 8
            if 0 <= number < len(raised):
                return raised[number]
        kind = getattr(builtins, name, None)
        if not (isinstance(kind, type) and issubclass(kind, BaseException)):
            return EffectError(f"PPy: native code raised `{name}`, which is not builtin")
        if kind is KeyError and text:
            import ast  # pylint: disable=import-outside-toplevel

            try:
                return KeyError(ast.literal_eval(text))
            except (ValueError, SyntaxError):
                return KeyError(text)
        return kind(text) if text else kind()

    def _string(self, handle: int) -> str:
        address = self.lib.ppy_str_data(handle)
        return _text(address or 0, self.lib.ppy_str_bytes(handle))

    # -- the hook ----------------------------------------------------------------

    def _raised(self) -> list[BaseException]:
        local = self.local
        raised = local.__dict__.get("raised")
        if raised is None:
            raised = local.raised = []
        return raised  # type: ignore[no-any-return]

    def _objects(self) -> dict[int, object]:
        local = self.local
        objects = local.__dict__.get("objects")
        if objects is None:
            objects = local.objects = {}
        return objects  # type: ignore[no-any-return]

    def _dispatch(self, op: int, a: int, b: int, c: int) -> int:
        try:
            if op == 1:
                stream = sys.stdout if c == 1 else sys.stderr
                if stream is not None:
                    stream.write(_text(a, b))
            elif op == 2:
                stream = sys.stdout if a == 1 else sys.stderr
                if stream is not None:
                    stream.flush()
            elif op in {3, 4}:
                arguments = self._arguments()
                name = _text(a, b)
                if op == 4:
                    result = getattr(arguments[0], name)(*arguments[1:])
                else:
                    result = _resolve(name)(*arguments)
                self._answer(result, c, name)
            elif op == 5:
                self._objects().pop(a, None)
            return 0
        except BaseException as error:  # noqa: BLE001 - every exception crosses back
            self._pend(error)
            return -1

    def _arguments(self) -> list[object]:
        state = self.lib.ppy_io_state()
        count = min(state[8], 32)
        values: list[object] = []
        for index in range(count):
            kind, word = state[18 + 2 * index], state[19 + 2 * index]
            if kind == _INT:
                values.append(word)
            elif kind == _FLOAT:
                values.append(_float(word))
            elif kind == _BOOL:
                values.append(bool(word))
            elif kind == _STR:
                values.append(self._string(word))
            elif kind == _OBJECT:
                values.append(self._objects()[word])
            else:
                values.append(None)
        return values

    def _answer(self, result: object, kind: int, name: str) -> None:
        lib = self.lib
        if kind == _NONE:
            lib.ppy_io_answer(_NONE, 0)
        elif kind == _STR:
            if not isinstance(result, str):
                raise _mismatch(name, "str", result)
            data = result.encode("utf-8", "surrogatepass")
            lib.ppy_io_answer_text(data, len(data))
            lib.ppy_io_answer(_STR, 0)
        elif kind == _FLOAT:
            if not isinstance(result, float):
                raise _mismatch(name, "float", result)
            lib.ppy_io_answer(_FLOAT, _bits(result))
        elif kind == _BOOL:
            if not isinstance(result, bool):
                raise _mismatch(name, "bool", result)
            lib.ppy_io_answer(_BOOL, int(result))
        elif kind == _INT:
            if not isinstance(result, int) or not _I64_LOW <= result <= _I64_HIGH:
                raise _mismatch(name, "int", result)
            lib.ppy_io_answer(_INT, int(result))
        else:
            objects = self._objects()
            number = id(result)
            objects[number] = result
            lib.ppy_io_answer(_OBJECT, number)

    def _pend(self, error: BaseException) -> None:
        traceback = error.__traceback__
        if traceback is not None and traceback.tb_frame.f_code is Effects._dispatch.__code__:
            # From where Python was called on: the hook's own frame is no part of it.
            error.__traceback__ = traceback.tb_next
        raised = self._raised()
        number = len(raised)
        raised.append(error)
        base = next((k for k in type(error).__mro__ if k.__module__ == "builtins"), BaseException)
        try:
            text = str(error)
        except Exception:  # noqa: BLE001 - str() of it is only for native code to read
            text = ""
        name = type(error).__name__.encode()
        data = text.encode("utf-8", "surrogatepass")
        self.lib.ppy_io_pending(
            _class_tag(f"builtins.{base.__name__}"), number, name, len(name), data, len(data)
        )


def _mismatch(name: str, wanted: str, result: object) -> TypeError:
    name = name.rpartition(":")[2]
    return TypeError(
        f"`{name}` returned {type(result).__name__}, where the compiled caller expects {wanted}"
    )


def _float(word: int) -> float:
    return ctypes.c_double.from_buffer(ctypes.c_int64(word)).value


def _bits(value: float) -> int:
    return ctypes.c_int64.from_buffer(ctypes.c_double(value)).value


def _resolve(spelled: str) -> Any:
    """The callable `module:dotted.name` names, looked up now, as Python would."""
    module, _, dotted = spelled.partition(":")
    head, *rest = dotted.split(".")
    namespace = _NAMESPACES.get(module)
    if namespace is None:
        loaded = sys.modules.get(module)
        namespace = vars(loaded) if loaded is not None else {}
    if head in namespace:
        found = namespace[head]
    elif hasattr(builtins, head):
        found = getattr(builtins, head)
    else:
        raise NameError(f"name '{head}' is not defined")
    for part in rest:
        found = getattr(found, part)
    return found


_LOCK = threading.Lock()
_BY_LIBRARY: dict[int, Effects] = {}


def effects_for(owner: object) -> Effects | None:
    """The effects of the runtime a native function calls: its own library's,
    where it carries the runtime, else the one `ppy run` loads."""
    library = owner if isinstance(owner, ctypes.CDLL) else None
    if library is not None:
        try:
            library.ppy_io_enter  # noqa: B018 - whether the library carries the runtime
        except AttributeError:
            library = None
    if library is None:
        from .collections import library_path  # pylint: disable=import-outside-toplevel

        path = library_path()
        if path is None:
            return None
        library = ctypes.CDLL(str(path))
    key = library._handle  # pylint: disable=protected-access
    with _LOCK:
        found = _BY_LIBRARY.get(key)
        if found is None:
            found = _BY_LIBRARY[key] = Effects(library)
        return found
