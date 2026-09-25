"""The Python boundary for collections: a `ppy.Vec` in, a native handle out, and back.

A native function that takes a collection works on a handle into the C
runtime (`collections.c`). When Python calls it, each reference-class
argument (`ppy._collections.Vec[int]` and the rest) is copied into native
memory first, collections inside collections included, and whatever the
function hands back is copied out into reference-class instances.

Identity is kept across the crossing. An object that appears twice among the
arguments becomes one handle with two references, and a handle that came
from an argument comes back as that same Python object: returning a
parameter returns the caller's object, as it does in Python. When the
function writes through a parameter, each argument's contents are copied
back into the caller's objects after the call, so Python sees the writes.

What crosses is what has a plain native form: numbers, tuples of numbers,
and collections of them. A dataclass element, an object, or a `LinkedList`
(whose node ids are the history of its insertions) keeps the function on its
Python body when Python calls it.
"""

from __future__ import annotations

import ctypes
import re
import struct
from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = ["RETURNS_NOTHING", "Boundary", "Spec", "parse", "runtime"]

#: A signature's `returned` for a function that returns `None`: its native
#: entry fills a placeholder word, which the boundary does not hand out.
RETURNS_NOTHING = "None"

_I64_LOW, _I64_HIGH = -(1 << 63), (1 << 63) - 1
_SCALARS = frozenset({"int", "float", "bool"})
#: The collections that cross, by the short name `ppy._collections` gives them.
_CROSSING = frozenset(
    {"Vec", "Deque", "Heap", "MaxHeap", "HashMap", "HashSet", "TreeMap", "TreeSet"}
)
_MAPS = frozenset({"HashMap", "TreeMap"})
_SETS = frozenset({"HashSet", "TreeSet"})
_FAMILY = {
    "Vec": "seq",
    "Deque": "seq",
    "Heap": "seq",
    "MaxHeap": "seq",
    "HashMap": "map",
    "HashSet": "map",
    "TreeMap": "tree",
    "TreeSet": "tree",
}


class Refused(Exception):
    """An argument that does not match its declared type: the Python body runs instead."""


@dataclass(frozen=True, slots=True)
class Spec:
    """One type as the boundary sees it: a scalar, a tuple of scalars, or a collection."""

    #: "int", "float", "bool", "tuple", or a collection's short name.
    kind: str
    parts: tuple[str, ...] = ()
    key: Spec | None = None
    value: Spec | None = None

    @property
    def collection(self) -> bool:
        return self.kind in _CROSSING

    @property
    def words(self) -> int:
        return len(self.parts) if self.kind == "tuple" else 1

    def _kinds(self) -> tuple[str, ...]:
        if self.kind == "tuple":
            return self.parts
        return ("handle",) if self.collection else (self.kind,)

    @property
    def floats(self) -> int:
        return sum(1 << i for i, kind in enumerate(self._kinds()) if kind == "float")

    @property
    def handles(self) -> int:
        return 1 if self.collection else 0

    def python(self) -> Any:
        """The type as `ppy._collections` spells it: `Vec[tuple[int, float]]`."""
        if self.kind in _SCALARS:
            return {"int": int, "float": float, "bool": bool}[self.kind]
        if self.kind == "tuple":
            return tuple[tuple({"int": int, "float": float, "bool": bool}[p] for p in self.parts)]
        from ppy import _collections  # pylint: disable=import-outside-toplevel

        base = getattr(_collections, self.kind)
        if self.kind in _MAPS:
            assert self.key is not None and self.value is not None
            return base[self.key.python(), self.value.python()]
        if self.kind in _SETS:
            assert self.key is not None
            return base[self.key.python()]
        assert self.value is not None
        return base[self.value.python()]


_TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z_0-9.]*|\[|\]|,)")


def parse(spelled: str) -> Spec | None:
    """A type as native signatures spell it (`ppy.HashMap[int, ppy.Vec[float]]`),
    or None where it does not cross."""
    tokens = _TOKEN.findall(spelled)
    if "".join(tokens) != "".join(spelled.split()):
        return None
    position = 0

    def one() -> Spec | None:
        nonlocal position
        if position >= len(tokens):
            return None
        name = tokens[position]
        position += 1
        if name in _SCALARS:
            return Spec(name)
        arguments: list[Spec | None] = []
        if position < len(tokens) and tokens[position] == "[":
            position += 1
            while True:
                arguments.append(one())
                if position >= len(tokens):
                    return None
                token = tokens[position]
                position += 1
                if token == "]":
                    break
                if token != ",":
                    return None
        if any(argument is None for argument in arguments):
            return None
        found = [a for a in arguments if a is not None]
        if name == "tuple":
            if not found or any(part.kind not in _SCALARS for part in found):
                return None
            return Spec("tuple", tuple(part.kind for part in found))
        short = name.removeprefix("ppy.")
        if not name.startswith("ppy.") or short not in _CROSSING:
            return None
        if short in _MAPS:
            return Spec(short, key=found[0], value=found[1]) if len(found) == 2 else None
        if len(found) != 1:
            return None
        if short in _SETS:
            return Spec(short, key=found[0])
        return Spec(short, value=found[0])

    found = one()
    if found is None or position != len(tokens) or not found.collection:
        return None
    return found if _keys_ok(found) else None


def _keys_ok(spec: Spec) -> bool:
    if spec.key is not None:
        key = spec.key
        if not (key.kind == "int" or (key.kind == "tuple" and set(key.parts) == {"int"})):
            return False
    return spec.value is None or not spec.value.collection or _keys_ok(spec.value)


# -- the runtime, through ctypes ------------------------------------------------

_P = ctypes.c_void_p
_I = ctypes.c_int64
_SIGNATURES: dict[str, tuple[Any, tuple[Any, ...]]] = {
    "ppy_seq_new": (_P, (_I, _I, _I, _I)),
    "ppy_map_new": (_P, (_I, _I, _I, _I)),
    "ppy_tree_new": (_P, (_I, _I, _I, _I)),
    "ppy_seq_push_many": (None, (_P, ctypes.c_char_p, _I)),
    "ppy_coll_put_many": (None, (_P, ctypes.c_char_p, ctypes.c_char_p, _I)),
    "ppy_coll_copy_out": (None, (_P, _P, _P)),
    "ppy_coll_len": (_I, (_P,)),
    "ppy_coll_retain": (None, (_P,)),
    "ppy_coll_release": (None, (_P,)),
}

_loaded: dict[str, Any] = {}


def runtime(library: Any = None) -> Any:
    """The collections runtime's functions: from `library` where it carries them
    (a `ppy build` library links them in), else the one `ppy run` compiles."""
    if library is not None and hasattr(library, "ppy_coll_copy_out"):
        found = library
    else:
        cached = _loaded.get("")
        if cached is not None:
            return cached
        from .collections import library_path  # pylint: disable=import-outside-toplevel

        path = library_path()
        if path is None:
            return None
        found = ctypes.CDLL(str(path))
        _loaded[""] = found
    for name, (result, arguments) in _SIGNATURES.items():
        function = getattr(found, name)
        function.restype = result
        function.argtypes = arguments
    return found


def _format(spec: Spec) -> str:
    """One element's words as `struct` spells them: `q` an integer or a handle, `d` a double."""
    kinds = spec.parts if spec.kind == "tuple" else (spec.kind,)
    return "".join("d" if kind == "float" else "q" for kind in kinds)


class Boundary:
    """One call's crossing: arguments in, results out, and the writes copied back.

    A collection moves as a whole: its elements are packed into one buffer
    with `struct` and handed to the runtime in one call, and read back the
    same way, so the crossing costs a pass in C over the words rather than a
    foreign call per element.
    """

    def __init__(self, rt: Any) -> None:
        self.rt = rt
        #: id of each Python object made native -> its handle, and the object.
        self._handles: dict[int, tuple[int, Any]] = {}
        #: handle -> the Python object it came from or went to.
        self._objects: dict[int, Any] = {}
        self._synced: set[int] = set()
        #: Handles this call owns a reference to, let go of at the end.
        self._owned: list[int] = []

    # -- in -------------------------------------------------------------------

    def argument(self, value: Any, spec: Spec) -> int:
        """A reference-class argument as a handle the call owns."""
        handle = self._native(value, spec)
        self._owned.append(handle)
        return handle

    def _native(self, value: Any, spec: Spec) -> int:
        """A handle holding one new reference to `value`'s native copy."""
        seen = self._handles.get(id(value))
        if seen is not None:
            self.rt.ppy_coll_retain(seen[0])
            return seen[0]
        if type(value) is not spec.python():
            raise Refused
        family = _FAMILY[spec.kind]
        value_spec = spec.value
        words = value_spec.words if value_spec is not None else 0
        floats = value_spec.floats if value_spec is not None else 0
        handles = value_spec.handles if value_spec is not None else 0
        if family == "seq":
            handle = self.rt.ppy_seq_new(0, words, floats, handles)
        else:
            assert spec.key is not None
            maker = self.rt.ppy_map_new if family == "map" else self.rt.ppy_tree_new
            handle = maker(spec.key.words, words, floats, handles)
        self._handles[id(value)] = (handle, value)
        self._objects[handle] = value
        # The call holds its own reference to every handle it made, so none is
        # freed, and its address given to something new, while the call runs:
        # an address seen again is the same collection.
        self.rt.ppy_coll_retain(handle)
        self._owned.append(handle)
        try:
            self._fill(handle, value, spec)
        except BaseException:
            self.rt.ppy_coll_release(handle)
            raise
        return handle

    def _fill(self, handle: int, value: Any, spec: Spec) -> None:
        if _FAMILY[spec.kind] == "seq":
            assert spec.value is not None
            items = list(value._items)
            packed = self._pack(spec.value, items)
            self.rt.ppy_seq_push_many(handle, packed, len(items))
            return
        assert spec.key is not None
        entries = list(value._walk())
        keys = self._pack(spec.key, [key for key, _ in entries])
        values = b""
        if spec.value is not None:
            values = self._pack(spec.value, [item for _, item in entries])
        self.rt.ppy_coll_put_many(handle, keys, values, len(entries))

    def _pack(self, spec: Spec, items: list[Any]) -> bytes:
        """Elements as native words, checked against their declared type."""
        if not items:
            return b""
        if spec.collection:
            words: list[Any] = []
            try:
                for item in items:
                    words.append(self._native(item, spec))  # noqa: PERF401 - kept on failure
            except BaseException:
                # The references meant for the parent it will never hold.
                for handle in words:
                    self.rt.ppy_coll_release(handle)
                raise
        elif spec.kind == "tuple":
            kinds = spec.parts
            for item in items:
                if type(item) is not tuple or len(item) != len(kinds):
                    raise Refused
                for kind, part in zip(kinds, item, strict=True):
                    _check(kind, part)
            words = [part for item in items for part in item]
        else:
            for item in items:
                _check(spec.kind, item)
            words = items
        try:
            return struct.pack(f"<{_format(spec) * len(items)}", *words)
        except (struct.error, OverflowError) as exc:
            raise Refused from exc

    # -- out ----------------------------------------------------------------------

    def result(self, handle: int, spec: Spec) -> Any:
        """A handle the call returned, owned: its Python object."""
        self._owned.append(handle)
        return self._python(handle, spec)

    def sync(self, arguments: list[tuple[Any, Spec]]) -> None:
        """Copy each argument's native contents back into the caller's objects."""
        for value, spec in arguments:
            handle = self._handles[id(value)][0]
            self._python(handle, spec, rewrite=True)

    def _python(self, handle: int, spec: Spec, rewrite: bool = False) -> Any:
        known = self._objects.get(handle)
        if known is not None and (not rewrite or handle in self._synced):
            return known
        self._synced.add(handle)
        made = known if known is not None else spec.python()()
        self._objects[handle] = made
        _replace(made, spec, self._read(handle, spec, rewrite))
        return made

    def _read(self, handle: int, spec: Spec, rewrite: bool) -> list[Any]:
        count = self.rt.ppy_coll_len(handle)
        key_spec = spec.key if _FAMILY[spec.kind] != "seq" else None
        value_spec = spec.value
        key_bytes = (ctypes.c_int64 * (count * key_spec.words if key_spec else 1))()
        value_bytes = (ctypes.c_int64 * (count * value_spec.words if value_spec else 1))()
        self.rt.ppy_coll_copy_out(handle, key_bytes, value_bytes)
        values = self._unpack(value_spec, value_bytes, count, rewrite) if value_spec else None
        if key_spec is None:
            assert values is not None
            return values
        keys = self._unpack(key_spec, key_bytes, count, rewrite)
        return list(zip(keys, values if values is not None else [None] * count, strict=True))

    def _unpack(self, spec: Spec, buffer: Any, count: int, rewrite: bool) -> list[Any]:
        if not count:
            return []
        words = struct.unpack_from(f"<{_format(spec) * count}", buffer)
        if spec.collection:
            return [self._python(word, spec, rewrite) for word in words]
        if spec.kind == "tuple":
            width = len(spec.parts)
            bools = [i for i, kind in enumerate(spec.parts) if kind == "bool"]
            found = []
            for start in range(0, len(words), width):
                item = list(words[start : start + width])
                for i in bools:
                    item[i] = item[i] != 0
                found.append(tuple(item))
            return found
        if spec.kind == "bool":
            return [word != 0 for word in words]
        return list(words)

    def close(self) -> None:
        """Let go of every handle the call owned: what the function kept is freed
        with it, and the native copies of the arguments go."""
        for handle in self._owned:
            self.rt.ppy_coll_release(handle)
        self._owned.clear()


def _check(kind: str, value: Any) -> None:
    """One word's value against its declared kind, as the reference class stored it."""
    if kind == "float":
        if type(value) is not float:
            raise Refused
    elif kind == "bool":
        if type(value) is not bool:
            raise Refused
    elif type(value) is not int or not _I64_LOW <= value <= _I64_HIGH:
        raise Refused


def _replace(made: Any, spec: Spec, entries: list[Any]) -> None:
    """A reference-class object's contents set to what native code holds, in place."""
    family = _FAMILY[spec.kind]
    if family == "seq":
        if spec.kind == "Deque":
            made._items = deque(entries)
        else:
            made._items = list(entries)
        return
    before = list(made)
    version = made._version
    if family == "map":
        made._reset()
    else:
        made._keys.clear()
        made._values.clear()
    for key, item in entries:
        made._put(key, 0 if item is None else item)
    # A walk over the caller's map notices a change as it would in Python:
    # only where keys came or went.
    made._version = version if list(made) == before else version + 1
