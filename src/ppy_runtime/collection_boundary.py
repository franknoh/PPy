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

What crosses is what has a plain native form: numbers, strings, tuples of
numbers, and collections of them, Python's own `list`, `dict`, and `set`
included (a string crosses as its UTF-8 bytes, made a native string), and
instances of the project classes the signature describes (`CrossingClass`):
an object as a handle to its one record of fields, a value class as its
fields' words in place. An object crosses whole, the objects its fields hold
with it, each once; one that comes back as a handle it went in as is the
caller's object again, its fields set to what native code left in them. A
`LinkedList` (whose node ids are the history of its insertions), or an
object whose class the signature does not describe, keeps the function on
its Python body when Python calls it.
"""

from __future__ import annotations

import array
import contextlib
import ctypes
import os
import re
import struct
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any

from .abi import CrossingClass

__all__ = [
    "RETURNS_NOTHING",
    "Boundary",
    "Spec",
    "attach",
    "field_spec",
    "parse",
    "resolver",
    "runtime",
]

#: A signature's `returned` for a function that returns `None`: its native
#: entry fills a placeholder word, which the boundary does not hand out.
RETURNS_NOTHING = "None"

_I64_LOW, _I64_HIGH = -(1 << 63), (1 << 63) - 1
_SCALARS = frozenset({"int", "float", "bool"})
#: Python's own containers, which cross as themselves.
_BUILTINS = {"list": list, "dict": dict, "set": set}
#: The collections that cross, by the short name `ppy._collections` gives them,
#: and Python's own by theirs.
_CROSSING = frozenset(
    {"Vec", "Deque", "Heap", "MaxHeap", "HashMap", "HashSet", "TreeMap", "TreeSet", *_BUILTINS}
)
_MAPS = frozenset({"HashMap", "TreeMap", "dict"})
_SETS = frozenset({"HashSet", "TreeSet", "set"})
_FAMILY = {
    "list": "seq",
    "dict": "map",
    "set": "map",
    "Vec": "seq",
    "Deque": "seq",
    "Heap": "seq",
    "MaxHeap": "seq",
    "HashMap": "map",
    "HashSet": "map",
    "TreeMap": "tree",
    "TreeSet": "tree",
}


#: A `random.Random` argument's kind: lent by address, never copied.
_GENERATOR = "random.Random"

#: `RandomObject`'s state after the object header: `int index`, then 624 words.
_STATE_BYTES = 4 + 624 * 4


class Refused(Exception):
    """An argument that does not match its declared type: the Python body runs instead."""


@dataclass(frozen=True, slots=True)
class Spec:
    """One type as the boundary sees it: a scalar, a tuple of scalars, or a collection."""

    #: "int", "float", "bool", "tuple", "str", a collection's short name,
    #: "object" (a handle to a project class's instance), "record" (a value
    #: class's fields in place), or "optional" (a number or `None`: the number's
    #: word, then a word that says whether there is one; `parts` is its kind).
    kind: str
    #: A tuple's or a record's words: scalar kinds.
    parts: tuple[str, ...] = ()
    key: Spec | None = None
    value: Spec | None = None
    #: An object's or a record's class.
    record: str = ""
    #: An object or a string that may be `None`, the null handle.
    nullable: bool = False

    @property
    def collection(self) -> bool:
        return self.kind in _CROSSING

    @property
    def words(self) -> int:
        if self.kind == "optional":
            return 2
        return len(self.parts) if self.kind in {"tuple", "record"} else 1

    def _kinds(self) -> tuple[str, ...]:
        if self.kind in {"tuple", "record"}:
            return self.parts
        if self.kind == "optional":
            return (self.parts[0], "bool")
        return ("handle",) if self.reference else (self.kind,)

    @property
    def reference(self) -> bool:
        """Held by handle: a collection, a string, or an object."""
        return self.collection or self.kind in {"str", "object"}

    @property
    def floats(self) -> int:
        return sum(1 << i for i, kind in enumerate(self._kinds()) if kind == "float")

    @property
    def handles(self) -> int:
        return 1 if self.reference else 0

    @property
    def leaves(self) -> int:
        """The handle words that are strings, which hold no handles themselves."""
        return 1 if self.kind == "str" else 0

    def python(self) -> Any:
        """The type as `ppy._collections` spells it: `Vec[tuple[int, float]]`."""
        if self.kind in _SCALARS:
            return {"int": int, "float": float, "bool": bool}[self.kind]
        if self.kind == "str":
            return str
        if self.kind in _BUILTINS:
            return _BUILTINS[self.kind]
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


_TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z_0-9.]*|\[|\]|,|\?|\|)")


def _or_none(found: Spec) -> Spec | None:
    """`X | NoneType`: a number as an optional, a string or an object nullable."""
    if found.kind in _SCALARS:
        return Spec("optional", (found.kind,))
    if found.kind in {"str", "object"}:
        return Spec(found.kind, found.parts, record=found.record, nullable=True)
    return None


def parse(spelled: str, classes: dict[str, CrossingClass] | None = None) -> Spec | None:
    """A type as native signatures spell it (`ppy.HashMap[int, ppy.Vec[float]]`),
    or None where it does not cross. A class `classes` describes crosses as
    one of its instances; `prog.Node?` is one that may be `None`."""
    classes = classes or {}
    tokens = _TOKEN.findall(spelled)
    if "".join(tokens) != "".join(spelled.split()):
        return None
    position = 0

    def one() -> Spec | None:
        nonlocal position
        found = atom()
        if found is not None and tokens[position : position + 2] == ["|", "NoneType"]:
            position += 2
            return _or_none(found)
        return found

    def atom() -> Spec | None:
        nonlocal position
        if position >= len(tokens):
            return None
        name = tokens[position]
        position += 1
        if name in _SCALARS or name == "str":
            return Spec(name)
        described = classes.get(name)
        if described is not None:
            nullable = position < len(tokens) and tokens[position] == "?"
            if nullable:
                position += 1
            if described.kind == "record":
                parts = tuple(kind for _field, _offset, kind in described.fields)
                if any(part not in _SCALARS for part in parts):
                    return None
                return Spec("record", parts, record=name)
            return Spec("object", record=name, nullable=nullable)
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
        if not (name.startswith("ppy.") or name in _BUILTINS) or short not in _CROSSING:
            return None
        if short in _MAPS:
            return Spec(short, key=found[0], value=found[1]) if len(found) == 2 else None
        if len(found) != 1:
            return None
        if short in _SETS:
            return Spec(short, key=found[0])
        return Spec(short, value=found[0])

    if tokens == ["random.Random"]:
        # A generator the caller lends: native code draws from its own state.
        return Spec(_GENERATOR)
    found = one()
    if found is None or position != len(tokens):
        return None
    if not found.collection and not (found.kind == "object" and found.record):
        return None
    return found if _keys_ok(found) else None


def _keys_ok(spec: Spec) -> bool:
    if spec.key is not None:
        key = spec.key
        text = key.kind == "str" and spec.kind in {"dict", "set"}
        if not (text or key.kind == "int" or (key.kind == "tuple" and set(key.parts) == {"int"})):
            return False
    return spec.value is None or not spec.value.collection or _keys_ok(spec.value)


class Classes:
    """The classes a signature crosses, each with its fields parsed and its
    Python class found when first wanted: the module is still running its
    body when a function of it is bound."""

    def __init__(self, described: tuple[CrossingClass, ...], resolve: Any = None) -> None:
        self.by_name = {c.qualname: c for c in described}
        self.by_tag = {c.tag: c for c in described if c.kind == "object"}
        self._resolve = resolve
        self._types: dict[type, CrossingClass] | None = None
        self._python: dict[str, type] = {}
        self.fields: dict[str, list[tuple[str, int, Spec]]] = {}
        for c in described:
            if c.kind != "object":
                continue
            parsed = []
            for name, offset, spelled in c.fields:
                spec = _field_spec(spelled, self.by_name)
                if spec is None:
                    self.fields.clear()
                    self.by_name.clear()
                    self.by_tag.clear()
                    return
                parsed.append((name, offset, spec))
            self.fields[c.qualname] = parsed

    def python(self, qualname: str) -> Any:
        """The Python class of `qualname`, or None where the module has none."""
        if qualname not in self._python:
            described = self.by_name.get(qualname)
            if described is None or self._resolve is None:
                return None
            resolved = self._resolve(described)
            if not isinstance(resolved, type):
                return None
            self._python[qualname] = resolved
        return self._python[qualname]

    def of(self, value: Any) -> CrossingClass | None:
        """The class `value` is an instance of, exactly, among those described."""
        if self._types is None:
            types: dict[type, CrossingClass] = {}
            for described in self.by_name.values():
                found = self.python(described.qualname)
                if found is not None:
                    types[found] = described
            self._types = types
        return self._types.get(type(value))


def field_spec(spelled: str, classes: dict[str, CrossingClass]) -> Spec | None:
    """A field's type as the boundary sees it (see `_field_spec`)."""
    return _field_spec(spelled, classes)


def resolver(signature: Any, function: Any) -> Any:
    """For a generated wrapper whose objects cross: a callable giving the Python
    class of each class the signature describes, in its order, or None
    while one is not defined yet. Found where `function` would find it."""
    if not signature.classes:
        return None
    classes = Classes(signature.classes, _finder(function))
    order = [c.qualname for c in signature.classes]

    def resolve() -> tuple[type, ...] | None:
        found = tuple(classes.python(qualname) for qualname in order)
        return found if all(isinstance(t, type) for t in found) else None

    return resolve


def _finder(function: Any) -> Any:
    """How a described class is found: in the namespace `function` reads, where
    the program defines it, else in its module (`binding._class_finder`)."""
    namespace = None
    while function is not None:
        namespace = getattr(function, "__ppy_globals__", None)
        if namespace is not None:
            break
        wrapped = getattr(function, "__wrapped__", None)
        if wrapped is None:
            namespace = getattr(function, "__globals__", None)
            break
        function = wrapped

    def find(described: CrossingClass) -> Any:
        if namespace is not None:
            found = namespace.get(described.name)
            if isinstance(found, type) and found.__qualname__ == described.name:
                return found
        module = sys.modules.get(described.module)
        return getattr(module, described.name, None) if module is not None else None

    return find


def _field_spec(spelled: str, classes: dict[str, CrossingClass]) -> Spec | None:
    """A field's type: a scalar, a string, a tuple of scalars, a number or a
    string that may be `None`, or what `parse` reads."""
    if spelled in _SCALARS or spelled == "str":
        return Spec(spelled)
    base, union, rest = spelled.partition("|")
    if union and rest.strip() == "NoneType" and base.strip() in {*_SCALARS, "str"}:
        return _or_none(Spec(base.strip()))
    if spelled.startswith("tuple["):
        parts = tuple(part.strip() for part in spelled[6:-1].split(","))
        if not parts or any(part not in _SCALARS for part in parts):
            return None
        return Spec("tuple", parts)
    found = parse(spelled, classes)
    if found is None:
        described = classes.get(spelled)
        if described is not None and described.kind == "record":
            parts = tuple(kind for _field, _offset, kind in described.fields)
            return Spec("record", parts, record=spelled)
    return found


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
    "ppy_coll_text_keys": (None, (_P, _I)),
    "ppy_str_new_many": (None, (ctypes.c_char_p, ctypes.c_char_p, _I, _P)),
    "ppy_str_gather": (_I, (_P, _I, _P, _P)),
    "ppy_random_external": (_P, (_I,)),
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


#: The runtime functions a generated wrapper copies containers with, in the
#: order its `ppy_runtime` takes their addresses (`crossing.c`).
_WRAPPER_FUNCTIONS = (
    "ppy_seq_new",
    "ppy_map_new",
    "ppy_seq_push_many",
    "ppy_coll_put_many",
    "ppy_coll_copy_out",
    "ppy_coll_len",
    "ppy_coll_retain",
    "ppy_coll_release",
    "ppy_coll_text_keys",
    "ppy_str_new_many",
    "ppy_coll_touched",
    "ppy_coll_adopt",
)


def attach(wrappers: Any, library: Any = None) -> bool:
    """Hand a generated wrapper module the runtime its native code makes handles
    in (`runtime`); whether it took it. Asked once per module."""
    hand = getattr(wrappers, "ppy_runtime", None)
    if hand is None:
        return False
    attached = getattr(wrappers, "__ppy_attached__", None)
    if attached is not None:
        return bool(attached)
    rt = runtime(library)
    taken = False
    if rt is not None:
        try:
            addresses = [
                ctypes.cast(getattr(rt, n), ctypes.c_void_p).value for n in _WRAPPER_FUNCTIONS
            ]
            taken = bool(hand(*addresses))
        except (AttributeError, TypeError, ValueError, OverflowError):
            taken = False
    if taken:
        _share_world(wrappers)
    with contextlib.suppress(AttributeError, TypeError):
        wrappers.__ppy_attached__ = taken
    return taken


#: The resident objects' world (`crossing.c`), which the first wrapper module
#: makes and every other one is handed: an object is resident in one place.
_world: list[Any] = []


def _share_world(wrappers: Any) -> None:
    """Give a wrapper module the process's world of resident objects; with
    `PPY_RESIDENT=0` in the environment, none, and objects are copied at
    every crossing."""
    share = getattr(wrappers, "ppy_world", None)
    if share is None or os.environ.get("PPY_RESIDENT", "1") == "0":
        return
    try:
        found = share(_world[0] if _world else None)
    except (TypeError, ValueError):
        return
    if found is not None and not _world:
        _world.append(found)
        if os.environ.get("PPY_RESIDENT_REPORT"):
            import atexit  # pylint: disable=import-outside-toplevel

            atexit.register(_report_world, wrappers)


def _report_world(wrappers: Any) -> None:
    """`PPY_RESIDENT_REPORT=1`: the world's objects at exit, on stderr."""
    stats = wrappers.ppy_world_stats()
    if stats is not None:
        live, stale, entries, enabled, admitted, calls = stats
        print(
            f"resident: {live} live, {stale} stale, {entries} entries, enabled {enabled}, "
            f"{admitted} admitted, {calls} calls",
            file=sys.stderr,
        )


def _format(spec: Spec) -> str:
    """One element's words as `struct` spells them: `q` an integer or a handle, `d` a double."""
    kinds = spec._kinds() if spec.kind == "optional" else None  # pylint: disable=protected-access
    if kinds is None:
        kinds = spec.parts if spec.kind in {"tuple", "record"} else (spec.kind,)
    return "".join("d" if kind == "float" else "q" for kind in kinds)


def _mask_format(floats: int, words: int) -> str:
    """A record's words as `struct` spells them, from its float mask."""
    return "".join("d" if floats >> i & 1 else "q" for i in range(words))


#: The header word an object's class tag is kept in (`lowering/collections.py`).
_TAG = 4


class Boundary:
    """One call's crossing: arguments in, results out, and the writes copied back.

    A collection moves as a whole: its elements are packed into one buffer
    with `struct` and handed to the runtime in one call, and read back the
    same way, so the crossing costs a pass in C over the words rather than a
    foreign call per element.
    """

    def __init__(self, rt: Any, classes: Classes | None = None) -> None:
        self.rt = rt
        self.classes = classes
        #: id of each Python object made native -> its handle, and the object.
        self._handles: dict[int, tuple[int, Any]] = {}
        #: handle -> the Python object it came from or went to.
        self._objects: dict[int, Any] = {}
        self._synced: set[int] = set()
        #: Handles this call owns a reference to, let go of at the end.
        self._owned: list[int] = []
        #: The generators arguments lent, by address, and their state before.
        self._generators: dict[int, bytes] = {}

    # -- in -------------------------------------------------------------------

    def argument(self, value: Any, spec: Spec) -> int:
        """A reference-class argument as a handle the call owns."""
        handle = self._native(value, spec)
        self._owned.append(handle)
        return handle

    def _native(self, value: Any, spec: Spec) -> int:
        """A handle holding one new reference to `value`'s native copy."""
        if spec.kind == "object":
            return self._object(value, spec)
        if spec.kind == _GENERATOR:
            return self._generator(value)
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
        if value_spec is not None and value_spec.leaves:
            handles |= value_spec.leaves << 32
        if family == "seq":
            handle = self.rt.ppy_seq_new(0, words, floats, handles)
        else:
            assert spec.key is not None
            maker = self.rt.ppy_map_new if family == "map" else self.rt.ppy_tree_new
            handle = maker(spec.key.words, words, floats, handles)
            if spec.key.kind == "str":
                self.rt.ppy_coll_text_keys(handle, 1)
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

    def _object(self, value: Any, spec: Spec) -> int:
        """An instance of a project class as a handle to its record, the objects
        and collections its fields hold made native with it."""
        if value is None:
            if spec.nullable:
                return 0
            raise Refused
        seen = self._handles.get(id(value))
        if seen is not None:
            self.rt.ppy_coll_retain(seen[0])
            return seen[0]
        classes = self.classes
        described = classes.of(value) if classes is not None else None
        if described is None or spec.record not in described.bases:
            raise Refused
        assert classes is not None
        handle = self.rt.ppy_seq_new(0, described.words, described.floats, described.handles)
        ctypes.c_int64.from_address(handle + 8 * _TAG).value = described.tag
        self._handles[id(value)] = (handle, value)
        self._objects[handle] = value
        self.rt.ppy_coll_retain(handle)
        self._owned.append(handle)
        try:
            words: list[Any] = [0] * described.words
            made: list[int] = []
            try:
                for name, offset, field in classes.fields[described.qualname]:
                    try:
                        item = getattr(value, name)
                    except AttributeError as exc:
                        raise Refused from exc
                    self._place(words, offset, field, item, made)
            except BaseException:
                for held in made:
                    self.rt.ppy_coll_release(held)
                raise
            packed = struct.pack("<" + _mask_format(described.floats, described.words), *words)
            self.rt.ppy_seq_push_many(handle, packed, 1)
        except BaseException:
            self.rt.ppy_coll_release(handle)
            raise
        return handle

    def _place(self, words: list[Any], offset: int, spec: Spec, item: Any, made: list[int]) -> None:
        """One field's value into its words of a record; a handle made for it
        is the record's reference, listed in `made` until the record holds it."""
        if spec.kind in _SCALARS:
            _check(spec.kind, item)
            words[offset] = item
        elif spec.kind == "optional":
            # `None` is the number 0 with its flag clear.
            if item is not None:
                _check(spec.parts[0], item)
            words[offset] = 0 if item is None else item
            words[offset + 1] = 0 if item is None else 1
        elif spec.kind == "str" and item is None and spec.nullable:
            words[offset] = 0
        elif spec.kind == "str":
            strings = self._strings([item])
            made.append(strings[0])
            words[offset] = strings[0]
        elif spec.kind == "tuple":
            if type(item) is not tuple or len(item) != len(spec.parts):
                raise Refused
            for index, (kind, part) in enumerate(zip(spec.parts, item, strict=True)):
                _check(kind, part)
                words[offset + index] = part
        elif spec.kind == "record":
            for index, part in enumerate(self._record_words(item, spec)):
                words[offset + index] = part
        else:
            handle = self._native(item, spec)
            if handle:
                made.append(handle)
            words[offset] = handle

    def _record_words(self, item: Any, spec: Spec) -> list[Any]:
        """A value class's fields, checked against their kinds."""
        classes = self.classes
        described = classes.of(item) if classes is not None else None
        if described is None or described.qualname != spec.record:
            raise Refused
        found = []
        for name, _offset, kind in described.fields:
            try:
                part = getattr(item, name)
            except AttributeError as exc:
                raise Refused from exc
            _check(kind, part)
            found.append(part)
        return found

    def _fill(self, handle: int, value: Any, spec: Spec) -> None:
        if _FAMILY[spec.kind] == "seq":
            assert spec.value is not None
            items = list(value) if spec.kind == "list" else list(value._items)
            packed = self._pack(spec.value, items)
            self.rt.ppy_seq_push_many(handle, packed, len(items))
            return
        assert spec.key is not None
        if spec.kind == "dict":
            entries = list(value.items())
        elif spec.kind == "set":
            entries = [(key, None) for key in value]
        else:
            entries = list(value._walk())
        made: list[int] = []
        keys = self._pack(spec.key, [key for key, _ in entries], made)
        values = b""
        if spec.value is not None:
            values = self._pack(spec.value, [item for _, item in entries])
        self.rt.ppy_coll_put_many(handle, keys, values, len(entries))
        # A map holds its own reference to each string key it keeps.
        for text in made:
            self.rt.ppy_coll_release(text)

    def _pack(self, spec: Spec, items: list[Any], made: list[int] | None = None) -> bytes:
        """Elements as native words, checked against their declared type; the
        strings made for them are added to `made` where it is given."""
        if not items:
            return b""
        if spec.kind == "str" and spec.nullable and None in items:
            # `None` is the null handle; the strings are made as ever.
            given = [item for item in items if item is not None]
            strings = list(self._strings(given)) if given else []
            if made is not None:
                made.extend(strings)
            taken = iter(strings)
            handles = [0 if item is None else next(taken) for item in items]
            return struct.pack(f"<{len(handles)}q", *handles)
        if spec.kind == "str":
            strings = self._strings(items)
            if made is not None:
                made.extend(strings)
            return bytes(strings)
        if spec.collection or spec.kind == "object":
            words: list[Any] = []
            try:
                for item in items:
                    words.append(self._native(item, spec))  # noqa: PERF401 - kept on failure
            except BaseException:
                # The references meant for the parent it will never hold.
                for handle in words:
                    if handle:
                        self.rt.ppy_coll_release(handle)
                raise
        elif spec.kind == "record":
            words = [part for item in items for part in self._record_words(item, spec)]
        elif spec.kind == "optional":
            words = []
            for item in items:
                if item is None:
                    words.extend((0.0 if spec.parts[0] == "float" else 0, 0))
                else:
                    _check(spec.parts[0], item)
                    words.extend((item, 1))
        elif spec.kind == "tuple":
            kinds = spec.parts
            for item in items:
                if type(item) is not tuple or len(item) != len(kinds):
                    raise Refused
                for kind, part in zip(kinds, item, strict=True):
                    _check(kind, part)
            words = [part for item in items for part in item]
        else:
            # One type test for the whole list, in C: a check per element in
            # Python cost more than the native work it fed. `struct.pack`
            # refuses an integer past a word itself.
            if set(map(type, items)) != {_TYPES[spec.kind]}:
                raise Refused
            words = items
        try:
            return struct.pack(f"<{_format(spec) * len(items)}", *words)
        except (struct.error, OverflowError) as exc:
            raise Refused from exc

    def _texts(self, words: tuple[int, ...]) -> list[str]:
        """The Python strings of native ones, their bytes copied out in one call."""
        count = len(words)
        if count == 0:
            return []
        handles = (ctypes.c_int64 * count)(*words)
        total = self.rt.ppy_str_gather(handles, count, None, None)
        lengths = (ctypes.c_int64 * count)()
        data = ctypes.create_string_buffer(total)
        self.rt.ppy_str_gather(handles, count, lengths, data)
        raw = data.raw
        if raw.count(0) == count:
            # No string holds a NUL, so the one after each string splits them.
            return raw[:-1].decode("utf-8").split("\0")
        return _split(raw, lengths)

    def _strings(self, items: list[Any]) -> Any:
        """Native strings for `items`, made in one call once every item is
        known to be a string: an array of their handles, each owned. The
        checks and the encoding run in C builtins, not a loop per item."""
        if not set(map(type, items)) <= {str}:
            raise Refused
        joined = "".join(items)
        try:
            data = joined.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise Refused from exc
        count = len(items)
        if len(data) == len(joined):
            # All ASCII: each string's bytes are its characters.
            lengths = array.array("q", map(len, items))
        else:
            lengths = array.array("q", (len(item.encode("utf-8")) for item in items))
        strings = (ctypes.c_int64 * count)()
        self.rt.ppy_str_new_many(data, lengths.tobytes(), count, strings)
        return strings

    # -- out ----------------------------------------------------------------------

    def result(self, handle: int, spec: Spec) -> Any:
        """A handle the call returned, owned: its Python object."""
        self._owned.append(handle)
        return self._python(handle, spec)

    def _generator(self, value: Any) -> int:
        """A `random.Random` lent: a handle drawing from its state in place, the
        state kept to put back if the call falls back (`restore`)."""
        import random  # pylint: disable=import-outside-toplevel

        if type(value) is not random.Random or not _generator_layout():
            raise Refused
        address = id(value) + object.__basicsize__
        if address not in self._generators:
            self._generators[address] = ctypes.string_at(address, _STATE_BYTES)
        return int(self.rt.ppy_random_external(address))

    def restore(self) -> None:
        """Put each generator an argument lent back as it was: the call fell
        back, and Python draws the same numbers again."""
        for address, state in self._generators.items():
            ctypes.memmove(address, state, _STATE_BYTES)

    def sync(self, arguments: list[tuple[Any, Spec]]) -> None:
        """Copy each argument's native contents back into the caller's objects."""
        for value, spec in arguments:
            if spec.kind == _GENERATOR:
                continue  # drawn from in place
            handle = self._handles[id(value)][0]
            self._python(handle, spec, rewrite=True)

    def _python(self, handle: int, spec: Spec, rewrite: bool = False) -> Any:
        if spec.kind == "object":
            return self._instance(handle, rewrite)
        known = self._objects.get(handle)
        if known is not None and (not rewrite or handle in self._synced):
            return known
        self._synced.add(handle)
        made = known if known is not None else spec.python()()
        self._objects[handle] = made
        _replace(made, spec, self._read(handle, spec, rewrite))
        return made

    def _instance(self, handle: int, rewrite: bool) -> Any:
        """The Python object of a native one: the one it came from, its fields
        set again when `rewrite`, or a new instance of its class, made without
        running `__init__` (native code ran it), its fields set."""
        if not handle:
            return None
        known = self._objects.get(handle)
        if known is not None and (not rewrite or handle in self._synced):
            return known
        self._synced.add(handle)
        classes = self.classes
        assert classes is not None
        tag = ctypes.c_int64.from_address(handle + 8 * _TAG).value
        described = classes.by_tag.get(tag)
        python = classes.python(described.qualname) if described is not None else None
        if described is None or python is None:
            raise RuntimeError("native code returned an object of a class it did not describe")
        made = known if known is not None else object.__new__(python)
        self._objects[handle] = made
        buffer = (ctypes.c_int64 * described.words)()
        self.rt.ppy_coll_copy_out(handle, None, buffer)
        words = struct.unpack_from("<" + _mask_format(described.floats, described.words), buffer)
        for name, offset, field in classes.fields[described.qualname]:
            object.__setattr__(made, name, self._field_value(words, offset, field, rewrite))
        return made

    def _field_value(self, words: tuple[Any, ...], offset: int, spec: Spec, rewrite: bool) -> Any:
        if spec.kind == "bool":
            return words[offset] != 0
        if spec.kind in _SCALARS:
            return words[offset]
        if spec.kind == "optional":
            if not words[offset + 1]:
                return None
            return words[offset] != 0 if spec.parts[0] == "bool" else words[offset]
        if spec.kind == "str":
            if not words[offset] and spec.nullable:
                return None
            return self._texts((words[offset],))[0]
        if spec.kind == "tuple":
            parts = words[offset : offset + len(spec.parts)]
            return tuple(
                part != 0 if kind == "bool" else part
                for kind, part in zip(spec.parts, parts, strict=True)
            )
        if spec.kind == "record":
            return self._record(spec, words[offset : offset + len(spec.parts)])
        return self._python(words[offset], spec, rewrite) if words[offset] else None

    def _record(self, spec: Spec, words: Any) -> Any:
        """A value class's instance from its fields' words."""
        classes = self.classes
        assert classes is not None
        python = classes.python(spec.record)
        described = classes.by_name.get(spec.record)
        if python is None or described is None:
            raise RuntimeError("native code returned a value of a class it did not describe")
        made = object.__new__(python)
        for (name, _offset, kind), part in zip(described.fields, words, strict=True):
            object.__setattr__(made, name, part != 0 if kind == "bool" else part)
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
        if spec.kind == "object":
            return [self._instance(word, rewrite) for word in words]
        if spec.kind == "record":
            width = len(spec.parts)
            return [
                self._record(spec, words[start : start + width])
                for start in range(0, len(words), width)
            ]
        if spec.kind == "str" and spec.nullable and 0 in words:
            texts = iter(self._texts(tuple(word for word in words if word)))
            return [next(texts) if word else None for word in words]
        if spec.kind == "str":
            return self._texts(words)
        if spec.kind == "optional":
            boolean = spec.parts[0] == "bool"
            return [
                (number != 0 if boolean else number) if flag else None
                for number, flag in zip(words[0::2], words[1::2], strict=True)
            ]
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


_LAYOUT: list[bool] = []


def _generator_layout() -> bool:
    """Whether a `random.Random` keeps its index and state words right after
    its object header, as `random._inst` does (`binding._random_state_address`):
    asked once, of a generator made for it."""
    if not _LAYOUT:
        import random  # pylint: disable=import-outside-toplevel

        found = False
        if sys.implementation.name == "cpython":
            probe = random.Random(20261002)
            _version, words, _gauss = probe.getstate()
            address = id(probe) + object.__basicsize__
            index = ctypes.c_int32.from_address(address).value
            state = (ctypes.c_uint32 * 624).from_address(address + 4)
            found = index == words[-1] and tuple(state) == tuple(words[:-1])
        _LAYOUT.append(found)
    return _LAYOUT[0]


#: The one Python type a word of each scalar kind is stored as.
_TYPES = {"int": int, "float": float, "bool": bool}


def _split(data: bytes, lengths: Any) -> list[str]:
    """Strings laid one after another, each followed by a NUL."""
    texts = []
    start = 0
    for length in lengths:
        texts.append(data[start : start + length].decode("utf-8"))
        start += length + 1
    return texts


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
    if spec.kind == "list":
        made[:] = entries
        return
    if spec.kind == "dict":
        made.clear()
        made.update(entries)
        return
    if spec.kind == "set":
        made.clear()
        made.update(key for key, _ in entries)
        return
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
