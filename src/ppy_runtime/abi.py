"""The PPY native ABI, as data: what a compiled artifact promises (spec 16.4).

This is the contract a binding manifest serializes and a runtime rebuilds.
Nothing here may depend on the compiler: the same dataclasses describe a
function to the lowering that emits it and to the runtime that calls it
years later.
"""

from __future__ import annotations

from ._record import record as dataclass

__all__ = [
    "OPTIONAL",
    "STATUS_FALLBACK",
    "STATUS_OK",
    "STATUS_RAISED",
    "TEXT",
    "VARIADIC",
    "CrossingClass",
    "NativeParam",
    "NativeSignature",
]

STATUS_OK = 0
STATUS_FALLBACK = 1
#: An exception is pending (`ppy_runtime/exceptions.c`): a caller in a `try`
#: catches it, any other returns this too. The boundary sees a status other
#: than `STATUS_OK` and runs the call as Python, which raises it; negative, so
#: it is never read as a sanitizer's.
STATUS_RAISED = -1
#: A sanitizer's check failed: `STATUS_SANITIZER_BASE + SANITIZERS.index(kind)`.
#: Unlike a fallback, the boundary raises; nothing Python could do would be right.
STATUS_SANITIZER_BASE = 2
SANITIZERS = ("bounds", "overflow", "pointer", "alignment")

#: `i8`/`u8` are byte-wide buffer elements; `bool` shares the width.
_ABI_NAMES = {
    "int": "i64",
    "float": "double",
    "bool": "i8",
    "i8": "i8",
    "u8": "i8",
    # A collection's handle: native callers pass it, and the Python boundary
    # copies a collection into one and out of one.
    "handle": "i8*",
}

#: A string at the Python boundary: its UTF-8 bytes and how many. As a
#: result, the bytes are a copy the boundary frees once it has read them.
TEXT = "text"


#: The `kind` of a parameter that is a number or `None` (`int | None`,
#: `float | None`, `bool | None`); its `element` is the number's kind.
OPTIONAL = "optional"


#: The `source` of a parameter that is a function's `*args` of numbers: the
#: positions Python spells after the named ones, which the boundary packs into
#: the list the native entry takes.
VARIADIC = "*"


def _abi_name(scalar: str) -> str:
    return _ABI_NAMES[scalar]


@dataclass(frozen=True, slots=True)
class NativeParam:
    """One source-level parameter and the ABI atoms it expands to."""

    name: str
    kind: str
    element: str = ""
    elements: tuple[str, ...] = ()
    fields: tuple[tuple[str, str], ...] = ()
    class_name: str = ""
    #: A collection the function writes through: the boundary copies its
    #: contents back into the caller's object after the call.
    written: bool = False
    #: A module global the function reads, passed as this parameter:
    #: `module:name`. Python does not pass it; the boundary reads the global
    #: when the function is called.
    source: str = ""
    #: An object parameter that may be `None`, the null handle.
    nullable: bool = False
    #: A `float` (or a tuple, list, or value class holding floats) whose
    #: int-ness the body would show: an `int` given for it keeps the call in
    #: Python, where it stays an `int`.
    exact: bool = False

    @property
    def is_buffer(self) -> bool:
        return self.kind in {"list", "sequence", "view"}

    @property
    def is_pointer(self) -> bool:
        """A `ppy.native.ptr[T]`: one machine address, no Python boundary."""
        return self.kind in {"ptr", "const_ptr"}

    @property
    def is_object(self) -> bool:
        return self.kind == "object"

    @property
    def is_handle(self) -> bool:
        """A `ppy.Vec` or another collection, or an object: its runtime handle."""
        return self.kind == "handle"

    @property
    def is_text(self) -> bool:
        """A `str` crossing from Python: its UTF-8 bytes and their count."""
        return self.kind == TEXT

    @property
    def is_borrowed(self) -> bool:
        """A `ppy.Buffer[T]` is borrowed in place; a list is copied out.

        A Python list holds boxed elements, so there is no contiguous array to
        point at. A buffer-protocol object already has one (spec 6.4, 13.8).
        """
        return self.kind == "view"

    @property
    def is_tuple(self) -> bool:
        return self.kind == "tuple"

    @property
    def is_optional(self) -> bool:
        """A number or `None` (`int | None`): the number's atom, then a byte
        that says whether there is one; `None` crosses as 0 and 0."""
        return self.kind == OPTIONAL

    @property
    def abi(self) -> tuple[str, ...]:
        if self.is_buffer:
            return (f"{_abi_name(self.element)}*", "i64")
        if self.is_pointer:
            return (f"{_abi_name(self.element)}*",)
        if self.is_handle:
            return ("i8*",)
        if self.is_text:
            return ("i8*", "i64")
        if self.is_tuple:
            return tuple(_abi_name(element) for element in self.elements)
        if self.is_optional:
            return (_abi_name(self.element), "i8")
        if self.is_object:
            return tuple(_abi_name(scalar) for _field, scalar in self.fields)
        return (_abi_name(self.kind),)

    def __str__(self) -> str:
        if self.is_buffer:
            borrow = " borrowed" if self.is_borrowed else ""
            return f"{_abi_name(self.element)}*{borrow} {self.name}, i64 {self.name}_len"
        if self.is_pointer:
            return f"{_abi_name(self.element)}* {self.name}"
        if self.is_handle:
            if self.element == "str":
                return f"str {self.name}"
            return f"{self.element} {self.name}"
        if self.is_text:
            return f"i8* {self.name}, i64 {self.name}_len"
        if self.is_tuple:
            return ", ".join(
                f"{_abi_name(element)} {self.name}{index}"
                for index, element in enumerate(self.elements)
            )
        if self.is_optional:
            return f"{_abi_name(self.element)} {self.name}, i8 {self.name}_present"
        if self.is_object:
            return ", ".join(
                f"{_abi_name(scalar)} {self.name}_{field}" for field, scalar in self.fields
            )
        return f"{_abi_name(self.kind)} {self.name}"


@dataclass(frozen=True, slots=True)
class CrossingClass:
    """A project class whose instances cross the Python boundary, as native
    code lays them out.

    An object (`kind` "object") is a handle: a sequence of one record, its
    fields at their word offsets, its class's tag in the header. A value
    class (`kind` "record") is its fields' words in place, inside a
    collection's element.
    """

    qualname: str
    #: Where Python finds the class: its module and its name there.
    module: str
    name: str
    kind: str
    #: Each field: its name, its first word, and its type as a signature
    #: spells an element (`int`, `str`, `list[int]`, `prog.Node`).
    fields: tuple[tuple[str, int, str], ...] = ()
    #: An object's record: its words, which are floats, which are handles,
    #: and (above bit 32) which handles are strings.
    words: int = 0
    floats: int = 0
    handles: int = 0
    tag: int = 0
    #: The classes an instance of this one is an instance of, itself first.
    bases: tuple[str, ...] = ()
    #: Whether an instance keeps its native record between calls (`crossing.c`):
    #: every field a number, a string, a tuple of numbers, or an object of a
    #: class that is resident too.
    resident: bool = False


def classes_to_json(classes: tuple[CrossingClass, ...]) -> list[dict]:
    """The classes a signature crosses, as a manifest or a cache writes them."""
    return [
        {
            "qualname": c.qualname,
            "module": c.module,
            "name": c.name,
            "kind": c.kind,
            "fields": [list(f) for f in c.fields],
            "words": c.words,
            "floats": c.floats,
            "handles": c.handles,
            "tag": c.tag,
            "bases": list(c.bases),
            "resident": c.resident,
        }
        for c in classes
    ]


def classes_from_json(raw: list[dict]) -> tuple[CrossingClass, ...]:
    return tuple(
        CrossingClass(
            qualname=str(c["qualname"]),
            module=str(c["module"]),
            name=str(c["name"]),
            kind=str(c["kind"]),
            fields=tuple((str(n), int(o), str(s)) for n, o, s in c["fields"]),
            words=int(c["words"]),
            floats=int(c["floats"]),
            handles=int(c["handles"]),
            tag=int(c["tag"]),
            bases=tuple(str(b) for b in c["bases"]),
            resident=bool(c.get("resident", False)),
        )
        for c in raw
    )


@dataclass(frozen=True, slots=True)
class NativeSignature:
    """The PPY native ABI for one function (spec 16.4)."""

    qualname: str
    symbol: str
    parameters: tuple[NativeParam, ...]
    returns: tuple[str, ...]
    #: The body touches no Python object once its arguments are unpacked, so
    #: the boundary may drop the GIL around the call (spec 16.6).
    releases_gil: bool = False
    #: CPU features the code was compiled for (`@ppy.cpu.target`); the
    #: boundary binds it only on a machine that has them all.
    cpu_features: tuple[str, ...] = ()
    #: A coroutine: the boundary hands back a future the runtime completes,
    #: carrying this kind -- `int`, `float`, `bool`, or `none`.
    future: str = ""
    #: A returned collection's type, spelled (`ppy.Vec[int]`), for the boundary
    #: to build the Python object from the handle.
    returned: str = ""
    #: The body draws from `random`'s generator: the boundary saves the state
    #: before the call and puts it back when the call falls back to Python,
    #: so the rerun draws the same numbers.
    draws: bool = False
    #: The project classes whose instances cross with its arguments or its
    #: result, and every subclass of those.
    classes: tuple[CrossingClass, ...] = ()
    #: The body prints, reads, or calls into Python (`ppy_runtime/effects.py`):
    #: the boundary holds its output until it answers, and raises what it
    #: raised after an effect it cannot take back.
    effects: bool = False
    #: The result is a number of this kind or `None` (`int | None`): its two
    #: atoms are the number and a byte that says whether there is one.
    optional: str = ""

    @property
    def crosses_collections(self) -> bool:
        """Whether a collection crosses the boundary, in or out."""
        return (bool(self.returned) and not self.returns_none) or any(
            p.is_handle for p in self.parameters
        )

    @property
    def returns_none(self) -> bool:
        """The function returns `None`: its native entry fills a placeholder
        word, and the boundary hands back `None`."""
        return self.returned == "None"

    @property
    def reads_globals(self) -> bool:
        """Whether module globals are passed to it, which the Python-level
        binding reads at each call."""
        return any(p.source for p in self.parameters)

    @property
    def ret(self) -> str:
        return self.returns[0] if len(self.returns) == 1 else "{" + ", ".join(self.returns) + "}"

    @property
    def returns_tuple(self) -> bool:
        return len(self.returns) > 1 and not self.optional

    @property
    def params(self) -> tuple[str, ...]:
        return tuple(atom for parameter in self.parameters for atom in parameter.abi)

    def __str__(self) -> str:
        rendered = ", ".join(str(p) for p in self.parameters)
        return f"{self.ret} {self.symbol}({rendered})"
