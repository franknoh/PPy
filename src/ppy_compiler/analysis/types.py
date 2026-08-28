"""Semantic types (spec 8, 10.1).

A semantic type is the language-level meaning of a value. It is deliberately
kept separate from refinements (spec 10.2) and from the physical representation
chosen by the backend (spec 10.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = [
    "Type",
    "AnyType",
    "UnknownType",
    "NeverType",
    "Instance",
    "Literal",
    "Tuple_",
    "Union_",
    "Callable_",
    "Param",
    "Module_",
    "ClassObject",
    "TypeVar_",
    "ANY",
    "UNKNOWN",
    "NEVER",
    "NONE",
    "BOOL",
    "INT",
    "FLOAT",
    "COMPLEX",
    "STR",
    "BYTES",
    "OBJECT",
    "ELLIPSIS_T",
    "BUILTIN_MRO",
    "instance",
    "list_of",
    "set_of",
    "dict_of",
    "union",
    "is_assignable",
    "join",
    "remove_none",
    "is_optional",
    "is_numeric",
    "numeric_rank",
    "is_exact_builtin",
]


class Type:
    """Base class for semantic types."""

    __slots__ = ()

    def __str__(self) -> str:  # pragma: no cover - overridden
        return repr(self)

    @property
    def is_dynamic(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class AnyType(Type):
    """Explicit `Any`. Always an optimization barrier (spec 8.2)."""

    def __str__(self) -> str:
        return "Any"

    @property
    def is_dynamic(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class UnknownType(Type):
    """An inference hole. Reported as an error rather than becoming `Any`."""

    reason: str = ""

    def __str__(self) -> str:
        return "<unknown>"

    @property
    def is_dynamic(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class NeverType(Type):
    def __str__(self) -> str:
        return "Never"


BUILTIN_MRO: dict[str, tuple[str, ...]] = {
    "object": ("object",),
    "type": ("type", "object"),
    "bool": ("bool", "int", "object"),
    "int": ("int", "object"),
    "float": ("float", "object"),
    "complex": ("complex", "object"),
    "str": ("str", "object"),
    "bytes": ("bytes", "object"),
    "bytearray": ("bytearray", "object"),
    "list": ("list", "object"),
    "tuple": ("tuple", "object"),
    "dict": ("dict", "object"),
    "set": ("set", "object"),
    "frozenset": ("frozenset", "object"),
    "range": ("range", "object"),
    "slice": ("slice", "object"),
    "memoryview": ("memoryview", "Buffer", "object"),
    "Buffer": ("Buffer", "Sequence", "object"),
    "array": ("array", "Buffer", "object"),
    "NoneType": ("NoneType", "object"),
    "ellipsis": ("ellipsis", "object"),
    "BaseException": ("BaseException", "object"),
    "Exception": ("Exception", "BaseException", "object"),
    "ValueError": ("ValueError", "Exception", "BaseException", "object"),
    "TypeError": ("TypeError", "Exception", "BaseException", "object"),
    "IndexError": ("IndexError", "LookupError", "Exception", "BaseException", "object"),
    "KeyError": ("KeyError", "LookupError", "Exception", "BaseException", "object"),
    "LookupError": ("LookupError", "Exception", "BaseException", "object"),
    "ZeroDivisionError": ("ZeroDivisionError", "ArithmeticError", "Exception", "BaseException", "object"),
    "ArithmeticError": ("ArithmeticError", "Exception", "BaseException", "object"),
    "OverflowError": ("OverflowError", "ArithmeticError", "Exception", "BaseException", "object"),
    "StopIteration": ("StopIteration", "Exception", "BaseException", "object"),
    "AttributeError": ("AttributeError", "Exception", "BaseException", "object"),
}

_ABSTRACT_MRO: dict[str, tuple[str, ...]] = {
    "Sequence": ("Sequence", "object"),
    "Iterable": ("Iterable", "object"),
    "Iterator": ("Iterator", "Iterable", "object"),
    "Mapping": ("Mapping", "object"),
    "Generator": ("Generator", "Iterator", "Iterable", "object"),
    "Coroutine": ("Coroutine", "Awaitable", "object"),
    "Awaitable": ("Awaitable", "object"),
    "AsyncIterator": ("AsyncIterator", "AsyncIterable", "object"),
    "AsyncIterable": ("AsyncIterable", "object"),
}


@dataclass(frozen=True, slots=True)
class Instance(Type):
    """An instance of a named class, with optional type arguments."""

    name: str
    args: tuple[Type, ...] = ()
    mro: tuple[str, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}[{', '.join(str(a) for a in self.args)}]"

    @property
    def resolved_mro(self) -> tuple[str, ...]:
        if self.mro:
            return self.mro
        return BUILTIN_MRO.get(self.name) or _ABSTRACT_MRO.get(self.name) or (self.name, "object")


@dataclass(frozen=True, slots=True)
class Literal(Type):
    """A literal type such as `Literal[4]` or `Literal["ok"]`."""

    value: object
    base: Instance = field(default=None)  # type: ignore[assignment]

    def __str__(self) -> str:
        return f"Literal[{self.value!r}]"


@dataclass(frozen=True, slots=True)
class Tuple_(Type):
    items: tuple[Type, ...] = ()
    homogeneous: bool = False

    def __str__(self) -> str:
        if self.homogeneous:
            return f"tuple[{self.items[0]}, ...]"
        if not self.items:
            return "tuple[()]"
        return f"tuple[{', '.join(str(i) for i in self.items)}]"

    @property
    def resolved_mro(self) -> tuple[str, ...]:
        return BUILTIN_MRO["tuple"]


@dataclass(frozen=True, slots=True)
class Union_(Type):
    members: tuple[Type, ...]

    def __str__(self) -> str:
        return " | ".join(str(m) for m in self.members)


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    type: Type
    has_default: bool = False
    kind: str = "positional_or_keyword"

    def __str__(self) -> str:
        return f"{self.name}: {self.type}"


@dataclass(frozen=True, slots=True)
class Callable_(Type):
    params: tuple[Param, ...]
    ret: Type
    qualname: str = ""
    is_async: bool = False
    is_generator: bool = False

    def __str__(self) -> str:
        rendered = ", ".join(str(p.type) for p in self.params)
        return f"({rendered}) -> {self.ret}"


@dataclass(frozen=True, slots=True)
class Module_(Type):
    name: str

    def __str__(self) -> str:
        return f"<module {self.name}>"


@dataclass(frozen=True, slots=True)
class ClassObject(Type):
    """The class object itself, as opposed to its instances."""

    name: str
    instance_type: Instance | None = None

    def __str__(self) -> str:
        return f"type[{self.name}]"


@dataclass(frozen=True, slots=True)
class TypeVar_(Type):
    name: str
    bound: Type | None = None

    def __str__(self) -> str:
        return self.name


ANY = AnyType()
UNKNOWN = UnknownType()
NEVER = NeverType()


def instance(name: str, *args: Type) -> Instance:
    return Instance(name, tuple(args), BUILTIN_MRO.get(name, ()))


NONE = instance("NoneType")
BOOL = instance("bool")
INT = instance("int")
FLOAT = instance("float")
COMPLEX = instance("complex")
STR = instance("str")
BYTES = instance("bytes")
OBJECT = instance("object")
ELLIPSIS_T = instance("ellipsis")

_NUMERIC_RANK = {"bool": 0, "int": 1, "float": 2, "complex": 3}


def numeric_rank(t: Type) -> int | None:
    if isinstance(t, Literal):
        t = t.base or OBJECT
    if isinstance(t, Instance):
        return _NUMERIC_RANK.get(t.name)
    return None


def is_numeric(t: Type) -> bool:
    return numeric_rank(t) is not None


def is_exact_builtin(t: Type) -> bool:
    return isinstance(t, Instance) and t.name in BUILTIN_MRO


def list_of(element: Type) -> Instance:
    return instance("list", element)


def set_of(element: Type) -> Instance:
    return instance("set", element)


def dict_of(key: Type, value: Type) -> Instance:
    return instance("dict", key, value)


def _flatten(types: Iterable[Type]) -> list[Type]:
    out: list[Type] = []
    for t in types:
        if isinstance(t, Union_):
            out.extend(_flatten(t.members))
        elif isinstance(t, NeverType):
            continue
        else:
            out.append(t)
    return out


def _widen_literals(t: Type) -> Type:
    """The same type with every literal replaced by what it is a literal of."""
    if isinstance(t, Literal):
        return t.base
    if isinstance(t, Tuple_):
        widened = tuple(_widen_literals(item) for item in t.items)
        return t if widened == t.items else Tuple_(widened, homogeneous=t.homogeneous)
    if isinstance(t, Instance) and t.args:
        widened = tuple(_widen_literals(arg) for arg in t.args)
        return t if widened == t.args else Instance(t.name, widened, t.mro)
    return t


def _all_never(args: tuple[Type, ...]) -> bool:
    return all(isinstance(a, NeverType) for a in args)


def union(*types: Type) -> Type:
    """Build a normalized union."""
    flat = _flatten(types)
    if not flat:
        return NEVER
    if any(isinstance(t, AnyType) for t in flat):
        return ANY
    unique: list[Type] = []
    for t in flat:
        if t not in unique:
            unique.append(t)
    # A literal is absorbed by the type it is a literal of, however deeply it
    # sits: `tuple[Literal[0.0], ...]` and `tuple[float, ...]` are one member,
    # which is what a function returning a constant from one branch produces.
    present = set(unique)
    unique = [
        t for t in unique
        if not (_widen_literals(t) != t and _widen_literals(t) in present)
    ]
    # `list[Never]` is the empty list, so it is absorbed by any populated list
    # of the same shape. Without this, `out = []` followed by `out.append(x)`
    # merges back into `list[Never] | list[X]` at the top of the loop.
    populated = {
        (t.name, len(t.args))
        for t in unique
        if isinstance(t, Instance) and t.args and not _all_never(t.args)
    }
    if populated:
        unique = [
            t for t in unique
            if not (
                isinstance(t, Instance)
                and t.args
                and _all_never(t.args)
                and (t.name, len(t.args)) in populated
            )
        ]
    if len(unique) == 1:
        return unique[0]
    return Union_(tuple(unique))


def join(*types: Type) -> Type:
    """The least common supertype usable as a flow merge result."""
    present = [t for t in types if not isinstance(t, NeverType)]
    if not present:
        return NEVER
    if any(isinstance(t, UnknownType) for t in present):
        return UNKNOWN
    first = present[0]
    if all(t == first for t in present):
        return first
    return union(*present)


def is_assignable(source: Type, target: Type) -> bool:
    """Is a value of `source` acceptable where `target` is expected?"""
    if isinstance(target, AnyType) or isinstance(source, AnyType):
        return True
    if isinstance(source, NeverType):
        return True
    if isinstance(source, UnknownType) or isinstance(target, UnknownType):
        return False
    if source == target:
        return True

    # A union source is decomposed first: every one of its members has to fit
    # somewhere in the target. Asking the other way round would reject
    # `int | None` against `None | int` purely because of member order.
    if isinstance(source, Union_):
        return all(is_assignable(m, target) for m in source.members)
    if isinstance(target, Union_):
        return any(is_assignable(source, m) for m in target.members)

    if isinstance(source, Literal):
        if isinstance(target, Literal):
            return source.value == target.value and type(source.value) is type(target.value)
        return is_assignable(source.base or OBJECT, target)
    if isinstance(target, Literal):
        return False

    if isinstance(source, Tuple_):
        return _tuple_assignable(source, target)
    if isinstance(target, Tuple_):
        return False

    if isinstance(source, Callable_) and isinstance(target, Callable_):
        return _callable_assignable(source, target)
    if isinstance(target, Callable_):
        return False

    if isinstance(source, TypeVar_):
        return is_assignable(source.bound or OBJECT, target)
    if isinstance(target, TypeVar_):
        return is_assignable(source, target.bound or OBJECT)

    if isinstance(source, ClassObject) and isinstance(target, ClassObject):
        return source.name == target.name
    if isinstance(source, Module_) or isinstance(target, Module_):
        return source == target

    if isinstance(source, Instance) and isinstance(target, Instance):
        return _instance_assignable(source, target)
    return False


def _instance_assignable(source: Instance, target: Instance) -> bool:
    if target.name == "object":
        return True
    rank_s, rank_t = numeric_rank(source), numeric_rank(target)
    if rank_s is not None and rank_t is not None:
        # Python's implicit numeric promotion: bool -> int -> float -> complex.
        if rank_s <= rank_t:
            return True
    if target.name not in source.resolved_mro:
        return False
    if not target.args:
        return True
    if len(source.args) != len(target.args):
        return False
    # Containers are treated invariantly except for immutable ones. An empty
    # display has a `Never` element type, which fits any element type.
    covariant = source.name in {"tuple", "frozenset", "Sequence", "Iterable", "Iterator", "Mapping"}
    if covariant:
        return all(is_assignable(a, b) for a, b in zip(source.args, target.args))
    return all(
        a == b or isinstance(b, AnyType) or isinstance(a, NeverType)
        for a, b in zip(source.args, target.args)
    )


def _tuple_assignable(source: Tuple_, target: Type) -> bool:
    if isinstance(target, Tuple_):
        if source.homogeneous or target.homogeneous:
            if target.homogeneous and source.homogeneous:
                return is_assignable(source.items[0], target.items[0])
            if target.homogeneous:
                return all(is_assignable(i, target.items[0]) for i in source.items)
            return False
        if len(source.items) != len(target.items):
            return False
        return all(is_assignable(a, b) for a, b in zip(source.items, target.items))
    if isinstance(target, Instance):
        if target.name in {"tuple", "object", "Sequence", "Iterable"}:
            if not target.args:
                return True
            element = join(*source.items) if source.items else NEVER
            return is_assignable(element, target.args[0])
    return False


def _callable_assignable(source: Callable_, target: Callable_) -> bool:
    if not is_assignable(source.ret, target.ret):
        return False
    required = [p for p in target.params if not p.has_default]
    if len(source.params) < len(required):
        return False
    for sp, tp in zip(source.params, target.params):
        if not is_assignable(tp.type, sp.type):
            return False
    return True


def is_optional(t: Type) -> bool:
    return isinstance(t, Union_) and any(m == NONE for m in t.members)


_IMMUTABLE_NAMES = frozenset({
    "int", "float", "complex", "bool", "str", "bytes", "NoneType",
    "frozenset", "range", "slice",
})


def is_immutable(t: Type) -> bool:
    """Can nothing a callee does to this value be seen by anyone else?"""
    base = strip_literal(t)
    if isinstance(base, NeverType):
        return True
    if isinstance(base, Union_):
        return all(is_immutable(m) for m in base.members)
    if isinstance(base, Tuple_):
        return all(is_immutable(item) for item in base.items)
    return isinstance(base, Instance) and base.name in _IMMUTABLE_NAMES and not base.args


def remove_none(t: Type) -> Type:
    if isinstance(t, Union_):
        return union(*[m for m in t.members if m != NONE])
    if t == NONE:
        return NEVER
    return t


def type_of_constant(value: object) -> Type:
    if value is None:
        return NONE
    if value is Ellipsis:
        return ELLIPSIS_T
    if isinstance(value, bool):
        return Literal(value, BOOL)
    if isinstance(value, int):
        return Literal(value, INT)
    if isinstance(value, float):
        return Literal(value, FLOAT)
    if isinstance(value, complex):
        return COMPLEX
    if isinstance(value, str):
        return Literal(value, STR)
    if isinstance(value, bytes):
        return Literal(value, BYTES)
    return OBJECT


def strip_literal(t: Type) -> Type:
    """Erase literal types, including inside containers."""
    if isinstance(t, Literal):
        return t.base or OBJECT
    if isinstance(t, Union_):
        return union(*[strip_literal(m) for m in t.members])
    if isinstance(t, Tuple_):
        return Tuple_(tuple(strip_literal(i) for i in t.items), t.homogeneous)
    if isinstance(t, Instance) and t.args:
        return Instance(t.name, tuple(strip_literal(a) for a in t.args), t.mro)
    return t


def members_of(t: Type) -> Sequence[Type]:
    return t.members if isinstance(t, Union_) else (t,)
