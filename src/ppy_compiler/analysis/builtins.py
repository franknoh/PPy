"""Signatures, effects, and refinements for builtin callables."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from . import types as T
from .effects import Effect, EffectSet
from .refinements import Facts, IntRange

__all__ = ["BUILTINS", "MODULE_EFFECTS", "BuiltinResult", "call_builtin", "is_builtin"]


@dataclass(frozen=True, slots=True)
class BuiltinResult:
    type: T.Type
    facts: Facts = field(default_factory=Facts)
    effects: EffectSet = field(default_factory=EffectSet)


@dataclass(frozen=True, slots=True)
class Arg:
    type: T.Type
    facts: Facts


Handler = Callable[[Sequence[Arg]], BuiltinResult]

_LEN_RANGE = IntRange(0, None)
_ALLOC = EffectSet.of(Effect.ALLOC)
_IO = EffectSet.of(Effect.IO)


def _element_of(t: T.Type) -> T.Type:
    base = T.strip_literal(t)
    if isinstance(base, T.Union_):
        # Iterating `list[float] | tuple[float, float]` yields a float either
        # way; the union is over containers, not over what they hand out.
        elements = [_element_of(member) for member in T.members_of(base)]
        if elements and not any(isinstance(e, T.UnknownType) for e in elements):
            return T.join(*elements)
        return T.UNKNOWN
    if isinstance(base, T.Tuple_):
        if base.homogeneous:
            return base.items[0]
        return T.join(*base.items) if base.items else T.NEVER
    if isinstance(base, T.Instance):
        if (
            base.name
            in {
                "list",
                "set",
                "frozenset",
                "Sequence",
                "Iterable",
                "Iterator",
                "Buffer",
                "memoryview",
                "array",
            }
            and base.args
        ):
            return base.args[0]
        if base.name == "dict" and base.args:
            return base.args[0]
        if base.name == "str":
            return T.STR
        if base.name == "bytes":
            return T.INT
        if base.name == "range":
            return T.INT
    return T.UNKNOWN


def element_type(t: T.Type) -> T.Type:
    return _element_of(t)


def _len(args: Sequence[Arg]) -> BuiltinResult:
    facts = Facts(int_range=_LEN_RANGE)
    if args and args[0].facts.length is not None:
        size = args[0].facts.length
        facts = Facts(int_range=IntRange(size, size), constant=size, has_constant=True)
    elif args:
        base = T.strip_literal(args[0].type)
        if isinstance(base, T.Tuple_) and not base.homogeneous:
            size = len(base.items)
            facts = Facts(int_range=IntRange(size, size), constant=size, has_constant=True)
    return BuiltinResult(T.INT, facts, EffectSet.of(raises=("TypeError",)))


def _range(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(
        T.instance("range"), Facts(), EffectSet.of(raises=("TypeError", "ValueError"))
    )


def _int(args: Sequence[Arg]) -> BuiltinResult:
    if args and args[0].facts.has_constant:
        try:
            value = int(args[0].facts.constant)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            value = None
        if value is not None:
            return BuiltinResult(
                T.INT, Facts(constant=value, has_constant=True, int_range=IntRange(value, value))
            )
    if args and T.strip_literal(args[0].type) == T.INT:
        return BuiltinResult(T.INT, args[0].facts)
    return BuiltinResult(T.INT, Facts(), EffectSet.of(raises=("ValueError", "TypeError")))


def _float(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.FLOAT, Facts(), EffectSet.of(raises=("ValueError", "TypeError")))


def _bool(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.BOOL, Facts(int_range=IntRange(0, 1)))


def _str(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.STR, Facts(), _ALLOC)


def _abs(args: Sequence[Arg]) -> BuiltinResult:
    if not args:
        return BuiltinResult(T.UNKNOWN)
    t = T.strip_literal(args[0].type)
    facts = Facts()
    if args[0].facts.int_range is not None:
        r = args[0].facts.int_range
        candidates = [abs(v) for v in (r.low, r.high) if v is not None]
        if len(candidates) == 2 and r.low is not None and r.high is not None:
            low = 0 if r.low <= 0 <= r.high else min(candidates)
            facts = Facts(int_range=IntRange(low, max(candidates)))
    return BuiltinResult(t if t in (T.INT, T.FLOAT) else T.UNKNOWN, facts)


def _min_max(args: Sequence[Arg]) -> BuiltinResult:
    if len(args) == 1:
        return BuiltinResult(
            _element_of(args[0].type), Facts(), EffectSet.of(raises=("ValueError",))
        )
    return BuiltinResult(T.join(*[a.type for a in args]) if args else T.UNKNOWN)


def _sum(args: Sequence[Arg]) -> BuiltinResult:
    if not args:
        return BuiltinResult(T.INT)
    element = _element_of(args[0].type)
    if element == T.FLOAT:
        return BuiltinResult(T.FLOAT)
    if element in (T.INT, T.BOOL):
        return BuiltinResult(T.INT)
    return BuiltinResult(T.join(element, T.INT))


def _list(args: Sequence[Arg]) -> BuiltinResult:
    element = _element_of(args[0].type) if args else T.NEVER
    return BuiltinResult(T.list_of(element), Facts(), _ALLOC)


def _tuple(args: Sequence[Arg]) -> BuiltinResult:
    if args:
        base = T.strip_literal(args[0].type)
        if isinstance(base, T.Tuple_):
            return BuiltinResult(base, args[0].facts, _ALLOC)
        return BuiltinResult(
            T.Tuple_((_element_of(args[0].type),), homogeneous=True), Facts(), _ALLOC
        )
    return BuiltinResult(T.Tuple_(()), Facts(), _ALLOC)


def _set(args: Sequence[Arg]) -> BuiltinResult:
    element = _element_of(args[0].type) if args else T.NEVER
    return BuiltinResult(T.set_of(element), Facts(), _ALLOC)


def _dict(args: Sequence[Arg]) -> BuiltinResult:
    if args:
        base = T.strip_literal(args[0].type)
        if isinstance(base, T.Instance) and base.name == "dict" and len(base.args) == 2:
            return BuiltinResult(base, Facts(), _ALLOC)
    return BuiltinResult(T.dict_of(T.ANY, T.ANY), Facts(), _ALLOC)


def _enumerate(args: Sequence[Arg]) -> BuiltinResult:
    element = _element_of(args[0].type) if args else T.UNKNOWN
    pair = T.Tuple_((T.INT, element))
    return BuiltinResult(T.instance("Iterator", pair), Facts(), _ALLOC)


def _zip(args: Sequence[Arg]) -> BuiltinResult:
    items = tuple(_element_of(a.type) for a in args)
    return BuiltinResult(T.instance("Iterator", T.Tuple_(items)), Facts(), _ALLOC)


def _sorted(args: Sequence[Arg]) -> BuiltinResult:
    element = _element_of(args[0].type) if args else T.UNKNOWN
    return BuiltinResult(T.list_of(element), Facts(), _ALLOC | EffectSet.of(Effect.PYTHON_CALLBACK))


def _reversed(args: Sequence[Arg]) -> BuiltinResult:
    element = _element_of(args[0].type) if args else T.UNKNOWN
    return BuiltinResult(T.instance("Iterator", element), Facts(), _ALLOC)


def _print(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.NONE, Facts(), _IO)


def _round(args: Sequence[Arg]) -> BuiltinResult:
    if len(args) >= 2:
        return BuiltinResult(T.FLOAT)
    return BuiltinResult(T.INT)


def _divmod(args: Sequence[Arg]) -> BuiltinResult:
    t = T.join(*[T.strip_literal(a.type) for a in args]) if args else T.INT
    return BuiltinResult(T.Tuple_((t, t)), Facts(), EffectSet.of(raises=("ZeroDivisionError",)))


def _isinstance(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.BOOL, Facts(int_range=IntRange(0, 1)))


def _identity_object(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.OBJECT)


def _iter(args: Sequence[Arg]) -> BuiltinResult:
    element = _element_of(args[0].type) if args else T.UNKNOWN
    return BuiltinResult(T.instance("Iterator", element))


def _next(args: Sequence[Arg]) -> BuiltinResult:
    base = T.strip_literal(args[0].type) if args else T.UNKNOWN
    element = base.args[0] if isinstance(base, T.Instance) and base.args else T.UNKNOWN
    if len(args) > 1:
        return BuiltinResult(T.join(element, args[1].type))
    return BuiltinResult(element, Facts(), EffectSet.of(raises=("StopIteration",)))


def _all_any(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.BOOL, Facts(int_range=IntRange(0, 1)))


def _open(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.OBJECT, Facts(), _IO | EffectSet.of(Effect.ALLOC, raises=("OSError",)))


def _input(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.STR, Facts(), _IO)


def _repr(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.STR, Facts(), _ALLOC | EffectSet.of(Effect.PYTHON_CALLBACK))


def _hash(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(
        T.INT, Facts(), EffectSet.of(Effect.PYTHON_CALLBACK, raises=("TypeError",))
    )


def _id(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.INT, Facts(int_range=IntRange(0, None)))


def _ord(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(
        T.INT, Facts(int_range=IntRange(0, 0x10FFFF)), EffectSet.of(raises=("TypeError",))
    )


def _chr(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.STR, Facts(length=1), EffectSet.of(raises=("ValueError",)))


def _type(args: Sequence[Arg]) -> BuiltinResult:
    if len(args) == 1:
        base = T.strip_literal(args[0].type)
        if isinstance(base, T.Instance):
            return BuiltinResult(T.ClassObject(base.name, base))
        return BuiltinResult(T.instance("type"))
    return BuiltinResult(T.instance("type"), Facts(), _ALLOC)


def _callable(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.BOOL, Facts(int_range=IntRange(0, 1)))


def _map_filter(args: Sequence[Arg]) -> BuiltinResult:
    """`filter` yields what it was given; `map` yields what it calls."""
    element = _element_of(args[1].type) if len(args) > 1 else T.UNKNOWN
    return BuiltinResult(
        T.instance("Iterator", element),
        Facts(),
        _ALLOC | EffectSet.of(Effect.PYTHON_CALLBACK),
    )


def _map(args: Sequence[Arg]) -> BuiltinResult:
    """`map(f, xs)` yields `f`'s results, not `xs`'s elements."""
    element = T.UNKNOWN
    if len(args) > 1:
        function = args[0].type
        if isinstance(function, T.Callable_):
            element = function.ret
        elif isinstance(function, T.ClassObject) and function.instance_type is not None:
            element = function.instance_type
    return BuiltinResult(
        T.instance("Iterator", element),
        Facts(),
        _ALLOC | EffectSet.of(Effect.PYTHON_CALLBACK),
    )


def _radix(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.STR, Facts(), _ALLOC | EffectSet.of(raises=("TypeError",)))


def _bytes(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.BYTES, Facts(), _ALLOC)


def _slice(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.instance("slice"))


def _memoryview(args: Sequence[Arg]) -> BuiltinResult:
    """A view over a contiguous buffer, carrying its element type through."""
    if args:
        base = T.strip_literal(args[0].type)
        if (
            isinstance(base, T.Instance)
            and base.args
            and base.name
            in {
                "array",
                "Buffer",
                "memoryview",
            }
        ):
            return BuiltinResult(T.instance("memoryview", base.args[0]))
    return BuiltinResult(T.instance("memoryview"), Facts(), EffectSet.of(raises=("TypeError",)))


def _getattr(args: Sequence[Arg]) -> BuiltinResult:
    default = args[2].type if len(args) > 2 else T.NEVER
    return BuiltinResult(
        T.join(T.UNKNOWN, default) if len(args) > 2 else T.UNKNOWN,
        Facts(),
        EffectSet.of(Effect.READ_OBJECT, raises=("AttributeError",)),
    )


def _hasattr(args: Sequence[Arg]) -> BuiltinResult:
    return BuiltinResult(T.BOOL, Facts(int_range=IntRange(0, 1)), EffectSet.of(Effect.READ_OBJECT))


def _pow(args: Sequence[Arg]) -> BuiltinResult:
    types = [T.strip_literal(a.type) for a in args]
    if all(t in (T.INT, T.BOOL) for t in types[:2]):
        return BuiltinResult(T.INT)
    return BuiltinResult(T.FLOAT)


BUILTINS: dict[str, Handler] = {
    "len": _len,
    "range": _range,
    "int": _int,
    "float": _float,
    "bool": _bool,
    "str": _str,
    "abs": _abs,
    "min": _min_max,
    "max": _min_max,
    "sum": _sum,
    "list": _list,
    "tuple": _tuple,
    "set": _set,
    "frozenset": _set,
    "dict": _dict,
    "enumerate": _enumerate,
    "zip": _zip,
    "sorted": _sorted,
    "reversed": _reversed,
    "print": _print,
    "round": _round,
    "divmod": _divmod,
    "isinstance": _isinstance,
    "issubclass": _isinstance,
    "iter": _iter,
    "next": _next,
    "all": _all_any,
    "any": _all_any,
    "open": _open,
    "input": _input,
    "repr": _repr,
    "hash": _hash,
    "id": _id,
    "ord": _ord,
    "chr": _chr,
    "pow": _pow,
    "object": _identity_object,
    "type": _type,
    "callable": _callable,
    "map": _map,
    "filter": _map_filter,
    "bin": _radix,
    "hex": _radix,
    "oct": _radix,
    "format": _str,
    "ascii": _repr,
    "bytes": _bytes,
    "bytearray": _bytes,
    "slice": _slice,
    "memoryview": _memoryview,
    "getattr": _getattr,
    "hasattr": _hasattr,
}

#: Effects attributed to calls into well-known standard-library modules.
MODULE_EFFECTS: dict[str, EffectSet] = {
    "random": EffectSet.of(Effect.RANDOM),
    "time": EffectSet.of(Effect.TIME),
    "datetime": EffectSet.of(Effect.TIME),
    "os": EffectSet.of(Effect.IO),
    "sys": EffectSet.of(Effect.IO),
    "io": EffectSet.of(Effect.IO),
    "pathlib": EffectSet.of(Effect.IO),
    "shutil": EffectSet.of(Effect.IO),
    "subprocess": EffectSet.of(Effect.PROCESS, Effect.IO),
    "multiprocessing": EffectSet.of(Effect.PROCESS),
    "threading": EffectSet.of(Effect.THREAD, Effect.SYNC),
    "asyncio": EffectSet.of(Effect.SYNC, Effect.IO),
    "socket": EffectSet.of(Effect.IO),
    "logging": EffectSet.of(Effect.IO),
    "secrets": EffectSet.of(Effect.RANDOM),
    "uuid": EffectSet.of(Effect.RANDOM),
}

#: Pure standard-library math functions usable inside `@ppy.pure`.
MATH_FUNCTIONS = {
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "exp",
    "log",
    "log2",
    "log10",
    "pow",
    "floor",
    "ceil",
    "fabs",
    "fmod",
    "hypot",
    "isnan",
    "isinf",
    "isfinite",
    "trunc",
    "copysign",
    "degrees",
    "radians",
}

_MATH_INT_RESULTS = {"floor", "ceil", "trunc"}
_MATH_BOOL_RESULTS = {"isnan", "isinf", "isfinite"}


def math_result(name: str) -> BuiltinResult | None:
    if name not in MATH_FUNCTIONS:
        return None
    if name in _MATH_INT_RESULTS:
        return BuiltinResult(T.INT, Facts(), EffectSet.of(raises=("ValueError", "OverflowError")))
    if name in _MATH_BOOL_RESULTS:
        return BuiltinResult(T.BOOL)
    return BuiltinResult(T.FLOAT, Facts(), EffectSet.of(raises=("ValueError",)))


def is_builtin(name: str) -> bool:
    return name in BUILTINS


def call_builtin(name: str, args: Sequence[tuple[T.Type, Facts]]) -> BuiltinResult | None:
    handler = BUILTINS.get(name)
    if handler is None:
        return None
    return handler([Arg(t, f) for t, f in args])
