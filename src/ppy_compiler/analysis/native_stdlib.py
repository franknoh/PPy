"""The standard-library calls native code makes itself: `random`, `heapq`,
and `bisect` over plain lists.

`stdlib` types these modules for any caller; what is here decides, from the
arguments, which calls have a native lowering and what each one gives. A
call this module answers for carries no native blocker, and the lowering
(`lowering/stdlib.py`) implements exactly these shapes, CPython's
algorithms draw for draw and comparison for comparison. Anything else it
answers None for, and the call stays a Python call.
"""

from __future__ import annotations

from typing import Protocol

from . import types as T
from .effects import Effect, EffectSet
from .refinements import Facts

__all__ = ["MODELS", "MUTATES_FIRST", "call"]


class _Argument(Protocol):
    type: T.Type
    facts: Facts


_RANDOM = EffectSet.of(Effect.RANDOM)
_RANDOM_RAISES = EffectSet.of(Effect.RANDOM, raises=("ValueError",))
_CHOOSES = EffectSet.of(Effect.RANDOM, raises=("IndexError",))
_SAMPLES = EffectSet.of(Effect.RANDOM, Effect.ALLOC, raises=("ValueError", "IndexError"))
_SHUFFLES = EffectSet.of(Effect.RANDOM, Effect.WRITE_OBJECT)
_HEAP_WRITE = EffectSet.of(Effect.WRITE_OBJECT, raises=("IndexError",))
_ALLOC = EffectSet.of(Effect.ALLOC)
_SEARCH = EffectSet.of(raises=("ValueError", "IndexError"))
_INSORT = EffectSet.of(Effect.WRITE_OBJECT, raises=("ValueError", "IndexError"))


def _fn(qualname: str, ret: T.Type, effects: EffectSet) -> tuple[T.Type, EffectSet]:
    return T.Callable_((), ret, qualname), effects


#: Every function here, typed as a caller sees it whatever its arguments.
MODELS: dict[str, tuple[T.Type, EffectSet]] = {
    "random.random": _fn("random.random", T.FLOAT, _RANDOM),
    "random.seed": _fn("random.seed", T.NONE, _RANDOM),
    "random.getrandbits": _fn("random.getrandbits", T.INT, _RANDOM_RAISES),
    "random.randrange": _fn("random.randrange", T.INT, _RANDOM_RAISES),
    "random.randint": _fn("random.randint", T.INT, _RANDOM_RAISES),
    "random.choice": _fn("random.choice", T.ANY, _CHOOSES),
    "random.choices": _fn("random.choices", T.list_of(T.ANY), _SAMPLES),
    "random.sample": _fn("random.sample", T.list_of(T.ANY), _SAMPLES),
    "random.shuffle": _fn("random.shuffle", T.NONE, _SHUFFLES),
    "random.uniform": _fn("random.uniform", T.FLOAT, _RANDOM),
    "random.triangular": _fn("random.triangular", T.FLOAT, _RANDOM),
    "random.gauss": _fn("random.gauss", T.FLOAT, _RANDOM),
    "random.normalvariate": _fn("random.normalvariate", T.FLOAT, _RANDOM),
    "random.lognormvariate": _fn("random.lognormvariate", T.FLOAT, _RANDOM),
    "random.expovariate": _fn(
        "random.expovariate", T.FLOAT, EffectSet.of(Effect.RANDOM, raises=("ZeroDivisionError",))
    ),
    "random.paretovariate": _fn(
        "random.paretovariate", T.FLOAT, EffectSet.of(Effect.RANDOM, raises=("ZeroDivisionError",))
    ),
    "random.weibullvariate": _fn(
        "random.weibullvariate",
        T.FLOAT,
        EffectSet.of(Effect.RANDOM, raises=("ZeroDivisionError",)),
    ),
    "random.gammavariate": _fn("random.gammavariate", T.FLOAT, _RANDOM_RAISES),
    "random.betavariate": _fn("random.betavariate", T.FLOAT, _RANDOM_RAISES),
    # Typed, and left to Python: no native lowering.
    "random.vonmisesvariate": _fn("random.vonmisesvariate", T.FLOAT, _RANDOM),
    "random.binomialvariate": _fn("random.binomialvariate", T.INT, _RANDOM_RAISES),
    "random.randbytes": _fn("random.randbytes", T.BYTES, _RANDOM),
    "random.getstate": _fn("random.getstate", T.ANY, _RANDOM),
    "random.setstate": _fn("random.setstate", T.NONE, _RANDOM_RAISES),
    "heapq.heappush": _fn("heapq.heappush", T.NONE, _HEAP_WRITE),
    "heapq.heappop": _fn("heapq.heappop", T.ANY, _HEAP_WRITE),
    "heapq.heapify": _fn("heapq.heapify", T.NONE, _HEAP_WRITE),
    "heapq.heappushpop": _fn("heapq.heappushpop", T.ANY, _HEAP_WRITE),
    "heapq.heapreplace": _fn("heapq.heapreplace", T.ANY, _HEAP_WRITE),
    "heapq.nlargest": _fn("heapq.nlargest", T.list_of(T.ANY), _ALLOC),
    "heapq.nsmallest": _fn("heapq.nsmallest", T.list_of(T.ANY), _ALLOC),
    "bisect.bisect_left": _fn("bisect.bisect_left", T.INT, _SEARCH),
    "bisect.bisect_right": _fn("bisect.bisect_right", T.INT, _SEARCH),
    "bisect.bisect": _fn("bisect.bisect", T.INT, _SEARCH),
    "bisect.insort_left": _fn("bisect.insort_left", T.NONE, _INSORT),
    "bisect.insort_right": _fn("bisect.insort_right", T.NONE, _INSORT),
    "bisect.insort": _fn("bisect.insort", T.NONE, _INSORT),
}

#: The calls that write to the list they are given first.
MUTATES_FIRST = frozenset(
    {
        "random.shuffle",
        "heapq.heappush",
        "heapq.heappop",
        "heapq.heapify",
        "heapq.heappushpop",
        "heapq.heapreplace",
        "bisect.insort_left",
        "bisect.insort_right",
        "bisect.insort",
    }
)

#: Elements native code orders as CPython does: numbers, strings, and tuples
#: of them (what `sort` and `sorted` order natively too).
_ORDERED = frozenset({"int", "float", "bool", "str"})


def _plain(t: T.Type) -> T.Type:
    return T.strip_literal(t)


def _is(t: T.Type, *names: str) -> bool:
    t = _plain(t)
    return isinstance(t, T.Instance) and t.name in names and not t.args


def _integer(argument: _Argument) -> bool:
    return _is(argument.type, "int", "bool")


def _number(argument: _Argument) -> bool:
    return _is(argument.type, "int", "bool", "float")


def _list_element(t: T.Type) -> T.Type | None:
    t = _plain(t)
    if isinstance(t, T.Instance) and t.name == "list" and len(t.args) == 1:
        element = t.args[0]
        if isinstance(element, (T.AnyType, T.UnknownType, T.NeverType, T.TypeVar_)):
            return None
        return element
    return None


def _orderable(t: T.Type) -> bool:
    t = _plain(t)
    if isinstance(t, T.Instance):
        return t.name in _ORDERED and not t.args
    if isinstance(t, T.Tuple_):
        return all(_orderable(item) for item in t.items)
    return False


def _range(argument: _Argument) -> bool:
    return _is(argument.type, "range")


def call(
    qualname: str, args: list[_Argument], keywords: dict[str, _Argument]
) -> tuple[T.Type, EffectSet] | None:
    """What a call gives where native code makes it, or None where it does not."""
    if qualname.startswith("math."):
        return _math(qualname, args, keywords)
    model = MODELS.get(qualname)
    if model is None:
        return None
    effects = model[1]
    module, _, name = qualname.partition(".")
    if module == "random":
        found = _random(name, args, keywords)
    elif module == "heapq":
        found = None if keywords else _heapq(name, args)
    else:
        found = None if keywords else _bisect(name, args)
    return (found, effects) if found is not None else None


def _random(name: str, args: list[_Argument], keywords: dict[str, _Argument]) -> T.Type | None:
    count = len(args)
    if name == "choices":
        if count not in (1, 2) or set(keywords) - {"weights", "cum_weights", "k"}:
            return None
        element = _list_element(args[0].type)
        weights = [args[1]] if count == 2 else []
        weights += [keywords[k] for k in ("weights", "cum_weights") if k in keywords]
        if element is None or len(weights) > 1 or ("k" in keywords and not _integer(keywords["k"])):
            return None
        if weights and _list_element(weights[0].type) not in (T.INT, T.FLOAT):
            return None
        return T.list_of(element)
    if keywords:
        return None
    if name == "random" and count == 0:
        return T.FLOAT
    if name == "seed" and (count == 0 or (count == 1 and _integer(args[0]))):
        return T.NONE
    if name == "getrandbits" and count == 1 and _integer(args[0]):
        return T.INT
    if name == "randrange" and 1 <= count <= 3 and all(map(_integer, args)):
        return T.INT
    if name == "randint" and count == 2 and all(map(_integer, args)):
        return T.INT
    if name == "choice" and count == 1:
        return _list_element(args[0].type)
    if name == "shuffle" and count == 1 and _list_element(args[0].type) is not None:
        return T.NONE
    if name == "sample" and count == 2 and _integer(args[1]):
        if _range(args[0]):
            return T.list_of(T.INT)
        element = _list_element(args[0].type)
        return T.list_of(element) if element is not None else None
    if name in {"uniform", "gauss", "normalvariate", "lognormvariate"}:
        if count in ({2} if name == "uniform" else {0, 2}) and all(map(_number, args)):
            return T.FLOAT
        return None
    if name == "triangular" and count <= 3 and all(map(_number, args)):
        return T.FLOAT
    if name == "expovariate" and count <= 1 and all(map(_number, args)):
        return T.FLOAT
    if name == "paretovariate" and count == 1 and _number(args[0]):
        return T.FLOAT
    if name in {"weibullvariate", "gammavariate", "betavariate"} and count == 2:
        return T.FLOAT if all(map(_number, args)) else None
    return None


def _heapq(name: str, args: list[_Argument]) -> T.Type | None:
    if name in {"nlargest", "nsmallest"}:
        if len(args) != 2 or not _integer(args[0]):
            return None
        element = _list_element(args[1].type)
        return T.list_of(element) if element is not None and _orderable(element) else None
    if not args:
        return None
    element = _list_element(args[0].type)
    if element is None or not _orderable(element):
        return None
    if name in {"heappop", "heapify"} and len(args) == 1:
        return element if name == "heappop" else T.NONE
    if name in {"heappush", "heappushpop", "heapreplace"} and len(args) == 2:
        if not T.is_assignable(_plain(args[1].type), element):
            return None
        return T.NONE if name == "heappush" else element
    return None


def _bisect(name: str, args: list[_Argument]) -> T.Type | None:
    if not 2 <= len(args) <= 4:
        return None
    element = _list_element(args[0].type)
    if element is None or not _orderable(element) or not all(map(_integer, args[2:])):
        return None
    if not T.is_assignable(_plain(args[1].type), element):
        return None
    return T.NONE if name.startswith("insort") else T.INT


#: `math`'s functions native code has beyond the machine's instructions,
#: and the constants.
MATH_NATIVE = frozenset(
    {
        "gcd", "lcm", "isqrt", "comb", "perm", "factorial", "prod", "fsum", "isclose",
        "hypot", "dist", "copysign", "atan2", "fmod", "degrees", "radians", "log",
        "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "log1p", "expm1", "erf", "erfc", "cbrt", "exp2", "fabs",
    }
)  # fmt: skip

_FLOAT_UNARY = frozenset(
    {
        "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "log1p", "expm1", "erf", "erfc", "cbrt", "exp2", "fabs", "degrees", "radians",
    }
)  # fmt: skip


def _tuple_of_numbers(t: T.Type) -> int | None:
    t = _plain(t)
    if isinstance(t, T.Tuple_) and not t.homogeneous and all(
        _is(item, "int", "bool", "float") for item in t.items
    ):
        return len(t.items)
    return None


def _math(
    qualname: str, args: list[_Argument], keywords: dict[str, _Argument]
) -> tuple[T.Type, EffectSet] | None:
    from . import stdlib  # pylint: disable=import-outside-toplevel

    name = qualname.removeprefix("math.")
    described = stdlib.lookup(qualname)
    if name not in MATH_NATIVE or described is None:
        return None
    effects = described[1]
    count = len(args)
    found: T.Type | None = None
    if name == "isclose":
        if count == 2 and all(map(_number, args)) and not set(keywords) - {"rel_tol", "abs_tol"}:
            if all(map(_number, keywords.values())):
                found = T.BOOL
    elif keywords:
        return None
    elif name in {"gcd", "lcm"} and all(map(_integer, args)):
        found = T.INT
    elif name in {"isqrt", "factorial"} and count == 1 and _integer(args[0]):
        found = T.INT
    elif name == "comb" and count == 2 and all(map(_integer, args)):
        found = T.INT
    elif name == "perm" and count in (1, 2) and all(map(_integer, args)):
        found = T.INT
    elif name in {"prod", "fsum"} and count == 1:
        element = _list_element(args[0].type)
        if element in (T.INT, T.FLOAT):
            found = element if name == "prod" else T.FLOAT
    elif name == "hypot" and count in (2, 3) and all(map(_number, args)):
        found = T.FLOAT
    elif name == "dist" and count == 2:
        sizes = {_tuple_of_numbers(a.type) for a in args}
        if len(sizes) == 1 and sizes & {2, 3}:
            found = T.FLOAT
        elif all(_list_element(a.type) == T.FLOAT for a in args):
            found = T.FLOAT
    elif name in {"copysign", "atan2", "fmod"} and count == 2 and all(map(_number, args)):
        found = T.FLOAT
    elif name == "log" and count == 2 and all(map(_number, args)):
        found = T.FLOAT
    elif name in _FLOAT_UNARY and count == 1 and _number(args[0]):
        found = T.FLOAT
    return (found, effects) if found is not None else None
