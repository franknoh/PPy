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

import string as _string

from . import types as T
from .effects import Effect, EffectSet
from .env import Binding

__all__ = ["MODELS", "MUTATES_FIRST", "STRING_CONSTANTS", "call"]


#: What the checker knows of an argument: its type and its facts.
_Argument = Binding


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

#: `string`'s constants, which native code holds as literals.
STRING_CONSTANTS = {
    f"string.{name}": getattr(_string, name)
    for name in (
        "ascii_letters",
        "ascii_lowercase",
        "ascii_uppercase",
        "digits",
        "hexdigits",
        "octdigits",
        "punctuation",
        "whitespace",
        "printable",
    )
}

MODELS.update(
    {
        f"itertools.{name}": _fn(f"itertools.{name}", T.instance("Iterator", T.ANY), _ALLOC)
        for name in (
            "product",
            "permutations",
            "combinations",
            "combinations_with_replacement",
            "pairwise",
            "chain",
            "islice",
            "accumulate",
            "repeat",
        )
    }
)

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
    qualname: str, args: list[_Argument], keywords: dict[str | None, _Argument]
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
    elif module == "itertools":
        found = _itertools(name, args, keywords)
    elif module == "heapq":
        found = None if keywords else _heapq(name, args)
    else:
        found = None if keywords else _bisect(name, args)
    return (found, effects) if found is not None else None


def _random(
    name: str, args: list[_Argument], keywords: dict[str | None, _Argument]
) -> T.Type | None:
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
        "log1p", "expm1", "erf", "erfc", "cbrt", "exp2", "fabs", "tan",
    }
)  # fmt: skip

_FLOAT_UNARY = frozenset(
    {
        "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "log1p", "expm1", "erf", "erfc", "cbrt", "exp2", "fabs", "degrees", "radians", "tan",
    }
)  # fmt: skip


def _tuple_of_numbers(t: T.Type) -> int | None:
    t = _plain(t)
    if (
        isinstance(t, T.Tuple_)
        and not t.homogeneous
        and all(_is(item, "int", "bool", "float") for item in t.items)
    ):
        return len(t.items)
    return None


def _math(
    qualname: str, args: list[_Argument], keywords: dict[str | None, _Argument]
) -> tuple[T.Type, EffectSet] | None:
    from . import stdlib  # pylint: disable=import-outside-toplevel

    name = qualname.removeprefix("math.")
    described = stdlib.lookup(qualname)
    if name not in MATH_NATIVE or described is None:
        return None
    if name == "isclose":
        tolerances = not set(keywords) - {"rel_tol", "abs_tol"}
        numbers = all(map(_number, [*args, *keywords.values()]))
        found = T.BOOL if len(args) == 2 and tolerances and numbers else None
    elif keywords:
        found = None
    else:
        found = _math_result(name, args)
    return (found, described[1]) if found is not None else None


#: How many integer arguments each integer function takes.
_INTEGER_ARITY = {"isqrt": {1}, "factorial": {1}, "comb": {2}, "perm": {1, 2}}


def _math_result(name: str, args: list[_Argument]) -> T.Type | None:
    count = len(args)
    if name in {"gcd", "lcm"} or name in _INTEGER_ARITY:
        arity = _INTEGER_ARITY.get(name)
        return T.INT if (arity is None or count in arity) and all(map(_integer, args)) else None
    if name in {"prod", "fsum"}:
        element = _list_element(args[0].type) if count == 1 else None
        if element not in (T.INT, T.FLOAT):
            return None
        return element if name == "prod" else T.FLOAT
    if name == "dist":
        sizes = {_tuple_of_numbers(a.type) for a in args}
        points = len(sizes) == 1 and bool(sizes & {2, 3})
        lists = all(_list_element(a.type) == T.FLOAT for a in args)
        return T.FLOAT if count == 2 and (points or lists) else None
    arity = {"hypot": {2, 3}, "copysign": {2}, "atan2": {2}, "fmod": {2}, "log": {2}}.get(
        name, {1} if name in _FLOAT_UNARY else set()
    )
    return T.FLOAT if count in arity and all(map(_number, args)) else None


#: Items `itertools` lays side by side in a tuple natively.
_SCALARS = frozenset({"int", "float", "bool"})


def _walked(argument: _Argument) -> T.Type | None:
    """The element a list or a `range` hands out."""
    if _range(argument):
        return T.INT
    return _list_element(argument.type)


def _constant(argument: _Argument) -> int | None:
    facts = argument.facts
    if facts.has_constant and isinstance(facts.constant, int):
        return int(facts.constant)
    return None


def _itertools(
    name: str, args: list[_Argument], keywords: dict[str | None, _Argument]
) -> T.Type | None:
    """`itertools`' finite iterators over lists and ranges, typed by what they
    hand out; the lowering makes each into a list where it is consumed."""
    if keywords and not (name == "accumulate" and set(keywords) == {"initial"}):
        return None
    if name == "repeat":
        if len(args) == 2 and _integer(args[1]):
            return T.instance("Iterator", _plain(args[0].type))
        return None
    elements = [_walked(a) for a in args[:1]] if args else []
    if not elements or elements[0] is None:
        return None
    element = elements[0]
    scalar = _is(element, *_SCALARS)
    if name == "product" and 1 <= len(args) <= 4:
        items: list[T.Type] = []
        for argument in args:
            item = _walked(argument)
            if item is None or not _is(item, *_SCALARS):
                return None
            items.append(item)
        return T.instance("Iterator", T.Tuple_(tuple(items)))
    if name in {"permutations", "combinations", "combinations_with_replacement"}:
        r = _constant(args[1]) if len(args) == 2 else None
        if r is None or not 0 <= r <= 16 or not scalar:
            return None
        return T.instance("Iterator", T.Tuple_((element,) * r))
    if name == "pairwise" and len(args) == 1 and scalar:
        return T.instance("Iterator", T.Tuple_((element, element)))
    if name == "chain" and 1 <= len(args) <= 4:
        if all(_walked(a) == element for a in args):
            return T.instance("Iterator", element)
        return None
    if name == "islice" and 2 <= len(args) <= 4 and all(map(_integer, args[1:])):
        return T.instance("Iterator", element)
    if name == "accumulate" and len(args) == 1 and element in (T.INT, T.FLOAT):
        initial = keywords.get("initial")
        if initial is None or _plain(initial.type) == element:
            return T.instance("Iterator", element)
    return None
