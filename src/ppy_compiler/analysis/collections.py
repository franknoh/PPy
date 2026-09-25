"""The types of the `ppy` collections.

A collection is a `T.Instance` named for its class, with its element as the
one argument (`ppy.Vec[int]`), or its key and value as two
(`ppy.HashMap[int, float]`). Elements and values are numbers, tuples of
numbers, dataclasses, and collections; keys are `int` or tuples of them.
Methods are typed here, each with the qualified name the native
lowering dispatches on (`ppy.Vec.push`), and the ones that change the
collection are listed so the checker records the write on the receiver,
which is what keeps a function that fills its own collection eligible for
native code.
"""

from __future__ import annotations

from . import types as T

__all__ = [
    "COLLECTIONS",
    "CONCATENATED",
    "FILLED",
    "INDEXED",
    "ITERABLE",
    "ITERABLE_PARAMETERS",
    "KEYED",
    "KEY_PARAMETER",
    "MAPS",
    "MUTATORS",
    "SETS",
    "SET_OPERATORS",
    "SHORT_NAMES",
    "instance",
    "is_collection",
    "method",
    "sort_key",
    "spelled",
]

#: Every collection, by the name the checker gives its type.
COLLECTIONS = frozenset(
    {
        "ppy.Vec",
        "ppy.Deque",
        "ppy.Heap",
        "ppy.MaxHeap",
        "ppy.LinkedList",
        "ppy.HashMap",
        "ppy.HashSet",
        "ppy.TreeMap",
        "ppy.TreeSet",
    }
)

#: The same names as `from ppy import Vec` spells them.
SHORT_NAMES = tuple(sorted(name.removeprefix("ppy.") for name in COLLECTIONS))

#: `v[i]` with an `int` index from 0.
INDEXED = frozenset({"ppy.Vec", "ppy.Deque"})
#: `m[key]`: a map's value.
MAPS = frozenset({"ppy.HashMap", "ppy.TreeMap"})
#: Collections of `int` keys, which `key in s` asks about.
KEYED = frozenset({"ppy.HashMap", "ppy.HashSet", "ppy.TreeMap", "ppy.TreeSet"})
#: What `for x in c` walks: elements in order, or keys. A heap has no order
#: to walk: it is read by `peek` and `pop` alone.
ITERABLE = frozenset({"ppy.Vec", "ppy.Deque", "ppy.LinkedList"}) | KEYED

#: How many type arguments each takes.
ARITY = {name: 2 if name in MAPS else 1 for name in COLLECTIONS}

#: What an element or a value may be, and what a key may be.
ELEMENTS = (T.INT, T.FLOAT)
KEYS = (T.INT,)

_HEAPS = ("ppy.Heap", "ppy.MaxHeap")
_TREES = ("ppy.TreeMap", "ppy.TreeSet")


def instance(name: str, *arguments: T.Type) -> T.Instance:
    return T.Instance(name, tuple(arguments), (name, "object"))


def is_collection(t: T.Type) -> bool:
    base = T.strip_literal(t)
    return isinstance(base, T.Instance) and base.name in COLLECTIONS


def key_of(base: T.Instance) -> T.Type:
    """A keyed collection's key type."""
    return base.args[0] if base.args else T.INT


def element_of(base: T.Instance) -> T.Type:
    """What iterating gives: the element, or for a keyed collection the key."""
    if base.name in KEYED:
        return key_of(base)
    return base.args[0] if base.args else T.UNKNOWN


def spelled(t: T.Type) -> str:
    """A collection type written out, the way the native lowering writes it, so a
    parameter and an argument can be matched by their spelling."""
    base = T.strip_literal(t)
    if isinstance(base, T.Tuple_):
        return f"tuple[{', '.join(spelled(item) for item in base.items)}]"
    if isinstance(base, T.Instance) and base.name in COLLECTIONS:
        return f"{base.name}[{', '.join(spelled(argument) for argument in base.args)}]"
    return str(base)


def value_of(base: T.Instance) -> T.Type:
    """What indexing gives: the element, or a map's value."""
    if base.name in MAPS:
        return base.args[1] if len(base.args) == 2 else T.UNKNOWN
    return base.args[0] if base.args else T.UNKNOWN


#: A class's methods: each one's parameters and result.
Signatures = dict[str, tuple[tuple[T.Param, ...], T.Type]]


#: The parameters that take any iterable of the element (`extend`, a
#: constructor): the checker holds what they iterate to the element itself.
ITERABLE_PARAMETERS = frozenset({"values"})

#: `sort`'s `key`: a function of one element, typed from the collection.
KEY_PARAMETER = "key"


def sort_key(element: T.Type) -> T.Callable_:
    """What `sort(key=...)` takes: a function of one element."""
    return T.Callable_((T.Param("item", element),), T.ANY)


def _sequence_extras(base: T.Instance, element: T.Type) -> Signatures:
    """What a `Vec` and a `Deque` share beyond pushing and popping."""
    value: tuple[T.Param, ...] = (T.Param("value", element),)
    iterable: tuple[T.Param, ...] = (T.Param("values", T.ANY),)
    extras: Signatures = {}
    extras["extend"] = (iterable, T.NONE)
    extras["insert"] = ((T.Param("index", T.INT), *value), T.NONE)
    extras["remove"] = (value, T.NONE)
    extras["index"] = (value, T.INT)
    extras["count"] = (value, T.INT)
    extras["copy"] = ((), base)
    return extras


def _signatures(base: T.Instance) -> Signatures:
    name = base.name
    element = value_of(base)
    value = (T.Param("value", element),)
    key_type = key_of(base) if name in KEYED else T.INT
    key = (T.Param("key", key_type),)
    node = (T.Param("node", T.INT),)
    nothing: tuple[T.Param, ...] = ()
    if name == "ppy.Vec":
        found: Signatures = {
            "push": (value, T.NONE),
            "pop": ((T.Param("index", T.INT, has_default=True),), element),
            "last": (nothing, element),
            "clear": (nothing, T.NONE),
            "sort": (
                (
                    T.Param(KEY_PARAMETER, sort_key(element), True, "keyword_only"),
                    T.Param("reverse", T.BOOL, True, "keyword_only"),
                ),
                T.NONE,
            ),
            "reverse": (nothing, T.NONE),
        }
        found.update(_sequence_extras(base, element))
        return found
    if name == "ppy.Deque":
        found: Signatures = {
            "push_back": (value, T.NONE),
            "push_front": (value, T.NONE),
            "pop_back": (nothing, element),
            "pop_front": (nothing, element),
            "front": (nothing, element),
            "back": (nothing, element),
            "clear": (nothing, T.NONE),
            "extendleft": ((T.Param("values", T.ANY),), T.NONE),
            "rotate": ((T.Param("steps", T.INT, has_default=True),), T.NONE),
        }
        found.update(_sequence_extras(base, element))
        return found
    if name in _HEAPS:
        found: Signatures = {
            "push": (value, T.NONE),
            "pop": (nothing, element),
            "peek": (nothing, element),
            "clear": (nothing, T.NONE),
            "pushpop": (value, element),
            "replace": (value, element),
            "to_sorted": (nothing, instance("ppy.Vec", element)),
            "copy": (nothing, base),
        }
        return found
    if name == "ppy.LinkedList":
        found: Signatures = {
            "push_back": (value, T.INT),
            "push_front": (value, T.INT),
            "insert_after": ((*node, *value), T.INT),
            "insert_before": ((*node, *value), T.INT),
            "remove": (node, element),
            "pop_front": (nothing, element),
            "pop_back": (nothing, element),
            "front": (nothing, element),
            "back": (nothing, element),
            "head": (nothing, T.INT),
            "tail": (nothing, T.INT),
            "next": (node, T.INT),
            "prev": (node, T.INT),
            "value": (node, element),
            "set": ((*node, *value), T.NONE),
            "clear": (nothing, T.NONE),
            "extend": ((T.Param("values", T.ANY),), T.NONE),
        }
        return found
    other = (T.Param("other", base),)
    table: Signatures = {
        "clear": (nothing, T.NONE),
        "update": (other, T.NONE),
        "copy": (nothing, base),
    }
    if name in MAPS:
        table["get"] = ((*key, T.Param("default", element)), element)
        table["pop"] = ((*key, T.Param("default", element, has_default=True)), element)
        table["setdefault"] = ((*key, T.Param("default", element)), element)
        table["keys"] = (nothing, T.instance("Iterator", key_type))
        table["values"] = (nothing, T.instance("Iterator", element))
        table["items"] = (nothing, T.instance("Iterator", T.Tuple_((key_type, element))))
    else:
        table["add"] = (key, T.NONE)
        table["remove"] = (key, T.NONE)
        table["discard"] = (key, T.NONE)
        for combined in ("union", "intersection", "difference", "symmetric_difference"):
            table[combined] = (other, base)
        for relation in ("issubset", "issuperset", "isdisjoint"):
            table[relation] = (other, T.BOOL)
    if name in _TREES:
        for bound in ("floor", "ceiling", "lower", "higher"):
            table[bound] = (key, key_type)
        table["min"] = (nothing, key_type)
        table["max"] = (nothing, key_type)
        table["between"] = (
            (T.Param("low", key_type), T.Param("high", key_type)),
            T.instance("Iterator", key_type),
        )
        end = key_type if name == "ppy.TreeSet" else T.Tuple_((key_type, element))
        table["pop_min"] = (nothing, end)
        table["pop_max"] = (nothing, end)
    return table


#: `a | b`, `a & b`, `a - b`, `a ^ b` of two sets of one type, by operator name.
SET_OPERATORS = {
    "BitOr": "union",
    "BitAnd": "intersection",
    "Sub": "difference",
    "BitXor": "symmetric_difference",
}

#: The sets, which those operators combine.
SETS = frozenset({"ppy.HashSet", "ppy.TreeSet"})

#: The collections `+` joins: two of one type make a third.
CONCATENATED = frozenset({"ppy.Vec", "ppy.Deque"})

#: The collections a constructor fills from an iterable.
FILLED = frozenset(
    {
        "ppy.Vec",
        "ppy.Deque",
        "ppy.Heap",
        "ppy.MaxHeap",
        "ppy.LinkedList",
        "ppy.HashSet",
        "ppy.TreeSet",
    }
)


#: The methods that change the collection they are called on.
_CHANGING = frozenset(
    {
        "push",
        "pop",
        "clear",
        "sort",
        "reverse",
        "push_back",
        "push_front",
        "pop_back",
        "pop_front",
        "insert_after",
        "insert_before",
        "remove",
        "set",
        "add",
        "discard",
        "extend",
        "extendleft",
        "insert",
        "rotate",
        "pushpop",
        "replace",
        "setdefault",
        "update",
        "pop_min",
        "pop_max",
    }
)
MUTATORS = frozenset(f"{name}.{attr}" for name in COLLECTIONS for attr in _CHANGING)


def method(base: T.Instance, attr: str) -> T.Callable_ | None:
    """The bound method `attr` of a collection, or None if it has none."""
    found = _signatures(base).get(attr)
    if found is None:
        return None
    parameters, result = found
    return T.Callable_(parameters, result, f"{base.name}.{attr}")
