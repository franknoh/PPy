"""The types of `ppy.Vec`, `ppy.Deque`, `ppy.Heap`, and `ppy.MaxHeap`.

A collection is `T.Instance("ppy.Vec", (element,))`, the element `int` or
`float`. Its methods are typed here, each with the qualified name the
native lowering dispatches on (`ppy.Vec.push`), and the ones that change
the collection are listed so the checker records the write on the
receiver, which is what keeps a function that fills its own collection
eligible for native code.
"""

from __future__ import annotations

from . import types as T

__all__ = [
    "COLLECTIONS",
    "INDEXED",
    "ITERABLE",
    "MUTATORS",
    "instance",
    "is_collection",
    "method",
]

#: Every collection, by the name the checker gives its type.
COLLECTIONS = frozenset({"ppy.Vec", "ppy.Deque", "ppy.Heap", "ppy.MaxHeap"})

#: The ones `v[i]` reads and writes and `for x in v` walks. A heap has no
#: order to walk: it is read by `peek` and `pop` alone.
INDEXED = frozenset({"ppy.Vec", "ppy.Deque"})
ITERABLE = INDEXED

#: What a collection may hold.
ELEMENTS = (T.INT, T.FLOAT)

_HEAPS = ("ppy.Heap", "ppy.MaxHeap")


def instance(name: str, element: T.Type) -> T.Instance:
    return T.Instance(name, (element,), (name, "object"))


def is_collection(t: T.Type) -> bool:
    base = T.strip_literal(t)
    return isinstance(base, T.Instance) and base.name in COLLECTIONS


def _signatures(element: T.Type) -> dict[tuple[str, str], tuple[tuple[T.Param, ...], T.Type]]:
    value = (T.Param("value", element),)
    table: dict[tuple[str, str], tuple[tuple[T.Param, ...], T.Type]] = {
        ("ppy.Vec", "push"): (value, T.NONE),
        ("ppy.Vec", "pop"): ((), element),
        ("ppy.Vec", "last"): ((), element),
        ("ppy.Vec", "clear"): ((), T.NONE),
        ("ppy.Vec", "sort"): ((), T.NONE),
        ("ppy.Vec", "reverse"): ((), T.NONE),
        ("ppy.Deque", "push_back"): (value, T.NONE),
        ("ppy.Deque", "push_front"): (value, T.NONE),
        ("ppy.Deque", "pop_back"): ((), element),
        ("ppy.Deque", "pop_front"): ((), element),
        ("ppy.Deque", "front"): ((), element),
        ("ppy.Deque", "back"): ((), element),
        ("ppy.Deque", "clear"): ((), T.NONE),
    }
    for heap in _HEAPS:
        table[(heap, "push")] = (value, T.NONE)
        table[(heap, "pop")] = ((), element)
        table[(heap, "peek")] = ((), element)
        table[(heap, "clear")] = ((), T.NONE)
    return table


#: The methods that change the collection they are called on.
MUTATORS = frozenset(
    {
        "ppy.Vec.push",
        "ppy.Vec.pop",
        "ppy.Vec.clear",
        "ppy.Vec.sort",
        "ppy.Vec.reverse",
        "ppy.Deque.push_back",
        "ppy.Deque.push_front",
        "ppy.Deque.pop_back",
        "ppy.Deque.pop_front",
        "ppy.Deque.clear",
        *(f"{heap}.{name}" for heap in _HEAPS for name in ("push", "pop", "clear")),
    }
)


def method(base: T.Instance, attr: str) -> T.Callable_ | None:
    """The bound method `attr` of a collection, or None if it has none."""
    element = base.args[0] if base.args else T.UNKNOWN
    found = _signatures(element).get((base.name, attr))
    if found is None:
        return None
    parameters, result = found
    return T.Callable_(parameters, result, f"{base.name}.{attr}")
