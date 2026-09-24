"""Collections that compile: `Vec`, `Deque`, `Heap`, and `MaxHeap`.

```python
from ppy import Deque, Heap, Vec

dist = Vec[int](n)          # n zeros; grows with push
dist.push(7)
q = Deque[int]()
q.push_back(0)
start = q.pop_front()
h = Heap[int]()
h.push(5)
smallest = h.pop()
```

Each is a generic class over `int` or `float`. This module is the reference:
it is what the types mean under CPython, and native code implements the same
operations over machine memory, with no Python object per element.

The rules are the same on every path, which is why some differ from a
`list`'s:

* An index runs from `0` to `len - 1`. A negative index is an `IndexError`,
  as is any other index out of range: there is no counting from the end.
* Removing from an empty collection is an `IndexError`.
* Iterating reads the length at every step, as iterating a `list` does, so
  a loop that pushes sees what it pushed.
* A heap has no iteration order to depend on: it is read by `peek` and `pop`
  alone.
"""

from __future__ import annotations

import bisect
import heapq
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any, ClassVar, TypeVar

__all__ = [
    "Deque",
    "HashMap",
    "HashSet",
    "Heap",
    "LinkedList",
    "MaxHeap",
    "TreeMap",
    "TreeSet",
    "Vec",
]

T = TypeVar("T", int, float)

#: One class per collection and element, made on first use: `Vec[float]` is
#: always the same class, and it knows its element is `float`.
_SPECIALIZED: dict[tuple[type, Any], type] = {}


class _Elements:
    """What subscripting a collection class gives: the class for one element type.

    The element decides what a value becomes when it is stored: `Vec[float]`
    holds floats, so `push(3)` stores `3.0` and `Vec[float](2)` starts with
    `0.0`, as native memory of doubles would.
    """

    __slots__ = ()
    _element: ClassVar[Any] = int
    _cast: ClassVar[Callable[[Any], Any]] = int

    def __class_getitem__(cls: type, element: Any) -> type:
        key = (cls, element)
        made = _SPECIALIZED.get(key)
        if made is None:
            base = getattr(element, "__origin__", element)
            if base not in (int, float):
                raise TypeError(f"a {cls.__name__} holds int or float, not {element!r}")
            name = getattr(element, "__name__", repr(element))
            made = type(
                f"{cls.__name__}[{name}]",
                (cls,),
                {"__slots__": (), "_element": element, "_cast": base, "__module__": cls.__module__},
            )
            _SPECIALIZED[key] = made
        return made


def _index(index: int, length: int) -> int:
    if not 0 <= index < length:
        raise IndexError(f"index {index} is out of range for length {length}")
    return index


class Vec(_Elements):
    """A growable array: `push` and `pop` at the end, any index in between."""

    __slots__ = ("_items",)
    _items: list[T]

    def __init__(self, count: int = 0) -> None:
        if count < 0:
            raise ValueError(f"a Vec cannot start with {count} elements")
        self._items = [self._cast(0)] * count

    def push(self, value: T) -> None:
        """Add `value` at the end."""
        self._items.append(self._cast(value))

    def pop(self) -> T:
        """Remove the last element and return it."""
        if not self._items:
            raise IndexError("pop from an empty Vec")
        return self._items.pop()

    def last(self) -> T:
        """The last element, left in place."""
        if not self._items:
            raise IndexError("last of an empty Vec")
        return self._items[-1]

    def clear(self) -> None:
        """Remove every element."""
        self._items.clear()

    def sort(self) -> None:
        """Sort in ascending order."""
        self._items.sort()

    def reverse(self) -> None:
        """Reverse the order in place."""
        self._items.reverse()

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> T:
        return self._items[_index(index, len(self._items))]

    def __setitem__(self, index: int, value: T) -> None:
        self._items[_index(index, len(self._items))] = self._cast(value)

    def __iter__(self) -> Iterator[T]:
        items = self._items
        i = 0
        while i < len(items):
            yield items[i]
            i += 1

    def __repr__(self) -> str:
        return f"Vec({self._items!r})"


class Deque(_Elements):
    """A double-ended queue: push and pop at either end, any index in between."""

    __slots__ = ("_items",)
    _items: deque[T]

    def __init__(self) -> None:
        self._items = deque()

    def push_back(self, value: T) -> None:
        """Add `value` at the back."""
        self._items.append(self._cast(value))

    def push_front(self, value: T) -> None:
        """Add `value` at the front."""
        self._items.appendleft(self._cast(value))

    def pop_back(self) -> T:
        """Remove the back element and return it."""
        if not self._items:
            raise IndexError("pop_back from an empty Deque")
        return self._items.pop()

    def pop_front(self) -> T:
        """Remove the front element and return it."""
        if not self._items:
            raise IndexError("pop_front from an empty Deque")
        return self._items.popleft()

    def front(self) -> T:
        """The front element, left in place."""
        if not self._items:
            raise IndexError("front of an empty Deque")
        return self._items[0]

    def back(self) -> T:
        """The back element, left in place."""
        if not self._items:
            raise IndexError("back of an empty Deque")
        return self._items[-1]

    def clear(self) -> None:
        """Remove every element."""
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> T:
        return self._items[_index(index, len(self._items))]

    def __setitem__(self, index: int, value: T) -> None:
        self._items[_index(index, len(self._items))] = self._cast(value)

    def __iter__(self) -> Iterator[T]:
        items = self._items
        i = 0
        while i < len(items):
            yield items[i]
            i += 1

    def __repr__(self) -> str:
        return f"Deque({list(self._items)!r})"


class Heap(_Elements):
    """A priority queue: `pop` returns the smallest element."""

    __slots__ = ("_items",)
    _items: list[T]

    def __init__(self) -> None:
        self._items = []

    def push(self, value: T) -> None:
        """Add `value`."""
        heapq.heappush(self._items, self._cast(value))

    def pop(self) -> T:
        """Remove the smallest element and return it."""
        if not self._items:
            raise IndexError("pop from an empty Heap")
        return heapq.heappop(self._items)

    def peek(self) -> T:
        """The smallest element, left in place."""
        if not self._items:
            raise IndexError("peek at an empty Heap")
        return self._items[0]

    def clear(self) -> None:
        """Remove every element."""
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Heap(size={len(self._items)})"


class MaxHeap(_Elements):
    """A priority queue: `pop` returns the largest element."""

    __slots__ = ("_items",)
    _items: list[T]

    def __init__(self) -> None:
        self._items = []

    def push(self, value: T) -> None:
        """Add `value`."""
        heapq.heappush(self._items, -self._cast(value))

    def pop(self) -> T:
        """Remove the largest element and return it."""
        if not self._items:
            raise IndexError("pop from an empty MaxHeap")
        return -heapq.heappop(self._items)

    def peek(self) -> T:
        """The largest element, left in place."""
        if not self._items:
            raise IndexError("peek at an empty MaxHeap")
        return -self._items[0]

    def clear(self) -> None:
        """Remove every element."""
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"MaxHeap(size={len(self._items)})"


class _KeysAndValues:
    """What subscripting a map class gives: the class for one key and value type."""

    __slots__ = ()
    _key: ClassVar[Any] = int
    _cast: ClassVar[Callable[[Any], Any]] = int

    def __class_getitem__(cls: type, types: Any) -> type:
        key, value = types if isinstance(types, tuple) else (types, None)
        made = _SPECIALIZED.get((cls, types))
        if made is None:
            if getattr(key, "__origin__", key) is not int:
                raise TypeError(f"a {cls.__name__} has int keys, not {key!r}")
            base = getattr(value, "__origin__", value)
            if value is not None and base not in (int, float):
                raise TypeError(f"a {cls.__name__} holds int or float values, not {value!r}")
            made = type(
                f"{cls.__name__}[{_spelled(types)}]",
                (cls,),
                {"__slots__": (), "_key": key, "_cast": base or int, "__module__": cls.__module__},
            )
            _SPECIALIZED[(cls, types)] = made
        return made


def _spelled(types: Any) -> str:
    parts = types if isinstance(types, tuple) else (types,)
    return ", ".join(getattr(part, "__name__", repr(part)) for part in parts)


def _changed(kind: str) -> RuntimeError:
    return RuntimeError(f"{kind.partition('[')[0]} changed during iteration")


class LinkedList(_Elements):
    """A doubly linked list whose nodes are named by integers, not pointers.

    `push_back` and the other insertions return the new node's id, which
    `next`, `prev`, `value`, `set`, `insert_after`, and `remove` take. Ids
    are handed out in order, and a removed node's id is the next one given
    out again. `-1` stands for no node: the `next` of the last one, the
    `head` of an empty list.
    """

    __slots__ = ("_alive", "_free", "_head", "_next", "_prev", "_size", "_tail", "_values")

    def __init__(self) -> None:
        self._values: list[Any] = []
        self._alive: list[bool] = []
        self._prev: list[int] = []
        self._next: list[int] = []
        self._free: list[int] = []
        self._head = -1
        self._tail = -1
        self._size = 0

    def _reset(self) -> None:
        self._values = []
        self._alive = []
        self._prev = []
        self._next = []
        self._free = []
        self._head = -1
        self._tail = -1
        self._size = 0

    def _node(self, value: T) -> int:
        stored = self._cast(value)
        if self._free:
            node = self._free.pop()
            self._values[node] = stored
            self._alive[node] = True
        else:
            node = len(self._values)
            self._values.append(stored)
            self._alive.append(True)
            self._prev.append(-1)
            self._next.append(-1)
        self._size += 1
        return node

    def _check(self, node: int) -> int:
        if not (0 <= node < len(self._values) and self._alive[node]):
            raise IndexError(f"node {node} is not in the list")
        return node

    def _link(self, node: int, before: int, after: int) -> int:
        self._prev[node] = before
        self._next[node] = after
        if before == -1:
            self._head = node
        else:
            self._next[before] = node
        if after == -1:
            self._tail = node
        else:
            self._prev[after] = node
        return node

    def push_back(self, value: T) -> int:
        """Add `value` at the back; its node id."""
        return self._link(self._node(value), self._tail, -1)

    def push_front(self, value: T) -> int:
        """Add `value` at the front; its node id."""
        return self._link(self._node(value), -1, self._head)

    def insert_after(self, node: int, value: T) -> int:
        """Add `value` after `node`; its node id."""
        self._check(node)
        return self._link(self._node(value), node, self._next[node])

    def insert_before(self, node: int, value: T) -> int:
        """Add `value` before `node`; its node id."""
        self._check(node)
        return self._link(self._node(value), self._prev[node], node)

    def remove(self, node: int) -> T:
        """Take `node` out and return its value; its id is given out next."""
        self._check(node)
        before, after = self._prev[node], self._next[node]
        if before == -1:
            self._head = after
        else:
            self._next[before] = after
        if after == -1:
            self._tail = before
        else:
            self._prev[after] = before
        self._prev[node] = self._next[node] = -1
        self._alive[node] = False
        self._free.append(node)
        self._size -= 1
        return self._values[node]

    def pop_front(self) -> T:
        """Remove the front element and return it."""
        if self._size == 0:
            raise IndexError("pop_front from an empty LinkedList")
        return self.remove(self._head)

    def pop_back(self) -> T:
        """Remove the back element and return it."""
        if self._size == 0:
            raise IndexError("pop_back from an empty LinkedList")
        return self.remove(self._tail)

    def front(self) -> T:
        """The front element, left in place."""
        if self._size == 0:
            raise IndexError("front of an empty LinkedList")
        return self._values[self._head]

    def back(self) -> T:
        """The back element, left in place."""
        if self._size == 0:
            raise IndexError("back of an empty LinkedList")
        return self._values[self._tail]

    def head(self) -> int:
        """The first node's id, or -1."""
        return self._head

    def tail(self) -> int:
        """The last node's id, or -1."""
        return self._tail

    def next(self, node: int) -> int:
        """The id after `node`, or -1."""
        return self._next[self._check(node)]

    def prev(self, node: int) -> int:
        """The id before `node`, or -1."""
        return self._prev[self._check(node)]

    def value(self, node: int) -> T:
        """What `node` holds."""
        return self._values[self._check(node)]

    def set(self, node: int, value: T) -> None:
        """Replace what `node` holds."""
        self._values[self._check(node)] = self._cast(value)

    def clear(self) -> None:
        """Remove every element; ids start from 0 again."""
        self._reset()

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[T]:
        node = self._head
        while node != -1:
            yield self._values[node]
            node = self._next[node]

    def __repr__(self) -> str:
        return f"LinkedList({list(self)!r})"


class HashMap(_KeysAndValues):
    """A hash map from `int` keys, in insertion order, as `dict` keeps it."""

    __slots__ = ("_alive", "_index", "_keys", "_size", "_values", "_version")

    def __init__(self) -> None:
        self._keys: list[int] = []
        self._values: list[Any] = []
        self._alive: list[bool] = []
        self._index: dict[int, int] = {}
        self._size = 0
        self._version = 0

    def _reset(self) -> None:
        self._keys = []
        self._values = []
        self._alive = []
        self._index = {}
        self._size = 0

    def _find(self, key: int) -> int:
        return self._index.get(key, -1)

    def _put(self, key: int, value: Any) -> None:
        entry = self._find(key)
        if entry >= 0:
            self._values[entry] = value
            return
        self._index[key] = len(self._keys)
        self._keys.append(key)
        self._values.append(value)
        self._alive.append(True)
        self._size += 1
        self._version += 1

    def _take(self, key: int) -> Any:
        entry = self._find(key)
        if entry < 0:
            raise KeyError(key)
        del self._index[key]
        self._alive[entry] = False
        self._size -= 1
        self._version += 1
        return self._values[entry]

    def __setitem__(self, key: int, value: Any) -> None:
        self._put(key, self._cast(value))

    def __getitem__(self, key: int) -> Any:
        entry = self._find(key)
        if entry < 0:
            raise KeyError(key)
        return self._values[entry]

    def get(self, key: int, default: Any) -> Any:
        """The value of `key`, or `default` where there is none."""
        entry = self._find(key)
        return self._values[entry] if entry >= 0 else self._cast(default)

    def pop(self, key: int) -> Any:
        """Remove `key` and return its value."""
        return self._take(key)

    def clear(self) -> None:
        """Remove every entry."""
        self._reset()
        self._version += 1

    def __contains__(self, key: int) -> bool:
        return self._find(key) >= 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[int]:
        version = self._version
        entry = 0
        while True:
            if self._version != version:
                raise _changed(type(self).__name__)
            if entry >= len(self._keys):
                return
            if self._alive[entry]:
                yield self._keys[entry]
            entry += 1

    def __repr__(self) -> str:
        return f"HashMap({ {key: self[key] for key in self}!r})"


class HashSet(HashMap):
    """A hash set of `int`, in insertion order."""

    __slots__ = ()

    def add(self, key: int) -> None:
        """Add `key`; nothing changes if it is there."""
        self._put(key, 0)

    def remove(self, key: int) -> None:
        """Remove `key`, which must be there."""
        self._take(key)

    def discard(self, key: int) -> None:
        """Remove `key` if it is there."""
        if self._find(key) >= 0:
            self._take(key)

    def __repr__(self) -> str:
        return f"HashSet({list(self)!r})"


class TreeMap(_KeysAndValues):
    """A map from `int` keys kept in order: the smallest, the largest, and the
    nearest key above or below any value."""

    __slots__ = ("_keys", "_values", "_version")

    def __init__(self) -> None:
        self._keys: list[int] = []
        self._values: list[Any] = []
        self._version = 0

    def _at(self, key: int) -> int:
        position = bisect.bisect_left(self._keys, key)
        found = position < len(self._keys) and self._keys[position] == key
        return position if found else -1

    def _put(self, key: int, value: Any) -> None:
        position = bisect.bisect_left(self._keys, key)
        if position < len(self._keys) and self._keys[position] == key:
            self._values[position] = value
            return
        self._keys.insert(position, key)
        self._values.insert(position, value)
        self._version += 1

    def _take(self, key: int) -> Any:
        position = self._at(key)
        if position < 0:
            raise KeyError(key)
        del self._keys[position]
        self._version += 1
        return self._values.pop(position)

    def __setitem__(self, key: int, value: Any) -> None:
        self._put(key, self._cast(value))

    def __getitem__(self, key: int) -> Any:
        position = self._at(key)
        if position < 0:
            raise KeyError(key)
        return self._values[position]

    def get(self, key: int, default: Any) -> Any:
        """The value of `key`, or `default` where there is none."""
        position = self._at(key)
        return self._values[position] if position >= 0 else self._cast(default)

    def pop(self, key: int) -> Any:
        """Remove `key` and return its value."""
        return self._take(key)

    def clear(self) -> None:
        """Remove every entry."""
        self._keys.clear()
        self._values.clear()
        self._version += 1

    def min(self) -> int:
        """The smallest key."""
        if not self._keys:
            raise IndexError(f"min of an empty {type(self).__name__}")
        return self._keys[0]

    def max(self) -> int:
        """The largest key."""
        if not self._keys:
            raise IndexError(f"max of an empty {type(self).__name__}")
        return self._keys[-1]

    def floor(self, key: int) -> int:
        """The largest key at most `key`."""
        position = bisect.bisect_right(self._keys, key) - 1
        if position < 0:
            raise KeyError(f"no key at most {key}")
        return self._keys[position]

    def ceiling(self, key: int) -> int:
        """The smallest key at least `key`."""
        position = bisect.bisect_left(self._keys, key)
        if position >= len(self._keys):
            raise KeyError(f"no key at least {key}")
        return self._keys[position]

    def lower(self, key: int) -> int:
        """The largest key below `key`."""
        position = bisect.bisect_left(self._keys, key) - 1
        if position < 0:
            raise KeyError(f"no key below {key}")
        return self._keys[position]

    def higher(self, key: int) -> int:
        """The smallest key above `key`."""
        position = bisect.bisect_right(self._keys, key)
        if position >= len(self._keys):
            raise KeyError(f"no key above {key}")
        return self._keys[position]

    def __contains__(self, key: int) -> bool:
        return self._at(key) >= 0

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[int]:
        version = self._version
        position = 0
        while True:
            if self._version != version:
                raise _changed(type(self).__name__)
            if position >= len(self._keys):
                return
            yield self._keys[position]
            position += 1

    def __repr__(self) -> str:
        return f"TreeMap({dict(zip(self._keys, self._values, strict=True))!r})"


class TreeSet(TreeMap):
    """A set of `int` kept in order."""

    __slots__ = ()

    def add(self, key: int) -> None:
        """Add `key`; nothing changes if it is there."""
        self._put(key, 0)

    def remove(self, key: int) -> None:
        """Remove `key`, which must be there."""
        self._take(key)

    def discard(self, key: int) -> None:
        """Remove `key` if it is there."""
        if self._at(key) >= 0:
            self._take(key)

    def __repr__(self) -> str:
        return f"TreeSet({self._keys!r})"
