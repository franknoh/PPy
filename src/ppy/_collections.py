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

import heapq
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any, ClassVar, TypeVar

__all__ = ["Deque", "Heap", "MaxHeap", "Vec"]

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

    def __class_getitem__(cls, element: Any) -> type:
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
