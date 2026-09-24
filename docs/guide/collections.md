# Collections

`ppy` has nine collection types that compile to native code: growable
arrays, queues, heaps, a linked list, hash maps and sets, and ordered maps
and sets. You use them like Python containers, with no pointers. They hold
numbers, tuples, dataclasses, or other collections, and a function that uses
them goes native under `ppy run`, in a standalone binary, and in emitted C
and C++.

```python
from ppy import Deque, Vec


def farthest(adjacent: Vec[Vec[int]]) -> int:
    seen = Vec[int](len(adjacent))  # len(adjacent) zeros
    queue = Deque[int]()
    queue.push_back(0)
    seen[0] = 1
    best: int = 0
    while queue:
        node: int = queue.pop_front()
        for other in adjacent[node]:
            if seen[other] == 0:
                seen[other] = seen[node] + 1
                best = max(best, seen[other])
                queue.push_back(other)
    return best - 1
```

Under CPython each type is a Python class in `ppy._collections`, and that
class is the reference: native code gives the same answers. The
[Collections example](../howto/47_collections.md) runs five problems with
them and times each path.

## The types

| Type | What it is |
|---|---|
| `Vec[T]` | a growable array |
| `Deque[T]` | a double-ended queue |
| `Heap[T]`, `MaxHeap[T]` | a priority queue, smallest or largest first |
| `LinkedList[T]` | a doubly linked list whose nodes are named by integer ids |
| `HashMap[K, V]` | a hash map in insertion order |
| `HashSet[K]` | a hash set in insertion order |
| `TreeMap[K, V]` | a map in key order |
| `TreeSet[K]` | a set in key order |

Make one by subscripting the class and calling it: `Vec[int]()`,
`HashMap[int, float]()`. `Vec[T](n)` starts with `n` zeros of `T`: `0`,
`0.0`, `False`, a tuple of zeros, a dataclass of zero fields, or a new empty
collection in each slot.

## What a collection holds

An element, or a map's value, is one of:

- a number: `int`, `float`, `bool`, or a fixed width such as `ppy.i32`,
  held as an `int`
- a tuple of numbers: `Vec[tuple[int, float]]`
- a dataclass whose fields are numbers: `Vec[Point]`
- another collection: `Vec[Vec[int]]`, `HashMap[int, Vec[int]]`

A key of a map or a set is an `int` or a tuple of them:
`HashMap[tuple[int, int], int]` for a grid, `TreeSet[tuple[int, int]]` for
pairs in order.

```python
from dataclasses import dataclass

from ppy import HashMap, Heap, Vec


@dataclass(order=True)
class Edge:
    cost: int
    to: int


def cheapest(edges: Vec[Vec[Edge]]) -> int:
    best = HashMap[int, int]()
    frontier = Heap[tuple[int, int]]()
    frontier.push((0, 0))
    while frontier:
        cost, node = frontier.pop()
        if node in best:
            continue
        best[node] = cost
        for edge in edges[node]:
            if edge.to not in best:
                frontier.push((cost + edge.cost, edge.to))
    return len(best)
```

A value is converted to the element type when it is stored, so
`Vec[float]` holds `3.0` after `push(3)`, as native memory of doubles would,
and a dataclass with a `float` field holds a float there.

### Order

`sort`, a heap, and a tree compare elements or keys with `<`, as Python
does. Numbers compare as numbers and tuples compare item by item. A
dataclass compares field by field when it is declared
`@dataclass(order=True)`; without that it has no order, and sorting or heaping
it is refused. A collection has no order either.

### Aliases

A collection element read into a name is the same collection, not a copy,
as it is in Python: after `row = grid[0]`, `row.push(1)` changes
`grid[0]`. A dataclass element is a value: it is read and written whole,
and setting a field of one in place keeps the function in Python.

## Generic functions

A generic function can take, make, and return collections of its type
parameter. Native code makes one instance per type it is called with, as a
C++ template is instantiated:

```python
from ppy import Heap, Vec


def smallest[T: int | float](values: Vec[T], count: int) -> Vec[T]:
    heap = Heap[T]()
    for value in values:
        heap.push(value)
    taken = Vec[T]()
    while heap and len(taken) < count:
        taken.push(heap.pop())
    return taken
```

`smallest(Vec[int]...)` and `smallest(Vec[float]...)` are two native
functions. Under CPython the type parameter is not known, so `Vec[T]`
stores values as they are given, and `Vec[T](n)` has no zero to start with.

A value is converted to the element type when it is stored, so
`Vec[float]` holds `3.0` after `push(3)`, as native memory of doubles would.

## Methods

### `Vec[T]`

| | |
|---|---|
| `v.push(x)` | add `x` at the end |
| `v.pop()` | remove and return the last element |
| `v.last()` | the last element |
| `v[i]`, `v[i] = x` | read or write index `i` |
| `v.sort()`, `v.reverse()` | in place; the sort is stable |
| `v.clear()`, `len(v)`, `for x in v` | |

### `Deque[T]`

| | |
|---|---|
| `d.push_back(x)`, `d.push_front(x)` | add at either end |
| `d.pop_back()`, `d.pop_front()` | remove and return from either end |
| `d.front()`, `d.back()` | the elements at the ends |
| `d[i]`, `d[i] = x` | read or write index `i`, counted from the front |
| `d.clear()`, `len(d)`, `for x in d` | |

### `Heap[T]` and `MaxHeap[T]`

| | |
|---|---|
| `h.push(x)` | add `x` |
| `h.pop()` | remove and return the smallest (`Heap`) or largest (`MaxHeap`) |
| `h.peek()` | the element `pop` would return |
| `h.clear()`, `len(h)` | |

A heap has no iteration order and no index. To keep a value with its
priority, pack both into one integer, such as `priority * N + node`.

### `LinkedList[T]`

Every insertion returns the new node's id, an `int`. The methods that take a
node use that id. `-1` means no node.

| | |
|---|---|
| `l.push_back(x)`, `l.push_front(x)` | add at either end; returns the id |
| `l.insert_after(node, x)`, `l.insert_before(node, x)` | add next to `node`; returns the id |
| `l.remove(node)` | take `node` out and return its value |
| `l.pop_front()`, `l.pop_back()` | remove and return from either end |
| `l.front()`, `l.back()` | the values at the ends |
| `l.head()`, `l.tail()` | the ids at the ends, or `-1` |
| `l.next(node)`, `l.prev(node)` | the neighboring ids, or `-1` |
| `l.value(node)`, `l.set(node, x)` | read or replace what `node` holds |
| `l.clear()`, `len(l)`, `for x in l` | |

Ids are handed out in order from 0, and the id of the most recently removed
node is the next one handed out. `clear` starts the ids from 0 again.

### `HashMap[K, V]` and `HashSet[K]`

| | |
|---|---|
| `m[key]`, `m[key] = value` | read or write; reading a missing key raises `KeyError` |
| `m.get(key, default)` | the value, or `default` |
| `m.pop(key)` | remove `key` and return its value |
| `s.add(key)`, `s.remove(key)`, `s.discard(key)` | the set's methods; `remove` raises `KeyError` for a missing key |
| `key in c`, `key not in c` | |
| `c.clear()`, `len(c)`, `for key in c` | keys in insertion order |

### `TreeMap[K, V]` and `TreeSet[K]`

These have the same methods as the hash types, iterate in ascending key
order, and add:

| | |
|---|---|
| `t.min()`, `t.max()` | the smallest and largest key |
| `t.floor(key)` | the largest key at most `key` |
| `t.ceiling(key)` | the smallest key at least `key` |
| `t.lower(key)` | the largest key below `key` |
| `t.higher(key)` | the smallest key above `key` |

`floor` and the others raise `KeyError` when there is no such key, and `min`
and `max` raise `IndexError` on an empty tree.

## Rules

These hold on every path, which is why a few differ from Python's own
containers:

- An index runs from `0` to `len - 1`. A negative index raises `IndexError`
  like any other index out of range; there is no counting from the end.
- Removing from an empty collection raises `IndexError`.
- A collection is true when it holds anything, so `while queue:` works.
- Iterating a `Vec` or `Deque` reads the length at every step, as iterating
  a `list` does, so a loop that pushes sees what it pushed.
- Iterating a linked list follows `next` from each node after the loop body
  ran. If the body removed that node, the walk ends there.
- Adding or removing a key while iterating a map or a set raises
  `RuntimeError`, as `dict` does. Replacing a value does not.

## In native code

A collection is a handle into a small C runtime
(`ppy_runtime/collections.c`) that works on eight-byte words and knows
nothing of types. The compiler writes the typed reads and writes for each
element type, which is where a `Vec[tuple[int, float]]` and a `Vec[Edge]`
differ.

The runtime is the same text in each place it runs:

- Under `ppy run`, it is a shared library compiled on first use and cached.
  That needs a C compiler; without one, functions that use collections
  stay in Python.
- In a standalone binary and in emitted C or C++, the functions a program
  calls are carried in as shims.

### Memory

References are counted. A name holding a collection keeps one reference,
and so does a collection holding another; a collection is freed when the
last reference goes, together with the collections it holds. Rebinding a
name, returning from a function, and removing an element all let go of what
they held, so a program frees what it made without saying so. The tests run
the emitted C under AddressSanitizer with leak detection.

### Guards

A check that CPython would raise for is a guard: an index out of range, a
pop from an empty collection, a missing key, a map changed while it is
walked. Under `ppy run`, a failed guard hands the call back to Python, which
runs it again and raises the same exception. A standalone binary stops with
exit status 70.

### Functions that take or return collections

A native function can take a collection as a parameter, as `farthest` above
takes `adjacent`, and can return one. Native callers pass and receive the
handle, and writes through a parameter land in the caller's collection.
Such a function has no Python boundary: Python code cannot call it
natively, so it runs as Python when called from Python.

## Limitations

- Keys are `int` or tuples of `int`.
- `get(key, default)` natively takes a map whose values are numbers.
- A dataclass element is read and written whole; setting a field of one in
  place keeps the function in Python.
- When a guard falls back under `ppy run`, the collections the native call
  made so far are not freed.
- A standalone binary reports a failed guard with a generic message rather
  than the exception's text.
- A collection holds a user class's instances natively when the class is a
  value class or an object class; see [Classes](classes.md).

Examples: [Collections](../howto/47_collections.md).
