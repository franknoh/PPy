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
`HashMap[int, float]()`. Where the target says the type, the subscript may
be left out: `v: Vec[int] = Vec()`. A collection of floats keeps it, as
[Classes](classes.md#generic-classes) explains. `Vec[T](n)` starts with `n` zeros of `T`: `0`,
`0.0`, `False`, a tuple of zeros, a dataclass of zero fields, or a new empty
collection in each slot.

Every type but the maps also starts from an iterable: `Vec[int]([3, 1, 2])`,
`Deque[int](range(5))`, `HashSet[int](other)`. A display, a `range`, or
another collection is what native code fills one from. `Heap[T](items)`
arranges its elements in linear time, as `heapq.heapify` does.

## What a collection holds

An element, or a map's value, is one of:

- a number: `int`, `float`, `bool`, or a fixed width such as `ppy.i32`,
  held as an `int`
- a string: `Vec[str]`, `HashMap[int, str]`
- a tuple of numbers: `Vec[tuple[int, float]]`
- a dataclass whose fields are numbers: `Vec[Point]`
- another collection: `Vec[Vec[int]]`, `HashMap[int, Vec[int]]`
- an instance of an object class: `Vec[Shape]`, which may hold subclasses

A key of a map or a set is an `int`, a `str`, a tuple of `int`, or an
instance of a class: `HashMap[str, int]` for counting words,
`HashMap[tuple[int, int], int]` for a grid, `TreeSet[tuple[int, int]]` for
pairs in order. A string key hashes and orders by its text. See
[Strings](strings.md).

A class is a key where Python could use it as one. A `HashMap` or `HashSet`
takes a class it can hash: one that defines `__hash__` (with `__eq__`, or
alone, which compares by identity), one that defines neither (identity
again), or a `@dataclass(frozen=True)`. `__eq__` without `__hash__`, or a
plain `@dataclass`, is unhashable, and the checker says so (`E1305`). A
`TreeMap` or `TreeSet` takes a class with `__lt__`, or a
`@dataclass(order=True)`.

```python
from ppy import HashMap, Heap


class Cell:
    def __init__(self, row: int, col: int) -> None:
        self.row: int = row
        self.col: int = col

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Cell) and self.row == other.row and self.col == other.col

    def __hash__(self) -> int:
        return self.row * 1000 + self.col


class Visit:
    def __init__(self, cost: int, cell: Cell) -> None:
        self.cost: int = cost
        self.cell: Cell = cell

    def __lt__(self, other: "Visit") -> bool:
        return self.cost < other.cost


def cheapest(size: int) -> int:
    best = HashMap[Cell, int]()
    frontier = Heap[Visit]()
    frontier.push(Visit(0, Cell(0, 0)))
    while frontier:
        visit = frontier.pop()
        if visit.cell in best:
            continue
        best[visit.cell] = visit.cost
        row, col = visit.cell.row, visit.cell.col
        if col + 1 < size:
            frontier.push(Visit(visit.cost + (row * 7 + col) % 10, Cell(row, col + 1)))
        if row + 1 < size:
            frontier.push(Visit(visit.cost + (row + col * 3) % 10, Cell(row + 1, col)))
    return best[Cell(size - 1, size - 1)]
```

Native code calls the class's compiled methods from inside the collection,
so `cheapest` goes native with its heap and its map. A method a subclass
overrides keeps the collection in Python, since the runtime cannot tell
which one an object needs, and so does a method native code cannot compile.
Write `__eq__`'s argument as `other: object`, as Python's typing wants, or
as the class.

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
`@dataclass(order=True)`, and an object by its class's `__lt__`; without
either it has no order, and sorting or heaping it is refused. A collection
has no order either.

A tree's two keys are the same key when neither is less than the other, as
in any ordered map. For numbers, strings, and tuples that is `==`; for a
class it is what its `__lt__` says, so `Task(4, "a")` and `Task(4, "b")` are
one key when `__lt__` compares costs.

### Aliases

A collection element read into a name is the same collection, not a copy,
as it is in Python: after `row = grid[0]`, `row.push(1)` changes
`grid[0]`, and the same holds for an object element. A dataclass element of
numbers is a value, held in the collection's own words. `points[i].x = 3`
and `table[k].y += 0.5` write the field where the element is held. A
function that also keeps a copy of such an element in a name, a loop target,
or a parameter stays in Python, since CPython's copy would be the same object
and see the write.

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

## Walking a collection

A `for` loop walks a collection, and so do these, natively:

| | |
|---|---|
| `for k, v in m.items()`, `m.keys()`, `m.values()` | a map's entries, keys, or values, in its order |
| `enumerate(c)`, `enumerate(c, 1)` | with a count |
| `zip(a, b)` | two or more in step, stopping at the shortest |
| `reversed(c)` | last to first; a heap has no order to reverse |
| `sorted(c)`, `sorted(c, reverse=True)` | a sorted copy, walked |
| `t.between(low, high)` | a tree's keys from `low` up to, and not including, `high` |
| `min(c)`, `max(c)`, `sum(c)` | over a collection or any of the walks above |

`sum` of floats adds as CPython does, with the rounding error carried and
added at the end, so `sum(Vec[float]([1e16, 1.0, -1e16]))` is `1.0`.

A collection a walk reads is held until the walk ends, so the loop body may
rebind the name it came from.

## Methods

### `Vec[T]`

| | |
|---|---|
| `v.push(x)` | add `x` at the end |
| `v.pop()`, `v.pop(i)` | remove and return the last element, or the one at `i` |
| `v.last()` | the last element |
| `v[i]`, `v[i] = x` | read or write index `i` |
| `v[a:b]`, `v[a:b:step]` | a new `Vec`, sliced as a list is |
| `v.insert(i, x)` | put `x` before index `i`, from 0 to `len(v)` |
| `v.extend(items)` | add each of `items` at the end |
| `v.remove(x)`, `v.index(x)`, `v.count(x)` | by equality, as a list does |
| `x in v`, `v == w`, `v + w`, `v.copy()` | a copy shares the collections it holds |
| `v.sort()`, `v.sort(reverse=True)`, `v.sort(key=f)` | in place; the sort is stable |
| `v.reverse()`, `v.clear()`, `len(v)`, `for x in v` | |

`sort`'s `key` is a lambda or a function's name. Native code calls it once
per element, as `list.sort` does, and sorts by what it returns: a number, a
tuple of numbers, or an ordered dataclass.

### `Deque[T]`

| | |
|---|---|
| `d.push_back(x)`, `d.push_front(x)` | add at either end |
| `d.pop_back()`, `d.pop_front()` | remove and return from either end |
| `d.front()`, `d.back()` | the elements at the ends |
| `d[i]`, `d[i] = x` | read or write index `i`, counted from the front |
| `d.extend(items)`, `d.extendleft(items)` | add at either end; `extendleft` reverses them |
| `d.rotate(n)` | move the last `n` elements to the front |
| `d.insert(i, x)`, `d.remove(x)`, `d.index(x)`, `d.count(x)` | as for a `Vec` |
| `x in d`, `d == e`, `d + e`, `d.copy()` | |
| `d.clear()`, `len(d)`, `for x in d` | |

### `Heap[T]` and `MaxHeap[T]`

| | |
|---|---|
| `h.push(x)` | add `x` |
| `h.pop()` | remove and return the smallest (`Heap`) or largest (`MaxHeap`) |
| `h.peek()` | the element `pop` would return |
| `h.pushpop(x)` | push `x`, then pop; `x` itself when it would come out first |
| `h.replace(x)` | pop, then push `x`; the heap must not be empty |
| `h.to_sorted()` | a new `Vec` in the order `pop` would give, the heap left as it was |
| `h.copy()`, `h.clear()`, `len(h)` | |

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
| `l.extend(items)` | add each of `items` at the back |
| `x in l`, `l == m`, `reversed(l)` | |
| `l.clear()`, `len(l)`, `for x in l` | |

Ids are handed out in order from 0, and the id of the most recently removed
node is the next one handed out. `clear` starts the ids from 0 again.

### `HashMap[K, V]` and `HashSet[K]`

| | |
|---|---|
| `m[key]`, `m[key] = value` | read or write; reading a missing key raises `KeyError` |
| `m.get(key, default)` | the value, or `default` |
| `m.setdefault(key, default)` | the value, first set to `default` where there is none |
| `m.pop(key)`, `m.pop(key, default)` | remove `key` and return its value, or `default` |
| `m.keys()`, `m.values()`, `m.items()` | walked in insertion order |
| `c.update(other)` | every entry of `other`, in its order |
| `s.add(key)`, `s.remove(key)`, `s.discard(key)` | the set's methods; `remove` raises `KeyError` for a missing key |
| `a \| b`, `a & b`, `a - b`, `a ^ b` | a new set; also `union`, `intersection`, `difference`, `symmetric_difference` |
| `a.issubset(b)`, `a.issuperset(b)`, `a.isdisjoint(b)` | |
| `key in c`, `c == d`, `c.copy()` | |
| `c.clear()`, `len(c)`, `for key in c` | keys in insertion order |

A set operation keeps the left set's keys in their order and then the right
set's, the order a `dict` built the same way would have.

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
| `t.between(low, high)` | the keys from `low` up to, and not including, `high` |
| `t.pop_min()`, `t.pop_max()` | remove the smallest or largest key; a `TreeMap` returns `(key, value)` |

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
- Equality compares contents, and a collection is never equal to one of
  another kind: a `Vec` is not a `Deque`, as a `list` is not a `deque`.
- A slice may count from the end and is clamped to the length, as a list's
  slice is; only indexing one element refuses a negative index.

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
they held, so a program frees what it made without saying so. A collector
frees the cycles counting cannot, such as a collection that holds itself
through another ([Classes](classes.md#memory) describes it). The tests run
the emitted C under AddressSanitizer with leak detection.

### Guards

A check that CPython would raise for is a guard: an index out of range, a
pop from an empty collection, a missing key, a map changed while it is
walked. Under `ppy run`, a failed guard hands the call back to Python, which
runs it again and raises the same exception. The collections the native call
made are freed the next time native code makes one: no handle outlives the
call that made it, so after a failed call nothing is reachable.

A standalone binary, and emitted C or C++, has no Python to hand the call
to. It prints the line CPython's traceback would end with, with the same
values (`KeyError: (3, -3)`, `IndexError: index 7 is out of range for
length 2`), and exits with status 1, as CPython does for an uncaught
exception. Where only native code can fail, an integer past 64 bits, it
says `OverflowError: the result does not fit in a 64-bit integer`.

### Functions that take or return collections

A native function can take a collection as a parameter, as `farthest` above
takes `adjacent`, and can return one. Native callers pass and receive the
handle, and writes through a parameter land in the caller's collection.

Python calls such a function natively too. Each argument is copied into
native memory, collections inside it included, and a returned collection is
copied out into the reference classes. The copy keeps identity: an object
passed twice is one handle, and a returned argument, or a collection taken
out of one, is the caller's own object. When the function writes through a
parameter, even through an element of it (`for row in rows: row.push(0)`),
the caller's objects are brought up to date after the call. A function that
returns nothing and fills what it was passed crosses the same way.

The copy is one pass over each argument and one over the result, so the
crossing pays off when the function does more than one pass of work over
what it is given. An argument of another type, a plain `list` for a `Vec` say, is
what the checker refuses; one whose element type differs at run time
(`Vec[float]` for `Vec[int]`) runs the Python body.

## Limitations

- A tuple with a string or an object in it is not a key.
- A class's keys natively hash and compare by its own `__hash__` and
  `__eq__`, or by identity: a value class (a class of numbers only, copied
  as a value) has no identity, so it is a key natively only as a
  `@dataclass(frozen=True)` of integers.
- `min` and `max` over objects, and `sort(key=...)` with an object as the
  key, stay in Python.
- A collection holds a user class's instances natively when the class is a
  value class or an object class; see [Classes](classes.md).
- Python calls a function natively when its collections hold numbers,
  tuples of numbers, and collections of those. A dataclass or object
  element, or a `LinkedList` (whose node ids are the history of its
  insertions), keeps the call on the Python body; native callers are not
  affected.
- `index`, `count`, `remove`, `in`, and `==` over floats hand a NaN back to
  Python: CPython counts the same NaN object as equal to itself, and native
  memory has no objects to tell apart.
- `pop_min` and `pop_max` of a `TreeMap` return a tuple natively when its
  keys and values are numbers.

Examples: [Collections](../howto/47_collections.md).
