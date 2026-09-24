# Collections

`ppy` has nine collection types that compile to native code: growable
arrays, queues, heaps, a linked list, hash maps and sets, and ordered maps
and sets. You use them like Python containers, with no pointers, and a
function that uses them goes native under `ppy run`, in a standalone binary,
and in emitted C and C++.

```python
from ppy import Deque, HashMap, Heap, LinkedList, TreeSet, Vec


def farthest(nodes: int, edges: Vec[int]) -> int:
    seen = Vec[int](nodes)  # nodes zeros
    queue = Deque[int]()
    queue.push_back(0)
    seen[0] = 1
    best: int = 0
    while queue:
        node: int = queue.pop_front()
        for k in range(2):
            other: int = edges[node * 2 + k]
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

| Type | What it is | Holds |
|---|---|---|
| `Vec[T]` | a growable array | `int` or `float` |
| `Deque[T]` | a double-ended queue | `int` or `float` |
| `Heap[T]`, `MaxHeap[T]` | a priority queue, smallest or largest first | `int` or `float` |
| `LinkedList[T]` | a doubly linked list whose nodes are named by integer ids | `int` or `float` |
| `HashMap[K, V]` | a hash map in insertion order | `int` keys, `int` or `float` values |
| `HashSet[K]` | a hash set in insertion order | `int` |
| `TreeMap[K, V]` | a map in key order | `int` keys, `int` or `float` values |
| `TreeSet[K]` | a set in key order | `int` |

Make one by subscripting the class and calling it: `Vec[int]()`,
`HashMap[int, float]()`. `Vec[int](n)` starts with `n` zeros. A fixed width
such as `ppy.i32` is accepted as an element and held as an `int`.

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

A collection a function makes is a handle into a small C runtime
(`ppy_runtime/collections.py`), and the function frees it before it
returns. The runtime is the same text in each place it runs:

- Under `ppy run`, it is a shared library compiled on first use and cached.
  That needs a C compiler; without one, functions that use collections
  stay in Python.
- In a standalone binary and in emitted C or C++, the functions a program
  calls are carried in as shims.

A check that CPython would raise for is a guard. Under `ppy run`, a failed
guard hands the call back to Python, which runs it again and raises the
same exception. A standalone binary stops with exit status 70.

A function can take a collection as a parameter, as `farthest` above takes
`edges`. Native callers pass the handle, and writes through it land in the
caller's collection. Such a function has no Python boundary: Python code
cannot call it natively, so it runs as Python when called from Python.

## Limitations

- A native function cannot return a collection or store one in another
  collection. Nested collections, such as a `Vec` of `Vec`s, are not
  available.
- Keys are `int`. Elements and values are `int` or `float`.
- A collection cannot be assigned to a second name in native code.
- When a guard falls back under `ppy run`, the collections the native call
  made so far are not freed.
- A standalone binary reports a failed guard with a generic message rather
  than the exception's text.

Examples: [Collections](../howto/47_collections.md).
