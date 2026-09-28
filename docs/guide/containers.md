# Lists, dicts, and sets

Python's own `list`, `dict`, and `set` lower to native code. A function that
builds and uses them goes native under `ppy run`, in a standalone binary, and
in emitted C and C++, with no rewrite to the [`ppy` collections](collections.md)
and the same answers as CPython.

```python
def busiest(lines: list[str]) -> str:
    hits: dict[str, int] = {}
    for line in lines:
        path = line.split()[1]
        hits[path] = hits.get(path, 0) + 1
    best, most = "", 0
    for path, count in hits.items():
        if count > most:
            best, most = path, count
    return best
```

Natively a list is a sequence of the collections runtime, a dict a hash map
kept in insertion order, and a set a hash set. They are the same memory as a
`ppy.Vec`, `ppy.HashMap`, and `ppy.HashSet`: counted references, cycles
collected, freed when the last reference goes. The `ppy` types stay the
explicit way to say which structure you mean; a plain container gets the one
Python's container behaves like.

## What lowers

A container's element, or a dict's value, is a number, a string, a tuple of
numbers, a value class or an object class, or another container:
`list[list[int]]`, `dict[str, list[int]]`, `list[tuple[int, float]]`. A key
of a dict or a set is an `int`, a `str`, a tuple of `int`, or an instance of
a class Python can hash (see [Collections](collections.md#what-a-collection-holds)).

| | |
|---|---|
| displays | `[a, b]`, `{k: v}`, `{a, b}`, and `[]`, `{}`, `set()` given a type by the name they go to |
| repetition | `[0] * n`, `[[0] * n for _ in range(m)]` |
| comprehensions | `[e for x in xs if c]`, `{k: v for ...}`, `{e for ...}`, nested, with the comprehension's names its own |
| constructors | `list(xs)`, `set(xs)`, `dict(d)` |
| a list | `xs[i]` and `xs[i] = v` from either end, slices with steps, `append`, `extend`, `insert`, `pop()`, `pop(i)`, `remove`, `index`, `count`, `sort()` with `key=` and `reverse=`, `reverse`, `clear`, `copy`, `del xs[i]`, `+` |
| a dict | `d[k]`, `d[k] = v`, `del d[k]`, `get(k, default)`, `setdefault`, `pop(k)`, `pop(k, default)`, `keys()`, `values()`, `items()`, `update`, `copy`, `clear` |
| a set | `add`, `remove`, `discard`, `update`, `copy`, `clear`, `\|`, `&`, `-`, `^` and their methods, `issubset`, `issuperset`, `isdisjoint` |
| any of them | `len`, `in`, `==`, `if xs:`, `for` loops, `sorted`, `min`, `max`, `sum`, `any`, `all`, `enumerate`, `zip`, `reversed` |

A value is evaluated before the slot it goes into is made, as Python
evaluates it, so `xs.append(len(xs))` appends the length before the append.

Where a check fails, the call hands back to Python under `ppy run`, which
raises what CPython raises. A standalone binary prints CPython's last line
(`IndexError: list index out of range`, `KeyError: 3`,
`ValueError: list.remove(x): x not in list`, `IndexError: pop from empty
list`) and exits with status 1.

## Aliases

Two names for one list are one list natively too: after `b = a`,
`b.append(1)` changes `a`, and `row = grid[0]` is the row inside `grid`.
`[[0] * n] * m` repeats one row `m` times, as in Python. A write through an
element lands in what the root name holds, so the compiler knows whether a
function writes only what it made or what it was passed.

## The order of a set

CPython walks a set in the order its hash table keeps, which depends on the
values and, for strings, on the process. Native memory keeps a different
order. So a `for` loop over a set, `list(s)`, a comprehension over one, and
printing one stay in Python. What does not depend on the order is native:
`len`, `in`, the set algebra, `sorted(s)`, `min`, `max`, `sum`, `any`, `all`,
and building one set from another.

A dict walks in insertion order, in Python and natively.

## Between functions

A native function takes and returns containers by handle when native code
calls it. A list of numbers that a function only reads is lent to it as a
buffer, which native code and Python both pass without copying the
elements; a list it writes, and every other container, goes by handle.

When Python calls such a function, each container argument is copied into
native memory (strings as their UTF-8 bytes), the result is copied out as a
new `list`, `dict`, or `set`, and a container the function wrote through is
copied back into the caller's object, which stays the same object. An
argument whose contents do not match the declared type runs the Python
body instead.

```python
def grow(xs: list[int], n: int) -> None:
    for i in range(n):
        xs.append(i * i)
```

`grow(xs, 3)` from Python runs natively and leaves `xs` holding the three
squares after what it held.

## Printing

In a standalone build, `print(xs)`, `print(d)`, and an f-string field
holding a list or a dict write what `repr` writes: numbers, strings with
their quotes, tuples, nested containers, and dataclasses shown as
`Point(x=1, y=2.0)`. A set is not printed natively, for the reason above.

## Limitations

- A tuple element holding a string or an object (`list[tuple[str, int]]`)
  keeps the function in Python.
- A dict keyed by `float` or `bool` stays in Python: equal floats are not
  always equal words (`0.0` and `-0.0`, NaN), and a bool key is the int it
  equals.
- `d.get(k)` with no default, which may give `None`, stays in Python; with a
  default it is native.
- `sort(key=...)` natively takes a key giving numbers or tuples of them.
- A standalone binary's `KeyError` for a string key says `KeyError` without
  the key.
- A container holding objects or dataclasses keeps a Python caller on the
  Python body; native callers are not affected.

Examples: [Collections](../howto/47_collections.md), [Strings](../howto/48_strings.md).
