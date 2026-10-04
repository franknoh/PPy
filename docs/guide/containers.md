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
| comprehensions | `[e for x in xs if c]`, `{k: v for ...}`, `{e for ...}`, nested, with the comprehension's names its own, and a generator given to `sum`, `min`, or `max` |
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

CPython walks a set in the order its hash table keeps. For a set of ints or
of tuples of ints that order follows from the values and from what was done
to the set, and native code keeps a copy of CPython's table beside its own:
the same slots, the same probing, the same resizes, and the same algorithms
for `|`, `&`, `-`, `^`, their in-place forms, `update`, `copy`, and `pop`.
So a `for` loop over such a set, `list(s)`, a comprehension over it, `s.pop()`,
and printing it are native and give what CPython gives:

```python
def pairs(n: int) -> set[tuple[int, int]]:
    return {(i % 7, i * 31 % 11) for i in range(n)}


def shown(n: int) -> str:
    s = pairs(n)
    s -= {(0, 0)}
    return f"{s} {s.pop()}"
```

A set made in Python and passed in has no such copy: its order is not known
natively, so a walk that shows it falls back to Python. A string's hash
changes from one process to the next unless `PYTHONHASHSEED` is fixed, so a
set of strings walked where its order shows stays in Python. What does not
depend on the order is native for every set: `len`, `in`, the set algebra,
`sorted(s)`, `min`, `max`, `sum`, `any`, `all`.

A dict walks in insertion order, in Python and natively.

## Between functions

A native function takes and returns containers by handle when native code
calls it. A list of numbers that a function only reads is lent to it as a
buffer, which native code and Python both pass without copying the
elements; a list it writes, and every other container, goes by handle.

When Python calls such a function, the generated wrapper copies each
container argument into native memory (strings as their UTF-8 bytes) and
copies the result out as a new `list`, `dict`, or `set`. After a call that
writes through a parameter, every container that came in is copied back
into the caller's object, which stays the same object. That includes one
the call took out of its parent: after `row = g[0]; row.append(1); g.pop(0)`
the caller's row has the 1, as in CPython. An element the call left as it
was keeps its identity. An argument whose contents do not match the
declared type runs the Python body instead.

Each object is copied once however often it is reached, and comes back as
one object: `f(xs, xs)` writes one list, the rows of `[[0] * n] * m` stay
one row, and a list in two dict values stays one list.

The copy costs time in proportion to what crosses. Measured on the
generated wrapper:

| element | copied in | and back, after a write |
|---|---|---|
| a number in a list | about 1 ns | about 2 ns, more where it changed |
| a dict's or a set's entry | about 20 ns | about the same again |
| a list inside a list | about 80 ns, plus its elements | its elements |
| a string | a native string, about 30 ns | a Python string |

A Python loop's pass costs 10 to 30 ns. So Python calls a function natively
only where the body does that much with each element: a loop over a list
of numbers; three or more operations per entry of a dict (`s += k * v + v`,
not `s += d[k]`); two or more per element of a list of lists; and, for a
container of strings, more than one pass, a loop inside its loop, as
`for w in words: for ch in w:` does. Otherwise Python calls its Python body,
and native callers still call it natively. `ppy explain` gives the reason.

```python
def grow(xs: list[int], n: int) -> None:
    for i in range(n):
        xs.append(i * i)
```

`grow(xs, 3)` from Python runs natively and leaves `xs` holding the three
squares after what it held.

## Printing

Natively, under `ppy run` and in a standalone build, `print(xs)`,
`print(d)`, and an f-string field holding a list or a dict write what
`repr` writes: numbers, strings with their quotes, tuples, nested
containers, and dataclasses shown as `Point(x=1, y=2.0)`. A set of ints or
of tuples of ints prints in CPython's order, and an empty one as `set()`.
A list parameter lent as a buffer (a list of numbers the function only
reads) has no `repr` natively, so printing one keeps the function in
Python.

## Limitations

- A tuple element holding a string or an object (`list[tuple[str, int]]`)
  keeps the function in Python.
- A dict or set keyed by `float` or `bool` is native, with `0.0` and `-0.0`
  one key. A NaN key, which only its own object finds, falls back. An `int`
  key given to a dict of floats (`d[1]` where `d: dict[float, int]`) stays in
  Python, since CPython keeps and prints the key as it came.
- `v = d.get(k)` with no default, of a dict of numbers, binds `v` as a
  number or `None`: native code keeps a flag beside the number. `v is None`,
  `v is not None`, `if v:`, `print(v)`, and `v` as a number once a test has
  shown it holds one are native; `v += x` raises CPython's `TypeError` where
  `v` is `None`. A `d.get(k)` used any other way (passed on, returned, a
  dict of strings) stays in Python. With a default, `get` is native.
- `sort(key=...)`, `sorted(key=...)`, `min(key=...)`, and `max(key=...)`
  natively take a key giving numbers or tuples of them, and `min` and `max`
  with a key pick among numbers. [Functions as values](closures.md) has the
  rest.
- `any` and `all` of a generator are native and stop at the first answer,
  as CPython's do; of a list they are native too.
- A standalone binary's `KeyError` for a string key says `KeyError` without
  the key.
- A container of objects or dataclasses crosses from Python as
  [objects do](classes.md#the-python-boundary). A container of objects
  whose class is an exception or generic keeps a Python caller on the Python
  body; native callers are not affected.

Examples: [Collections](../howto/47_collections.md), [Strings](../howto/48_strings.md).
