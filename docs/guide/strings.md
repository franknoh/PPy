# Strings

A `str` lowers to native code. A function that builds, splits, searches, or
formats text goes native under `ppy run`, in a standalone binary, and in
emitted C and C++, and gives CPython's answer on every path.

```python
from ppy import HashMap


def busiest(log: str) -> str:
    hits = HashMap[str, int]()
    for line in log.splitlines():
        path = line.split()[1]
        hits[path] = hits.get(path, 0) + 1
    best, most = "", 0
    for path in hits:
        if hits[path] > most:
            best, most = path, hits[path]
    return f"{best:<12}{most:>6,}"
```

The [Strings example](../howto/48_strings.md) parses and reports on a
200,000-line log and times each path.

## What a string is natively

A string is a handle into the C runtime (`ppy_runtime/strings.c`), counted
the way a [collection](collections.md) is and freed when the last reference
goes. It holds:

- its UTF-8 bytes, with a NUL after them
- its length in code points, kept, so `len(s)` reads one word
- whether every byte is ASCII, so an index into an ASCII string is one step
  and an index into any other walks the bytes
- its hash, worked out the first time a map asks

A string is immutable. `s += t` appends in place when nothing else holds
`s`, as CPython does, and makes a new string otherwise. The 128
one-character ASCII strings are static and never allocated, so `for c in s`
and `s[i]` over ASCII text allocate nothing.

Indices, lengths, and slices count code points, as Python's do. Comparing
two strings compares their bytes, which in UTF-8 orders them the way
Python orders code points.

## What lowers

| | |
|---|---|
| literals, `+`, `*` by an `int`, `+=`, `*=` | |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | by code point |
| `s[i]`, `s[a:b]`, `s[a:b:c]` | negative indices and steps as Python has them |
| `sub in s`, `s in names`, `c in ("a", "b")` | |
| `for c in s`, `len(s)`, `if s:` | |
| `ord`, `chr`, `min` and `max` of strings | |
| `str(x)`, `repr(x)`, `format(x, spec)` | of an `int`, `float`, `bool`, or `str` |
| `"...".format(...)` | positional, numbered, and keyword fields, with conversions and specs |
| `"..." % values` | `%s`, `%r`, `%d`, `%i`, `%x`, `%X`, `%o`, `%e`, `%f`, `%g` and their capitals, `%%`, with flags, width, and precision |
| `int(s)`, `int(s, base)`, `float(s)` | the grammar CPython reads, underscores and `inf` included |
| f-strings | fields of numbers, bools, and strings, with `!r`, `!s`, `!a` |

A float is written the way `repr` writes it: the shortest digits that read
back as the same double.

### Format specs

An f-string field or `format` takes fill and alignment (`<`, `>`, `^`, `=`),
a sign, `z`, `#`, `0`, a width, grouping with `,` or `_`, a precision, and
the types `d`, `b`, `o`, `x`, `X`, `e`, `E`, `f`, `F`, `g`, `G`, `%`, and
`s`. A spec with a nested field (`{x:{width}}`) stays in Python.

`str.format` and `%` of a string literal lower as the f-string they mean, so
they take the same specs. `"%-6s|%05d" % (name, n)` is `f"{name!s:<6}|{n:05d}"`.
A format string that is not a literal, a `%` whose right side may be a tuple
or a mapping (`"%s" % t` where `t` is a tuple), a field that reaches into its
value (`{0.x}`, `{a[0]}`), and arguments read out of order where reading them
twice would show stay in Python.

### Methods

| | |
|---|---|
| `find`, `rfind`, `index`, `rindex`, `count` | with `start` and `end` |
| `startswith`, `endswith` | with `start` and `end`; a tuple of prefixes stays in Python |
| `replace` | with a count |
| `strip`, `lstrip`, `rstrip` | on whitespace or on the characters given |
| `split`, `rsplit` | on whitespace or a separator, with `maxsplit` |
| `splitlines` | every line boundary Python has, with `keepends` |
| `join` | of a list of strings, a `Vec[str]`, or a literal list or tuple |
| `partition`, `rpartition` | unpacked into three names |
| `removeprefix`, `removesuffix`, `zfill` | |
| `ljust`, `rjust`, `center` | with a fill character |
| `lower`, `upper`, `capitalize`, `title`, `swapcase` | |
| `isdigit`, `isalpha`, `isalnum`, `isupper`, `islower`, `isspace`, `isdecimal`, `isnumeric` | |

## Lists of strings

`split`, `rsplit`, and `splitlines` hand back a `list[str]`, and a function
may make one from a literal (`["a", "b"]`). Natively it is a sequence of
string handles. It indexes from either end, iterates, unpacks
(`key, value = line.split("=")`), and has `append`, `insert`, `pop`,
`sort`, `reverse`, `clear`, `index`, `count`, `in`, `len`, and `print`.

## Strings in collections and classes

A string is an element (`Vec[str]`, `Deque[str]`, `Heap[str]`) and a key
(`HashMap[str, int]`, `HashSet[str]`, `TreeMap[str, float]`,
`TreeSet[str]`). A map hashes a key by its text and a tree orders keys by
it, and each holds a reference to every key it keeps. A heap and `sort`
order strings as `<` does.

A string is a field of a dataclass or of an object class, which makes the
class an object class: native code holds it by handle. See
[Classes](classes.md).

## At the Python boundary

A native function with `str` parameters or a `str` result is called from
Python under `ppy run`. The boundary hands the string's UTF-8 bytes in,
without a copy, and a returned string back out as a new Python string.
A string with a lone surrogate has no UTF-8, and a call with one runs as
Python.

A `list[str]`, `dict[str, V]`, or `set[str]` parameter or result is
passed by handle between native functions. From Python, a call that writes
through none of its parameters reads such a container in place: each
string in it is the Python string's own UTF-8, borrowed for the call, and a
string it hands back is the caller's object. A call that writes copies the
strings in. Either way native code pays to hold each string and to hash
each key it looks up, so Python calls the function natively only when it
does more than one pass of work over its strings (`for w in words: for ch
in w:`); see [Lists, dicts, and sets](containers.md#between-functions).

## Reading and printing

In a standalone binary, `ppy.input[str]()` and `input()` read one line
without its newline, `ppy.scan[str]()` reads the next whitespace-delimited
token, and `print` writes strings and lists of strings. A line that is not
UTF-8 is `UnicodeDecodeError`, as CPython reads it.

Under `ppy run`, a native `print` writes into output held until the call
returns, and `input()` calls Python's own `input`, as
[Effects in native code](native-effects.md) describes.

## Where it falls back

Some answers need Python's Unicode tables. These check first and fall back:
under `ppy run` the call runs as Python, and a standalone binary stops.

- the case methods and `is...` methods of text outside ASCII (`isspace` is
  native for all text)
- `repr` of text outside ASCII, which decides what is printable
- `int(s)` of digits outside ASCII

Everything CPython raises for (an index out of range, `int("x")`, an empty
separator, `ord` of two characters) is a guard of the same kind.

## Limitations

- `encode`, `translate`, `expandtabs`, and `casefold` stay in Python.
- The case of text outside ASCII falls back, as above.

Examples: [Strings](../howto/48_strings.md).
