# Reading input

PPy has three ways to read standard input, all sharing one grammar:

- `ppy.input` reads lines.
- `ppy.scan` reads tokens.
- `ppy.read_*` fill a buffer you already have.

Both `ppy.input` and `ppy.scan` also read JSON into a dataclass, a pydantic
model, or a `TypedDict` ([below](#json-into-a-schema)).

```python
import ppy
from ppy import Buffer

line = ppy.input[str]()  # one line, the newline removed
n = ppy.input[int]()  # one line, read as int(input()) reads it
a, b = ppy.input[tuple[int, int]]()  # one line, exactly two fields
names = ppy.input[list[str]]()  # one line, split into its fields
row = ppy.input[Buffer[int]]()  # one line of integers, straight into a buffer
small = ppy.input[ppy.i32]()  # read as int, then it must fit 32 bits

word = ppy.scan[str]()  # one whitespace-delimited token, lines irrelevant
k = ppy.scan[int]()  # one integer token
values = ppy.scan[Buffer[int]](n)  # n integer tokens, straight into a buffer
```

## `ppy.input`: lines

`ppy.input[T]()` reads one line and means what the builtin `input()` means.
That is what lets `ppy convert` turn `int(input())` into `ppy.input[int]()`
without changing what the program reads.

- `ppy.input[str]()` is the line without its trailing newline. Spaces and
  tabs inside it are kept, an empty line is `""`, the last line of the input
  needs no newline, and reading past the end raises `EOFError`.
- `ppy.input[int]()` and `ppy.input[float]()` read one line and parse the
  whole line, as `int(input())` and `float(input())` do: `"  42  "` is `42`,
  `"1_000"` is `1000`, and `"1 2"` is `ValueError`. It is not `1` with the
  `2` quietly kept for later.
- `ppy.input[tuple[int, int]]()` reads one line, splits it as
  `input().split()` would, and requires exactly the fields the tuple names.
  A short line is `ValueError`, and a field is never taken from the next
  line. The fields are `int`, `float`, or `str`. A tuple of `int` alone is
  read in C, the way a buffer line is (below).
- `ppy.input[list[int]]()` reads one line and converts each of its fields,
  as `list(map(int, input().split()))` does: however many there are, `[]`
  for an empty line. `list[float]` and `list[str]` work the same way.
- `ppy.input[Buffer[int]]()` reads one line of integers into a new buffer,
  as `array.array("q", map(int, input().split()))` does. It takes no count:
  the line decides. See the next section for how its fields are parsed.
  `ppy.input[Buffer[float]]()` is the same for floats, as
  `array.array("d", map(float, input().split()))`.
- A fixed width, such as `ppy.input[ppy.i32]()`, reads the field as `int`
  and then requires it to fit: a value outside the width is
  `OverflowError`. This holds for a scalar, a tuple field, and a list
  element. `ppy.f32` and `ppy.f64` read as `float`.

A read takes no argument. A prompt is a `print` before it, the way any other
output is written, so reading and printing stay separate.

### How a buffer line is parsed

The fields of `ppy.input[Buffer[int]]()` are parsed in C without a Python
object each: a sign, digits, single underscores between them, the ASCII forms
`int()` accepts. A field outside what C reads is handed to `int()` and the
array themselves, so the read means what the idiom means:

- a field that is no integer is `int()`'s `ValueError`
- one past 64 bits is the array's `OverflowError`
- non-ASCII digits are read as `int()` reads them

A tuple of `int` is parsed the same way. There a field past 64 bits is simply
the integer, as `map(int, input().split())` gives it.

A block of integers spread over several lines is `ppy.scan[Buffer[int]](n)`.

## `ppy.scan`: tokens

`ppy.scan[T]()` reads the next whitespace-delimited token wherever it is.
Newlines are whitespace to it, as they are to `scanf`, so a million numbers
spread over any number of lines read the same way.

- `ppy.scan[str]()` is the next token, however long.
- `ppy.scan[int]()` is the next token as an integer, and the token must be
  one: an optional sign and digits, within 64 bits. `abc` where an integer is
  expected is `ValueError` naming it (it is never skipped), and the end of
  the input is `EOFError`.
- `ppy.scan[float]()` is the next token as a float.
- `ppy.scan[tuple[int, int]]()` is two tokens, wherever they fall.
- `ppy.scan[Buffer[float]](n)` reads `n` float tokens into a new buffer.
- `ppy.scan[Buffer[int]](n)` reads `n` integer tokens straight into a new
  buffer, fewer where the input ends. Reading goes into memory rather than
  through a Python object per field, which is what takes a million numbers
  quickly: faster than `sys.stdin.read().split()` and faster than C's
  `scanf`, measured in [Algorithms](../howto/15_algorithms.md).

### Errors in `ppy.scan[Buffer[int]](n)`

`ppy.scan[Buffer[int]](n)` reads exactly `n` integers into a new buffer.

- the input ending first is `EOFError`
- a token that is not an integer is `ValueError`
- one outside 64 bits is the scanner's `ValueError` for it
- a negative `n` is `ValueError`

A value that was never read is not a zero. The tokens before the offending one
were read, and the reader stands after it. `ppy.read_ints(buffer)` is the
partial read: it fills what it can and says how many it got.

### Mixing token and line reads

A token read stops before the whitespace that ends it, so a line read that
follows starts there. After `ppy.scan[int]()` reads the `5` of `5\nabc`,
`ppy.input[str]()` returns `""`, the rest of that line, and the next returns
`abc`.

## JSON into a schema

Give `ppy.input` or `ppy.scan` a class you already have, and it reads JSON
into it:

```python
from dataclasses import dataclass

import ppy


@dataclass
class Item:
    name: str
    price: float


@dataclass
class Order:
    id: int
    items: list[Item]
    note: str | None = None


order = ppy.input[Order]()  # one line of JSON
orders = ppy.input[list[Order]]()  # one line holding a JSON array
pretty = ppy.scan[Order]()  # the next JSON value, over as many lines as it spans
```

The class can be a dataclass, a pydantic model, or a `TypedDict`, or a
`list`, `tuple`, `dict`, or optional of one. `list[int]` is still a line of
fields; a list of a schema class is a JSON array.

A dataclass or a `TypedDict` is built field by field:

- `int` takes a JSON integer, not `true` and not `1.5`.
- `float` takes any JSON number.
- `str`, `bool`, and `None` take their own kind.
- Containers, unions, `Literal`, enums, and fixed widths such as `ppy.u8` are
  read as declared.
- A field with a default may be missing.
- A key the class does not declare is ignored, as pydantic ignores it by
  default.

A mismatch is a `ValueError` that says where it is and what was there, such
as `$.items[2].price: expected a number, got a string`. JSON that does not
parse is the `json` module's `JSONDecodeError`, also a `ValueError`.

A pydantic model, or anything that holds one, is validated by pydantic
itself with its own rules, and its `ValidationError` is a `ValueError` too.

`ppy.scan[Model]()` starts at the next line that is not blank and reads up
to the line that closes the value's last bracket. The next read starts on
the line after it.

## `ppy.read_ints` and `ppy.read_token`: buffers

The scanner's low-level forms fill a buffer the caller made.

- `ppy.read_ints(buffer)` fills a writable buffer of 64-bit integers with
  integer tokens and returns how many it read. That is fewer than
  `len(buffer)` only where the input ended.
- `ppy.read_token(buffer)` reads one token into a buffer of bytes or of
  64-bit integers, one byte per slot, and returns how many slots it wrote.

The buffer's length is the capacity the caller chose. `read_token` cuts a
longer token there and consumes the rest, which is what a fixed buffer means.
`ppy.scan[str]()` is the form with no such limit.

`ppy.buffer[T](n)` makes a buffer of `n` zeroed elements of `T`:
`array.array` under CPython and a native allocation in a standalone binary.
That is what lets the same source build both ways.

## One grammar, two implementations

Whitespace is ASCII: space, tab, newline, carriage return, form feed, vertical
tab. An integer token is `[+-]?[0-9]+` within a signed 64-bit word.

The scanner is a few C functions (`ppy_runtime.scanner`) compiled once and
cached. The same text is what a standalone binary links in as its runtime, so
`ppy.scan` means the same thing under CPython and with no interpreter at all.

Without a C compiler a pure-Python fallback reads the same grammar. A
differential test holds the two to the same answers on malformed and boundary
input, and the only difference is speed.

## Effects and `sys.stdin`

All of these carry the IO effect, so a function that reads is never mistaken
for a pure one, and all of them work on every path.

!!! warning
    The reader owns file descriptor 0 and buffers it itself, so a program that
    uses it must not also read `input()` or `sys.stdin`. `ppy convert`
    rewrites a module's `input` idioms only where the module never touches
    `sys.stdin`.

## Native code and standalone binaries

A standalone binary reads with the C runtime, the same scanner text. Where
Python would raise, the binary names the same exception on standard error
and stops, for example `ppy: ValueError: could not convert string to float`.

| Read | In a standalone binary |
|---|---|
| `ppy.input[int]()`, `ppy.input[float]()` | yes |
| `ppy.input[ppy.i32]()` and the other fixed widths | yes |
| `ppy.input[tuple[int, float]]()` of `int`, `float`, and fixed widths | yes |
| `ppy.input[Buffer[int]]()`, `ppy.input[Buffer[float]]()` | yes |
| `ppy.input[list[int]]()`, `ppy.input[list[float]]()` | yes, as a buffer: reading and iterating work, growing it does not |
| `ppy.scan[int]()`, `ppy.scan[float]()`, `ppy.scan[tuple[...]]()` of numbers | yes |
| `ppy.scan[Buffer[int]](n)`, `ppy.scan[Buffer[float]](n)` | yes |
| `ppy.input[str]()`, `ppy.scan[str]()`, `list[str]` | no: native code has no string type yet |
| JSON into a schema class | no: it builds Python objects |

A buffer or list read is bound to a name (`values = ppy.input[Buffer[int]]()`).
A tuple read is unpacked or bound to a name. A native `print` writes a float
the way `repr` does, with the shortest digits that read back as the same
value.

In `ppy run`, a function that reads keeps running as Python, since reading
is an effect, and every form above works there.

`E1305` names a misuse:

- an argument to a scalar read
- a `scan` of a buffer without its count
- a count given to the line read of one
