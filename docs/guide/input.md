# Reading input

Three ways to read standard input, one grammar underneath, and a rule for
telling them apart: `ppy.input` reads lines, `ppy.scan` reads tokens, and
`ppy.read_*` fill a buffer you already have.

```python
import ppy
from ppy import Buffer

line = ppy.input[str]()  # one line, the newline removed
n = ppy.input[int]()  # one line, read as int(input()) reads it
a, b = ppy.input[tuple[int, int]]()  # one line, exactly two fields
names = ppy.input[list[str]]()  # one line, split into its fields
row = ppy.input[Buffer[int]]()  # one line of integers, straight into a buffer

word = ppy.scan[str]()  # one whitespace-delimited token, lines irrelevant
k = ppy.scan[int]()  # one integer token
values = ppy.scan[Buffer[int]](n)  # n integer tokens, straight into a buffer
```

## `ppy.input`: lines

`ppy.input[T]()` reads one line and means what the builtin `input()`
means, which is what lets `ppy convert` turn `int(input())` into
`ppy.input[int]()` without changing what the program reads:

- `ppy.input[str]()` is the line without its trailing newline. Spaces and
  tabs inside it are kept, an empty line is `""`, the last line of the input
  needs no newline, and reading past the end raises `EOFError`.
- `ppy.input[int]()` and `ppy.input[float]()` read one line and parse the
  whole line, as `int(input())` and `float(input())` do: `"  42  "` is `42`,
  `"1_000"` is `1000`, and `"1 2"` is `ValueError`, never `1` with the `2`
  quietly kept for later.
- `ppy.input[tuple[int, int]]()` reads one line, splits it as
  `input().split()` would, and requires exactly the fields the tuple names;
  a short line is `ValueError`, and a field is never taken from the next
  line. The fields are `int`, `float`, or `str`. A tuple of `int` alone is
  read in C, the way a buffer line is, below.
- `ppy.input[list[int]]()` reads one line and converts each of its fields,
  as `list(map(int, input().split()))` does: however many there are, `[]`
  for an empty line. `list[float]` and `list[str]` likewise.
- `ppy.input[Buffer[int]]()` reads one line of integers into a new buffer,
  as `array.array("q", map(int, input().split()))` does, and takes no count:
  the line decides. The fields are read in C without a Python object each --
  a sign, digits, single underscores between them, the ASCII forms `int()`
  accepts -- and a field that is not one is `ValueError`, one outside 64
  bits `OverflowError`, as the array would raise; either way the line was
  read whole first, as `input()` had read it. That is the one place the
  typed read is narrower than `input()`: non-ASCII digits and integers
  past 64 bits, which `int()` takes, are refused by a buffer line read and
  by a tuple of `int`. A block of integers spread over several lines is
  `ppy.scan[Buffer[int]](n)`.

A read takes no argument. A prompt is a `print` before it, the way any
other output is written, so reading and printing stay two things.

## `ppy.scan`: tokens

`ppy.scan[T]()` reads the next whitespace-delimited token wherever it is.
Newlines are whitespace to it, as they are to `scanf`, so a million numbers
spread over any number of lines read the same way:

- `ppy.scan[str]()` is the next token, however long.
- `ppy.scan[int]()` is the next token as an integer, and the token must be
  one: an optional sign and digits, within 64 bits. `abc` where an integer
  is expected is `ValueError` naming it -- it is never skipped -- and the
  end of the input is `EOFError`.
- `ppy.scan[float]()` is the next token as a float; `ppy.scan[tuple[int,
  int]]()` two tokens, wherever they fall.
- `ppy.scan[Buffer[int]](n)` reads `n` integer tokens straight into a new
  buffer, fewer where the input ends. Reading goes into memory rather than
  through a Python object per field, which is what takes a million numbers
  quickly -- faster than `sys.stdin.read().split()` and faster than C's
  `scanf`, measured in [Algorithms](../howto/15_algorithms.md).

A token read stops before the whitespace that ends it, so a line read that
follows starts there: after `ppy.scan[int]()` reads the `5` of `5\nabc`,
`ppy.input[str]()` returns `""`, the rest of that line, and the next
returns `abc`.

## `ppy.read_ints` and `ppy.read_token`: buffers

The scanner's low-level forms fill a buffer the caller made:
`ppy.read_ints(buffer)` fills a writable buffer of 64-bit integers with
integer tokens and returns how many it read, which is fewer than
`len(buffer)` only where the input ended; `ppy.read_token(buffer)` reads one
token into a buffer of bytes or of 64-bit integers, one byte per slot, and
returns how many slots it wrote. The buffer's length is the capacity the
caller chose: `read_token` cuts a longer token there and consumes the rest,
which is what a fixed buffer means, and `ppy.scan[str]()` is the form with
no such limit. `ppy.buffer[T](n)` makes a buffer of `n` zeroed elements of
`T` -- `array.array` under CPython and a native allocation in a standalone
binary, which is what lets the same source build both ways.

## One grammar, two implementations

Whitespace is ASCII -- space, tab, newline, carriage return, form feed,
vertical tab -- and an integer token is `[+-]?[0-9]+` within a signed 64-bit
word. The scanner is a few C functions
(`ppy_runtime.scanner`) compiled once and cached, and the same text is what
a standalone binary links in as its runtime, so `ppy.scan` means the same
thing under CPython and with no interpreter at all. Without a C compiler a
pure-Python fallback reads exactly the same grammar; a differential test
holds the two to the same answers on malformed and boundary input, and the
only difference is speed.

All of these carry the IO effect, so a function that reads is never
mistaken for a pure one, and all of them work on every path. The reader
owns file descriptor 0 and buffers it itself, so a program that uses it must
not also read `input()` or `sys.stdin`; `ppy convert` rewrites a module's
`input` idioms only where the module never touches `sys.stdin`.

In native code, `ppy.input[int]()` and `ppy.scan[int]()` are calls into the
runtime; in a standalone binary they are the scanner itself, and where
Python would raise, the binary says what happened on standard error and
stops. `ppy.input[Buffer[int]]()` has no standalone lowering yet. `E1305`
names a misuse -- an argument to a scalar read, a `scan` of a buffer
without its count, a count given to the line read of one.
