# Reading input and native lowering

## Reading input

`ppy.input[T]()` reads the next value the way `T` says to read it, and the
checker types the result from the same `T`:

```python
import ppy
from ppy import Buffer

n = ppy.input[int]()  # one integer
a, b = ppy.input[tuple[int, int]]()  # two fields, line breaks irrelevant
word = ppy.input[str]("name? ")  # a token, after printing the prompt
values = ppy.input[Buffer[int]](n)  # n integers, straight into a buffer
```

Whitespace and newlines are the same thing to it, as they are to `scanf`.
Reading goes into memory rather than through a Python object per field, so
the buffer form is what takes a million numbers quickly — faster than
`sys.stdin.read().split()` and faster than C's `scanf`, measured in
[Algorithms](../howto/15_algorithms.md). The call takes a prompt for a
scalar, printed before reading the way the builtin `input` does, or how
many values to read for a buffer. `ppy.read_ints` and `ppy.read_token` are
the lower-level forms that fill a buffer you already have, and
`ppy.buffer[T](n)` makes one: `n` elements of `T`, all zero. It is
`array.array` under CPython and a native allocation in a standalone binary,
which is what lets the same source build both ways.

All of them carry the IO effect, so a function that reads is never mistaken
for a pure one, and all of them work on every path including plain CPython:
the small C reader is compiled once and cached, with a pure-Python fallback
where no compiler exists. The reader owns file descriptor 0 and buffers it
itself, so a program that uses it must not also read `input()` or
`sys.stdin`. Reading past the end raises `EOFError`.

## Native lowering

Eligibility and profitability are different questions, and the compiler asks
both: `can_lower_native` decides whether correct native code exists, and
`should_lower_native` whether crossing the Python/native boundary pays for
it. A function with a loop, a buffer parameter, enough straight-line work,
or an explicit `@ppy.native`/`@ppy.jit`/`@ppy.specialize`/`@ppy.parallel`
gets the boundary; a two-instruction helper stays on the Python side
(remarked as `R3004`) — while native callers keep calling its native symbol
directly, boundary or not.

A buffer's element is a machine word unless it says otherwise:
`Buffer[ppy.i8]` and `Buffer[ppy.u8]` are one byte each, which is what text
and packed data want — four million characters cost four megabytes rather
than thirty-two. The width is storage, not type: reading one hands out an
`int`, arithmetic on it is integer arithmetic, and writing a value that does
not fit in a byte falls back to CPython rather than wrapping. An
`array.array("b", ...)` is what such a buffer is made of.

A function lowers when its types are scalars, `Buffer[T]`, homogeneous
`list[int]`/`list[float]`, `Sequence` of those, or all-scalar `@dataclass`
value classes (flattened into scalar arguments), and its body stays inside the
modeled subset. That subset includes what a loop is normally made of:
`break` and `continue` (a `continue` in a `for` still advances the counter),
statement-level calls whose result is discarded, buffers handed on to
another native function, and the bitwise operators including `~`. A module
constant written as an expression — `MOD = 10**9 + 7`, `LIMIT = 1 << 20` —
folds into the code rather than staying a global read. A function whose
writes all happen inside a callee it handed a buffer to lowers too: the
write lands in the caller's memory either way, and so does the reverse —
filling memory you allocated and then passing it on. A call to a function
that did not lower keeps its caller on the Python side, because the call
would otherwise name a symbol nothing defines. The generated wrapper
releases the GIL around the native call, so `@ppy.native` functions scale
across threads. `ppy explain FILE.ppy:name` reports the decision and, when
the answer is no, the first blocking construct.

Examples: [Algorithms](../howto/15_algorithms.md),
[Buffers and JIT](../howto/12_buffers_and_jit.md).
