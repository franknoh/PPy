# Native lowering

This page covers when the compiler lowers a function to native code and what
a function may contain to lower.

## Eligibility and profitability

The compiler asks two different questions:

- `can_lower_native` decides whether correct native code exists.
- `should_lower_native` decides whether crossing the Python/native boundary
  pays for it.

A function with a loop, a buffer parameter, enough straight-line work, or an
explicit `@ppy.native`/`@ppy.jit`/`@ppy.specialize`/`@ppy.parallel` gets the
boundary. A two-instruction helper stays on the Python side (remarked as
`R3004`). Native callers still call its native symbol directly, boundary or
not.

`ppy explain FILE.ppy:name` reports the decision and, when the answer is no,
the first blocking construct.

## Byte-wide buffers

A buffer's element is a machine word unless it says otherwise.
`Buffer[ppy.i8]` and `Buffer[ppy.u8]` are one byte each, which is what text
and packed data want: four million characters cost four megabytes rather than
thirty-two.

The width is storage. Reading one hands out an `int`, and arithmetic on it is
integer arithmetic. Writing a value that does not fit in a byte falls back to
CPython rather than wrapping. An `array.array("b", ...)` is what such a buffer
is made of.

## Types that lower

A function lowers when its types are:

- scalars
- `Buffer[T]`
- homogeneous `list[int]`/`list[float]`, or `Sequence` of those
- all-scalar `@dataclass` value classes (flattened into scalar arguments)

and its body stays inside the modeled subset.

## What the body may contain

The subset includes what a loop is normally made of:

- `break` and `continue` (a `continue` in a `for` still advances the counter)
- statement-level calls whose result is discarded
- buffers handed on to another native function
- the bitwise operators, including `~`

A module constant written as an expression, such as `MOD = 10**9 + 7` or
`LIMIT = 1 << 20`, folds into the code rather than staying a global read.

A function whose writes all happen inside a callee it handed a buffer to
lowers too: the write lands in the caller's memory either way. The reverse
also lowers: filling memory you allocated and then passing it on.

A call to a function that did not lower keeps its caller on the Python side,
because the call would otherwise name a symbol nothing defines.

## Threads

The generated wrapper releases the GIL around the native call, so
`@ppy.native` functions scale across threads.

Reading input is its own guide: [Reading input](input.md).

Examples: [Algorithms](../howto/15_algorithms.md),
[Buffers and JIT](../howto/12_buffers_and_jit.md).
