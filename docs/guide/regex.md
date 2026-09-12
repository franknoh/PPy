# Regular expressions

A pattern compiled from a bytes literal at module level is matched natively
over a byte buffer. The compiler reads the pattern with CPython's own parser,
compiles it into a matcher function of its own, and a function that searches
with it lowers like any other:

```python
import re

import ppy
from ppy import Buffer

WORD = re.compile(rb"[A-Za-z]+")
PAIR = re.compile(rb"(?P<key>\w+)\s*=\s*(\d+)", re.MULTILINE)


def count_words(text: Buffer[ppy.u8]) -> int:
    n = 0
    pos = 0
    while True:
        m = WORD.search(text, pos)
        if m is None:
            return n
        n += 1
        pos = m.end()


def sum_values(text: Buffer[ppy.u8]) -> int:
    total = 0
    pos = 0
    while pos < len(text):
        m = PAIR.search(text, pos)
        if not m:
            break
        value = 0
        for i in range(m.start(2), m.end(2)):
            value = value * 10 + (text[i] - 48)
        total += value
        pos = m.end()
    return total
```

The same source runs unchanged on CPython: `re` accepts an `array.array("B")`
or a `memoryview` for a bytes pattern, which is what `Buffer[ppy.u8]` is
there. Natively, `WORD.search(text, pos)` is one call into a matcher the
compiler wrote for that pattern, and `m.end()` is a load.

## What is native

- **The pattern** is a bytes literal, either compiled once at module level
  with `re.compile(rb"...", flags)` and bound to a name that is never
  rebound, or written into the call: `re.search(rb"[a-z]+", text)`. The
  flags are spelled with `re.` names -- `re.IGNORECASE`, `re.MULTILINE`,
  `re.DOTALL`, `re.ASCII`, `re.VERBOSE`, their one-letter forms, and `|`
  between them -- or inline, `(?im)`.
- **The subject** is a `Buffer[ppy.u8]` local or parameter, one byte per
  element. `search`, `match`, and `fullmatch` take it, then `pos` and
  `endpos` positionally, with the meaning `re` gives them: `endpos` is where
  the string ends, `pos` is where matching starts, and `^` and `\b` still
  see the bytes before `pos`.
- **The match** is a local. `if m:`, `if m is None:`, and `if m is not None:`
  narrow it; `m.start()`, `m.end()`, and `a, b = m.span()` read a span, and
  each takes a group by constant index or by the name the pattern gave it.
  A group that took no part reads -1, as it does in `re`. `m.group()` hands
  back bytes, which has no native form: read the buffer between `start()`
  and `end()` instead, and the function stays on Python if it calls `group`.
- **The syntax** is `re`'s for bytes: literals and escapes, `.`, classes,
  `\d \w \s` and their negations, `^ $ \A \Z \b \B`, groups, named groups,
  non-capturing groups, alternation, and `* + ? {m,n}` with their lazy
  forms. Backreferences, lookahead and lookbehind, atomic groups,
  possessive repeats, and locale categories are not compiled; a function
  using one keeps running on Python and `ppy explain` names the construct.

## What the matcher does

The pattern becomes a `regex.search`, `regex.match`, or `regex.fullmatch`
operation in the IR, and the `lower-regex` pass compiles each distinct pattern
into one private function of core operations, so both the LLVM and the C
backend run it without a regex library. `ppy emit ir` shows the operation and
the function; `ppy emit c` shows the matcher as C.

The matcher backtracks the way CPython's does, on an explicit stack: an entry
records where to resume, the position, and every loop's count, so undoing one
puts the matcher back exactly. Alternatives are tried in order, a greedy
repeat gives back one iteration at a time, a lazy one takes one more, a group
keeps the position of its last iteration, and an iteration that took nothing
is the last one tried -- the same choices `re` makes, so the spans agree byte
for byte. A repeat of a single byte class, `[A-Za-z]+` or `\s*`, scans the run
and gives back one byte per backtrack rather than pushing an entry per byte,
which is what keeps a tokenizer's stack flat. The tests hold the matcher to
`re` on random inputs, bounds, and flags across both backends.

The stack holds 4096 entries. A match that would need more -- `(a|b)*c` over
a long input with no `c` -- fails a guard, and the calling function takes its
fallback: `re` answers, the program is right, and `ppy run --report-opt` says
which guard failed. Write such a loop as a single class, `[ab]*c`, and it
never pushes.

## Types and effects

The checker types `re.compile(...)` as `re.Pattern`, its `search`, `match`,
and `fullmatch` as `re.Match | None`, `start`, `end`, and `span` as integers
and a pair, `group` as `bytes | str | None`, and the flags as `int`. A pattern
bound at module level counts as a constant, so reading it is not a global
dependency and a function that only searches with it can be `@ppy.pure`.
Searching carries the allocation effect of the match object it makes on
CPython and may raise `TypeError`; `re.compile` may raise `re.error`.

Read on: [The IR: the regex dialect](../internals/ir.md#the-regex-dialect) ·
[Regular expressions](../howto/43_regex.md), the example.
