# Regular expressions

A pattern compiled from a bytes literal is matched natively over a byte
buffer. The compiler reads the pattern with CPython's own parser, writes a
matcher function for it, and a function that searches with it lowers like any
other.

## One pattern, one matcher

```python
WORD = re.compile(rb"[A-Za-z]+")
PAIR = re.compile(rb"(?P<key>\w+)\s*=\s*(\d+)")


def sum_values(text: Buffer[ppy.u8]) -> int:
    total = 0
    pos = 0
    while True:
        m = PAIR.search(text, pos)
        if m is None:
            return total
        value = 0
        for i in range(m.start(2), m.end(2)):
            value = value * 10 + (text[i] - 48)
        total += value
        pos = m.end()
```

`PAIR.search(text, pos)` becomes a `regex.search` operation in the IR, and
the `lower-regex` pass compiles the pattern into a private function of core
operations, so the LLVM and C backends run it without a regex library. The
match is a local: `m is None` and `if m:` narrow it, `m.start(2)`, `m.end()`,
and `start, end = m.span()` are loads. The matcher backtracks the way `re`
does -- ordered alternation, greedy and lazy repeats, a group's last
iteration -- so the spans agree byte for byte; a repeat of one byte class
such as `[A-Za-z]+` scans its run rather than pushing per byte. The same
source runs unchanged on CPython, where `re` accepts the `array.array("B")`
the buffer is.

## What stays with Python

`m.group()` hands back bytes, which has no native form, so a function that
calls it keeps running on Python; read the buffer between `start()` and
`end()` instead. Backreferences, lookahead and lookbehind, atomic groups, and
possessive repeats are refused when the pattern is analysed, and `ppy explain`
names the construct. A match that would need more than the matcher's stack of
4096 entries -- `(a|b)*c` over a long input without a `c` -- fails a guard and
the function falls back to `re` for that call.

The text is 50,000 generated lines of `name = value` pairs, words, and hex
numbers; the line starting with `# ` is a timing and differs between machines
and between the two runs below.

## Run it

```bash
python  patterns.ppy
ppy run patterns.ppy
ppy emit ir patterns.ppy
ppy emit c patterns.ppy
```

<!-- outputs:start -->
## What it prints

**`python  patterns.ppy`**

```text
989986 bytes
119454 7 25007528 7143
True False
# four passes: 72.1 ms
```

**`ppy run patterns.ppy`**

```text
989986 bytes
119454 7 25007528 7143
True False
# four passes: 9.1 ms
```

**`ppy emit ir patterns.ppy`**

*1480 lines: [outputs/03-ppy-emit-ir-patterns-ppy.txt](outputs/03-ppy-emit-ir-patterns-ppy.txt)*

**`ppy emit c patterns.ppy`**

*1270 lines: [outputs/04-ppy-emit-c-patterns-ppy.txt](outputs/04-ppy-emit-c-patterns-ppy.txt)*

<!-- outputs:end -->

Read on: [Regular expressions](../../docs/guide/regex.md) ·
[The IR: the regex dialect](../../docs/internals/ir.md) ·
[Substring search](../15_algorithms/15c_kmp/README.md)

`patterns.ppy` is hand-written; there is no `.py` source and no conversion step.
