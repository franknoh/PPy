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

## Compared with CPython's `re` and Rust's `regex`

Four passes over 400,000 generated lines (7.9 MB): count the words, find the
longest, sum the `name = value` pairs, count the hex numbers. The programs
are in [`compare/`](compare/) -- [`patterns_bench.ppy`](compare/patterns_bench.ppy),
[`patterns_re.py`](compare/patterns_re.py), [`patterns.rs`](compare/patterns.rs)
with its [`Cargo.toml`](compare/Cargo.toml) -- and the site shows them
whole. Milliseconds, best of five passes, mean and spread over five processes.

**PPY** -- a pattern compiled from a bytes literal, `search` from a position,
`m.end()`; the same file runs on CPython:

```python
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

**CPython `re`** -- the idiomatic spelling, `finditer` and `int(m.group(2))`:

```python
def sum_values(text: bytes) -> int:
    return sum(int(m.group(2)) for m in PAIR.finditer(text))
```

**Rust `regex`** -- `captures_iter` over `&[u8]`, the number parsed from the
captured bytes, `cargo build --release`:

```rust
fn sum_values(pair: &Regex, text: &[u8]) -> u64 {
    pair.captures_iter(text)
        .map(|c| std::str::from_utf8(&c[2]).unwrap().parse::<u64>().unwrap())
        .sum()
}
```

<!-- compare:start -->
| | PPY `ppy run` | CPython `re` | Rust `regex` |
|---|---:|---:|---:|
| count_words | **6.57 ± 0.05** | 107.07 ± 1.27 | 18.23 ± 0.13 |
| longest_word | **6.53 ± 0.05** | 133.05 ± 8.69 | 36.85 ± 2.99 |
| sum_values | **56.70 ± 0.33** | 198.20 ± 1.86 | 65.62 ± 0.79 |
| count_hex | 6.59 ± 0.03 | 10.22 ± 0.10 | **2.33 ± 0.10** |
<!-- compare:end -->

PPY compiles each pattern into a function of core operations -- a
backtracker in which a byte-class repeat scans its run instead of stepping
byte by byte -- and a match is a few locals, not an object, so a search
loop is a native loop. That is the whole distance to `re`, whose matcher is
a bytecode interpreter allocating a match object per hit. Against Rust the
pattern decides: `[A-Za-z]+` is a scan in both engines, `\w+\s*=\s*(\d+)`
makes `regex` fall back to its slower capturing engine, and `0x[0-9a-f]+`
under `(?i)` is where its lazy DFA wins outright -- a literal-prefix
prefilter is the obvious next step for the PPY matcher there.

Intel Core Ultra 9 386H; rustc 1.95.0 with regex 1.x, CPython 3.12.13 for
`re`, PPY on CPython 3.13.13, from a checkout on a native filesystem.

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
