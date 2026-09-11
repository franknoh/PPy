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

The same four passes over 400,000 generated lines -- 7.9 MB -- each written
the way its tool is written, in [`compare/`](compare/):
[`patterns_bench.ppy`](compare/patterns_bench.ppy) (this example's functions),
[`patterns_re.py`](compare/patterns_re.py), and [`patterns.rs`](compare/patterns.rs)
with its [`Cargo.toml`](compare/Cargo.toml). Each prints the best of five
passes; [`examples/compare.py`](../compare.py) runs each program five times
and reports the mean and standard deviation across processes, in
milliseconds. All three print the same four answers.

| | PPY `ppy run` | CPython `re` | Rust `regex` |
|---|---:|---:|---:|
| count_words | **6.54 ± 0.04** | 106.91 ± 2.51 | 18.49 ± 0.08 |
| longest_word | **6.47 ± 0.04** | 129.53 ± 2.90 | 37.11 ± 2.49 |
| sum_values | **56.22 ± 0.39** | 199.87 ± 3.33 | 66.22 ± 0.90 |
| count_hex | 6.55 ± 0.10 | 10.39 ± 0.26 | **2.34 ± 0.09** |

What each port asked for:

- **PPY** is the source above: `re.compile` of a bytes pattern, `search`
  from a position, `m.end()`, over a `Buffer[ppy.u8]`. The same file runs on
  CPython, where the buffer is an `array.array("B")` and `re` does the work.
- **CPython's `re`** is the idiomatic spelling -- `finditer` in a generator
  expression, `int(m.group(2))` -- over `bytes`. Its matcher is a bytecode
  interpreter, and each match builds a match object.
- **Rust's `regex`** is `find_iter(...).count()` and `captures_iter` over
  `&[u8]`, compiled with `cargo build --release`. The crate compiles a
  pattern to a lazy DFA where it can, which is what makes `count_hex` fast;
  `sum_values` needs capture groups, which is a slower engine, and the
  number is parsed from the captured bytes.

PPY's matcher is compiled from the pattern into a function of core
operations, a backtracker with the run of a byte class scanned rather than
stepped, and a match is a local rather than an object; that is the whole
difference, and it is the difference of a compiled matcher against an
interpreted one. Against Rust the pattern is what decides: a byte-class
repeat is a scan in both, and a capture group is a backtrack in PPY where
`regex` runs its slower engine.

Intel Core Ultra 9 386H (16 threads); rustc 1.95.0 with regex 1.x, CPython 3.12.13 for `re`, PPY on
CPython 3.13.13, from a checkout on a native filesystem.

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
