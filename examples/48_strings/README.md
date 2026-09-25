# Strings

A 200,000-line log built with f-strings, then split, parsed, counted by
path in `HashMap[str, float]`, ordered in a `TreeMap`, and run-length
encoded one character at a time, all with native `str`. Every function goes
native, and the program prints the same thing on all three paths.

## Run it

```bash
python  text.ppy
ppy run text.ppy
ppy build --standalone text.ppy -o dist && ./dist/text
```

<!-- outputs:start -->
## What it prints

**`python  text.ppy`**, **`ppy run text.ppy`**, **`ppy build --standalone text.ppy -o dist && ./dist/text`**

```text
/api/orders        451.74 ms  x40,000
/api/search        448.74 ms  x40,000
/api/users         447.74 ms  x40,000
/health            449.73 ms  x40,000
/static/app.js     450.75 ms  x40,000
levels: INFO=50000 DEBUG=50000 ERROR=50000 WARN=50000
a3b1c2d5 46075322
```

<!-- outputs:end -->

## What it does

- `make_log` writes each line with an f-string: a zero-padded day, a level
  padded to five with `:<5`, a path, a time with `:.2f`, and a user id.
- `slowest_paths` splits each line on whitespace, strips `took=` and `ms`
  with `removeprefix` and `removesuffix`, reads the number with `float()`,
  and sums per path in a `HashMap[str, float]`. The report walks the paths
  in order through a `TreeMap[str, float]` and formats each with `:<16`,
  `:>9.2f`, and `:,`.
- `levels` slices the level out of each line (`line[11:16].strip()`) and
  counts them.
- `compress` walks a string with `for c in s`, comparing one character with
  the last, and builds the encoding with `+=`.
- `checksum` uppercases each line, encodes it, and folds `ord` of each
  character of the result.

`report` calls them all and returns one string. It is marked
`@ppy.native`: it has no loop of its own, and without the mark the
compiler would judge its body too small to be worth the Python boundary,
though the functions it calls do all the work.

## How it runs natively

A string is a counted handle into the C runtime in
`ppy_runtime/strings.c`, holding its UTF-8 bytes, its length in code
points, and whether it is ASCII. The line list is a `Vec[str]`; a map keyed
by `str` hashes and compares the text.

Two things keep allocation down, as CPython does the same two:

- the 128 one-character ASCII strings are static, so `for c in text` and
  `text[0]` allocate nothing
- `out += part` appends in place when nothing else holds `out`

`ppy explain text.compress` and the others each say
`llvm backend: native`.

## Timing

One machine, five runs each, median wall time for the whole program:

| | seconds |
|---|---:|
| `python text.ppy` | 2.05 |
| `idiomatic.py`: the same work with `str`, `dict`, `Counter`, `defaultdict`, and `sorted` | 1.85 |
| `ppy run text.ppy`, after the first run built the cache | 2.27 |
| `./dist/text`, the standalone binary | 1.16 |

Measured inside the process, `report` itself takes about 1.8 seconds under
`ppy run` and 1.1 in the standalone binary; the rest of `ppy run`'s time is
starting the interpreter and loading the cached native code. Under
CPython, `text.ppy` runs its `Vec` and `HashMap` as the reference classes
in Python, which is why it trails `idiomatic.py`.

## Where the code comes from

`text.ppy` and `idiomatic.py` are hand-written.

Read on: [the strings guide](../../docs/guide/strings.md).
