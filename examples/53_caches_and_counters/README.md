# Caches and counters

Word statistics with `collections.Counter` and `defaultdict`, words drawn
from a seeded `random.Random`, and two dynamic programs memoized with
`functools.cache` and `lru_cache`. All of it compiles to native code, on
`ppy run` and in a standalone binary, with CPython's answers.

## Run it

```bash
python  words.ppy
ppy run words.ppy
ppy build --standalone words.ppy -o dist && ./dist/words
```

<!-- outputs:start -->
## What it prints

**`python  words.ppy`**, **`ppy run words.ppy`**, **`ppy build --standalone words.ppy -o dist && ./dist/words`**

```text
300000 words, 99370 different
  ne     5090
  mi     5005
  ant    4999
a e i o: [314741, 209800, 210487, 210676]
lengths 2 to 12: [39719, 9943, 32288, 16067, 27578, 19151, 25111, 20810, 24311, 21702, 23411]
nearest of 150 words: mean distance 4.927, most 7
cheapest order for a chain of 250 matrices: 1041250 multiplications
```

<!-- outputs:end -->

## The program

- `make_words(seed, count)` builds 300,000 words of one to six syllables,
  each drawn with `rng.randint(1, 6)` and `rng.choice(syllables)` from a
  `random.Random(seed)`.
- `report` counts the words with `Counter(words)` and prints the three most
  common with `most_common(3)`.
- `letter_counts(words, "aeio")` counts every letter into a
  `Counter[str]` and reads four of them back. A missing key reads 0, as in
  CPython.
- `length_counts(words)` counts word lengths into a
  `defaultdict(int)` and reads the lengths 2 to 12, including those no word
  has, which the `defaultdict` adds with 0.
- `distance(a, b)` is the edit distance written top-down, as
  TheAlgorithms/Python writes it: a nested function `d(i, j)` under
  `@cache` that reads `a` and `b` from the function around it.
  `nearest(words)` gives each of 150 words, sampled with
  `random.Random(seed + 1).sample(sorted(counts), 150)`, the distance to
  its nearest neighbour among the others: 22,350 calls of `distance`.
- `cheapest(i, j)` under `@lru_cache(maxsize=None)` is the cheapest order
  to multiply a chain of 250 matrices, whose sizes `size(i)` gives. Each
  entry tries every split point `k` and looks up `cheapest(i, k)` and
  `cheapest(k, j)`.

## How it runs natively

`ppy explain --summary words.ppy` counts all ten functions as native.
`report` is one call from Python into native code; everything it calls
runs natively from there, and what it prints is held until it returns.

- **`random.Random(seed)`** is a native generator: CPython's Mersenne
  Twister state, seeded as CPython seeds it, and `randint`, `choice`, and
  `sample` drawing the same numbers as `random.py`. The sample therefore
  picks the same 150 words on every path.
- **`Counter` and `defaultdict`** are the runtime's dict with a flavor. A
  `Counter`'s missing key reads 0 without being added; a `defaultdict`'s
  missing key gets its factory's value (`int()` is 0) before the read.
  `most_common` ranks as `sorted(..., reverse=True)` does, so ties keep
  insertion order, as in CPython.
- **`@cache` on a nested function** gives `d` a table of its own each time
  `distance` runs, held by its closure, and its recursive calls look in the
  table first. The arguments are two ints, the result an int.
- **`@lru_cache` on a module function** is the same lookup in one table
  that lives as long as the program. A function that read
  `cheapest.cache_info()` would keep CPython's cache and run as Python, so
  that the counts it reports are CPython's.

## Timing

Wall time for the whole program, the mean of five runs, measured from a
checkout under `/tmp` on one machine (Python 3.14, an Intel Core Ultra 9
386H under WSL2):

| | seconds |
|---|---:|
| `python words.ppy` | 1.45 |
| `ppy run words.ppy`, after the first run built the cache | 0.66 |
| `./dist/words`, the standalone binary | 0.64 |

Timed one part at a time inside the program, in seconds, the mean of five
runs:

| part | CPython | `ppy run` |
|---|---:|---:|
| `make_words` | 0.154 | 0.106 |
| `letter_counts` | 0.134 | 0.098 |
| `nearest` (22,350 `distance` calls) | 0.606 | 0.235 |
| `cheapest(0, 250)` | 0.492 | 0.106 |

`cheapest` gains the most: each entry runs a loop of lookups into one
table. `nearest` gains less: every `distance` call makes a new table for
`d`, and most of its entries are computed once and read once, so the
native code spends much of its time on the table rather than on the
arithmetic. `make_words` and `letter_counts` gain the least, since most of
their work is making strings and counting them in dicts.
CPython's `cache` is written in C, and a hit costs it little more than a
call.

## Where the code comes from

`words.ppy` is hand-written.

Read on: [the standard library guide](../../docs/guide/stdlib.md).
