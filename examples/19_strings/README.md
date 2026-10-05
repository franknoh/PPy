# Strings

Three small string functions, checked in strict mode and proven pure, and
all three compiled to native code: `initials` splits and uppercases,
`is_palindrome` indexes a string from both ends, and `word_count` counts
words into a `dict`. Python calls the first two natively. It calls
`word_count` on its Python body, and `ppy explain` says why.

## Run it

```bash
python  strings.ppy
ppy     strings.ppy
ppy run strings.ppy
```

<!-- outputs:start -->
## What it prints

**`python  strings.ppy`**, **`ppy     strings.ppy`**, **`ppy run strings.ppy`**

```text
ALK
True False
[('a', 3), ('b', 2), ('c', 1)]
```

<!-- outputs:end -->

## Which functions lower

```bash
ppy explain strings.initials        # llvm backend: native
ppy explain strings.is_palindrome   # llvm backend: native
ppy explain strings.word_count      # llvm backend: native; called from Python, its Python body runs: copying its strings across costs what one pass over them saves
```

`is_palindrome` indexes `cleaned[left]` and `cleaned[right]`. Natively a
string knows its length in code points and whether it is all ASCII, so an
index into ASCII text is one step, and the one-character strings the
comparison reads are static and never allocated.

`word_count` makes one pass over its text. Called from Python, the text
itself is borrowed, but every word in the `dict` it returns is a string
made natively and then decoded into a Python string, which costs about
what that one pass saves, so the cost model leaves Python's call on the
Python body; a native caller calls it natively. The
[Strings example](../48_strings/README.md) counts words over 200,000 lines,
where the work is worth the copy.

## Very large text

Text of millions of characters can still come in as bytes:
`Buffer[ppy.u8]` is one byte per character and is read without building a
Python string, which is how
[substring search](../15_algorithms/15c_kmp/README.md) scans four million
characters.

## Where the code comes from

`strings.ppy` is hand-written; there is no `.py` source and no conversion step.

Read on: [Strings](../../docs/guide/strings.md) ·
[Reading input](../../docs/guide/input.md) ·
[Native lowering](../../docs/guide/native-lowering.md) ·
[Algorithms: substring search](../15_algorithms/15c_kmp/README.md)
