# Strings

Three small string functions, checked in strict mode and proven pure. Two
of them go native: `initials` splits and uppercases, and `is_palindrome`
indexes a string from both ends. The third returns a `dict`, which has no
native form, so it runs as Python, and `ppy explain` says why.

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
ppy explain strings.word_count      # llvm backend: boxed: returns `dict[str, int]`, which has no native ABI
```

`is_palindrome` indexes `cleaned[left]` and `cleaned[right]`. Natively a
string knows its length in code points and whether it is all ASCII, so an
index into ASCII text is one step, and the one-character strings the
comparison reads are static and never allocated.

`word_count` would lower with a `HashMap[str, int]` in place of the `dict`;
the [Strings example](../48_strings/README.md) counts that way.

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
