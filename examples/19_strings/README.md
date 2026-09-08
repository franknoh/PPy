# Strings stay on CPython, and say so

PPY does not pretend to lower `str.split`. String methods are fully typed —
`name.split(" ")` is a `list[str]`, `part[0].upper()` a `str` — so the code
checks in strict mode, and it runs as ordinary Python on every path. The
compiler's job here is to say clearly which calls kept a function boxed,
and `ppy explain` does.

## Typed, checked, not lowered

```python
@ppy.pure
def is_palindrome(text: str) -> bool:
    cleaned: str = text.lower().replace(" ", "")
    left: int = 0
    right: int = len(cleaned) - 1
    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1
    return True
```

The loop is native-shaped, but `cleaned[left]` indexes a `str`, and a
Python string has no native representation. The function is pure — the
checker proves that — and it stays on the Python side.

```bash
ppy explain strings.ppy:is_palindrome    # the first construct that blocked lowering
```

Text that needs to be fast goes through a byte buffer instead:
`Buffer[ppy.u8]` is one byte per character and lowers, which is how
[substring search](../15_algorithms/15c_kmp/README.md) scans four million
characters natively.

## Run it

```bash
python  strings.ppy
ppy     strings.ppy
ppy run strings.ppy
```

<!-- outputs:start -->
## What it prints

**`python  strings.ppy`**

```text
ALK
True False
[('a', 3), ('b', 2), ('c', 1)]
```

**`ppy     strings.ppy`**

```text
ALK
True False
[('a', 3), ('b', 2), ('c', 1)]
```

**`ppy run strings.ppy`**

```text
ALK
True False
[('a', 3), ('b', 2), ('c', 1)]
```

<!-- outputs:end -->

## Read on

- [Input and native lowering](../../docs/guide/native-lowering.md) — what lowers, what does not, and `ppy explain`.
- [Algorithms: substring search](../15_algorithms/15c_kmp/README.md) — text as bytes, at C speed.

`strings.ppy` is hand-written; there is no `.py` source and no conversion step.
