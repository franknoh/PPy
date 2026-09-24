# Strings

String code checks in strict mode and is proven pure, but it stays on
CPython, and the compiler says so. String methods are fully typed
(`name.split(" ")` is a `list[str]`, `part[0].upper()` a `str`). A Python
string has no native representation, though, so these functions run as
ordinary Python on every path.

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

The loop is native-shaped, but `cleaned[left]` indexes a `str`. The checker
proves the function pure, and it stays on the Python side.
`ppy explain strings.ppy:is_palindrome` reports the first construct that
blocked lowering.

## Fast text goes through bytes

Text that needs to be fast goes through a byte buffer instead.
`Buffer[ppy.u8]` is one byte per character and lowers, which is how
[substring search](../15_algorithms/15c_kmp/README.md) scans four million
characters natively.

## Where the code comes from

`strings.ppy` is hand-written; there is no `.py` source and no conversion step.

Read on: [Reading input](../../docs/guide/input.md) ·
[Native lowering](../../docs/guide/native-lowering.md) ·
[Algorithms: substring search](../15_algorithms/15c_kmp/README.md)
