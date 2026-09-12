# Strings

String work stays on CPython, and the compiler says so. String methods are
fully typed — `name.split(" ")` is a `list[str]`, `part[0].upper()` a
`str` — so the code checks in strict mode and is proven pure, but a Python
string has no native representation, so these functions run as ordinary
Python on every path.

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

The loop is native-shaped, but `cleaned[left]` indexes a `str`. The
function is pure — the checker proves that — and it stays on the Python
side. `ppy explain strings.ppy:is_palindrome` reports the first construct
that blocked lowering.

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

**`python  strings.ppy`**, **`ppy     strings.ppy`**, **`ppy run strings.ppy`**

```text
ALK
True False
[('a', 3), ('b', 2), ('c', 1)]
```

<!-- outputs:end -->

Read on: [Reading input](../../docs/guide/input.md) ·
[Native lowering](../../docs/guide/native-lowering.md) ·
[Algorithms: substring search](../15_algorithms/15c_kmp/README.md)

`strings.ppy` is hand-written; there is no `.py` source and no conversion step.
