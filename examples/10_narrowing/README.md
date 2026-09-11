# Narrowing

Every form the checker narrows on, one function each. Flow typing is what
lets a strict checker accept ordinary Python: after `if name is None:
return 0`, `name` is a `str`; after `isinstance(value, str)` fails, `value`
is an `int`.

## Guards, boolean operators, the walrus

```python
@ppy.pure
def boolean_guard(name: str | None) -> bool:
    return name is not None and len(name) > 0


@ppy.pure
def walrus_guard(values: list[int]) -> int:
    if (count := len(values)) == 0:
        return 0
    return count
```

`and` narrows its right operand: `len(name)` is checked with `name: str`.
`or` narrows the code after it: in `early_default`, the body after
`if name is None or len(name) == 0: return -1` sees a non-empty `str`. A
walrus binding is a local the checker tracks like any other. `by_class`
narrows with `isinstance`.

## `match`

```python
@ppy.pure
def by_match(value: int | str | None) -> str:
    match value:
        case None:
            return "none"
        case int():
            return f"int:{value + 1}"
        case _:
            return f"str:{value.upper()}"
```

A case sees the subject with the earlier cases subtracted: the `int()` case
sees `int | str`, the wildcard sees only `str`, so `value.upper()` checks
without a cast. A case with a guard rules nothing out for the cases after
it, because a guard can fail for reasons the pattern does not express. All
six functions are pure and native.

## Run it

```bash
python  narrowing.ppy
ppy     narrowing.ppy
ppy run narrowing.ppy
```

<!-- outputs:start -->
## What it prints

**`python  narrowing.ppy`**, **`ppy     narrowing.ppy`**, **`ppy run narrowing.ppy`**

```text
3 0
True False False
3 -1 -1
4 7
none int:42 str:HI
3 0
```

<!-- outputs:end -->

Read on: [Classes](../04_classes/README.md) ·
[The subset](../../docs/guide/subset.md)

`narrowing.ppy` is hand-written; there is no `.py` source and no conversion
step.
