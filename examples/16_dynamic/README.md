# Dynamic boundaries

The escape hatch, and what it costs. Strict PPY rejects `getattr` with a
computed name, attributes it cannot resolve, and values it cannot type;
`ppy.dynamic` is where a piece of code is declared Python in the old sense,
and the checker holds the line at its edge.

## Three spellings

```python
def reflective(name: str) -> str:
    with ppy.dynamic():
        return str(getattr("abc", name)())


@ppy.dynamic
def duck_typed(obj: object) -> str:
    return f"{obj.__class__.__name__}"


def normalize(payload: ppy.Dynamic) -> int:
    return int(payload)
```

As a context manager, `ppy.dynamic()` permits the computed `getattr` for a
block. As a decorator, it makes a whole function dynamic: `obj.__class__` on
an `object` is fine inside. As an annotation, `ppy.Dynamic` marks a value
that arrives untyped — `Any` at run time, but spelled as a decision rather
than an inference failure. `int(payload)` is the conversion that turns it
into something typed; `ppy.check[int]` is the checked form. Nothing dynamic
leaks out untyped.

## What a boundary costs

A boundary is an optimization barrier. Native code stops at it, and
`static_only` — the plain function in the file — is the one that lowers. A
project that wants no boundaries sets `[tool.ppy] dynamic-boundaries =
"deny"` and gets `E1505` at the first one. Migration goes the other way:
`ppy migrate` converts a dynamic file faithfully and marks each site, and
`ppy check` then asks for the boundary.

## Run it

```bash
python  dynamic.ppy
ppy     dynamic.ppy
ppy run dynamic.ppy
```

<!-- outputs:start -->
## What it prints

**`python  dynamic.ppy`**, **`ppy     dynamic.ppy`**, **`ppy run dynamic.ppy`**

```text
10
ABC
int str list
42
```

<!-- outputs:end -->

Read on: [The subset](../../docs/guide/subset.md) ·
[Effects and contracts](../03_effects_and_contracts/README.md) ·
[Migration](../30_migrate/README.md)

`dynamic.ppy` is hand-written; there is no `.py` source and no conversion step.
