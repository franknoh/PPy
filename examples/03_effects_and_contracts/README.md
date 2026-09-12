# Effects and contracts

What `@ppy.pure` refuses, and how a value from a dynamic boundary gets back
into typed code.

## Effects are a set

Every function carries an inferred effect set — I/O, global writes,
randomness, time, mutation of arguments, each tracked on its own. A
`@ppy.pure` decorator is a claim against that set, and the checker either
proves it or names the effect that breaks it.

```python
@ppy.pure
def clamp(x: Annotated[int, ppy.Range(0, 255)]) -> Annotated[int, ppy.Range(0, 255)]:
    if x < 0:
        return 0
    if x > 255:
        return 255
    return x
```

`ppy.Range(0, 255)` is a refinement the checker propagates: inside `clamp`
the arithmetic on `x` needs no overflow guard, and a caller passing `300` is
told so at check time. `scale` builds a new list and is pure — local
allocation is fine, mutating an argument is not. `mix` is float arithmetic
on `f32`, pure by construction. `log_and_total` prints, so it carries `io`;
with `@ppy.pure` on it the checker answers `E1601` with the effect named.
The rule is interprocedural: a pure function that calls something with
unknown effects is `E1602`, and a callee that mutates the caller's argument
charges the write to the caller.

## A boundary for the dynamic part

```python
@ppy.dynamic
def evaluate(source: str) -> int:
    with ppy.dynamic:
        return ppy.check[int](eval(source))
```

`eval` is rejected in strict code (`E1501`). Inside a `ppy.dynamic` boundary
it is allowed, but what comes out is `Dynamic`, and `Dynamic` does not fit a
declared `-> int`. `ppy.check[int]` validates the value at run time — raising
`TypeError` if it is not an `int` — and hands back a typed one. The boundary
suspends the dynamic-feature rules; it does not suspend the types.

## Run it

```bash
python  effects.ppy
ppy     effects.ppy
ppy run effects.ppy
```

<!-- outputs:start -->
## What it prints

**`python  effects.ppy`**, **`ppy     effects.ppy`**, **`ppy run effects.ppy`**

```text
[2.0, 4.0] 255 1.5
1
2
3
42
```

<!-- outputs:end -->

Read on: [Effects and purity](../../docs/guide/effects.md) ·
[The subset](../../docs/guide/subset.md) ·
[Dynamic boundaries](../16_dynamic/README.md)

`effects.ppy` is hand-written; there is no `.py` source and no conversion step.
