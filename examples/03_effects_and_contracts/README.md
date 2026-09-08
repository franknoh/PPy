# What `@ppy.pure` refuses

Every function in a `.ppy` file carries an inferred effect set — I/O, global
writes, randomness, time, mutation of arguments, each tracked on its own. A
`@ppy.pure` decorator is a claim against that set, and the checker either
proves it or names the effect that breaks it. Five functions here make the
claim; the one that prints does not, and could not.

## Contracts the checker can hold

```python
@ppy.pure
def clamp(x: Annotated[int, ppy.Range(0, 255)]) -> Annotated[int, ppy.Range(0, 255)]:
    if x < 0:
        return 0
    if x > 255:
        return 255
    return x
```

`ppy.Range(0, 255)` is a refinement the checker propagates: inside `clamp`,
arithmetic on `x` needs no overflow guard, and a caller passing `300` is told
so at check time rather than at run time. `scale` builds a new list and is
pure — local allocation is fine, mutating an argument is not. `mix` is float
arithmetic on `f32`, pure by construction.

`log_and_total` prints, so it carries `io`. Put `@ppy.pure` on it and the
checker answers `E1601` with the effect named. The rule is interprocedural:
a pure function that calls something with unknown effects is `E1602`, and
a callee that mutates the caller's argument charges the write to the caller.

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
suspends the dynamic-feature rules; it never suspends the types.

## Run it

```bash
python  effects.ppy
ppy     effects.ppy
ppy run effects.ppy
```

<!-- outputs:start -->
## What it prints

**`python  effects.ppy`**

```text
[2.0, 4.0] 255 1.5
1
2
3
42
```

**`ppy     effects.ppy`**

```text
[2.0, 4.0] 255 1.5
1
2
3
42
```

**`ppy run effects.ppy`**

```text
[2.0, 4.0] 255 1.5
1
2
3
42
```

<!-- outputs:end -->

## Read on

- [Effects and purity](../../docs/guide/effects.md) — the effect vocabulary every pass reads.
- [The subset](../../docs/guide/subset.md) — unknown, `Any`, and `Dynamic`, held apart on purpose.
- [Dynamic boundaries](../16_dynamic/README.md) — more of what a boundary permits.

`effects.ppy` is hand-written; there is no `.py` source and no conversion step.
