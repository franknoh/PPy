# Generics, inferred at the call and monomorphized in native code

`largest(1, 2)` is an `int`. `largest(1.5, 2.5)` is a `float`. One
definition, Python 3.12's `def largest[T: int | float]` syntax, and the
call site decides `T`, checks it against the bound, and gets the declared
return type with `T` substituted. Native code goes further: every tuple of
type arguments a native caller uses becomes its own compiled instance, and
the call goes straight to it.

## Bounds lend what they promise

```python
class Named(Protocol):
    def name(self) -> str: ...


def label[T: Named](thing: T) -> str:
    return "at " + thing.name()
```

An unbounded `T` has no operators at all — nothing says it does. A bound
of `int | float` lends comparison and arithmetic; a bound that is a
`Protocol` lends its methods, so `label` may call `name()`, and `Point`,
whose members cover `Named`'s, is an instance of it without declaring so.
A type argument that does not satisfy its bound is `E1721`.

## One instance per tuple of type arguments

```python
def sweep(n: int) -> float:
    total = 0.0
    for i in range(n):
        total += clamp(i * 0.25, 1.0, 10.0) + largest(i, 3)
    return total
```

`sweep` is native and calls `clamp` with floats and `largest` with ints.
Each generic is lowered once per tuple of type arguments under a name that
spells them — `ppy emit ir` shows the instances, marked `ppy.generic` — and
the calls inside `sweep` are direct native calls to those instances. The
generics keep their Python bodies for every other caller.
`[tool.ppy.generics]` bounds the process (`max-specializations` per
generic, `max-depth` of a type argument), and a generic that calls itself
with its own parameter wrapped in a type is refused outright (`E1723`)
because its specializations would never end.

## Run it

```bash
python  generic.ppy
ppy run generic.ppy
ppy emit ir generic.ppy
```

<!-- outputs:start -->
## What it prints

**`python  generic.ppy`**

```text
2 2.5 at (3, 4) 7
10 0.0 509303.5
```

**`ppy run generic.ppy`**

```text
2 2.5 at (3, 4) 7
10 0.0 509303.5
```

**`ppy emit ir generic.ppy`**

```text
ppyir 1
module @generic
dialect core 1

func @generic_sweep(%n: i64) -> f64 attrs {effects = ["may_raise"], ppy.abi = "ppy", ppy.qualname = "generic.sweep", ppy.releases_gil = true, ppy.symbol = "ppy_generic_sweep"} loc("examples/40_generics/generic.ppy":33:0) {
^entry:
    %n_addr = core.alloca : ptr<i64, stack> loc("examples/40_generics/generic.ppy":33:0)
    core.store %n, %n_addr
    %0 = core.const 0.0 : f64 loc("examples/40_generics/generic.ppy":34:4)
    %total_addr = core.alloca : ptr<f64, stack>
    core.store %0, %total_addr
    %1 = core.load %n_addr : i64 loc("examples/40_generics/generic.ppy":35:4)
    %n_entry = core.load %n_addr : i64
    %2 = core.const 0 : i64
    %3 = core.const 1 : i64
    %i_addr = core.alloca : ptr<i64, stack>
    %4 = core.const 3 : i64
    core.store %2, %i_addr loc("examples/40_generics/generic.ppy":35:4)
    core.br ^for.head3
^for.head3:
```

*97 lines in all — [full output](outputs/03-ppy-emit-ir-generic-ppy.txt).*

<!-- outputs:end -->

## Read on

- [Generics](../../docs/guide/generics.md) — bounds, monomorphization, static dispatch.
- [Value classes](../13_value_classes/README.md) — operators on a value class, dispatched statically.

`generic.ppy` is hand-written; there is no `.py` source and no conversion step.
