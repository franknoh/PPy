# Generics

Type parameters the way Python 3.12 spells them, inferred at each call and
checked against their bounds; native code monomorphizes, one instance per
tuple of type arguments.

## Provenance

Hand-written. `generic.ppy` is written directly; there is no `.py` source
and no conversion step involved.

## What it shows

- `largest(1, 2)` is an `int` and `largest(1.5, 2.5)` a `float`: the call
  infers `T`, checks it against `int | float` (`E1721` otherwise), and has
  the declared return type with `T` substituted.
- A bound that is a `Protocol` lends its methods: `label` may call `name()`
  because `Named` says so, a class whose members cover the protocol's is an
  instance of it, and an unbounded `T` has no operators at all.
- `sweep` is native and calls `clamp` and `largest` with `float` and `int`
  arguments: each generic is lowered once per tuple of type arguments under a
  name that spells them (`ppy emit ir` shows them, marked `ppy.generic`), and
  the calls go straight to those instances. The generics keep their Python
  bodies for every other caller.
- `[tool.ppy.generics]` bounds the process -- `max-specializations` per
  generic and `max-depth` of a type argument -- and a generic that calls
  itself with its own parameter wrapped in a type is refused outright.

## Run it

```bash
python  generic.ppy
ppy run generic.ppy
ppy emit ir generic.ppy   # the monomorphized instances
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
    %5 = core.load %i_addr : i64 loc("examples/40_generics/generic.ppy":35:4)
    %6 = core.cmp.lt %5, %1 : bool
    core.cond_br %6, ^for.body4, ^for.end6
^for.body4:
    %7 = core.load %total_addr : f64 loc("examples/40_generics/generic.ppy":36:8)
    %8 = core.load %i_addr : i64
    %9 = core.const 0.25 : f64
    %10 = core.cast %8 : f64
    %11 = core.mul %10, %9 : f64
    %12 = core.const 1.0 : f64
    %13 = core.const 10.0 : f64
    %14 = core.call %11, %12, %13 {callee = @generic_clamp__float} : f64
    %15 = core.load %i_addr : i64
    %16 = core.call %15, %4 {callee = @generic_largest__int} : i64
    %17 = core.cast %16 : f64
    %18 = core.add %14, %17 : f64
    %19 = core.add %7, %18 : f64
    core.store %19, %total_addr
    %20 = core.load %i_addr : i64
    %21 = core.add %20, %3 {overflow = "python"} : i64
… 57 more lines
```

<!-- outputs:end -->
