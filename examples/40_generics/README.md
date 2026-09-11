# Generics

Type parameters the way Python 3.12 spells them, inferred at each call and
checked against their bounds; native code monomorphizes, one instance per
tuple of type arguments.

## Bounds lend what they promise

```python
class Named(Protocol):
    def name(self) -> str: ...


def label[T: Named](thing: T) -> str:
    return "at " + thing.name()
```

`largest(1, 2)` is an `int` and `largest(1.5, 2.5)` a `float`: the call
infers `T`, checks it against `int | float` (`E1721` otherwise), and has the
declared return type with `T` substituted. An unbounded `T` has no
operators at all. A bound that is a `Protocol` lends its methods, so `label`
may call `name()`, and `Point`, whose members cover `Named`'s, is an
instance of it without declaring so.

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
the calls inside `sweep` go straight to those instances. The generics keep
their Python bodies for every other caller. `[tool.ppy.generics]` bounds the
process (`max-specializations` per generic, `max-depth` of a type
argument), and a generic that calls itself with its own parameter wrapped in
a type is refused outright (`E1723`) because its specializations would never
end.

## Compared with Numba

`sweep` over eight million values, in [`compare/`](compare/):
[`generic_bench.ppy`](compare/generic_bench.ppy) -- under `ppy run` and, the
same file, under `python` -- and
[`generic_numba.py`](compare/generic_numba.py). Milliseconds, best of five,
over five processes.

**PPY** is the generics as Python 3.12 spells them, with bounds, called
from a plain function; **Numba** is the same three functions under
`@njit`, generic by dispatch: each is compiled once per tuple of argument
types it meets, which is the same instance-per-type-tuple rule without the
declaration:

```python
def largest[T: int | float](a: T, b: T) -> T:
    return a if a > b else b


def clamp[T: int | float](x: T, lo: T, hi: T) -> T:
    return largest(lo, x) if x < hi else hi
```

```python
@njit
def largest(a, b):
    return a if a > b else b


@njit
def clamp(x, lo, hi):
    return largest(lo, x) if x < hi else hi
```

<!-- compare:start -->
| | PPY `ppy run` | CPython, the same file | Numba `@njit` |
|---|---:|---:|---:|
| sweep, eight million clamps and comparisons | 5.77 ± 0.03 | 485.64 ± 62.82 | **5.04 ± 0.09** |
<!-- compare:end -->

Both compile `largest` twice -- once for ints, once for floats -- and call
the instances directly from the loop, and the loop is the same code
either way. What PPY adds is at the source: the bound is written, so
`largest("a", 1)` is refused before anything runs, and the file is still
the file `python` runs. What Numba adds is nothing to write; the types are
whatever arrives first.

Intel Core Ultra 9 386H; Numba 0.67.0 on CPython 3.12.13, PPY on CPython
3.14.5, from a checkout on a native filesystem.

## Run it

```bash
python  generic.ppy
ppy run generic.ppy
ppy emit ir generic.ppy
```

<!-- outputs:start -->
## What it prints

**`python  generic.ppy`**, **`ppy run generic.ppy`**

```text
2 2.5 at (3, 4) 7
10 0.0 509303.5
```

**`ppy emit ir generic.ppy`**

<details markdown="1">
<summary>97 lines</summary>

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
    core.store %21, %i_addr
    core.br ^for.head3
^for.end6:
    %22 = core.load %total_addr : f64 loc("examples/40_generics/generic.ppy":37:4)
    core.ret %22
}

func @generic_clamp__float(%x: f64, %lo: f64, %hi: f64) -> f64 attrs {effects = [], ppy.abi = "ppy", ppy.generic = "generic.clamp", ppy.qualname = "generic.clamp__float", ppy.releases_gil = true, ppy.symbol = "ppy_generic_clamp__float", ppy.type_arguments = ["float"]} loc("examples/40_generics/generic.ppy":29:0) {
^entry:
    %x_addr = core.alloca : ptr<f64, stack> loc("examples/40_generics/generic.ppy":29:0)
    core.store %x, %x_addr
    %lo_addr = core.alloca : ptr<f64, stack>
    core.store %lo, %lo_addr
    %hi_addr = core.alloca : ptr<f64, stack>
    core.store %hi, %hi_addr
    %0 = core.load %x_addr : f64 loc("examples/40_generics/generic.ppy":30:4)
    %1 = core.load %hi_addr : f64
    %2 = core.cmp.lt %0, %1 : bool
    %3 = core.load %lo_addr : f64
    %4 = core.load %x_addr : f64
    %5 = core.call %3, %4 {callee = @generic_largest__float} : f64
    %6 = core.load %hi_addr : f64
    %7 = core.select %2, %5, %6 : f64
    core.ret %7
}

func @generic_largest__float(%a: f64, %b: f64) -> f64 attrs {effects = [], ppy.abi = "ppy", ppy.generic = "generic.largest", ppy.qualname = "generic.largest__float", ppy.releases_gil = true, ppy.symbol = "ppy_generic_largest__float", ppy.type_arguments = ["float"]} loc("examples/40_generics/generic.ppy":17:0) {
^entry:
    %a_addr = core.alloca : ptr<f64, stack> loc("examples/40_generics/generic.ppy":17:0)
    core.store %a, %a_addr
    %b_addr = core.alloca : ptr<f64, stack>
    core.store %b, %b_addr
    %0 = core.load %a_addr : f64 loc("examples/40_generics/generic.ppy":18:4)
    %1 = core.load %b_addr : f64
    %2 = core.cmp.gt %0, %1 : bool
    %3 = core.load %a_addr : f64
    %4 = core.load %b_addr : f64
    %5 = core.select %2, %3, %4 : f64
    core.ret %5
}

func @generic_largest__int(%a: i64, %b: i64) -> i64 attrs {effects = [], ppy.abi = "ppy", ppy.generic = "generic.largest", ppy.qualname = "generic.largest__int", ppy.releases_gil = true, ppy.symbol = "ppy_generic_largest__int", ppy.type_arguments = ["int"]} loc("examples/40_generics/generic.ppy":17:0) {
^entry:
    %a_addr = core.alloca : ptr<i64, stack> loc("examples/40_generics/generic.ppy":17:0)
    core.store %a, %a_addr
    %b_addr = core.alloca : ptr<i64, stack>
    core.store %b, %b_addr
    %0 = core.load %a_addr : i64 loc("examples/40_generics/generic.ppy":18:4)
    %a_entry = core.load %a_addr : i64
    %1 = core.load %b_addr : i64
    %b_entry = core.load %b_addr : i64
    %2 = core.cmp.gt %0, %1 : bool
    %3 = core.load %a_addr : i64
    %4 = core.load %b_addr : i64
    %5 = core.select %2, %3, %4 : i64
    core.ret %5
}
```

</details>

<!-- outputs:end -->

Read on: [Generics](../../docs/guide/generics.md) ·
[Value classes](../13_value_classes/README.md)

`generic.ppy` is hand-written; there is no `.py` source and no conversion step.
