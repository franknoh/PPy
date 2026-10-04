# Generics

Generic functions use the Python 3.12 type-parameter syntax, and PPy infers
the type arguments at each call and checks them against their bounds. Native
code monomorphizes: one instance per tuple of type arguments.

## Run it

```bash
python  generic.ppy
ppy run generic.ppy
ppy emit ir generic.ppy
```

## Bounds lend what they promise

```python
class Named(Protocol):
    def name(self) -> str: ...


def label[T: Named](thing: T) -> str:
    return "at " + thing.name()
```

`largest(1, 2)` is an `int` and `largest(1.5, 2.5)` a `float`. The call
infers `T`, checks it against the bound `int | float` (`E1721` otherwise),
and gets the declared return type with `T` substituted.

- An unbounded `T` has no operators at all.
- A bound that is a `Protocol` lends its methods, so `label` may call
  `name()`.
- `Point`, whose members cover `Named`'s, is an instance of `Named` without
  declaring so.

## One instance per tuple of type arguments

```python
def sweep(n: int) -> float:
    total = 0.0
    for i in range(n):
        total += clamp(i * 0.25, 1.0, 10.0) + largest(i, 3)
    return total
```

`sweep` is native and calls `clamp` with floats and `largest` with ints.
Each generic is lowered once per tuple of type arguments, under a name that
spells them. `ppy emit ir` shows the instances, marked `ppy.generic`. The
calls inside `sweep` go straight to those instances, and the generics keep
their Python bodies for every other caller.

## Limits on specialization

`[tool.ppy.generics]` bounds the process:

- `max-specializations`: instances per generic.
- `max-depth`: depth of a type argument.

A generic that calls itself with its own parameter wrapped in a type is
refused outright (`E1723`), because its specializations would never end.

## Compared with Numba

`sweep` over eight million values, in [`compare/`](compare/):

- [`generic_bench.ppy`](compare/generic_bench.ppy), under `ppy run` and,
  the same file, under `python`
- [`generic_numba.py`](compare/generic_numba.py)

Times are milliseconds, best of five, over five processes.

**PPy** is the generics as Python 3.12 spells them, with bounds, called from
a plain function. **Numba** is the same three functions under `@njit`,
generic by dispatch: each is compiled once per tuple of argument types it
meets. That is the same instance-per-type-tuple rule without the
declaration.

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
| | PPy `ppy run` | CPython, the same file | Numba `@njit` |
|---|---:|---:|---:|
| sweep, eight million clamps and comparisons | 5.87 ± 0.03 | 428.92 ± 9.67 | **5.09 ± 0.06** |
<!-- compare:end -->

Both compile `largest` twice (once for ints, once for floats) and call the
instances directly from the loop, and the loop is the same code either way.

- PPy adds a check at the source. The bound is written, so
  `largest("a", 1)` is refused before anything runs, and the file is still
  the file `python` runs.
- Numba asks you to write nothing. The types are whatever arrives first.

Intel Core Ultra 9 386H; Numba 0.67.0 on CPython 3.12.13, PPy on CPython
3.14.5, from a checkout on a native filesystem.

<!-- outputs:start -->
## What it prints

**`python  generic.ppy`**, **`ppy run generic.ppy`**

```text
2 2.5 at (3, 4) 7
10 0.0 509303.5
```

**`ppy emit ir generic.ppy`**

<details markdown="1">
<summary>273 lines</summary>

```text
ppyir 1
module @generic
dialect core 1
attrs {ppy.libraries = ["ppy_collections"]}

private global @ppy.str.0 : buffer<u8> = "("
private global @ppy.str.1 : buffer<u8> = ", "
private global @ppy.str.2 : buffer<u8> = ")"

func @generic_Point___init__(%self: ptr<i8>, %x: i64, %y: i64) -> () attrs {effects = ["write_memory", "write_object"], ppy.abi = "ppy", ppy.qualname = "generic.Point.__init__", ppy.releases_gil = false, ppy.symbol = "ppy_generic_Point___init__"} loc("examples/40_generics/generic.ppy":9:4) {
^entry:
    %self_addr = core.alloca {ppy.owns = true} : ptr<ptr<i8>, stack> loc("examples/40_generics/generic.ppy":9:4)
    core.store %self, %self_addr
    core.call_extern %self {abi = "c", callee = "ppy_coll_retain"}
    %x_addr = core.alloca : ptr<i64, stack>
    core.store %x, %x_addr
    %y_addr = core.alloca : ptr<i64, stack>
    core.store %y, %y_addr
    %0 = core.load %self_addr : ptr<i8> loc("examples/40_generics/generic.ppy":10:8)
    %1 = core.cast %0 : i64
    %2 = core.const 0 : i64
    %3 = core.cmp.ne %1, %2 : bool
    core.guard %3 {kind = "bounds", message = "`None` has no attribute `x`", raises = "AttributeError: 'NoneType' object has no attribute 'x'"}
    %4 = core.cast %0 {ppy.reads = true} : ptr<i64>
    %5 = core.const 3 : i64
    %6 = core.ptr_offset %4, %5 : ptr<i64>
    %7 = core.load %6 : i64
    %8 = core.cast %0 {ppy.reads = true} : ptr<i64>
    %9 = core.const 1 : i64
    %10 = core.ptr_offset %8, %9 : ptr<i64>
    %11 = core.load %10 : i64
    %12 = core.cast %0 {ppy.reads = true} : ptr<i64>
    %13 = core.const 15 : i64
    %14 = core.ptr_offset %12, %13 : ptr<i64>
    %15 = core.load %14 : i64
    %16 = core.cast %0 {ppy.reads = true} : ptr<ptr<i64>>
    %17 = core.const 2 : i64
    %18 = core.ptr_offset %16, %17 : ptr<ptr<i64>>
    %19 = core.load %18 : ptr<i64>
    %20 = core.cmp.ge %7, %11 : bool
    %21 = core.sub %7, %11 {overflow = "wrap"} : i64
    %22 = core.select %20, %21, %7 : i64
    %23 = core.mul %22, %15 {overflow = "wrap"} : i64
    %24 = core.ptr_offset %19, %23 : ptr<i64>
    %25 = core.cast %24 : ptr<i8>
    %26 = core.load %x_addr : i64
    %x_entry = core.load %x_addr : i64
    %27 = core.cast %25 : ptr<i64>
    core.store %26, %27
    %28 = core.load %self_addr : ptr<i8> loc("examples/40_generics/generic.ppy":11:8)
    %29 = core.cast %28 : i64
    %30 = core.const 0 : i64
    %31 = core.cmp.ne %29, %30 : bool
    core.guard %31 {kind = "bounds", message = "`None` has no attribute `y`", raises = "AttributeError: 'NoneType' object has no attribute 'y'"}
    %32 = core.cast %28 {ppy.reads = true} : ptr<i64>
    %33 = core.const 3 : i64
    %34 = core.ptr_offset %32, %33 : ptr<i64>
    %35 = core.load %34 : i64
    %36 = core.cast %28 {ppy.reads = true} : ptr<i64>
    %37 = core.const 1 : i64
    %38 = core.ptr_offset %36, %37 : ptr<i64>
    %39 = core.load %38 : i64
    %40 = core.cast %28 {ppy.reads = true} : ptr<i64>
    %41 = core.const 15 : i64
    %42 = core.ptr_offset %40, %41 : ptr<i64>
    %43 = core.load %42 : i64
    %44 = core.cast %28 {ppy.reads = true} : ptr<ptr<i64>>
    %45 = core.const 2 : i64
    %46 = core.ptr_offset %44, %45 : ptr<ptr<i64>>
    %47 = core.load %46 : ptr<i64>
    %48 = core.cmp.ge %35, %39 : bool
    %49 = core.sub %35, %39 {overflow = "wrap"} : i64
    %50 = core.select %48, %49, %35 : i64
    %51 = core.mul %50, %43 {overflow = "wrap"} : i64
    %52 = core.ptr_offset %47, %51 : ptr<i64>
    %53 = core.cast %52 : ptr<i8>
    %54 = core.cast %53 : ptr<i64>
    %55 = core.const 1 : i64
    %56 = core.ptr_offset %54, %55 : ptr<i64>
    %57 = core.cast %56 : ptr<i8>
    %58 = core.load %y_addr : i64
    %y_entry = core.load %y_addr : i64
    %59 = core.cast %57 : ptr<i64>
    core.store %58, %59
    %60 = core.load %self_addr : ptr<i8>
    core.call_extern %60 {abi = "c", callee = "ppy_coll_release"}
    core.ret
}

func @generic_Point_name(%self: ptr<i8>) -> ptr<i8> attrs {effects = ["alloc", "read_object"], ppy.abi = "ppy", ppy.qualname = "generic.Point.name", ppy.releases_gil = false, ppy.symbol = "ppy_generic_Point_name"} loc("examples/40_generics/generic.ppy":13:4) {
^entry:
    %self_addr = core.alloca {ppy.owns = true} : ptr<ptr<i8>, stack> loc("examples/40_generics/generic.ppy":13:4)
    core.store %self, %self_addr
    core.call_extern %self {abi = "c", callee = "ppy_coll_retain"}
    %0 = core.const 0 : i64 loc("examples/40_generics/generic.ppy":14:8)
    %1 = core.call_extern %0 {abi = "c", callee = "ppy_str_builder"} : ptr<i8>
    %2 = core.call_intrinsic {intrinsic = "ppy.string_data", symbol = "ppy.str.0"} : ptr<u8>
    %3 = core.const 1 : i64
    core.call_extern %1, %2, %3 {abi = "c", callee = "ppy_str_add_bytes"}
    %4 = core.load %self_addr : ptr<i8>
    %5 = core.cast %4 : i64
    %6 = core.const 0 : i64
    %7 = core.cmp.ne %5, %6 : bool
    core.guard %7 {kind = "bounds", message = "`None` has no attribute `x`", raises = "AttributeError: 'NoneType' object has no attribute 'x'"}
    %8 = core.cast %4 {ppy.reads = true} : ptr<i64>
    %9 = core.const 3 : i64
    %10 = core.ptr_offset %8, %9 : ptr<i64>
    %11 = core.load %10 : i64
    %12 = core.cast %4 {ppy.reads = true} : ptr<i64>
    %13 = core.const 1 : i64
    %14 = core.ptr_offset %12, %13 : ptr<i64>
    %15 = core.load %14 : i64
    %16 = core.cast %4 {ppy.reads = true} : ptr<i64>
    %17 = core.const 15 : i64
    %18 = core.ptr_offset %16, %17 : ptr<i64>
    %19 = core.load %18 : i64
    %20 = core.cast %4 {ppy.reads = true} : ptr<ptr<i64>>
    %21 = core.const 2 : i64
    %22 = core.ptr_offset %20, %21 : ptr<ptr<i64>>
    %23 = core.load %22 : ptr<i64>
    %24 = core.cmp.ge %11, %15 : bool
    %25 = core.sub %11, %15 {overflow = "wrap"} : i64
    %26 = core.select %24, %25, %11 : i64
    %27 = core.mul %26, %19 {overflow = "wrap"} : i64
    %28 = core.ptr_offset %23, %27 : ptr<i64>
    %29 = core.cast %28 : ptr<i8>
    %30 = core.cast %29 : ptr<i64>
    %31 = core.load %30 : i64
    core.call_extern %1, %31 {abi = "c", callee = "ppy_str_add_int"}
    %32 = core.call_intrinsic {intrinsic = "ppy.string_data", symbol = "ppy.str.1"} : ptr<u8>
    %33 = core.const 2 : i64
    core.call_extern %1, %32, %33 {abi = "c", callee = "ppy_str_add_bytes"}
    %34 = core.load %self_addr : ptr<i8>
    %35 = core.cast %34 : i64
    %36 = core.const 0 : i64
    %37 = core.cmp.ne %35, %36 : bool
    core.guard %37 {kind = "bounds", message = "`None` has no attribute `y`", raises = "AttributeError: 'NoneType' object has no attribute 'y'"}
    %38 = core.cast %34 {ppy.reads = true} : ptr<i64>
    %39 = core.const 3 : i64
    %40 = core.ptr_offset %38, %39 : ptr<i64>
    %41 = core.load %40 : i64
    %42 = core.cast %34 {ppy.reads = true} : ptr<i64>
    %43 = core.const 1 : i64
    %44 = core.ptr_offset %42, %43 : ptr<i64>
    %45 = core.load %44 : i64
    %46 = core.cast %34 {ppy.reads = true} : ptr<i64>
    %47 = core.const 15 : i64
    %48 = core.ptr_offset %46, %47 : ptr<i64>
    %49 = core.load %48 : i64
    %50 = core.cast %34 {ppy.reads = true} : ptr<ptr<i64>>
    %51 = core.const 2 : i64
    %52 = core.ptr_offset %50, %51 : ptr<ptr<i64>>
    %53 = core.load %52 : ptr<i64>
    %54 = core.cmp.ge %41, %45 : bool
    %55 = core.sub %41, %45 {overflow = "wrap"} : i64
    %56 = core.select %54, %55, %41 : i64
    %57 = core.mul %56, %49 {overflow = "wrap"} : i64
    %58 = core.ptr_offset %53, %57 : ptr<i64>
    %59 = core.cast %58 : ptr<i8>
    %60 = core.cast %59 : ptr<i64>
    %61 = core.const 1 : i64
    %62 = core.ptr_offset %60, %61 : ptr<i64>
    %63 = core.cast %62 : ptr<i8>
    %64 = core.cast %63 : ptr<i64>
    %65 = core.load %64 : i64
    core.call_extern %1, %65 {abi = "c", callee = "ppy_str_add_int"}
    %66 = core.call_intrinsic {intrinsic = "ppy.string_data", symbol = "ppy.str.2"} : ptr<u8>
    %67 = core.const 1 : i64
    core.call_extern %1, %66, %67 {abi = "c", callee = "ppy_str_add_bytes"}
    %68 = core.call_extern %1 {abi = "c", callee = "ppy_str_finish"} : ptr<i8>
    %69 = core.load %self_addr : ptr<i8>
    core.call_extern %69 {abi = "c", callee = "ppy_coll_release"}
    core.ret %68
}

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
    %21 = core.add %20, %3 {overflow = "proven"} : i64
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
    core.cond_br %2, ^ifexp.then1, ^ifexp.else2
^ifexp.then1:
    %3 = core.load %lo_addr : f64 loc("examples/40_generics/generic.ppy":30:4)
    %4 = core.load %x_addr : f64
    %5 = core.call %3, %4 {callee = @generic_largest__float} : f64
    core.br ^ifexp.end3(%5)
^ifexp.else2:
    %6 = core.load %hi_addr : f64 loc("examples/40_generics/generic.ppy":30:4)
    core.br ^ifexp.end3(%6)
^ifexp.end3(%ifexp: f64):
    core.ret %ifexp loc("examples/40_generics/generic.ppy":30:4)
}

func @generic_largest__float(%a: f64, %b: f64) -> f64 attrs {effects = [], ppy.abi = "ppy", ppy.generic = "generic.largest", ppy.qualname = "generic.largest__float", ppy.releases_gil = false, ppy.symbol = "ppy_generic_largest__float", ppy.type_arguments = ["float"]} loc("examples/40_generics/generic.ppy":17:0) {
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

func @generic_largest__int(%a: i64, %b: i64) -> i64 attrs {effects = [], ppy.abi = "ppy", ppy.generic = "generic.largest", ppy.qualname = "generic.largest__int", ppy.releases_gil = false, ppy.symbol = "ppy_generic_largest__int", ppy.type_arguments = ["int"]} loc("examples/40_generics/generic.ppy":17:0) {
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
