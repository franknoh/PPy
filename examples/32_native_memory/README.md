# Native memory

Typed pointers, memory a function owns, a C function bound from `libm`, and
a function exported as a C symbol — each with a reference implementation
under plain CPython.

## Pointers

```python
@ppy.native
def fill(p: native.ptr[float], n: int, scale: float) -> float:
    total = 0.0
    for i in range(n):
        q = native.offset(p, i)
        native.store(q, scale * i)
        total += native.load(q)
    return total
```

`native.ptr[T]` and `native.const_ptr[T]` are typed pointers; `load`,
`store`, and `offset` read, write, and move. A store through a `const_ptr`
is `E1631`. `native.cast[U](p)` reads the same memory as another element
type — `bytes_of(native.cast[ppy.u8](memory), 8)` sums the first eight
bytes of the four doubles `fill` wrote. `sizeof` and `alignof` are
constants the checker knows.

## Memory the function owns

`native.stack_alloc[int](8)` is eight zeroed words on the function's own
stack. Native code needs a constant count, and the pointer cannot be
returned or stored where it outlives the call: the IR's verifier refuses a
`core.ret` or a `core.store` of a stack pointer, the same way it refuses a
borrowed parameter.

## Into C, and out of it

```python
libm = ffi.library("m")


@ffi.bind(libm, symbol="hypot", pure=True)
def c_hypot(x: float, y: float) -> float: ...


@native.export(name="ppy_norm")
def norm(p: native.const_ptr[float], n: int) -> float:
```

`@ffi.bind` makes a stub a C function: ctypes under CPython, a direct call
in native code, and `ppy emit ir` shows the `core.call_extern` with
`ppy.libraries = ("m",)` on the module. `@native.export` makes `norm` a
public symbol in the library `ppy build` writes, declared in the header
beside it. A C caller has no Python to fall back to, so a failed guard
traps. `ppy bind header foo.h` writes a module of such bindings from a C
header.

## Run it

```bash
python  native_memory.ppy
ppy run native_memory.ppy
ppy emit ir native_memory.ppy
```

<!-- outputs:start -->
## What it prints

**`python  native_memory.ppy`**, **`ppy run native_memory.ppy`**

```text
9.0 93 0
5.612486 5.612486
```

**`ppy emit ir native_memory.ppy`**

<details markdown="1">
<summary>202 lines</summary>

```text
ppyir 1
module @native_memory
dialect core 1
attrs {ppy.libraries = ["m"]}

func @native_memory_fill(%p: ptr<f64>, %n: i64, %scale: f64) -> f64 attrs {effects = ["may_raise", "read_memory", "write_memory"], ppy.abi = "ppy", ppy.qualname = "native_memory.fill", ppy.releases_gil = true, ppy.symbol = "ppy_native_memory_fill"} loc("examples/32_native_memory/native_memory.ppy":14:0) {
^entry:
    %p_addr = core.alloca : ptr<ptr<f64>, stack> loc("examples/32_native_memory/native_memory.ppy":14:0)
    core.store %p, %p_addr
    %n_addr = core.alloca : ptr<i64, stack>
    core.store %n, %n_addr
    %scale_addr = core.alloca : ptr<f64, stack>
    core.store %scale, %scale_addr
    %0 = core.const 0.0 : f64 loc("examples/32_native_memory/native_memory.ppy":15:4)
    %total_addr = core.alloca : ptr<f64, stack>
    core.store %0, %total_addr
    %1 = core.load %n_addr : i64 loc("examples/32_native_memory/native_memory.ppy":16:4)
    %n_entry = core.load %n_addr : i64
    %2 = core.const 0 : i64
    %3 = core.const 1 : i64
    %i_addr = core.alloca : ptr<i64, stack>
    %q_addr = core.alloca : ptr<ptr<f64>, stack>
    core.store %2, %i_addr loc("examples/32_native_memory/native_memory.ppy":16:4)
    core.br ^for.head3
^for.head3:
    %4 = core.load %i_addr : i64 loc("examples/32_native_memory/native_memory.ppy":16:4)
    %5 = core.cmp.lt %4, %1 : bool
    core.cond_br %5, ^for.body4, ^for.end6
^for.body4:
    %6 = core.load %p_addr : ptr<f64> loc("examples/32_native_memory/native_memory.ppy":17:8)
    %7 = core.load %i_addr : i64
    %8 = core.ptr_offset %6, %7 : ptr<f64>
    core.store %8, %q_addr
    %9 = core.load %q_addr : ptr<f64> loc("examples/32_native_memory/native_memory.ppy":18:8)
    %10 = core.load %scale_addr : f64
    %11 = core.load %i_addr : i64
    %12 = core.cast %11 : f64
    %13 = core.mul %10, %12 : f64
    core.store %13, %9
    %14 = core.load %total_addr : f64 loc("examples/32_native_memory/native_memory.ppy":19:8)
    %15 = core.load %q_addr : ptr<f64>
    %16 = core.load %15 : f64
    %17 = core.add %14, %16 : f64
    core.store %17, %total_addr
    %18 = core.load %i_addr : i64
    %19 = core.add %18, %3 {overflow = "python"} : i64
    core.store %19, %i_addr
    core.br ^for.head3
^for.end6:
    %20 = core.load %total_addr : f64 loc("examples/32_native_memory/native_memory.ppy":20:4)
    core.ret %20
}

func @native_memory_norm(%p: ptr<f64, generic, const>, %n: i64) -> f64 attrs {effects = ["may_raise", "read_memory"], ppy.abi = "ppy", ppy.export = "ppy_norm", ppy.qualname = "native_memory.norm", ppy.releases_gil = true, ppy.symbol = "ppy_native_memory_norm"} loc("examples/32_native_memory/native_memory.ppy":24:0) {
^entry:
    %p_addr = core.alloca : ptr<ptr<f64, generic, const>, stack> loc("examples/32_native_memory/native_memory.ppy":24:0)
    core.store %p, %p_addr
    %n_addr = core.alloca : ptr<i64, stack>
    core.store %n, %n_addr
    %0 = core.const 0.0 : f64 loc("examples/32_native_memory/native_memory.ppy":25:4)
    %acc_addr = core.alloca : ptr<f64, stack>
    core.store %0, %acc_addr
    %1 = core.load %n_addr : i64 loc("examples/32_native_memory/native_memory.ppy":26:4)
    %n_entry = core.load %n_addr : i64
    %2 = core.const 0 : i64
    %3 = core.const 1 : i64
    %i_addr = core.alloca : ptr<i64, stack>
    core.store %2, %i_addr loc("examples/32_native_memory/native_memory.ppy":26:4)
    core.br ^for.head3
^for.head3:
    %4 = core.load %i_addr : i64 loc("examples/32_native_memory/native_memory.ppy":26:4)
    %5 = core.cmp.lt %4, %1 : bool
    core.cond_br %5, ^for.body4, ^for.end6
^for.body4:
    %6 = core.load %acc_addr : f64 loc("examples/32_native_memory/native_memory.ppy":27:8)
    %7 = core.load %p_addr : ptr<f64, generic, const>
    %8 = core.load %i_addr : i64
    %9 = core.ptr_offset %7, %8 : ptr<f64, generic, const>
    %10 = core.load %9 : f64
    %11 = core.call_extern %6, %10 {abi = "c", callee = "hypot"} : f64
    core.store %11, %acc_addr
    %12 = core.load %i_addr : i64
    %13 = core.add %12, %3 {overflow = "python"} : i64
    core.store %13, %i_addr
    core.br ^for.head3
^for.end6:
    %14 = core.load %acc_addr : f64 loc("examples/32_native_memory/native_memory.ppy":28:4)
    core.ret %14
}

func @native_memory_scratch(%n: i64) -> i64 attrs {effects = ["alloc", "may_raise", "read_memory", "write_memory"], ppy.abi = "ppy", ppy.qualname = "native_memory.scratch", ppy.releases_gil = true, ppy.symbol = "ppy_native_memory_scratch"} loc("examples/32_native_memory/native_memory.ppy":31:0) {
^entry:
    %n_addr = core.alloca : ptr<i64, stack> loc("examples/32_native_memory/native_memory.ppy":31:0)
    core.store %n, %n_addr
    %0 = core.alloca {count = 8} : ptr<i64, stack> loc("examples/32_native_memory/native_memory.ppy":32:4)
    %buf_addr = core.alloca : ptr<ptr<i64, stack>, stack>
    core.store %0, %buf_addr
    %1 = core.const 0 : i64 loc("examples/32_native_memory/native_memory.ppy":33:4)
    %acc_addr = core.alloca : ptr<i64, stack>
    core.store %1, %acc_addr
    %2 = core.const 8 : i64 loc("examples/32_native_memory/native_memory.ppy":34:4)
    %3 = core.const 0 : i64
    %4 = core.const 1 : i64
    %i_addr = core.alloca : ptr<i64, stack>
    %n_entry = core.load %n_addr : i64
    %5 = core.const 8 : i64
    %6 = core.const 0 : i64
    %7 = core.const 1 : i64
    %8 = core.const 8 : i64
    %9 = core.const 1 : i64
    %10 = core.const 7 : i64
    %11, %12 = core.call_intrinsic %3, %n_entry {intrinsic = "ppy.checked_mul"} : i64, bool
    %13, %14 = core.call_intrinsic %3, %n_entry {intrinsic = "ppy.checked_mul"} : i64, bool
    %15 = core.or %12, %14 : bool
    %16, %17 = core.call_intrinsic %10, %n_entry {intrinsic = "ppy.checked_mul"} : i64, bool
    %18 = core.or %15, %17 : bool
    %19, %20 = core.call_intrinsic %10, %n_entry {intrinsic = "ppy.checked_mul"} : i64, bool
    %21 = core.or %18, %20 : bool
    %22 = core.const true : bool
    %23 = core.xor %21, %22 : bool
    core.guard %23 {kind = "overflow", message = "hoisted guard"}
    core.store %3, %i_addr loc("examples/32_native_memory/native_memory.ppy":34:4)
    core.br ^for.head3
^for.head3:
    %24 = core.load %i_addr : i64 loc("examples/32_native_memory/native_memory.ppy":34:4)
    %25 = core.cmp.lt %24, %2 : bool
    core.cond_br %25, ^for.body4, ^for.end6
^for.body4:
    %26 = core.load %buf_addr : ptr<i64, stack> loc("examples/32_native_memory/native_memory.ppy":35:8)
    %27 = core.load %i_addr : i64
    %28 = core.ptr_offset %26, %27 : ptr<i64, stack>
    %29 = core.load %i_addr : i64
    %30 = core.load %n_addr : i64
    %31 = core.mul %29, %30 {overflow = "proven"} : i64
    core.store %31, %28
    %32 = core.load %i_addr : i64
    %33 = core.add %32, %4 {overflow = "python"} : i64
    core.store %33, %i_addr
    core.br ^for.head3
^for.end6:
    core.store %6, %i_addr loc("examples/32_native_memory/native_memory.ppy":36:4)
    core.br ^for.head9
^for.head9:
    %34 = core.load %i_addr : i64 loc("examples/32_native_memory/native_memory.ppy":36:4)
    %35 = core.cmp.lt %34, %5 : bool
    core.cond_br %35, ^for.body10, ^for.end12
^for.body10:
    %36 = core.load %acc_addr : i64 loc("examples/32_native_memory/native_memory.ppy":37:8)
    %37 = core.load %buf_addr : ptr<i64, stack>
    %38 = core.load %i_addr : i64
    %39 = core.ptr_offset %37, %38 : ptr<i64, stack>
    %40 = core.load %39 : i64
    %41 = core.add %36, %40 {overflow = "python"} : i64
    core.store %41, %acc_addr
    %42 = core.load %i_addr : i64
    %43 = core.add %42, %7 {overflow = "python"} : i64
    core.store %43, %i_addr
    core.br ^for.head9
^for.end12:
    %44 = core.load %acc_addr : i64 loc("examples/32_native_memory/native_memory.ppy":38:4)
    %45 = core.add %44, %8 {overflow = "python"} : i64
    %46 = core.add %45, %9 {overflow = "python"} : i64
    core.ret %46
}

func @native_memory_bytes_of(%p: ptr<u8, generic, const>, %n: i64) -> i64 attrs {effects = ["may_raise", "read_memory"], ppy.abi = "ppy", ppy.qualname = "native_memory.bytes_of", ppy.releases_gil = true, ppy.symbol = "ppy_native_memory_bytes_of"} loc("examples/32_native_memory/native_memory.ppy":41:0) {
^entry:
    %p_addr = core.alloca : ptr<ptr<u8, generic, const>, stack> loc("examples/32_native_memory/native_memory.ppy":41:0)
    core.store %p, %p_addr
    %n_addr = core.alloca : ptr<i64, stack>
    core.store %n, %n_addr
    %0 = core.const 0 : i64 loc("examples/32_native_memory/native_memory.ppy":42:4)
    %total_addr = core.alloca : ptr<i64, stack>
    core.store %0, %total_addr
    %1 = core.load %n_addr : i64 loc("examples/32_native_memory/native_memory.ppy":43:4)
    %n_entry = core.load %n_addr : i64
    %2 = core.const 0 : i64
    %3 = core.const 1 : i64
    %i_addr = core.alloca : ptr<i64, stack>
    core.store %2, %i_addr loc("examples/32_native_memory/native_memory.ppy":43:4)
    core.br ^for.head3
^for.head3:
    %4 = core.load %i_addr : i64 loc("examples/32_native_memory/native_memory.ppy":43:4)
    %5 = core.cmp.lt %4, %1 : bool
    core.cond_br %5, ^for.body4, ^for.end6
^for.body4:
    %6 = core.load %total_addr : i64 loc("examples/32_native_memory/native_memory.ppy":44:8)
    %7 = core.load %p_addr : ptr<u8, generic, const>
    %8 = core.load %i_addr : i64
    %9 = core.ptr_offset %7, %8 : ptr<u8, generic, const>
    %10 = core.load %9 : u8
    %11 = core.cast %10 : i64
    %12 = core.add %6, %11 {overflow = "python"} : i64
    core.store %12, %total_addr
    %13 = core.load %i_addr : i64
    %14 = core.add %13, %3 {overflow = "python"} : i64
    core.store %14, %i_addr
    core.br ^for.head3
^for.end6:
    %15 = core.load %total_addr : i64 loc("examples/32_native_memory/native_memory.ppy":45:4)
    core.ret %15
}
```

</details>

<!-- outputs:end -->

Read on: [Native memory and FFI](../../docs/guide/native.md) ·
[Lanes and the machine](../33_simd_and_cpu/README.md)

`native_memory.ppy` is hand-written; there is no `.py` source and no
conversion step.
