# Native memory

Typed pointers, memory a function owns, a C function bound from `libm`,
and a function exported as a C symbol -- with a reference implementation
under plain CPython, so the three paths agree.

## Provenance

Hand-written. `native_memory.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- `native.ptr[T]` and `native.const_ptr[T]` are typed pointers; `load`,
  `store`, and `offset` read, write, and move, and `sizeof`/`alignof` are
  constants the checker knows.
- `native.stack_alloc[T](n)` is memory the function owns; it cannot be
  returned or stored, and the IR's verifier holds that.
- `native.cast[U](p)` reads the same memory as another element: the four
  floats are also thirty-two bytes.
- `ffi.library("m")` and `@ffi.bind(..., symbol="hypot")` bind a C function:
  ctypes under CPython, a direct call in native code, and `ppy emit ir` shows
  the `core.call_extern` with `ppy.libraries = ("m",)` on the module.
- `@native.export(name="ppy_norm")` makes `norm` a public C symbol in the
  library `ppy build` writes, declared in the header beside it; a C caller
  has no Python to fall back to, so a failed guard traps.

## Run it

```bash
python  native_memory.ppy
ppy run native_memory.ppy
ppy build native_memory.ppy -o dist   # dist/native_memory.h declares ppy_norm
ppy emit ir native_memory.ppy         # the pointers, the extern call, the export
```

<!-- outputs:start -->
## What it prints

**`python  native_memory.ppy`**

```text
9.0 93 0
5.612486 5.612486
```

**`ppy run native_memory.ppy`**

```text
9.0 93 0
5.612486 5.612486
```

**`ppy build native_memory.ppy -o dist`**

*(prints nothing; exits 0)*

**`ppy emit ir native_memory.ppy`**

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
… 162 more lines
```

<!-- outputs:end -->
