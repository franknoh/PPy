# Typed pointers, stack memory, a C function, a C symbol

`ppy.native` is the directive it always was and, since 0.2.0, the namespace
of typed native memory. This program fills a buffer through a `native.ptr`,
allocates scratch on the stack, reinterprets four doubles as thirty-two
bytes, calls `hypot` from `libm`, and exports `norm` as a public C symbol —
and every one of those has a reference implementation under plain CPython,
so `python native_memory.ppy` prints what `ppy run` prints.

## Pointers you can read, write, and move

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
`store`, and `offset` are the three operations. A store through a
`const_ptr` is `E1631`. `native.cast[U](p)` reads the same memory as
another element type — `bytes_of(native.cast[ppy.u8](memory), 8)` sums the
first eight bytes of the four doubles `fill` wrote. `sizeof` and `alignof`
are constants the checker knows.

## Memory the function owns

`native.stack_alloc[int](8)` is eight zeroed words on the function's own
stack. Native code needs a constant count, and the pointer cannot be
returned or stored where it outlives the call — the IR's verifier refuses a
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
traps rather than answering wrongly. `ppy bind header foo.h` writes a
module of such bindings from a C header.

## Run it

```bash
python  native_memory.ppy
ppy run native_memory.ppy
ppy emit ir native_memory.ppy
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
```

*202 lines in all — [full output](outputs/03-ppy-emit-ir-native-memory-ppy.txt).*

<!-- outputs:end -->

## Read on

- [Native memory and FFI](../../docs/guide/native.md) — every operation, and the ownership rules.
- [Lanes and the machine](../33_simd_and_cpu/README.md) — vectors over the same pointers.

`native_memory.ppy` is hand-written; there is no `.py` source and no
conversion step.
