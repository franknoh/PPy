# Native memory: `ppy.native` and `ppy.ffi`

This page covers typed native memory in `ppy.native` and the C binding layer
`ppy.ffi` built on it.

## Example

`ppy.native` is still the directive it was (`@ppy.native`,
`@ppy.native(require=True)`). It is also the namespace of typed native memory:

```python
from ppy import native


@native
def fill(p: native.ptr[float], n: int) -> None:
    for i in range(n):
        native.store(native.offset(p, i), 0.5 * i)


def scratch() -> int:
    buf = native.stack_alloc[int](8)  # the function's own memory
    native.store(buf, native.sizeof[int]())
    return native.load(buf)
```

## Reference

| | |
|---|---|
| `native.ptr[T]`, `native.const_ptr[T]` | a typed pointer. `T` is `int`, `float`, `bool`, or any fixed-width marker (`ppy.i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f32`, `f64`), each stored at its own width. A `const_ptr` refuses `store` (`E1631`). |
| `native.ptr[T, N, ...]`, `native.const_ptr[T, N, ...]` | borrowed contiguous C arrays with static positive inner extents; `p[row, col, ...]` reads or writes a scalar. The pointer's element is a complete row, so `native.offset(p, 1)` advances by one row. Supported by C, C++, and LLVM. |
| `native.load(p)`, `native.store(p, v)`, `native.offset(p, n)` | read, write, move. Reading a byte hands out an `int`, as a buffer does. |
| `native.cast[U](p)` | the same memory read as `U`. |
| `native.sizeof[T]()`, `native.alignof[T]()` | constants the checker knows. |
| `native.stack_alloc[T](n)` | `n` zeroed elements the function owns. Native code needs a constant `n`, and the memory cannot be returned or stored; the IR's verifier holds that. |
| `@native.extern("sin", library="m", pure=True)` | a stub is a C function. Under CPython the call goes through ctypes; in native code, straight to the symbol. The directive says what the C function does (`pure=True` or, by default, reads and writes of native memory), and the signature must be fully annotated (`E1633`). |
| `native.compiled(f)` | whether calling `f` runs its native form here: `True` under `ppy run` or a built artifact once the runtime has bound it, `False` under plain CPython. It answers for the object standing in the namespace, which under `ppy run` may be the C entry point itself (named `module.f`). |
| `@native.export(name="ppy_dot")` | a public C symbol in the built library, declared in the header `ppy build` writes beside it. A C caller has no Python to fall back to, so a failed guard traps. |

## Under CPython

Under plain CPython each of these has a reference implementation over `array`
memory for scalar native operations. Row-pointer indexing is a native-code
interface. A function whose parameter is a pointer has no automatic Python
boundary and is called from native code.

## C integer and array interfaces

Explicit integer markers select both width and signedness in native parameters,
results, pointers, and `Buffer` storage. For example, `i32` maps to `int32_t`,
`u32` to `uint32_t`, and `native.ptr[i32]` to `int32_t *`. The ordinary `int`
remains 64-bit. Conversions to narrower integers are checked; an incompatible
pointer argument is rejected instead of being reinterpreted. An explicit
`native.cast` remains the programmer's responsibility.

Full-range `u64` values are supported at native boundaries and in unsigned
storage. Mixed operations with negative signed integers, and unary operations
that require a signed result outside the default integer range, can fail a
native conversion guard. A Python caller then uses its Python body; a C export
traps on a failed guard.

```python
from ppy import i32, native

@native.export
def game_init() -> None:
    pass

@native.export
def process(ice: native.ptr[i32, 8]) -> None:
    ice[2, 3] = ice[1, 7] + 1
```

The C declarations are `void game_init(void)` and
`void process(int32_t (*ice)[8])`. Generated C/C++ may spell the row type with
an equivalent array typedef. `native.ptr[i32, 3, 4]` similarly maps to
`int32_t (*)[3][4]`. Supply one index for the outer row plus one per inner
extent. Array rows cannot be loaded, assigned, or copied as whole values.
`Buffer[i32]` retains its existing pointer-and-length ABI; it is a different
interface from a raw pointer-to-array.

These pointers borrow the caller's allocation. No descriptor, allocation,
copy, or free is introduced. The caller must keep correctly aligned, contiguous
storage alive throughout every access and provide a valid outer row index.
Inner indices are checked in safe code. `const_ptr` forbids writes through that
pointer; it does not prevent another alias from changing the allocation.
There is no runtime lifetime enforcement or automatic ownership transfer.
When passing a C `int` array to `ptr[i32, ...]`, verify the target's `int` is
the same type as `int32_t`; the example below includes a compile-time check.

`--int-width 32` remains an unsafe standalone C/C++ option for the default
integer model. Its existing rewrite cannot safely change fixed runtime or
foreign ABIs. Pointer parameters and explicitly sized integer programs are
therefore refused by that mode; use the default 64-bit model with `i32` where
the C interface needs 32 bits. Library mode still refuses `--int-width`.

See [the complete C caller example](https://github.com/franknoh/PPy/tree/main/examples/32_native_memory/c_arrays).

## `ppy.ffi`

`ppy.ffi` is the binding layer over `extern`:

- `ffi.library("m")` loads a library.
- `@ffi.bind(lib, symbol="sin", pure=True)` binds a symbol.
- `ffi.nullable[native.ptr[T]]` is a pointer that may be null.
- `ffi.LengthOf("xs")` is an integer that is another parameter's length.
- `ppy.Owned`/`ppy.Borrowed` say who keeps memory.

`ppy bind header foo.h` writes such a module from a C header
([CLI](../cli.md)).

Examples: [Native memory](../howto/32_native_memory.md),
[The toolbox](../howto/42_toolbox.md).
