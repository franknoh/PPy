# Native memory: `ppy.native` and `ppy.ffi`

`ppy.native` is still the directive it was -- `@ppy.native`,
`@ppy.native(require=True)` -- and also the namespace of typed native memory:

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

| | |
|---|---|
| `native.ptr[T]`, `native.const_ptr[T]` | a typed pointer; `T` is `int`, `float`, `bool`, or any fixed-width marker -- `ppy.i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f32`, `f64` -- each stored at its own width. A `const_ptr` refuses `store` (`E1631`). |
| `native.load(p)`, `native.store(p, v)`, `native.offset(p, n)` | read, write, move. Reading a byte hands out an `int`, as a buffer does. |
| `native.cast[U](p)` | the same memory read as `U`. |
| `native.sizeof[T]()`, `native.alignof[T]()` | constants the checker knows. |
| `native.stack_alloc[T](n)` | `n` zeroed elements the function owns; native code needs a constant `n`, and the memory cannot be returned or stored -- the IR's verifier holds that. |
| `@native.extern("sin", library="m", pure=True)` | a stub is a C function. Under CPython the call goes through ctypes; in native code straight to the symbol. The directive says what the C function does (`pure=True` or, by default, reads and writes of native memory), and the signature must be fully annotated (`E1633`). |
| `@native.export(name="ppy_dot")` | a public C symbol in the built library, declared in the header `ppy build` writes beside it. A C caller has no Python to fall back to, so a failed guard traps. |

Under plain CPython every one of these has a reference implementation over
`array` memory, so the three paths agree; a function whose parameter is a
pointer has no Python boundary and is called only from native code.

`ppy.ffi` is the binding layer over `extern`: `ffi.library("m")`,
`@ffi.bind(lib, symbol="sin", pure=True)`, `ffi.nullable[native.ptr[T]]`
for a pointer that may be null, `ffi.LengthOf("xs")` for an integer that
is another parameter's length, and `ppy.Owned`/`ppy.Borrowed` for who keeps
memory. `ppy bind header foo.h` writes such a module from a C header
([CLI](../cli.md)).

Examples: [Native memory](../howto/32_native_memory.md),
[The toolbox](../howto/42_toolbox.md).
