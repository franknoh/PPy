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
