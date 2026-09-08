# `ppy`

Importing `ppy` is cheap: it installs the `.ppy` import hook and exposes
directives and annotation markers. It never initializes LLVM, loads
compiler services, or starts background processes. The namespaces a program
writes against are each a reference implementation under CPython that
answers the same as the compiled one. How to use them is in the
[guide](../guide/index.md); this page is the signatures.

## Directives

::: ppy._directives
    options:
      show_root_heading: false
      members:
        - Directive
        - pure
        - opt
        - native
        - jit
        - specialize
        - dynamic
        - parallel
        - inline
        - noinline
        - fastmath
        - reflective
        - jax
        - attach
        - directives_of

## Markers

::: ppy._markers
    options:
      show_root_heading: false

## Input and buffers

::: ppy._io
    options:
      show_root_heading: false

::: ppy._alloc
    options:
      show_root_heading: false
      members:
        - buffer

## The import hook

::: ppy._importer
    options:
      show_root_heading: false

## `ppy.native`

`ppy.native` is an object, not a module: the directive, and the namespace of
typed native memory — `native.ptr[T]`, `native.const_ptr[T]`,
`native.stack_alloc[T](n)`, `native.load`, `native.store`, `native.offset`,
`native.cast[T]`, `native.sizeof[T]`, `native.extern(...)`,
`native.export(...)`. [Native memory and FFI](../guide/native.md) describes
each.

::: ppy._native_api
    options:
      show_root_heading: false

## `ppy.ffi`

::: ppy.ffi

## `ppy.simd`

::: ppy.simd

## `ppy.cpu`

::: ppy.cpu

## `ppy.atomic`

::: ppy.atomic

## `ppy.concurrent`

::: ppy.concurrent

## Derivatives

::: ppy.autodiff

## `ppy.aio`

::: ppy.aio

## `ppy.cuda`

::: ppy.cuda

## `ppy.hip`

::: ppy.hip

## `ppy.xla`

::: ppy.xla
