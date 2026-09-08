# Directives and markers

## Directives

All directives work bare (`@ppy.pure`) and called (`@ppy.pure()`), and are
contracts the compiler verifies, not hints it trusts.

| directive | meaning |
|---|---|
| `@ppy.pure` | no observable effects: no I/O, no global or nonlocal writes, no mutation of arguments. Local allocation and mutation of locally created values are fine. Violations are `E1601`/`E1602`. |
| `@ppy.opt(n)` | per-function optimization level 0–3, overriding the project default. |
| `@ppy.native` | lower to LLVM. `require=True` makes any fallback to the Python body an error (`E1702`). |
| `@ppy.parallel` | parallelize the eligible loop. `require=True` makes failure an error (`E1701`). |
| `@ppy.jit` | specialize at runtime on the argument classes actually seen. |
| `@ppy.specialize` | ahead-of-time specialization on declared value classes. |
| `@ppy.inline` / `@ppy.noinline` | force or forbid inlining into callers. |
| `@ppy.fastmath` | permit floating-point reassociation in this function; without it, reduction order is preserved bit-for-bit. |
| `@ppy.jax` | stage this function for build-time StableHLO export (a plain `@jax.jit` decorator marks it too). |
| `@ppy.dynamic` / `with ppy.dynamic():` | an explicit boundary inside which dynamic features are allowed; every value it produces is `Dynamic`, and stays `Dynamic` through attribute hops and arithmetic until a `ppy.check[T]` clears it. |
| `@ppy.reflective` | the function's annotations are runtime-visible state, exactly as written: `ppy convert`/`ppy migrate` will never add to or rely on rewriting them. |

A typo in a directive name is `E1205`, with a suggestion.

## Markers

Ordinary `Annotated` aliases from `ppy`:

| marker | meaning |
|---|---|
| `i8 i16 i32 i64 u8 u16 u32 u64` | fixed-width integer contract. A value provably outside the range is `E1401`; a check the contract mode forbids is `E1402`. |
| `f16 f32 f64` | floating-point width. |
| `Buffer[T]` | a borrowed writable buffer (`memoryview` over `array.array`) — zero-copy in and out of native code. `T` may be `int`, `float`, or `ppy.i8`/`ppy.u8` for one byte per element. |
| `Array[T]`, `Vector[T]` | contiguous numeric containers with a known element type. |
| `Range(lo, hi)` | an integer refinement the checker propagates. |
| `Dynamic` | an explicit Python-dynamic boundary value. Entering is free; leaving is not: `Dynamic -> Dynamic` flows freely, but `Dynamic -> int` is `E1508` until a `ppy.check[T]` validates it. `Any` at runtime. |
| `Length(n)`, `Shape(...)`, `DType("f32")`, `Contiguous`, `NoAlias` | container refinements; `Shape`/`DType` are what makes a `@ppy.jax` function exportable. |
| `Owned[T]`, `Borrowed[T]`, `Mut[T]` | how a parameter holds its value. `Borrowed` is read for the call and no longer: not returned (`E1611`), not stored where it outlives the call (`E1612`), not written (`E1613`). `Mut` is a borrow the function may write through. `Owned` hands the value over. A `Buffer[T]` is `Borrowed` unless the program says otherwise. |

Examples: [Basics](../howto/01_basics.md),
[Effects and contracts](../howto/03_effects_and_contracts.md),
[Buffers and JIT](../howto/12_buffers_and_jit.md).
