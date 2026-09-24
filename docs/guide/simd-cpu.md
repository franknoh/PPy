# Lanes and the machine: `ppy.simd` and `ppy.cpu`

This page covers two namespaces beside `ppy.native`: `ppy.simd` for vector
lanes and `ppy.cpu` for the machine you run on. Each has a reference
implementation under CPython and a lowering to a dialect of the IR
([The IR](../internals/ir.md)), so a program using them runs the same on
every path.

## `ppy.simd`

`ppy.simd` operates on a few scalars at once. `simd.Vector[T, N]` is `N`
lanes of `T`, where `T` is `int`, `float`, `bool`, `ppy.i8`, or `ppy.u8`.

### Building and reading vectors

- `splat[T, N](x)` fills one.
- `load[T, N](p)` reads one from a `native.ptr[T]`.
- `store(v, p)` writes it back.
- `extract(v, i)` and `insert(v, i, x)` reach one lane. The index is guarded.
- `shuffle(a, b, mask)` builds a vector from the lanes of two by a constant
  mask.

### Reductions

`reduce_add`, `reduce_min`, and `reduce_max` fold a vector to a scalar in lane
order, first to last. A floating-point sum is therefore the same number
everywhere, and a NaN lane is passed over or kept as Python's `min` would.

### Operators

- `+ - *` work lane by lane on numbers. Integer lanes wrap at their width.
- `/` works on float lanes.
- `& | ^` work on integer and bool lanes.
- The comparisons give a `Vector[bool, N]`, and `select(mask, a, b)` chooses
  by it.

### Limits

A vector lives in a local. It does not cross the Python boundary. `E1640`
names a misuse.

## `ppy.cpu`

`ppy.cpu` is the machine as a facade, with no instruction named.

- `cpu.features()` is what this machine has, in LLVM's spelling (`avx2`,
  `fma`, `neon`, `sse4.2`). The compiler folds `"avx2" in cpu.features()` to
  a constant for the machine compiling.
- `cpu.vector_width[T]()` is how many `T` a vector register holds here, also
  a constant.
- `cpu.prefetch(p, write=False, locality=3)` and `cpu.pause()` are hints and
  change no value.
- `@cpu.target("avx2", "fma")` compiles a function with those features on.
  The boundary binds it only on a machine that has them all and runs its
  Python definition elsewhere, so a program stays correct on each machine it
  reaches.

`E1643` names a misuse.

Examples: [SIMD and the CPU](../howto/33_simd_and_cpu.md).
