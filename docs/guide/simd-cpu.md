# Lanes and the machine: `ppy.simd` and `ppy.cpu`

Two namespaces beside `ppy.native`, each with a reference implementation
under CPython and a lowering to a dialect of the IR
([The IR](../internals/ir.md)), so a program using them runs the same on
every path.

## `ppy.simd`

`ppy.simd` is a few scalars operated on at once. `simd.Vector[T, N]` is
`N` lanes of `T` -- `int`, `float`, `bool`, `ppy.i8`, or `ppy.u8`;
`splat[T, N](x)` fills one, `load[T, N](p)` reads one from a
`native.ptr[T]`, `store(v, p)` writes it back, `extract(v, i)` and
`insert(v, i, x)` reach one lane (the index is guarded), `shuffle(a, b,
mask)` builds a vector from the lanes of two by a constant mask, and
`reduce_add`, `reduce_min`, `reduce_max` fold one to a scalar in lane
order -- first to last, so a floating-point sum is the same number
everywhere, and a NaN lane is passed over or kept exactly as Python's
`min` would. `+ - *` work lane by lane on numbers and integer lanes wrap
at their width, `/` on float lanes, `& | ^` on integer and bool lanes,
the comparisons give a `Vector[bool, N]`, and `select(mask, a, b)`
chooses by it. A vector lives in a local; it does not cross the Python
boundary. `E1640` names a misuse.

## `ppy.cpu`

`ppy.cpu` is the machine as a facade, with no instruction named.
`cpu.features()` is what this machine has, in LLVM's spelling (`avx2`,
`fma`, `neon`, `sse4.2`); the compiler folds `"avx2" in cpu.features()`
to a constant for the machine compiling. `cpu.vector_width[T]()` is how
many `T` a vector register holds here, also a constant. `cpu.prefetch(p,
write=False, locality=3)` and `cpu.pause()` are hints and change no
value. `@cpu.target("avx2", "fma")` compiles a function with those
features on; the boundary binds it only on a machine that has them all
and runs its Python definition elsewhere, so a program stays correct on
every machine it reaches. `E1643` names a misuse.

Examples: [SIMD and the CPU](../howto/33_simd_and_cpu.md).
