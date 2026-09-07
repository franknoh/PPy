# The IR

Between analysis and every backend sits one typed, SSA-form intermediate
representation with an explicit control-flow graph. Analysis decides what a
program means; the IR keeps that meaning in a form a backend can lower
without reading Python again; dialects extend the IR; passes transform it;
backends lower it. This page is the reference for the core; `docs/solver.md`
and the plugin pages say what the other dialects add.

## Shape

```text
func @abs(%x: i64) -> i64 {
^entry:
    %zero = core.const 0 : i64
    %negative = core.cmp.lt %x, %zero : bool
    core.cond_br %negative, ^neg, ^positive
^neg:
    %value = core.neg %x {overflow = "python"} : i64
    core.br ^exit(%value)
^positive:
    core.br ^exit(%x)
^exit(%result: i64):
    core.ret %result
}
```

- A **module** holds functions and globals under one symbol table, and
  names every dialect it uses with the version it was written against.
- A **function** has typed parameters, result types, attributes, and a body
  region. A body-less function is a declaration (`extern func`).
- A **region** is a list of **blocks**; the first is the entry. A block has
  typed arguments and operations, and ends in exactly one terminator.
  Branches pass arguments to the block they reach: there is no phi.
- A **value** is a block argument or the result of one operation, defined
  once and carrying exactly one type. A use must be dominated by its
  definition; the verifier checks that along every path.
- An **operation** is `dialect.name`, operands, results, attributes,
  successors (for terminators), and optional nested regions.

## Types

```text
void  bool  i8 i16 i32 i64  u8 u16 u32 u64  f16 f32 f64  index
ptr<T>  ptr<T, space>  ptr<T, space, const>
buffer<T>                    contiguous elements with a readable length
vector<T, N>
tuple<T, ...>
struct<Name, field: T, ...>
future<T>
dialect.name<args>           a type a dialect owns; the core parses it, the dialect verifies it
```

`ptr<T, stack>` is what `core.alloca` yields; a stack pointer may not be
returned or stored, and the verifier says so. `generic` is the host's
address space; a dialect adds its own (`global`, `shared`, ...).

## Core operations

| operation | meaning |
|---|---|
| `core.const V : T` | a constant; `V` must fit `T` |
| `core.add`, `sub`, `mul`, `div`, `mod`, `neg` | arithmetic on numbers or vectors of them. Integer forms carry `overflow` (`python` \| `checked` \| `wrap` \| `proven`); `div`/`mod` also `rounding` (`floor` \| `trunc`). The backend never guesses either. |
| `core.and`, `or`, `xor`, `shl`, `shr` | bitwise on integers or bools |
| `core.cmp.<eq,ne,lt,le,gt,ge>` | comparison to `bool`, or `vector<bool, N>` |
| `core.select` | `bool ? a : b` |
| `core.cast` | scalar-to-scalar, or pointer-to-pointer within one address space |
| `core.br`, `cond_br`, `ret`, `unreachable` | terminators |
| `core.alloca` | stack memory: `ptr<T, stack>` |
| `core.load`, `store`, `ptr_offset` | pointer access; a store through `const` is refused |
| `core.buffer_data`, `buffer_len`, `buffer_load`, `buffer_store` | buffers |
| `core.tuple_make`, `tuple_extract {index}` | fixed tuples |
| `core.struct_make`, `struct_extract {field}` | structs |
| `core.call @f`, `call_extern {callee}`, `call_intrinsic {intrinsic}` | calls; a `core.call` is checked against the callee's signature. A failed call takes the caller's fallback, unless `capture_status`, which hands the status back as a trailing `i64` for a caller that has threads to join first. |
| `core.guard %cond {kind}` | a runtime check the function fails on: `overflow`, `bounds`, `zero_division`, `range`, `contract`, `assert` |

Overflow semantics live on the operation. `python` means the true value is
what Python computes -- the backend guards and falls back; `checked` means
overflow is a guard failure; `wrap` means two's-complement wrap like C;
`proven` means a proof -- a corner check hoisted ahead of the loop, or the
solver -- established that the value fits, so the backend emits the plain
operation and may tell the optimizer it never wraps.
Bounds checks are explicit `core.guard`s the frontend emits; a sanitizer
pass adds more.

## The math dialect

`math.sqrt`, `sin`, `cos`, `tan`, `exp`, `exp2`, `log`, `log2`, `log10`,
`floor`, `ceil`, `trunc`, `abs`, and `pow`: elementary functions named once
for every backend, pure, over a floating-point value or a vector of them.
The frontend writes `math.sqrt %x : f64` for `math.sqrt(x)`; the LLVM
backend lowers it to the intrinsic of that name, a C backend to libm, a GPU
backend to its device library. A math function of a constant folds, and
`floor(floor(x))` is `floor(x)`.

## The simd dialect

`vector<T, N>` is `N` scalars operated on at once, and the core dialect's
arithmetic, comparisons, `select`, and `cast` already take vectors (a
vector's integer arithmetic carries `wrap` or `proven`). The simd dialect
says what the core cannot: `simd.splat %x : vector<T, N>`, `simd.load %p :
vector<T, N>` and `simd.store %v, %p` through a pointer to the element,
`simd.extract %v, %i : T` and `simd.insert %v, %x, %i`, `simd.shuffle %a,
%b {mask = (...)}` over the lanes of both by a constant mask, and
`simd.reduce_add`, `reduce_min`, `reduce_max` to a scalar. A reduction
walks the lanes in order, first to last: an integer sum is the reduction
intrinsic, a floating-point sum and every minimum and maximum a chain of
lane operations, so every backend and the reference implementation give
one number.

## The cpu dialect

`cpu.prefetch %p {rw, locality}` asks for a cache line, `cpu.pause` is the
spin-wait hint; a backend without the instruction drops the hint. A
function compiled for a feature set carries `cpu.features = ("avx2", ...)`
as an attribute, which the LLVM backend turns into the function's target
features and the boundary into a check before binding.

## The atomic dialect

`atomic.load`, `store`, `exchange`, `compare_exchange` (two results: the
value found and whether it swapped, with a `success` and a `failure`
order), `fetch_add`, `fetch_sub`, `fetch_and`, `fetch_or`, `fetch_xor`,
and `fence`, each with its memory `order` -- `relaxed`, `acquire`,
`release`, `acq_rel`, `seq_cst`, the vocabulary of C11 and LLVM -- so a
backend lowers to the instruction of that order and never guesses. The
verifier holds the orders a load and a store may carry and the failure
order of a compare-exchange.

## The concurrency dialect

`concurrency.spawn @f(%args...) : i64` starts a thread running the IR
function `@f` (which returns nothing, and takes exactly those arguments)
and hands back its handle; `join %h : i64` waits and gives the status
`@f` returned -- zero, or the fallback status a guard failed with.
`mutex_lock`, `mutex_unlock`, `condition_wait`, `condition_notify`,
`barrier`, and `thread_id` complete it. The synchronization objects are
memory the program owns: a mutex one `i64` slot, a condition one slot
counting notifications, a barrier two slots (arrivals, generation); every
backend implements them over the atomic dialect and the pause hint the
same way, so a program built for one runs exactly like one built for
another. Threads are pthreads on the targets that have them; another is
refused with the reason.

## The parallel dialect

`parallel.for @body(%captures...) %begin, %end` runs `@body(captures...,
begin, end)` -- a function that loops over its own chunk and returns
nothing -- over the range; `parallel.reduce @body(%captures...) %begin,
%end, %init {op, reassociate} : T` folds the chunks' results with `op`
(`add`, `mul`, `min`, `max`) from `init`, `@body` accumulating its chunk
from the accumulator it is handed; `parallel.map @body(%captures...)
%begin, %end, %out` stores `@body(captures..., i)` at `out[i]`. The
operations say nothing about how the range is split: `lower-parallel`
decides that once per build from the configuration -- one chunk on the
calling thread, or chunks spawned through the concurrency dialect and
joined, with a floating-point reduction whose `reassociate` is false
never split -- and the C backend spells what the pass left as OpenMP
regions when that backend was selected. The frontend writes these for
`parallel.range` loops (`docs/language.md`) and outlines each body into a
private function marked `ppy.synthesized`.

## Shapes and layouts

A tensor's shape is a tuple of dimensions, each an integer, a symbol
(`N`), or an expression of them in parentheses -- `(N * M)`, `(N + 1)`,
`ceil_div(N, 32)`, `max(N, M)` -- kept in a canonical form so two
spellings of one size compare equal (`ir/shape.py`). Shape inference for
broadcasting, matmul, reshape, transpose, slice, reduce, and concat lives
there too, and every tensor operation's verifier holds its written result
to what inference says. The layout dialect names how elements sit in
memory: `layout.row_major`, `layout.col_major`, or `layout.strided<s0,
s1, ..., offset, K, align, A>` with a stride per axis, elements before the
first, and the alignment of the first. A tensor operation's meaning is on
the elements; the layout is how a backend reaches them.

## The tensor dialect

`tensor.tensor<f64, 4, 8>` is a 4-by-8 array of `f64`, row-major unless a
trailing layout says otherwise. `tensor.load %buffer` and `tensor.store
%t, %buffer` move a tensor to and from a buffer of its elements in
row-major order, guarded by the buffer's length; `empty`, `fill %scalar`,
`unary {op}` (the math dialect's functions, `neg`, and the special
functions, over every element), `add`, `sub`, `mul`, `div`, `pow`, `min`,
`max` (with broadcasting; a NaN wins `min` and `max`, as NumPy has it),
`broadcast`, `reshape`, `transpose {perm}`, `slice {starts, stops,
steps}`, `concat {axis}`, `reduce {axes, op, keepdims}`, `matmul`,
`convert`, and `fused` -- a region over one element of each operand,
ending in `tensor.yield`, with an optional `reduce` at its root -- are
the values. `lower-tensor` makes memory of a tensor -- a
view onto memory that exists already for `load`, `fill`, `broadcast`,
`transpose`, `slice`, and a contiguous `reshape`; fresh memory, on the
stack when small and static and the heap otherwise, freed where the
function returns, for the rest -- and loops of the operations. A symbolic
dimension is bound where a tensor naming it is loaded: `N` in a load of
`tensor<f64, N, 3>` is the buffer's length over 3, guarded to divide
exactly, and every extent, stride, and allocation over `N` is arithmetic
on that value; a shape naming a symbol nothing has bound, or two unknown
dimensions in one load, is refused with the reason. The linalg, fft, and
sparse operations work on static shapes. A tensor that crosses a call is
refused too: it travels as a buffer.

## The linalg, fft, special, and sparse dialects

`linalg.dot`, `matmul`, `solve`, `triangular_solve {lower, unit}`, and
`cholesky` lower to loops (a singular or non-positive-definite matrix
fails a guard); `linalg.qr`, `svd`, and `eig` call LAPACK where the build
has it -- `dgeqrf`/`dorgqr`, `dgesvd`, `dgeev` on a column-major copy --
and are refused with the reason where it does not. Complex values are
real tensors whose last dimension is 2: `fft.fft`, `ifft`, `rfft`,
`irfft {n}`, `fftn`, and `ifftn` lower to the definition of the transform
in loops, the sum over every input for every output, until a build
selects an FFT library. `special.erf`, `erfc`, `gamma`, `gammaln`,
`ndtr`, `logit`, and the Bessel functions are scalar operations like the
math dialect's, lowered to libm by both backends. `sparse.csr<T, I, rows,
cols>` (and `csc`, `coo`) hold a matrix by its non-zeros with an explicit
index type; `from_parts` borrows the program's buffers, `to_dense`,
`matmul` by a dense matrix, and `reduce {axis}` give dense tensors,
`transpose` of a CSR matrix is the CSC matrix of the same parts, `convert`
changes the format, and `add` merges two matrices of one format into
memory of its own.

## The columnar and arrow dialects

`columnar.column<f64>` is a column of `f64` with no nulls;
`columnar.column<f64, nullable>` carries a validity bitmap. A column's
length is a run-time fact, not part of its type. `columnar.table<a,
column<i64>, b, column<f64, nullable>>` is a table of named columns. The
operations are the ones pandas and PyArrow share: `from_parts %values,
%validity, %length` and `store` move a column to and from Arrow's layout
(values one per row -- one bit for `bool` -- and a bit-packed validity
bitmap); `add`, `sub`, `mul`, `div`, the six comparisons, `and`, `or`,
`xor`, `negate`, `abs`, `invert` give a null where an input is null;
`is_null`, `is_valid`, `fill_null`, `cast`, `select`; `filter` by a bool
column, `take` by positions (a null position is a null row), `concat`;
`sort_indices` (ascending, nulls last, stable); `aggregate {function}` --
`sum`, `mean`, `min`, `max`, `count`, `any`, `all` over the valid rows, a
one-row column that is null when no row was valid; and over tables `make
{names}`, `column_of {name}`, `project {names}`, `filter`, `take`,
`concat`, `group_by {key, aggregates}` (an integer or bool key without
nulls; `sum`, `count`, `min`, `max` per group), and an inner `join {key}`
on one integer key. `lower-tensor` makes loops of them all: `filter`
counts then copies, `sort_indices` is a bottom-up merge sort, `group_by`
sorts by the key and folds each run, `join` is a sort-merge.

`arrow.array<f64>` is an Arrow array as the C Data Interface hands it
over: `arrow.import %p` reads an `ArrowArray` struct through a `ptr<u8>`
-- length, null count, offset, the validity and values buffers -- with no
copy; `arrow.length`, `arrow.null_count`, and `arrow.offset` read the
counts; `arrow.to_column` is that memory as a `columnar.column<T,
nullable>`, the values borrowed at the array's offset and the validity
bitmap re-based to bit zero (all ones when the array has none). A bool
array sliced inside a byte is refused by the guard rather than copied
wrongly.

## Effects and ownership on the IR

A function carries its `effects` -- the lower-case names of the analysis's
vocabulary (`docs/language.md`), with a write through a buffer spelled as
the `write_memory` it is -- and each parameter its `ownership`
(`borrowed`, `mut`, `owned`) and `noalias`. Passes read the effects (an
unused `core.call` to a callee with none but allocation and reads is dead),
and the verifier holds the ownership: a `borrowed` or `mut` parameter is
refused in a `core.ret` and as the value of a `core.store`, the same way a
stack pointer is.

## Text and `.ppyir`

`ppy_compiler.ir.encode` prints a module; `decode` reads it back. The text
is the on-disk format:

```text
ppyir 1                      the schema version
module @name
dialect core 1               every dialect used, with its version
attrs {...}                  module metadata (optional)

global @scale : f64 = 2.0
extern func @sin(f64) -> f64
func @f(%x: i64 {ownership = "borrowed"}) -> i64 attrs {effects = ["pure"]} { ... }
```

Printing is deterministic: values are named in definition order (a hint is
kept unless an earlier value took it; the rest are numbered), attributes
print sorted, and a module printed after being parsed prints the same
text. A reader refuses a schema it does not have and a dialect it does not
have or has only at an older version, with the reason, rather than
guessing. The format is experimental in 0.2.0.

## Verification

`verify(module)` returns every error with the function, block, and
operation it sits on; `verify_or_raise` turns the list into one exception.
The core checks structure (terminators, branch targets and arguments,
dominance, unique names, symbols), then each operation against its
dialect's `OpSpec` (arity, required attributes, types), then the dialect's
own rule for that operation. A dialect adds its rules through `OpSpec.verify`
and `Dialect.verify_type`; a new dialect never touches the verifier.

## Dialects

A dialect is a namespace of operations and types with a version:

```python
class Dialect:
    name: str
    version: int

    def register_types(self, registry): ...
    def register_operations(self, registry): ...  # registry.add_op(OpSpec(...))
    def register_patterns(self, registry): ...
    def register_lowerings(self, registry): ...
    def verify_type(self, t): ...
    def address_spaces(self): ...
```

`ppy_compiler.ir.registry()` is the process-wide registry with the builtin
dialects; a plugin registers its own through the plugin API. Two dialects
of one name from different classes are refused.

## Passes and patterns

A **pattern** roots at one operation name and rewrites through the
`Rewriter`, which is the only way anything changes: every replacement is
recorded, and every operation a change touched is looked at again. The
`GreedyRewriteDriver` applies a `PatternSet` until nothing applies; a
pattern that never settles hits the iteration cap and is named in the
error. A rewrite happens only where the types and the operation's own
semantics allow it -- an integer `x * 0` is `0`, a float `x * 0.0` stays,
`neg(neg(x))` folds under `python` and `wrap` but not `checked` overflow,
and a folded constant that does not fit its type is left for the guard.

Each dialect contributes its patterns through `register_patterns`, so
`canonicalize` is the union of what every registered dialect knows.

A **pass** transforms a module and declares the analyses it `requires`,
`preserves`, and `invalidates`; the `PassContext` serves analyses from a
cache that those declarations empty. With `verify_after_each` the manager
verifies the module after every pass and names the pass that broke it.
The shared passes are `canonicalize`, `constant-fold`, `simplify-cfg`
(constant branches, one-target conditional branches, unreachable blocks,
single-predecessor chains), `dce` (unused pure operations, dead blocks),
`tensor-canonicalize` (the tensor dialect's own patterns: views that
change nothing go away, two transposes or reshapes are one, arithmetic
over `fill`s is a `fill`, `x * fill 1` is `x`), `tensor-fusion` (a chain
of elementwise tensor operations whose intermediates have one reader, and
a `reduce` at its root, become one `tensor.fused` -- a region computing
one element from one element of each input -- so `lower-tensor` makes one
loop of them with no temporaries), `lower-tensor`, and `lower-parallel`;
`transforms.default_pipeline(level)` orders them and marks the
stages -- `after-ir-generation`, `after-canonicalization`,
`before-optimization`, `after-optimization`, `before-backend` -- where a
plugin's `register_stage_pass` puts a pass of its own.

## From Python to the IR

`ppy_compiler.lowering.ast_to_ir` reads an analyzed function once and
writes IR: parameters become entry-block arguments held in stack slots,
Python control flow becomes blocks with arguments (`and`/`or` join through
a block argument, loops have a header, a body, a latch, and an exit), and
every place the program must be handed back to CPython -- a division by
zero, an index out of range, a shift past the word, a byte that does not
fit -- is a `core.guard`. Integer arithmetic carries `overflow = "python"`,
or `wrap` when the safeguards are off, and `//` and `%` carry
`rounding = "floor"`. What the native subset excludes is refused with the
reason, and a caller of a refused function is refused with it.

`backend/llvm/from_ir` reads that IR and nothing else. It gives every
function the native ABI the runtime binds -- machine atoms in, result
slots out, an `i32` status back -- lowers `python` overflow to the
`with.overflow` intrinsics and a branch to the function's fallback block,
`floor` rounding to the sign-corrected sequence (or a shift for a
power-of-two divisor), and block arguments to phis.

`backend/c/emit` reads the same IR and writes C11 or C++17: the same
ABI, `python` overflow through checking helpers (the compiler's
`__builtin_*_overflow` where it has them, plain C otherwise), `floor`
rounding as the sign-corrected sequence, blocks as labels and block
arguments as parallel assignments before a `goto`. `ppy emit c` and
`ppy emit cpp` print it; `tests/test_c_backend.py` compiles it and calls
it on the LLVM road's inputs.

`[tool.ppy.llvm] pipeline = "ir"` selects this road; `"ast"` (the default
while the two are compared) is the direct AST-to-LLVM lowering.
`PPY_LOWERING=ast|ir` overrides the setting for one process, which is how
a differential run of the whole test suite is made; every cache and run
artifact is keyed on the road, so the two never serve each other's
objects. `tests/test_lowering.py` JIT-compiles the same functions both
ways and calls them on the same inputs, fallbacks included.
