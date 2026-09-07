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
| `core.call @f`, `call_extern {callee}`, `call_intrinsic {intrinsic}` | calls; a `core.call` is checked against the callee's signature |
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
single-predecessor chains), and `dce` (unused pure operations, dead
blocks); `transforms.default_pipeline(level)` orders them and marks the
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

`[tool.ppy.llvm] pipeline = "ir"` selects this road; `"ast"` (the default
while the two are compared) is the direct AST-to-LLVM lowering.
`PPY_LOWERING=ast|ir` overrides the setting for one process, which is how
a differential run of the whole test suite is made; every cache and run
artifact is keyed on the road, so the two never serve each other's
objects. `tests/test_lowering.py` JIT-compiles the same functions both
ways and calls them on the same inputs, fallbacks included.
