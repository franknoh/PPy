# Language reference

A `.ppy` file is valid Python 3.12+. Everything PPY adds is carried by
annotations and decorators from the `ppy` package, all of which are inert at
runtime: under plain CPython the decorators return the function unchanged and
the markers are ordinary `typing.Annotated` aliases. What the compiler adds is
enforcement and speed, never behavior.

## The subset

The compiler analyzes the whole project as one call graph. Inside it:

- Every parameter and return type must be declared or inferable — an implicit
  `Any` is an error (`E1201`), not a silence.
- Attributes are resolved on statically known types; `__init__` declares the
  instance fields.
- `eval`, `exec`, `from x import *`, computed imports, monkey-patching,
  frame manipulation, computed base classes, and unvouched metaclasses are
  rejected (`E15xx`) unless isolated behind `ppy.dynamic`.
- Class construction must be declarative: a class body that executes
  statements, a body value constructing a project descriptor whose
  `__set_name__` runs at creation, or a base whose `__init_subclass__` does
  real work are all `E1507`. The strict checker and the safe hoister judge
  this from the same shared facts (`class_construction` in
  `analysis/decorators.py`), so they cannot disagree about what a class body
  runs.
- A decorator must have vouched semantics — the built-in table, a plugin, or
  `ppy.*` — because a decorator may replace the decorated object: believing
  the `def` while the runtime holds whatever the decorator returned would be
  unsound. An unvouched decorator is `E1204` unless the definition is marked
  `@ppy.dynamic`. `@partial(vouched, ...)` counts as the vouched decorator it
  binds.
- Everything else — classes, generators, closures, `match`, comprehensions,
  decorators the compiler knows, the stdlib it models — is ordinary Python.

Strict mode is the default. `--no-strict` downgrades only the errors that have
a sound fallback.

## Compatibility policy

Three different claims, deliberately held to three different standards:

- **Syntax compatibility — very high.** A `.ppy` file is valid Python; the
  tooling, editors, and formatters that read Python read PPY.
- **Library compatibility — high, through plugins and boundaries.** NumPy,
  PyTorch, JAX/Flax, pydantic, FastAPI and the modeled stdlib work as-is;
  everything else works behind an explicit `ppy.dynamic` boundary.
- **Semantic compatibility — intentionally incomplete.** PPY does not aim to
  preserve arbitrary dynamic Python behavior. `exec`/`eval`, monkey-patching,
  dynamic namespace mutation, computed class construction, and unrestricted
  runtime reflection are restricted in exchange for reliable analysis,
  optimization, and native compilation.

Running existing Python is a migration feature (`ppy migrate`), not the
definition of the language: a valid Python program is not necessarily a valid
PPY program.

## Directives

All directives work bare (`@ppy.pure`) and called (`@ppy.pure()`), and are
contracts the compiler verifies, not hints it trusts.

| directive | meaning |
|---|---|
| `@ppy.pure` | no observable effects: no I/O, no global or nonlocal writes, no mutation of arguments. Local allocation and mutation of locally created values are fine (spec 11.2). Violations are `E1601`/`E1602`. |
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

## Unknown, Any, and Dynamic

Three different absences of a type, held apart on purpose:

- **Unknown** is internal compiler state — inference has not resolved the
  value. It must not survive strict compilation: it is reported (`E1201`,
  `E1304`), never silently widened.
- **`typing.Any`** is the permissive legacy spelling. It absorbs anything and
  the compiler polices nothing about it — use it for interop annotations you
  already trust.
- **`ppy.Dynamic`** is the policed boundary. Any value may become `Dynamic`;
  a `Dynamic` value fits only `Dynamic`, `Any`, or `object`. Crossing into
  typed code — a typed return, parameter, field, or declared variable — is
  `E1508` until it passes through `ppy.check[T](value)`, which validates at
  runtime (raising `TypeError`) and hands back a typed value. `ppy.check` is
  the inverse of `typing.cast`: it checks and asserts nothing, where `cast`
  asserts and checks nothing. Validation is shallow — `list[int]` is checked
  to be a `list`, not walked — because the check runs on the boundary.

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

## Native memory: `ppy.native` and `ppy.ffi`

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
| `native.ptr[T]`, `native.const_ptr[T]` | a typed pointer; `T` is `int`, `float`, `bool`, `ppy.i8`, or `ppy.u8`. A `const_ptr` refuses `store` (`E1631`). |
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
memory.

## Lanes, hints, atomics, threads: `ppy.simd`, `ppy.cpu`, `ppy.atomic`, `ppy.concurrent`

Four namespaces beside `ppy.native`, each with a reference implementation
under CPython and a lowering to a dialect of the IR (`docs/ir.md`), so a
program using them runs the same on every path.

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

`ppy.atomic` is shared memory, one operation at a time, on a
`native.ptr[int]` (or a byte pointer; `load`, `store`, and `exchange` take
`float` too): `load`, `store`, `exchange`, `compare_exchange` -- which
answers `(the value found, whether it swapped)` -- `fetch_add`,
`fetch_sub`, `fetch_and`, `fetch_or`, `fetch_xor`, and `fence`. Each
takes `order="seq_cst"` unless said otherwise (`relaxed`, `acquire`,
`release`, `acq_rel`), and the checker holds what C11 holds: a load is
not `release`, a store is not `acquire`, a relaxed fence orders nothing
(`E1641`). Under CPython the operations serialize under one lock.

`ppy.concurrent` is threads and what keeps them apart. `spawn(f, *args)`
runs a function of the module -- one that returns nothing, with the
parameters a native call takes -- on a new thread and hands back its
handle; `join(handle)` waits for it, and a thread that failed a guard
fails its joiner, which falls back as a whole. The synchronization
objects are memory the program owns, so they are pointers: a mutex is one
`int` slot (`lock`, `unlock`; zero is unlocked), a condition is one `int`
slot counting notifications (`wait(condition, mutex)`, `notify`), a
barrier is two `int` slots (`barrier(slots, parties)`), and every path
implements them the same way over the atomics, spinning with the CPU's
pause hint. `thread_id()` names the running thread. `E1642` names a
misuse. Native code links pthreads for these; a target without them is
refused with the reason.

## Parallel loops: `ppy.parallel.range`

```python
from ppy import Buffer, native, parallel


def squares(out: native.ptr[int], n: int) -> None:
    for i in parallel.range(n):
        native.store(native.offset(out, i), i * i)


def dot(a: Buffer[float], b: Buffer[float]) -> float:
    total = 0.0
    for i in parallel.range(len(a)):
        total += a[i] * b[i]
    return total
```

`for i in parallel.range(...)` says the iterations may run at once. Under
CPython they run in order, which is one of the orders allowed. Natively
the body becomes a function over a chunk of the range and a `parallel.for`
in the IR runs it; the build's `[tool.ppy.parallel] backend` decides how
-- `threads` splits the range across the worker count and joins,
`serial` runs it on the calling thread, `simd` hands the serial loop to
the vectorizer, `openmp` spells it as OpenMP regions in `ppy emit c` --
and every choice gives the same answer. A range too small to be worth the
threads runs serially whatever the setting.

The body may read anything the function holds and write through pointers
and buffers; it may not assign a variable that lives outside the loop,
`break`, or `return`, and a function that does is refused with the reason.
One `+=` or `*=` into an outer `int` or `float` is a reduction, lowered as
a `parallel.reduce`: an integer reduction splits freely, a floating-point
one keeps its order -- the answer would change otherwise -- unless the
function is `@ppy.fastmath`, which permits the reassociation. A thread
whose chunk fails a guard fails the whole loop, and the function falls
back to Python as a whole.

`@ppy.parallel` on a function asks the same of its outermost `range`
loops without spelling `parallel.range`: each that passes the analysis
becomes parallel, and each that does not stays serial and says why in an
optimization remark. `E1650` names a misuse of `parallel.range` itself.
The fused NumPy loops `@ppy.parallel` already split are unchanged.

## Generics

A function may declare type parameters the way Python 3.12 spells them:

```python
def largest[T: int | float](a: T, b: T) -> T:
    return a if a > b else b
```

A call infers the type arguments from what it passes, checks each against
its bound (`E1721`), and has the declared return type with them
substituted: `largest(1, 2)` is an `int`, `largest(1.5, 2.5)` a `float`. An
unbounded parameter has no operators -- nothing says it does -- and a
bound that is a Protocol lends its methods' return types.

Native code monomorphizes: a generic called from a native function is
lowered once per tuple of type arguments, under a name that spells them,
and the call goes straight to that instance; the generic itself keeps its
Python body for every other caller. `[tool.ppy.generics]` bounds the
process -- `max-specializations` per generic (`E1722`) and `max-depth` of
a type argument -- and a generic that calls itself with its own parameter
wrapped in a type is refused outright (`E1723`), because its
specializations never end.

Static dispatch: `a + b` on a value class inside native code calls the
class's own `__add__`, lowered like any native function; native code never
falls back to Python's dynamic dispatch, and a class without a native
operator is refused with the reason.

## Effects and purity

Every function gets an inferred effect set. The vocabulary is the one
every consumer shares -- purity, native and GPU eligibility, code motion,
inlining, parallelization, fusion, the async lowering, plugin contracts:

```text
alloc  read_object  write_object  read_memory  write_memory
read_global  write_global  io  network  random  time
thread  process  sync  atomic  may_raise
python_callback  python_dynamic  gpu_launch  device_memory  external_unknown
```

`read_memory`/`write_memory` are native memory a pointer or a buffer
reaches; `read_object`/`write_object` are Python objects. `@ppy.pure`
asserts the set is empty of the forbidden ones -- everything but
allocation, reads, and raising -- and the checker proves it
interprocedurally: a pure function calling something with unknown effects
is `E1602`, and a callee that mutates the caller's argument is charged to
the caller. The IR carries each function's effects, and passes read them:
an unused call to a function that only allocates and reads is dead code.

## The three execution paths

| | |
|---|---|
| `python f.ppy` | plain CPython. `import ppy` installs a `sys.meta_path` finder so `.py` files can import `.ppy` modules -- natively when the compiler is installed and the module checks clean, as Python source otherwise. |
| `ppy f.ppy` | the optimized Python backend: AST-level optimization (folding, inlining, LICM, loop transforms) executed by CPython. |
| `ppy run f.ppy` | eligible functions compile through LLVM; everything else runs the Python body. |

Any observable difference between the three is a compiler bug. The test suite
and `examples/run_all.py` compare all three on every example. `ppy build`
produces the third path ahead of time: its launcher runs through
`ppy_runtime` with machine code from the library built next to it, and
`ppy build --standalone` goes all the way — a native executable with no
CPython inside, for programs whose reachable graph is entirely native.

## Reading input

`ppy.input[T]()` reads the next value the way `T` says to read it, and the
checker types the result from the same `T`:

```python
n = ppy.input[int]()  # one integer
a, b = ppy.input[tuple[int, int]]()  # two fields, line breaks irrelevant
word = ppy.input[str]("name? ")  # a token, after printing the prompt
values = ppy.input[Buffer[int]](n)  # n integers, straight into a buffer
```

Whitespace and newlines are the same thing to it, as they are to `scanf`.
Reading goes into memory rather than through a Python object per field, so
the buffer form is what takes a million numbers quickly — faster than
`sys.stdin.read().split()` and faster than C's `scanf`, measured in
[examples/15_algorithms/15f_input](../examples/15_algorithms/15f_input/README.md).
The call takes a prompt for a scalar, printed before reading the way the
builtin `input` does, or how many values to read for a buffer.
`ppy.read_ints` and `ppy.read_token` are the lower-level forms that fill a
buffer you already have, and `ppy.buffer[T](n)` makes one: `n` elements of
`T`, all zero. It is `array.array` under CPython and a native allocation in a
standalone binary, which is what lets the same source build both ways.

All of them carry the IO effect, so a function that reads is never mistaken
for a pure one, and all of them work on every path including plain CPython:
the small C reader is compiled once and cached, with a pure-Python fallback
where no compiler exists. The reader owns file descriptor 0 and buffers it
itself, so a program that uses it must not also read `input()` or
`sys.stdin`. Reading past the end raises `EOFError`.

## Native lowering

Eligibility and profitability are different questions, and the compiler asks
both: `can_lower_native` decides whether correct native code exists, and
`should_lower_native` whether crossing the Python/native boundary pays for
it. A function with a loop, a buffer parameter, enough straight-line work,
or an explicit `@ppy.native`/`@ppy.jit`/`@ppy.specialize`/`@ppy.parallel`
gets the boundary; a two-instruction helper stays on the Python side
(remarked as `R3004`) — while native callers keep calling its native symbol
directly, boundary or not.

A buffer's element is a machine word unless it says otherwise:
`Buffer[ppy.i8]` and `Buffer[ppy.u8]` are one byte each, which is what text
and packed data want — four million characters cost four megabytes rather
than thirty-two. The width is storage, not type: reading one hands out an
`int`, arithmetic on it is integer arithmetic, and writing a value that does
not fit in a byte falls back to CPython rather than wrapping. An
`array.array("b", ...)` is what such a buffer is made of.

A function lowers when its types are scalars, `Buffer[T]`, homogeneous
`list[int]`/`list[float]`, `Sequence` of those, or all-scalar `@dataclass`
value classes (flattened into scalar arguments), and its body stays inside the
modeled subset. That subset includes what a loop is normally made of:
`break` and `continue` (a `continue` in a `for` still advances the counter),
statement-level calls whose result is discarded, buffers handed on to
another native function, and the bitwise operators including `~`. A module
constant written as an expression — `MOD = 10**9 + 7`, `LIMIT = 1 << 20` —
folds into the code rather than staying a global read. A function whose
writes all happen inside a callee it handed a buffer to lowers too: the
write lands in the caller's memory either way, and so does the reverse —
filling memory you allocated and then passing it on. That last one is not
`@ppy.pure`, because the callee saw the object before the function returned,
and the two are different questions about one write: purity asks whether
anything could observe it, lowering asks whether CPython has to perform it. A call to a function that did
not lower keeps its caller on the Python side, because the call would
otherwise name a symbol nothing defines. The generated wrapper releases the GIL around the native
call, so `@ppy.native` functions scale across threads. `ppy explain FILE.ppy:name` reports the decision and, when the
answer is no, the first blocking construct.
