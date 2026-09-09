# Changelog

## 0.3.0 — unreleased

Work toward the next release, on `dev`; alphas of it are tagged `v0.3.0aN`.

- Three examples compare PPY with the tools that do the same job, code and
  numbers side by side: the parallel ranges against Numba, Taichi, Mojo, and
  NumPy; the CUDA kernels against CuPy and Numba; the eight algorithm kernels
  against Numba, Mojo, and Codon. Each counterpart is in the example's
  `compare/` folder, written the way its tool wants it, and
  `examples/compare.py` holds them all to one answer and tabulates the
  timings. The CUDA table says plainly what PPY lacks: an array that lives
  on the device between launches.
- Regular expressions run natively. A pattern compiled from a bytes literal
  at module level -- `WORD = re.compile(rb"[A-Za-z]+")` -- or written into
  `re.search(rb"...", buf)` becomes a `regex.search`, `regex.match`, or
  `regex.fullmatch` operation over a `Buffer[ppy.u8]`, and the new
  `lower-regex` pass compiles each pattern into a matcher function of core
  operations, so the LLVM and C backends run it as they run anything else.
  The matcher backtracks the way CPython's does and answers exactly what
  `re` answers -- ordered alternation, greedy and lazy repeats, the group's
  last iteration, `$` before a trailing newline, `\b` at ASCII word edges,
  `pos` and `endpos` -- on random inputs across both backends. A match is a
  local whose `start`, `end`, and `span` are native; `m is None` and `if m:`
  narrow it; `group()` stays on Python. Backreferences, lookaround, atomic
  groups, possessive repeats, and locale categories are refused with the
  reason, and a match that would need more than the matcher's stack falls
  back to `re`. The checker types `re.compile`, `re.Pattern`, `re.Match`,
  and the flags. A `while True:` loop that only leaves by returning lowers
  too, which is how a search loop is written.
- The project scan skips a virtual environment by any name -- a directory
  holding `pyvenv.cfg` -- not only `.venv` and `venv`; a second environment
  kept beside the first no longer costs a scan of every package in it.
- `ppy.input[T]()` takes no argument, and `ppy.input[Buffer[T]](n)` takes
  only how many values to read: reading and printing are two things, so a
  prompt is a `print` before the read rather than an argument that meant a
  prompt for one type and a count for another. The checker says so (`E1305`
  for an argument to a scalar read or a missing count, `E1301` for a count
  that is not an `int`), and the converter writes `input("p")`'s prompt as
  `print("p", end="", flush=True)` before the statement that reads -- or,
  inside a loop's test or a comprehension, as the one-expression
  `print(...) or ppy.input[T]()` so it still prints each time. A fill loop
  whose reads carry a prompt stays a loop rather than becoming one bulk read.
- The C backend writes structured code. Loops are `while`, branches are
  `if`/`else` with `break`, `continue`, and `return`, rebuilt from the IR's
  dominator tree and loops; a stack slot that is only loaded and stored is a
  variable named after it, a parameter keeps its Python name and is the
  variable its slot was, and a value read once is written where it is read
  with the parentheses C's precedence needs and no others. Python's floor
  division by a positive constant is `(a % b + b) % b`, a failed guard is
  `if (b == 0) return 1;`, a checked addition into a variable writes the
  variable itself. A graph the reconstruction cannot express falls back, for
  that function alone, to the labels-and-`goto` writer, so every unit is
  still correct; the tests hold both writers to the LLVM road's answers.
  CUDA and HIP get the same treatment, with C++'s `int64_t(x)` casts.
  `ppy emit --format` runs `c`, `cpp`, `cuda`, `hip`, and `header` output
  through clang-format, with the project's `.clang-format` where it has one.

## 0.2.1 — unreleased

- The lowering cache dropped a coroutine's future kind from its signature,
  so the second `ppy run` of a program whose entry coroutine was served from
  the cache bound it through the plain boundary and handed `aio.run` a bare
  handle instead of a future (`TypeError` from asyncio). The cache carries
  the kind now, its schema moved to 7 so no stale entry is served, and a test
  holds the round trip. Found by recording the examples' outputs twice.
- Every example README is rewritten in a plain voice with its detail kept,
  and ends with the commands and what each one prints. A short output is
  inline, a longer one is folded into a `<details>` block, a very long one is
  a file under the folder's `outputs/` that the documentation site embeds;
  each exists once. The six judge problems gained a small `input.txt` so
  their commands run as written.

## 0.2.0 — 2026-09-08

The release that turns the compiler into a platform: a typed, multi-dialect
IR between analysis and every backend, a pass and pattern infrastructure
over it, and plugins that extend it through explicit APIs. The entries
below are in the order the work landed.

- One version, `0.2.0`, in every place that states it: the compiler
  constant the packaging build reads, `ppy_runtime.version`, and
  `ppy.__version__`, held together by a test. Every cache and artifact
  schema moved with it -- the frontend cache, the lowering cache, the
  project scan, and the binding manifest's ABI, now 2 -- so nothing a
  0.1.x compiler produced is served by this one; a 0.1.x manifest is
  refused with the rebuild message.
- The typed canonical IR, `ppy_compiler.ir`: SSA values with one type
  each, blocks with arguments and one terminator, an explicit control-flow
  graph, and operations named in dialects. The core dialect spells overflow
  and rounding on the operation rather than leaving a backend to guess. A
  verifier checks structure, dominance, symbols, and every dialect's own
  rules and returns a list of errors with their positions; the printer
  writes one deterministic text per module, the parser reads it back, and
  `.ppyir` is that text with a schema and dialect-version header a reader
  refuses rather than guesses at. `docs/internals/ir.md` is the reference.
- Passes and patterns over the IR. A pattern rewrites one operation
  through a rewriter that records every change and revisits what it
  touched; the greedy driver runs a pattern set to a fixed point and names
  a pattern that never settles. The core dialect's patterns fold constants
  by the operation's own overflow and rounding attributes, remove identity
  elements, double negations, lossless cast round trips, and settled
  selects and comparisons, and leave alone what floating point or checked
  semantics forbid. A pass manager runs passes that declare what analyses
  they require, preserve, and invalidate, caches those analyses
  accordingly, verifies the module after every pass on request and names
  the pass that broke it, and runs plugin passes at named stages. The
  shared passes are canonicalize, constant-fold, simplify-cfg, and dce.
- The LLVM backend has a second road: Python AST to canonical IR
  (`ppy_compiler.lowering`), the shared passes, and IR to LLVM
  (`backend/llvm/from_ir`), which reads the IR and nothing else. It covers
  the whole native subset the direct road covers -- scalars, fixed tuples,
  value classes, borrowed buffers and their loops, calls between native
  functions, math intrinsics, the standalone shims, specialization with
  pinned constants -- with the same guard hoisting and the same solver
  proofs, and answers alike on every input, fallbacks included: the whole
  suite and every example pass on it. `[tool.ppy.llvm] pipeline = "ir"`
  selects it, `PPY_LOWERING=ast|ir` overrides for one process, every cache
  is keyed on the road, and CI runs the suite on both. One thing the IR
  road does that the direct road did not: `MIN // -1` takes the fallback
  instead of trapping in the division.
- Plugin interface 2. `Plugin` is a base class with a no-op default for
  every hook, so the compiler calls `operator`, `subscript`,
  `instance_attribute`, `call_alias`, `decorator_semantics`, and
  `adjust_call` directly instead of probing for them; the builtin plugins
  extend it. What a call answers about lowering is a typed spec --
  `IntrinsicSpec`, `DialectOperationSpec`, `DirectCallSpec`,
  `GraphRegionSpec`, `FallbackSpec`, `RejectSpec` -- never backend code.
  A plugin registers dialects, patterns, and passes for the IR, and the
  pipeline runs its passes at their stages, verified, naming one that
  breaks the IR (`E1902`). External plugins are discovered through the
  `ppy.plugins` entry-point group without being imported, and load only
  for a project that names them; two plugins claiming one module are a
  reported problem (`E1901`), never a question of who registered last.
- Three builtin plugins: `scipy` (special functions, transforms, dense
  linear algebra, sparse matrices, typed and named as dialect operations;
  the callback-driven families carry their effect), `pandas` (frames,
  series, and indexes typed as what they are, the curated surface named as
  `columnar` operations, everything the model does not capture exactly
  left to pandas), and `pyarrow` (Arrow typed as Arrow, the curated
  compute named as the same `columnar` operations).
- Effect system v2. The vocabulary every consumer shares -- purity, native
  and GPU eligibility, code motion, fusion, the async lowering, plugin
  contracts -- now names native memory apart from Python objects
  (`read_memory`/`write_memory`) and adds `network`, `atomic`,
  `python_dynamic`, `gpu_launch`, and `device_memory`; the socket and
  urllib surface carries `network`. Every IR function carries its effects
  and the passes read them: an unused call to a callee with none but
  allocation and reads is dead code.
- A light ownership model: `ppy.Owned[T]`, `ppy.Borrowed[T]`, `ppy.Mut[T]`.
  A borrow lasts the call -- not returned (`E1611`), not stored where it
  outlives the call (`E1612`), not written unless `Mut` (`E1613`) -- and a
  `Buffer[T]` is borrowed unless the program says otherwise. The IR carries
  `ownership` and `noalias` on parameters, and its verifier refuses a
  borrowed parameter in a return or a store.
- Generics. `def f[T: Bound](...)` declares type parameters; a call infers
  the arguments, checks the bounds (`E1721`), and substitutes them into the
  result. Native code monomorphizes: a generic called from native code is
  lowered once per tuple of type arguments under a name that spells them,
  and the call goes straight to the instance. `[tool.ppy.generics]` bounds
  the specializations (`E1722`), and a generic that feeds its own type
  parameter back into itself wrapped is refused (`E1723`). Inside native
  code `a + b` on a value class dispatches statically to the class's own
  `__add__`; native code never falls back to dynamic dispatch.
- The `math` dialect: elementary functions named once for every backend,
  pure, folding on constants, `floor(floor(x))` once. The frontend writes
  `math.sqrt` where the source says so; the LLVM backend lowers it to the
  intrinsic of that name.
- `ppy.native` is the directive it was and a namespace of typed native
  memory: `ptr[T]`/`const_ptr[T]`, `load`, `store`, `offset`, `cast[U]`,
  `sizeof[T]()`, `alignof[T]()`, `stack_alloc[T](n)`, with a reference
  implementation over `array` memory under CPython so the three paths
  agree, and pointer operations in native code. `@native.extern` binds a
  stub to a C symbol -- ctypes under CPython, a direct call in native code,
  the library linked and loaded -- and `@native.export` gives a function a
  public C symbol, with a header written beside the built library and a
  trap where Python would have taken the fallback. `ppy.ffi` is the
  binding layer over it: `library`, `bind`, `nullable`, `LengthOf`.
- `ppy emit ir|llvm-ir TARGET [-o]` prints a compiler stage as text, one
  rule for every kind; `.ppyir` is the IR's on-disk form, self-describing
  down to each function's ABI, and `ppy build foo.ppyir` builds an
  object, a library, and a manifest from it alone.

Speed of the compiler itself, measured before being changed.

- A warm `ppy run` no longer imports the compiler. The first run of a
  program builds its artifact into the cache and launches it; the next run
  finds the artifact by a key over everything that could change it and goes
  straight to `ppy_runtime`. On a small program that was 1.65 s of imports,
  LLVM initialization, and re-analysis before the first line ran; it is now
  the launcher's few dozen milliseconds plus the key. Programs that
  specialize at runtime, fuse NumPy kernels, or use the JAX plugin keep the
  in-process JIT, and say so in a `needs-jit` note.
- A built artifact carries its torch ATen regions. The extension compiled
  against the installed PyTorch is copied beside the manifest and recorded
  under `regions`, and `ppy_runtime` loads it with no compiler in the
  process; a region library that has gone missing is the Python body. So a
  program that imports torch is cached and launched like any other, and a
  `.ppy` kernel that imports torch is served natively by `import ppy`.
- `ppy build --warm TARGET` builds ahead of time what `ppy run` and
  `import ppy` build on first use, for one module or every `.ppy` under a
  directory, under the project configuration alone -- flags that would key
  a different artifact are refused. It is the step before a launch that
  starts many ranks at once: each finds the build instead of making its own.
- `examples/31_torchrun`: a trainer with two `.ppy` kernels -- native
  preprocessing loops and an ATen-region model -- that runs the same under
  `python`, `torchrun`, and `accelerate launch`, with `import ppy` as the
  whole integration.
- A development tree fingerprints the compiler by the sizes and mtimes of
  its sources rather than by reading them all, which was a third of a
  second on every command.
- Inference stops re-checking what nothing moved. Each round used to seed
  every function's summary again and then check every module twice to
  confirm it; now the seed runs once per project, and a module whose
  inputs -- its own signatures and fields, and those of everything it
  imports -- have the digest they had when it was last checked keeps its
  analysis instead of being checked again. The answers are byte for byte
  what they were.
- `ppy convert` and `ppy migrate` stop repeating themselves on large
  files. The reflection index walked a module's whole tree once per
  function it held; the write index and the reflection index each walked,
  parsed, and lexically scanned the whole project on their own (the latter
  twice); the lexical scan snapshotted every statement's environment onto
  the two `Load`/`Store` singletons and spent most of its time joining
  them; and every migration pass walked every module whether or not the
  module spelled anything it rewrites. The alias analysis, which runs for
  every function on every inference round, did the same snapshot-and-join
  on the `Load`/`Store` singletons inside its loop fixpoint -- a loop inside
  a loop multiplied it -- and now records without joining and joins only
  what is asked about. The alias map is also computed once per function
  per project rather than once per checker pass -- it depends on the body
  and on which parameters are immutable, and on nothing else, while the
  checker runs at least twice per analysis and once per inference round. A
  3270-line module migrates in 5.6 s where it took 10.5 s, with
  byte-identical output; the compiler's own 26,600 lines migrate in 50 s
  where 85 s used to end in a crash.
- An error that would have to say `<unknown>` is not reported. It only
  restates a type that was never resolved, and that unresolved type already
  has its line -- the untyped parameter (`E1201`, `E1304`) or the call with
  no signature (`E1306`). A 3270-line module went from 332 reported errors
  to 35, and the 35 are the findings; one `W2006` line says how many were
  withheld and names the unknown signatures behind them.
- A field is typed by everything its class assigns to it, joined, rather
  than by the first assignment in `__init__` alone: `self.buffer = None`
  there and a tensor in `setup()` is `Tensor | None`, where it used to be
  `None` and every later assignment an error. A field the class body
  annotated keeps the annotation.
- Three checker gaps the compiler's own source exposed are closed: the set
  algebra (`a | b`, `a & b`, `a - b`, `a ^ b` on sets and frozensets) was
  "not defined", `frozenset(...)` was typed as a `set`, and a project
  class's own `__or__`, `__add__` and the other operator methods were never
  consulted. Together they were 193 of the compiler's 627 self-reported
  errors, and none of them was a finding.
- A name imported from a library module is what the library says it is:
  `from pathlib import Path` then `Path(p)` was an unknown signature while
  `pathlib.Path(p)` was not. `pathlib.Path` is modeled -- construction, the
  filesystem calls with their IO effect, the spellings of other paths, the
  string parts -- and so are the `ast` functions (`parse`, `walk`,
  `unparse`, and the rest). `x: type` is an annotation, and what a class
  held as a value exposes is known. libcst's node classes are opaque
  types.
- `f(*pair)` is as many arguments as the pair holds, and `[first, *rest]`
  holds what `rest` holds. `type[Base]` is the class object of `Base` or a
  subclass, a class is callable, and `dict[str, dict[str, int]]` is a
  `dict[str, Any]`: `Any` is anything at any depth of an invariant
  argument. `str.partition` and the rest of the `str` methods with a known
  result are modeled; an external class's bases are spelled the way the
  annotation spells them, so a `libcst.Call` is a `libcst.BaseExpression`;
  `sqlite3`'s classes are annotations.
- A class whose members cover a project `Protocol`'s is an instance of it,
  and one that defines `__iter__` is an `Iterable`. A function, a class, a
  module: everything is an `object`. `type(x)` of something unresolved is
  some class and fits `type[Base]`; `list | dict` in an `isinstance` is a
  type; `table.get(type(node.op))` may ask with a wider key than the table
  holds. A library exception is a `BaseException`.
- `program or {}` is never `None`: an operand of `or` that is not the last
  is the result only when it is truthy.
- A constant subscript is a place a check can be about: after
  `if args[0].facts is not None`, `args[0].facts` is not `None`, until
  `args[0]` is written. A union of tuples unpacks position by position, so
  `count, first = seen.get(key, (0, node))` gives an `int` count. Being one
  of the constants in `x in {"a", "b"}` is being of their type.
- `ppy check`, `run`, and `build` know the fields `__init__` assigns.
  `self.width = width` says as much about the field as an annotation
  would, and only `convert` and `migrate` used to hear it: the single-pass
  path reported `has no attribute` on every read of such a field. The
  analysis now settles those fields in its own fixpoint, from what its seed
  pass saw assigned, so every path starts from the same fields and the
  reporting pass reads fields that are known. No extra pass: `ppy check`
  costs what it did.
- Every whole-tree scan iterates a node tuple the tree's owner walked once:
  a conversion made a dozen passes over the same trees through `ast.walk`,
  which was a tenth of its time. A branch merge keeps a binding both sides
  share instead of joining it with itself, and the alias snapshot's revisit
  check is a set lookup rather than a scan of a list that grew with every
  loop iteration the fixpoint needed.
- A project class that subclasses a builtin has the builtin's methods:
  `class Reached(list)` may call `self.append`, and `super().__init__(...)`
  past a base the project does not define is a call that returns nothing.
- A migration pass runs only over a module that has the shape it rewrites,
  decided on the syntax tree rather than on a substring: `getattr` is in
  most files, `getattr(x, "name")` in few, and a pass that found nothing
  still paid for a position-annotated traversal of the whole module. The
  builtin method table is built once per receiver type and attribute
  rather than on every attribute read; an annotation's expression is
  parsed once per spelling; the rewriter hands its module to the
  normalizer instead of printing and parsing its own output. A path is
  resolved once per directory and a module's file is probed for once per
  graph build: every importer of `pkg.util` used to stat the same four
  paths, and resolving walked every component with an lstat. A migration
  pass resolves positions once, after it ran, and only for what it
  rewrote; and each function's environment starts from the module's
  imports and classes seeded once per pass rather than rebuilt per
  function. Signatures are wrapped only when some line of the module is
  past the limit.
- The whole-project scan behind `Final` and annotation materialization
  keeps, per file, only what the two indexes need -- the attribute writes
  it makes on other modules, the annotation readers it holds and calls --
  and serves that record from the cache store when the file's path, size,
  and modification time match one written by this compiler. Converting one
  file of a large project costs a stat per file, not a parse.
- A class's MRO is derived from its name, not part of a type's identity:
  `pathlib.Path` reached through the analyzer's own table and through the
  library's `__mro__`, which Python 3.13 spells with more bases than 3.14,
  was a union of two types that printed alike, and `path / "x"` was an
  error on one Python and not the other.
- A module the analysis did not need to recheck still reports what it
  found. The fixpoint keeps a module whose inputs have not moved between
  rounds, and the kept module's diagnostics lived in the earlier round's
  discarded bag: `ppy check b.ppy`, with `b` importing an `a` whose error
  was already settled, said "no errors". Each module now keeps its own
  report, and the round that settles reports every module.
- `llvm.prover = "z3"`, or `--prover z3` on `ppy run` and `ppy build`:
  the solver proves overflow guards away where the analysis allows it. A
  chain of `+`, `-`, `*` is stated as an obligation over integers -- each
  load a variable with the range the analysis recorded, each induction
  variable carrying `start <= i <= stop - 1` from its `range()` -- settled
  by intervals when they suffice and by Z3 otherwise; a proven chain lowers
  to the plain `nsw` instruction, an unproven one keeps its guard exactly
  as before. A function whose guards a proof may leave out checks its
  parameters' declared ranges once on entry, and a call outside them takes
  the fallback, so the three paths agree on every call. Needs
  `ppy-lang[solver]`; `ppy doctor` reports the solver; the artifact and
  the warm run directory are keyed by the prover and its version.
  `docs/internals/solver.md` says where a solver fits, where it does not, and what is
  next.
- A type alias imported from another module is read in the module that
  defines it: `from .obligations import Term` with `Term = Union[Var,
  Const, BinOp]` was "not a type the project can analyze", because the
  alias table was the importing module's. Arithmetic on a union of numbers
  is arithmetic on the widest of them: `x / 2` with `x: int | float` is a
  `float`, not an undefined operator.
- A dataclass is built the way `dataclasses` builds it: the fields of each
  dataclass base first, a field with a default or a `field(default=...)`
  optional, `kw_only=True`, `field(kw_only=True)` and everything after a
  `KW_ONLY` marker by keyword only, `InitVar[T]` a constructor parameter
  and not an attribute, and a required field left out is an error.
  `Stats(*totals)` is checked by the list, not by position.
- `PAGE = 8` in a class body is a class attribute, read through the class
  and through every instance, and not a dataclass field.
- `self.a, self.b = x, y` sets two fields. `mask | other` on arrays is
  `bitwise_or`, with `&`, `^` and the shifts. Being one of the constants in
  `x in {"small", "medium", "large"}` is being those literals, which a
  `Literal[...]` return type accepts.
- An instance of a project class is callable through its `__call__`, or
  through the method a plugin names for an external base: an `nn.Module`
  subclass is called through `forward`. `torch.nn.Module`'s members are
  modeled -- `parameters`, `state_dict`, `to`, `eval`, and the rest -- and
  a method that returns the module returns the subclass it was called on.
  A bare container iterates: `isinstance(x, tuple)` leaves a `tuple` that a
  `for` can walk. `sys.path`, `sys.modules`, `sys.version_info` are known.
  `--no-strict` is accepted after the subcommand as well as before it,
  except on `convert`, which has no strictness escape hatch by design. A
  bare `tuple` annotation is `tuple[Any, ...]`, any length, and a
  `dict[str, tuple]` holds a `dict[str, tuple[A, B]]`.
- A run whose Python lacks its headers says so: the ctypes boundary is
  correct and several times slower per call than the generated CPython
  ABI, and a node built from a system interpreter without `python3-dev`
  used to find that out only from `ppy doctor`. On one real kernel the
  difference was a native call at 14 us against the interpreted 5.6 us,
  becoming 1.3 us once the headers were there.
- `ppy doctor` prints the C library it found, and `docs/reference/compatibility.md`
  says what the platform floor is: the wheel is pure Python, everything
  native is compiled where it runs and binds to that machine's libc, and
  the dependencies' wheels set the minimum -- glibc 2.17 for `llvmlite`,
  `libcst` 1.7, `z3-solver` 4.x and `numpy` up to 2.2; 2.27 for `z3-solver`
  5.x; 2.28 for `numpy` 2.3 and `libcst` 1.8. `ppy-lang` pins `libcst<1.8`
  on Python 3.13 and earlier: from 1.8 it ships `manylinux_2_28` wheels
  only, and on Ubuntu 18.04 the install fell back to building its Rust
  sources and failed. Python 3.14 has no 1.7 wheel and takes 1.8.
- `import ppy` is the whole integration. With the compiler installed, a
  `.ppy` module imported from a plain `.py` program is served from its
  native build: the first process builds it into the project's cache --
  the artifact `ppy run` builds -- and binds its functions through the
  prebuilt binder, every later process finds the build, and a kernel that
  does not check clean, or needs the in-process JIT, loads as Python source
  with one line on stderr saying why. A program carved into `.ppy` kernels
  needs no bootstrap and no launcher. `PPY_IMPORT=python` turns it off for
  a process, `native-import = false` under `[tool.ppy]` for a project,
  `PPY_QUIET=1` silences the notes, `ppy.native_import(False)` decides in
  code, and `ppy.native_imports()` says which modules came in native. The
  build itself is the compiler's (`ppy_compiler.driver.native_import`), asked
  for by name: the runtime still never requires the compiler, and a `.ppy`
  program without it loads as source.
- `scripts/dogfood.py` counts the errors in a target's own files: a module
  reached through an import is another target's business.
- `ppy check pkg/a.py` checks `pkg.a`. A file inside a package used to be
  named from its own directory -- `a` -- so every `from . import b` in it
  was unresolved, on the one command people run most against the file they
  are editing. And `from pkg import mod` now follows `mod` into the program
  when it is a module rather than a symbol: `from . import types as T` left
  `T.Type` with nothing behind it whenever `types` was not itself an entry,
  which was the compiler's own largest remaining self-reported error and
  every one of its 3,000 withheld consequences.
- A name a package re-exports resolves to where it is defined: `from
  ..diagnostics import Diagnostic` follows `diagnostics/__init__.py` to
  `.model`. A bound method passed along as a value (`notify=reporter.note`)
  is a callable without its receiver, however it is later called. A check
  on an attribute narrows that attribute: after `if self.end is not None:`,
  `self.end - 1` is an `int`, and so for `isinstance(self.node, Call)` and a
  truthiness test, until the attribute or its owner is assigned again. A
  walrus inside an `and` chain binds its name for the body. `dir` is a
  builtin. The compiler's self-reported errors went from 627 to 195 across
  these rounds, and its runtime's from 182 to 84.
- `super()` in a class whose base is not the project's -- an `Exception`
  subclass -- is `super()`, not an undefined name. A list, set, or dict
  written out beside a declared type is held to that type's elements
  rather than typed from its first element and then rejected
  (`stack: list[ast.AST] = [stmt]`). Dictionary views (`keys()`, `items()`)
  are set-like, so `left.keys() | right.keys()` is defined.
- An annotation may name a common standard-library class -- `pathlib.Path`,
  every `ast` node, `re.Pattern`, `datetime`, `collections.deque`,
  `argparse.Namespace`, and others -- without the analyzer modeling it.
  Each carries its real hierarchy, read from the class, so an `ast.Call` is
  accepted where an `ast.AST` is expected; `Path / "name"` is a `Path`; and
  `Ellipsis` and `NotImplemented` are names. `p: Path` used to be "not a
  type the project can analyze" in every file that took a path. Resolving
  these surfaced a few findings the unknown had been hiding, so the
  runtime's dogfood ceiling moved from 119 to 127 while the compiler's
  came down from 315 to 287.
- The `math` module is modeled: all 57 functions and the five constants,
  with the exceptions the C implementation raises. Every numeric kernel
  imports it, and `math.tanh` in a reward function was an unknown
  signature that everything computed from it followed.
- Every builtin exception is a known name. Thirteen were listed by hand and
  `AssertionError`, `RuntimeError`, `OSError` and fifty-three others were
  "not defined at this point"; the table now reads the interpreter's own
  hierarchy.
- [docs/internals/migrating.md](docs/internals/migrating.md) says what to hand `ppy migrate` on
  a real project: profile, find the two or three files that do the numeric
  work, migrate those, and leave the orchestration as `.py` importing them
  through the loader. It also says how to read the report -- `E1304` is the
  to-do list, `W2006` is the count of what follows from it, and the rest is
  about the code.
- The compiler migrates itself in CI. `scripts/dogfood.py` runs the
  converter over `src/ppy_compiler`, `src/ppy_runtime`, and `src/ppy` on
  every push and fails on a crash, on `<unknown>` in any message, or on
  more errors than the recorded ceiling, which only ratchets down. Thirty
  thousand lines of real Python found the crash below, the missing
  exceptions above, and the cascades; now they keep finding things.
- `ppy migrate` no longer crashes on a module whose first statement is a
  relative import: placing the `ppy` import spelled the missing module name
  as an empty identifier, which libcst refuses.
- The C backend, `ppy emit c`, and its C++ form, `ppy emit cpp`. Both read
  the canonical IR and write one translation unit per module: every
  function in the ABI the runtime binds, every export behind its public
  signature (`extern "C"` in C++), the overflow helpers and runtime shims
  the unit uses and no others, so it compiles on its own and answers what
  the LLVM road answers, fallbacks included. The C++ output is the
  emitter making C++ choices, never C text rewritten. `--header-only`
  makes every function `static inline` under a guard and refuses, with
  `E1804`, a feature that needs state the process owns; `--standalone`
  emits a whole program from `main`; `ppy emit header` prints the export
  declarations a built library ships. `tests/test_c_backend.py` compiles
  the C and the C++ and calls them on the LLVM road's inputs.
- `TargetInfo`, and builds for another machine. One record holds what the
  compiler knows about a target -- triple, CPU and features, pointer
  width, endianness, ABI, OS, object format, data layout -- the host being
  one target among others, and the cache keys, the warm key, `ppy doctor`,
  and the linker ask it instead of `sys.platform`. `ppy build --target
  TRIPLE` (or `[tool.ppy.llvm] target`) retargets the objects and links
  them with a toolchain for the triple; the wrapper and the launcher, which
  only the running interpreter can build, are left out with a note, and
  the manifest names its target so a runtime elsewhere refuses it.
- `ppy build --python-extension`: one importable CPython module of the
  native code, the generated boundary, and the module's own Python, bound
  as it is defined through the same hook the launcher uses; and `ppy build
  --library`: the exports laid out as `lib/`, `include/`, a pkg-config
  file, and the manifest.
- `ppy bind header foo.h`: PPY bindings for a C header, read through
  libclang (`ppy-lang[bind]`) -- functions as typed `@ffi.bind` stubs,
  typedefs, enums, structs of scalars as dataclasses, numeric `#define`s
  as constants -- with what has no spelling yet listed by name rather than
  guessed at.
- Four dialects and their namespaces: `simd` (`vector<T, N>` made, moved,
  shuffled, and reduced in lane order), `cpu` (prefetch and pause hints,
  and the `cpu.features` a function is compiled for), `atomic` (every
  operation with its C11 memory order, verified), and `concurrency`
  (spawn, join, and mutexes, conditions, and barriers that are memory the
  program owns, implemented over the atomics the same way on every
  backend). The LLVM and C backends lower all four; `ppy.simd`, `ppy.cpu`,
  `ppy.atomic`, and `ppy.concurrent` carry them into the language with
  reference implementations under CPython, the checker's rules
  (`E1640`-`E1643`), and the frontend's lowering, so a program using them
  runs the same on every path. `@cpu.target("avx2")` compiles a function
  with the features on, and the boundary binds it only where they are.
- `ppy.parallel` v2. The parallel dialect -- `parallel.for`, `reduce`,
  `map` over an outlined body -- and `lower-parallel`, which decides once
  per build how a range is split: one chunk on the calling thread
  (`serial`, `simd`), chunks spawned through the concurrency dialect and
  joined (`threads`), or OpenMP regions the C backend spells (`openmp`);
  every choice gives the same answer, a floating-point reduction keeps its
  order unless `@ppy.fastmath` permits otherwise, and a chunk that fails
  a guard fails the loop. `for i in parallel.range(n)` in the language,
  with one `+=`/`*=` reduction, and `@ppy.parallel` asking the same of a
  function's outermost loops; the checker (`E1650`) and the frontend refuse
  what cannot run at once and say why, and optimization remarks say what
  became parallel.
- The tensor family of the IR: shapes with symbols and expressions and the
  inference the verifiers hold operations to, the layout dialect, the
  tensor dialect and `lower-tensor` (views where memory exists, loops
  where it must be made, on the stack or the heap), the linalg dialect
  (loops for dot, matmul, the solves and Cholesky; LAPACK for QR, SVD, and
  eigendecomposition where a build has it), the fft dialect (the transform
  by its definition, complex as `(re, im)` pairs), the special dialect
  (erf, gamma, Bessel and friends through libm on both backends), and the
  sparse dialect (CSR, CSC, COO with explicit index types; matmul, add,
  transpose, convert, reduce). All of it is checked against NumPy on both
  backends.
- The numeric plugins converge onto the tensor IR. A plugin names the
  shared operation a call is (`tensor_operation`): `numpy.multiply`,
  `torch.mul`, and `jax.numpy.multiply` are `tensor.mul`; `numpy.sum` and
  `torch.sum` are `tensor.reduce {op = add}`; `scipy.special.erf` over
  arrays is `tensor.unary {op = erf}`. The fused kernels are built from
  that vocabulary as tensor IR -- `tensor.fill`, `tensor.unary`, `pow`,
  `min`, `max` joined the dialect, and `lower-tensor` binds a symbolic
  dimension from the buffer a tensor is loaded from -- so nothing in the
  fusion path writes LLVM IR any more, and one kernel runs over a NumPy
  array or a CPU torch tensor alike behind each library's guards. The
  torch plugin reports its curated arithmetic as the shared operations
  while `matmul` and the other dispatcher-sensitive calls stay with the
  dispatcher; `ppy explain` shows the shared operation beside a call's
  lowering.
- Tensor canonicalization and fusion on the IR. The tensor dialect's
  patterns remove views that change nothing, compose transposes and
  reshapes, fold arithmetic over `fill`s into one scalar computation, and
  drop neutral elements where that is exact. `tensor-fusion` turns a chain
  of elementwise operations whose intermediates have a single reader --
  with a `reduce` at the root where there is one -- into `tensor.fused`, a
  region computing one element from one element of each input, and
  `lower-tensor` makes a single loop of it; a result whose only reader is
  a `tensor.store` is written straight into the store's buffer. A fused
  NumPy or torch kernel is now one loop with no temporary and no copy, and
  the pass reports `tensor ops fused` for every group it makes.
- Autodiff. `ppy.grad(f)` and `ppy.value_and_grad(f)` -- with `argnums`
  -- differentiate a function of floats whose body is assignments and a
  return over arithmetic, the math functions, `abs`, `erf`. Under CPython
  the derivative is made from the source; natively the `autodiff`
  transform differentiates the function's IR in reverse mode -- scalar
  arithmetic, `select`, casts, and over tensors the elementwise
  operations, `fill`, `broadcast`, `reshape`, `transpose`, `reduce` by
  sum, `matmul`, `convert`, with a buffer parameter's gradient written to
  a buffer -- after `promote-slots` has turned the frontend's stack slots
  back into values. One rule table and one order of accumulation on both
  paths, so they agree bit for bit; a branch, a loop, a write, or an
  effect is refused with the reason (`E1660`-`E1662`).
- The columnar and arrow dialects. A `columnar.column<T, nullable>` is a
  column with a validity bitmap and a run-time length, a `columnar.table`
  a set of named columns; the operations pandas and PyArrow share --
  arithmetic, comparison, and boolean logic with null propagation,
  `cast`, `is_null`, `fill_null`, `select`, `filter`, `take`, `concat`,
  `sort_indices`, `aggregate`, and over tables `project`, `group_by`, and
  an inner `join` -- are values the verifier checks, and `lower-tensor`
  makes loops of them in Arrow's layout: bit-packed validity, bit-packed
  bools, a merge sort for ordering, sort-based grouping and a sort-merge
  join. `arrow.import` reads an array from the Arrow C Data Interface
  struct without a copy and `arrow.to_column` makes it a column. Checked
  against NumPy on both backends and against an `ArrowArray` built by
  hand.
- PyArrow converges onto the columnar IR. The plugin names the dialect's
  operations, and an expression tree of `pyarrow.compute` calls over
  `float64` and `bool` arrays -- arithmetic, comparison, boolean logic,
  `if_else`, `fill_null`, `is_null`, `is_valid` -- is fused into one
  kernel built as columnar IR, run over the arrays' own buffers behind
  the same guards NumPy's kernels have, and answered as an Arrow array
  over the buffers the kernel filled. pandas spells the same operations
  by the same names.
- pandas converges onto the same kernels. A tree of Series arithmetic,
  comparisons, `fillna`, `isna`/`notna` is one columnar kernel; an
  Arrow-backed Series is read as its Arrow array, a NumPy-backed `float64`
  Series as its values behind a bitmap of ones, and the answer is a Series
  over the callers' index with the same backing. Alignment of different
  indexes, mixed backings, nullable extension dtypes, and bool answers
  over NumPy storage stay with pandas; a Series operator now resolves to
  the plugin's operation (it was composed into a name no plugin knew).
  `ppy_runtime.arrow.exported` lends
  a PyArrow array to native code as the C Data Interface's `ArrowArray`
  struct and releases it once the borrow ends.
- `ppy.xla`, the StableHLO backend, and the PJRT bridge. `@xla.jit` marks
  a function of scalars whose body is arithmetic and math; the compiler
  lowers it to the IR, `backend/stablehlo` writes it as an MLIR module --
  scalars as rank-0 tensors, the tensor dialect one to one onto StableHLO
  (`broadcast_in_dim`, `reshape`, `transpose`, `reduce`, `dot_general`,
  `convert`, a fused region as its body), `erf` as CHLO -- `ppy emit
  stablehlo` shows it, and the build stages it. At run time
  `ppy_runtime.xla` compiles the module through XLA's own bindings, caches
  the executable by digest, bindings, platform, and device (in memory and
  serialized on disk), and runs each call on the device; `xla.devices()`
  names them. JAX is not on the compile path; the bridge places buffers
  through JAX's `device_put` while that is the one public way to the
  client. What XLA cannot take is reported as `W2007` and runs as written.
- The gpu dialect: one execution model every GPU backend meets. A function
  is `host`, `device`, or `kernel` by its `gpu.kind`; device code reads
  `gpu.thread_id`, `block_id`, `block_dim`, `grid_dim` (each `.x`, `.y`,
  `.z`), waits at `barrier` and `subgroup_barrier`, trades values with
  `subgroup_shuffle`, takes `shared_alloc` and `private_alloc` memory, and
  is handed pointers into `global` and `constant` memory; the host launches
  a kernel with `gpu.launch`. The verifier holds the kinds -- a device
  operation in a host function; a guard, a buffer, a host call, or another
  dialect in device code; a kernel that returns or takes stack memory --
  through the new `Dialect.verify_function` hook, and the LLVM backend
  leaves device code to the GPU backends. A program without a kernel is
  untouched.
- The IR's text reader ends an operation at its line. An operation with
  nothing after its name -- `gpu.barrier`, a bare `core.ret` before the next
  block -- used to take the following line's value or label as its own
  operand or successor; the printed form always was one operation per line,
  and the reader now holds it to that.
- `ppy.cuda` and `ppy.hip`, and the CUDA/HIP source backend. `@cuda.kernel`
  and `@cuda.device` (or `hip.`) mark device code; `thread_id`, `block_id`,
  `block_dim`, `grid_dim`, `global_id`, `syncthreads`, `syncwarp`,
  `shared[T, N]()`, `local[T, N]()`, the `shfl` family, and `launch(kernel,
  grid, block, *args)` are the vocabulary, lowered to the gpu dialect by
  the one frontend -- inside device code `int` arithmetic wraps and nothing
  guards -- and under CPython a launch runs the grid on threads that know
  their position, a reference with real barriers and shuffles. `ppy emit
  cuda` and `ppy emit hip` write a module as CUDA or HIP C++: `__global__`
  kernels, `__device__` functions, `__shared__` memory, `__shfl_sync` or
  `__shfl`, and `<<<grid, block>>>` launches followed by a device
  synchronization whose status the host function reports. The CPU backends
  leave device code alone, and a launching function stays in Python until
  the launch runtime. `E1644` names a misuse. A C++ unit now includes a
  header that is not C's standard library -- `pthread.h`, `omp.h` -- as it
  is spelled rather than as `<cpthread>`.
- The NVVM backend and the CUDA launch runtime. `backend/nvvm` writes a
  module's kernels and device functions as LLVM IR for NVPTX -- the LLVM
  backend's own lowerings under the `nvptx64-nvidia-cuda` triple, a kernel
  a `ptx_kernel`, positions the `llvm.nvvm.read.ptx.sreg.*` registers,
  `barrier0`, `addrspace(3)` shared memory, `shfl.sync` shuffles (a 64-bit
  value as two halves), libdevice's `__nv_*` for the math library -- and
  as PTX through LLVM's NVPTX backend with libdevice linked and pruned;
  `ppy emit nvvm-ir` and `ppy emit ptx` show them. A build stages each
  kernel's PTX with the kinds of its parameters (cached like every
  artifact), and under `ppy run` `cuda.launch` runs it through the CUDA
  driver by ctypes -- no toolkit needed -- copying a `native` pointer's
  array to the device and back; `cuda.compiled(kernel)` says whether a
  launch runs there. Without the driver, a device, or the NVPTX backend
  the reference launch runs and `W2008` names the reason; `PPY_CUDA_ARCH`
  picks the PTX architecture (`sm_70` by default). A built artifact now
  carries its staged exports -- a kernel's PTX, an `@xla.jit` function's
  StableHLO -- as files beside the manifest, and the launcher binds them
  without the compiler, so the warm `ppy run` path and `ppy run --prebuilt`
  route them as the JIT path does.
- `ppy.aio`, the async dialect, and the native async runtime. `aio.sleep`,
  `accept`, `connect`, `read`, `write` are awaitables, `spawn` starts a
  task, `listen`, `port`, `close` are immediate, and `run` drives a
  coroutine; sockets are ints and answer a negative errno rather than
  raising, so every path says the same. An `async def` of scalars and
  pointers whose awaits are these and other coroutines lowers to the async
  dialect -- `create`, `start`, `await`, the IO operations -- and
  `lower-async` makes it a starter and a resume function over a frame the
  runtime owns, spilling what lives across an await; the LLVM and C
  backends call the runtime, one C file compiled once into the cache
  (Linux, epoll; elsewhere asyncio runs everything). Calling a compiled
  coroutine hands back a `NativeFuture` that `aio.run` drives and asyncio
  can await; a built artifact links the runtime in. A guard failing inside
  a running coroutine fails its future and `aio.run` raises
  `NativeGuardFailed`. `E1645` names a misuse.
- The IR linker and the package-level build. A call from one module into
  another's native function now lowers as a declaration (`ppy.external`)
  the driver only allows for a callee it has already lowered, so the JIT
  resolves it across modules and `ppy build` links the modules' IR into one
  program: definitions answer declarations, shared generic instances are
  kept once, colliding private symbols are renamed with their references,
  dialects and libraries are merged. Whole-program optimization then
  internalizes what Python never binds, inlines small callees and
  `@ppy.inline` ones across module seams (`@ppy.noinline` holds), and drops
  dead private code, and one object comes out; `ppy emit linked-ir` prints
  the program. `.ppyir` is public from 0.2.0 at schema 1.
- Sanitizers, the IR stage debugger, and the optimization report. `ppy run`
  and `ppy build` take `--sanitize bounds,overflow,pointer,alignment` (or
  `[tool.ppy.llvm] sanitize`): the `sanitize` pass instruments the IR
  before optimization -- every buffer index, every wrapping or proven
  integer operation, every load and store through a pointer -- and a check
  that fails returns a sanitizer status the boundary turns into
  `SanitizerFailure` rather than a fallback; `lifetime` and `alias` are
  refused with the reason. The IR gained the null pointer constant and the
  pointer-to-integer cast the checks need. `ppy inspect --stage` prints
  the program at any stage -- analysis, ir, canonical, tensor, columnar,
  optimized, gpu, stablehlo, llvm -- and `ppy build --report-opt` (or
  `--report-opt-json FILE`) reports what became native, what stayed in
  Python and why, the guards proofs removed, and every remark under a
  stable category.
- Profile-guided optimization. `ppy run --profile foo.ppy` is a JIT run
  with the `instrument-profile` pass in the pipeline: every native
  function counts its blocks and the taken edge of every branch into a
  counter array the module carries next to its legend, the boundary
  records the kinds of value -- shapes, dtypes, column schemas -- each
  function is called with, and when the program ends the counters are read
  back and `foo.ppyprof` written, merged into one already there. `ppy
  build --pgo foo.ppyprof` (or `[tool.ppy.llvm] pgo`, or `ppy run --pgo`)
  applies it at the same point of the pipeline to every function whose
  graph still matches: hot and cold functions, branch weights, loop trip
  counts, all as IR attributes the inliner reads -- a hot callee at four
  times the budget, a cold one or an unreached call left alone -- and the
  LLVM backend writes as `!prof` metadata and `hot`/`cold` attributes. A
  changed function is named by `W2009` and built without the profile; the
  report shows the profile first; every cache key and the warm run
  directory carry the profile's content. The prof dialect is public in
  `.ppyir`. The lowering cache (schema 6) now keeps each module's remarks
  and proved guards, so a warm build's report says what the cold build's
  did.
- The direct AST-to-LLVM road is gone. The IR road -- the frontend in
  `ppy_compiler.lowering`, the passes, `from_ir` -- is the LLVM backend,
  and the default `pipeline`; `"ast"` in a project or `PPY_LOWERING=ast`
  is still read, answered with `W2004`, and builds on the IR road. The
  differential test that held the two roads to each other now holds the
  one road to CPython's own answers, and CI runs one suite. What stays of
  the old module is the native ABI and the eligibility rules both roads
  shared.
- Artifact determinism, held by a test: a build's program object,
  library, boundary wrapper, manifest, generated Python, and header, and
  every text `ppy emit` prints, come out byte for byte the same from an
  empty cache, a full one, and an emptied one. The one artifact that did
  not -- the boundary wrapper, compiled from a draft named after the
  process that wrote it, a name the C compiler records in the object --
  is compiled from its final name now.
- The checker reads the `ppy` package's own modules as ordinary code: a
  helper `ppy.aio` defines and calls is not a use of the aio namespace,
  and the same for every namespace the compiler models, so `ppy migrate`
  over the package itself no longer reports the namespace rules against
  their own implementations.
- `import ppy` is light again. `ppy.aio` had imported asyncio, socket, and
  the async runtime for every program, the native API `ctypes.util`,
  `ppy.autodiff` `inspect`, and the CPU probe `platform` and `subprocess`,
  together doubling the cost of importing the package and adding thirty
  milliseconds to every launched artifact; each loads when first used, and
  a test holds the async ones out of a plain import.
- The artifact a warm `ppy run` launches is compiled for the machine's own
  CPU again, as JIT code always was and `ppy build` never is without
  `--host-cpu`; the run directory's name and the program object's key carry
  the CPU's features, so a cache carried to another machine never serves
  it. A test holds both.
- Eleven examples for what 0.2.0 added, each hand-written and each held to
  the three paths: native memory and a `libm` binding, `ppy.simd` and
  `ppy.cpu`, atomics and threads, `parallel.range`, derivatives, `ppy.aio`
  coroutines, `ppy.cuda` kernels, `ppy.xla`, generics, columnar pandas
  expressions, and the toolbox of `ppy emit`, `ppy inspect --stage`, the
  report, the sanitizers, and PGO; the existing examples point at them. The
  runners skip an example that needs pandas, pyarrow, or scipy where those
  are missing. Found on the way: a type parameter bounded by a Protocol now
  lends its methods to attribute calls, as it already did to operators.
- The documentation is a site, [ppy.franknoh.dev](https://ppy.franknoh.dev/),
  built with MkDocs from `docs/` on every push to `main` and every release
  tag, versioned by mike (`dev` for the tip, `X.Y` and `latest` for a
  release). The language reference is split into one page per topic under
  `docs/guide/`, the configuration, diagnostics, and compatibility pages
  live under `docs/reference/`, the architecture, IR, conversion,
  migration, plugin, and solver pages under `docs/internals/`, and the
  example gallery, the performance tables, the API pages, the contributing
  page, and this changelog are generated at build time from the examples'
  READMEs, the recorded measurements, the docstrings, and the repository's
  own files, so none of them is a second copy. The gate builds the site
  with `--strict`, so a broken link fails it.

## 0.1.0a1

The first release, and an alpha in the ordinary sense: the language and the
diagnostics are in use and tested, and neither is promised to stay put. Pin
an exact version.

On PyPI as **`ppy-lang`**; the packages it installs are `ppy`,
`ppy_compiler`, and `ppy_runtime`, so a program still writes `import ppy`.

### The language

- A `.ppy` file is valid Python 3.12+. Everything PPY adds is carried by
  decorators and annotations from the `ppy` package, all inert under plain
  CPython.
- Strict analysis by default: an implicit `Any` is an error, dynamic features
  need a `ppy.dynamic` boundary, and a decorator must have vouched semantics.
- `ppy convert` staticizes Python into strict PPY; `ppy migrate` is the
  permissive form. Both are deterministic, both refuse to write anything when
  the settled analysis holds an error.

### Running it

Three paths, held to one answer — plain CPython, an optimized Python backend,
and LLVM-lowered native code. Any observable difference between them is a
compiler bug; the suite and `examples/run_all.py` compare all three on every
example.

- `ppy build` produces an artifact that runs through `ppy_runtime` with
  machine code from the library beside it, and keeps working with the
  compiler uninstalled.
- `ppy build --standalone` links a native executable with no CPython inside,
  for a program whose reachable graph is entirely native.
- `ppy.input[T]`, `ppy.buffer[T]`, and `Buffer[T]` (including one-byte
  `ppy.i8`/`ppy.u8` elements) read and hold data without a Python object per
  value.

### Plugins

NumPy, PyTorch, JAX/Flax, pydantic, and FastAPI/Uvicorn are modeled; each is
an optional extra, and a missing runtime disables only its plugin.

### Known limits

- `ppy.read_token` has no standalone lowering yet, which is the one thing
  keeping the substring-search example off that path.
- Floats do not print from a standalone binary, pending native formatting
  that reproduces CPython's shortest round-trip repr exactly.
