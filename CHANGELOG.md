# Changelog

## 0.1.1a1 — unreleased

Speed of the compiler itself, measured before being changed.

- A warm `ppy run` no longer imports the compiler. The first run of a
  program builds its artifact into the cache and launches it; the next run
  finds the artifact by a key over everything that could change it and goes
  straight to `ppy_runtime`. On a small program that was 1.65 s of imports,
  LLVM initialization, and re-analysis before the first line ran; it is now
  the launcher's few dozen milliseconds plus the key. Programs that
  specialize at runtime, fuse NumPy kernels, or use the torch or JAX plugins
  keep the in-process JIT, and say so in a `needs-jit` note.
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
  `docs/solver.md` says where a solver fits, where it does not, and what is
  next.
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
- [docs/migrating.md](docs/migrating.md) says what to hand `ppy migrate` on
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
