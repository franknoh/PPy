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
