# Changelog

## 0.1.0a1

The first published release, and an alpha in the ordinary sense: the language
and the diagnostics are in use and tested, and neither is promised to stay
put. Pin an exact version.

Published to PyPI as **`ppy-lang`**; the packages it installs are `ppy`,
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
