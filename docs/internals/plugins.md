# Plugins

Each supported library is a plugin (spec 19–23): the compiler learns that
library's types and effects, takes a faster path where it can prove
equivalence, and falls back to the ordinary Python call everywhere else. A
guard that fails at runtime is a fallback, never a different answer.

Every plugin has a `fingerprint()` — its own version plus the library build it
found (`v1:torch=2.11+cu128:cxx11abi=1:cuda=12.8:...`) — and every cache key
includes the fingerprints of the plugins whose library the module actually
imports. An artifact built against one build of a library is never reused
against another, and a module that imports none of them pays for none of them.
`ppy doctor` prints what each plugin detected.

## NumPy

- Array expressions are typed with dtype and shape refinements; `tolist()`
  and friends follow the declared dtype.
- Elementwise expressions and whole-array reductions converge onto the
  tensor dialect -- `numpy.multiply` is `tensor.mul`, `numpy.sin` is
  `tensor.unary {op = sin}`, `numpy.sum` is `tensor.reduce {op = add}` --
  and a maximal expression tree of them becomes one kernel: tensor IR over
  buffers whose length the call supplies, lowered by `lower-tensor` to one
  strided loop with no temporaries and compiled through LLVM. An exact
  `float64` C-contiguous array of one shape across the operands is guarded
  at runtime; anything else runs NumPy.
- `dot`, `matmul`, `inner`, `vdot`, `tensordot` route to the linear-algebra
  path.
- Reduction order is preserved bit-for-bit unless the function is
  `@ppy.fastmath`.

## PyTorch

- A function whose body is entirely curated tensor operations (55 ops,
  `plugins/torch_plugin.CURATED_OPS`) compiles into one C++ region calling
  ATen directly — one Python round trip per call instead of one per operator.
- The region still calls through the dispatcher, so autograd, device
  selection, and backend keys behave identically; a tensor subclass or
  `__torch_function__` override fails the guard and the Python body runs.
- Building the region needs a C++ compiler and `ninja`; `toolchain_ready()`
  reports what is missing. CUDA is used when available.
- A built artifact carries its regions: the extension is copied beside the
  manifest and recorded under `regions`, and `ppy_runtime` loads it with no
  compiler in the process -- so a warm `ppy run`, a built launcher, and a
  `.ppy` served by `import ppy` all get the region, under any launcher
  (`examples/31_torchrun`). A region library that has gone missing is the
  Python body, not a broken artifact.
- Worth ~20% on small CPU tensors; nothing on an accelerator, where kernel
  launch latency dominates. Measured honestly in `examples/21_training_torch`.
- The curated arithmetic and reductions -- `add`, `sub`, `mul`, `div`,
  `pow`, `neg`, `abs`, `sum`, `prod`, `mean`, `max`, `min` -- are the same
  tensor operations NumPy's are, and an expression tree of them over
  tensors fuses into the same kind of kernel, which runs over an exact CPU
  `float64` contiguous tensor outside autograd; a tensor that records its
  history, or lives elsewhere, runs torch. `matmul` and the other
  dispatcher-sensitive operations stay with the dispatcher.

## JAX

- A `@jax.jit` function (or one marked `@ppy.jax`) whose inputs carry
  `ppy.Shape` and `ppy.DType` exports to StableHLO at build time via
  `jax.export`, so the trace is not repeated at startup; shapes may be
  symbolic, so one artifact serves every batch size.
- Export executes project code at build time, so it is governed by
  `[tool.ppy] build-execution` and **off by default** (`"deny"`).
- At runtime the artifact executes through PJRT; a mismatch falls back to the
  ordinary jitted call.
- `jax.numpy` spells the shared tensor operations as NumPy does, and the
  plugin names them (`jax.numpy.add` is `tensor.add`); an eager call
  outside an exported region still runs on the Python path.
- The plugin also models the Flax (linen) and optax surface: layer
  constructors and activations, `Module.init`/`apply` resolved through a
  class's external MRO (`class Mlp(nn.Module)` gets them from a base only
  the plugin knows), optimizers and `tx.update`. A Flax training loop checks
  under `strict = true` with nothing extra — `examples/29_flax` converts one
  whose three paths train identically.

## Pydantic

- Models are typed from their fields; the constructor signature and the
  validated output shape are kept distinct.
- Field constraints become integer-range refinements the checker propagates —
  both spellings: `Annotated[int, Field(ge=0, le=100)]` and
  `count: int = Field(ge=0, le=100)` (`ge`/`gt`/`le`/`lt`, `conint`).
- Schema-building code execution is policy-gated like JAX export.

## Uvicorn (and FastAPI on top of it)

- `uvicorn.run(app)` with a statically resolvable application skips the
  per-worker re-import by module string.
- The reloader is told to watch `*.ppy` alongside `*.py`.
- FastAPI rides the same serving path — it is an ASGI framework — so this
  plugin models its surface too: `FastAPI()`/`APIRouter()`, route decorators,
  dependency markers (`Depends`, `Query`, ...), and the in-process
  `TestClient` with its responses. A FastAPI project checks under
  `strict = true` with nothing extra installed; the plugin activates when
  the project imports the library.
- Route handlers stay exactly as their author wrote them: `@app.get` is an
  unvouched decorator and FastAPI reads `__annotations__` at import to build
  validation, so the conversion policy refuses to touch them — while the
  pydantic plugin types the models they exchange. `examples/27_uvicorn`
  converts a FastAPI service whose three paths return byte-identical
  responses.

## SciPy

- A curated surface, not a reimplementation: `scipy.special` (one- and
  two-argument functions over scalars and arrays), `scipy.fft`,
  `scipy.linalg` (dense solves, factorizations, norms, determinants), and
  `scipy.sparse` (the CSR/CSC/COO constructors and their methods) are
  typed and named as operations of the `special`, `fft`, `linalg`, and
  `sparse` dialects, so a backend that has those lowers them and every
  other runs SciPy. A one-argument special function over arrays is also
  `tensor.unary` of that function, so `special.erf(x) * 2.0` fuses into
  one loop with the arithmetic around it.
- `scipy.optimize`, `scipy.integrate`, and `scipy.stats` drive Python
  callbacks; their calls carry that effect and stay Python calls.

## pandas

- `DataFrame`, `Series`, `Index`, and grouped frames are typed by what they
  are. Selection, boolean filters, `assign`, arithmetic and comparison
  operators, `isna`/`notna`/`fillna`, `astype`, `sort_values`, the
  aggregations, grouped aggregation, `merge`/`join`, and `concat` are named
  as `columnar` operations -- the same ones PyArrow's compute names.
- An expression tree of Series arithmetic, comparison, `fillna`,
  `isna`/`notna`, and the boolean operators fuses into one loop -- a
  `columnar.map` -- the same kernel PyArrow's compute gets, with nothing
  materialized between the operations. At run time an Arrow-backed Series
  (`pd.ArrowDtype`) is its Arrow array, read in place with its validity
  bits; a NumPy-backed `float64`
  Series goes to the kernel's twin under NumPy's null model, where a NaN
  is the null `fillna` fills and `isna` finds, a bool mask is a byte per
  row, and `!=` is IEEE's; the answer is a Series over the callers' index
  with the same backing, written straight into the array it wraps. Series
  whose indexes are not one index, a mix of backings, or a nullable
  extension dtype run pandas: the index alignment, the copy-or-view rule,
  and the dtype are pandas' own semantics and are not approximated (spec
  59).
- `ppy_runtime.arrow.exported(array)` lends a PyArrow array to native code
  as the Arrow C Data Interface's `ArrowArray` struct -- the buffers
  shared, no `PyObject` in the ABI -- and releases it, once, when the
  borrow ends; `arrow.import` in the IR reads such a struct.
- What the model does not capture exactly -- the index, nullable dtypes,
  `NA` against `NaN`, categoricals, time zones, extension dtypes, duplicate
  column names, copy-or-view -- keeps the pandas implementation; a frame is
  never treated as a 2-D tensor. `apply`/`map`/`transform` carry the
  callback effect; `read_*`/`to_*` carry IO.

## PyArrow

- `pyarrow.compute` over `float64` and `bool` arrays converges onto the
  columnar dialect -- `pc.add` is `columnar.add`, `pc.greater` is
  `columnar.greater`, `pc.if_else` is `columnar.select`, `pc.fill_null`
  is `columnar.fill_null` -- and a maximal expression tree of them becomes
  one kernel: columnar IR over the arrays' own values and validity
  buffers, lowered to one loop that keeps Arrow's null semantics. The
  kernel reads an `Array` where it lies (no copy, no `PyObject` in the
  ABI) and answers an `Array` built over the buffers it filled; a chunked
  array, another type, or a bitmap sliced inside a byte runs Arrow's own
  compute.

- `Array`, `ChunkedArray`, `Table`, `RecordBatch`, `Schema`, `Field`,
  `DataType`, `Buffer`, and `Scalar` are typed as Arrow, with their
  representation-level attributes (`null_count`, `offset`, `buffers`,
  `chunks`, `num_rows`, `schema`). The curated `pyarrow.compute` surface --
  cast, filter, take, sort, arithmetic, comparison, boolean, aggregation,
  null handling -- is named as `columnar` operations shared with pandas.
- `pyarrow.parquet`, `pyarrow.dataset`, and `pyarrow.csv` carry IO.
  `to_numpy` says what it is: a zero-copy view only for a fixed-width array
  without nulls.

## Writing against the interface

A plugin extends `ppy_compiler.plugins.Plugin` (interface 2). Every hook
has a no-op default, so a plugin implements what it knows and the compiler
calls the rest without probing: `external_types`, `attribute_type`,
`instance_attribute`, `subscript`, `call`, `operator`, `call_alias`,
`decorator_semantics`, `adjust_call`, `stage`, `tensor_operation` (the
shared tensor operation a call converges onto, whatever its own lowering
is), and the IR hooks `register_dialects`, `register_passes`,
`register_patterns`, `register_lowerings`. `fingerprint()` names the
library's version and enters every cache key.

What a `call` answers about lowering is a typed spec a backend reads, never
backend code: `IntrinsicSpec`, `DialectOperationSpec`, `DirectCallSpec`,
`GraphRegionSpec`, `FallbackSpec`, or `RejectSpec` (the bare `Lowering`
kinds still work).

An external plugin is a Python package with an entry point:

```toml
[project.entry-points."ppy.plugins"]
foo = "foo.ppy_plugin:create_plugin"
```

Discovery reads the entry points without importing anything; the plugin
is imported and loaded only for a project that names it and does not
disable it:

```toml
[tool.ppy.plugins.foo]
enabled = true
```

Two plugins claiming one module are reported (`E1901`) rather than settled
by registration order; a plugin written against another interface version
is refused with the reason; a plugin pass that leaves the IR invalid is
named in the error (`E1902`).
