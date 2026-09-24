# Plugins

Each supported library is a plugin (spec 19–23). A plugin teaches the
compiler that library's types and effects. The compiler takes a faster path
where it can prove equivalence, and falls back to the ordinary Python call
everywhere else. A guard that fails at runtime is a fallback, never a
different answer.

This page covers how each builtin plugin works, then the interface for
writing your own.

## Fingerprints and the cache

Every plugin has a `fingerprint()`: its own version plus the library build it
found (`v1:torch=2.11+cu128:cxx11abi=1:cuda=12.8:...`). Every cache key
includes the fingerprints of the plugins whose library the module imports.

- An artifact built against one build of a library is never reused against
  another.
- A module that imports none of them pays for none of them.

`ppy doctor` prints what each plugin detected.

## NumPy

- Array expressions are typed with dtype and shape refinements; `tolist()`
  and friends follow the declared dtype.
- Elementwise expressions and whole-array reductions converge onto the
  tensor dialect: `numpy.multiply` is `tensor.mul`, `numpy.sin` is
  `tensor.unary {op = sin}`, `numpy.sum` is `tensor.reduce {op = add}`.
  A maximal expression tree of them becomes one kernel: tensor IR over
  buffers whose length the call supplies, lowered by `lower-tensor` to one
  strided loop with no temporaries and compiled through LLVM.
- An exact `float64` C-contiguous array of one shape across the operands is
  guarded at runtime; anything else runs NumPy.
- `dot`, `matmul`, `inner`, `vdot`, `tensordot` route to the linear-algebra
  path.
- Reduction order is preserved bit-for-bit unless the function is
  `@ppy.fastmath`.

## PyTorch

### Regions

A function whose body is entirely curated tensor operations (68 ops,
`plugins/torch_plugin.CURATED_OPS`) compiles into one C++ region calling ATen
directly. That is one Python round trip per call instead of one per operator.

What a region may emit is the narrower table
`plugins/torch_region.ATEN_CALLS`: 45 operations, each described by the C++
signature it is called through rather than by an arity.

| argument | C++ type |
|---|---|
| a dimension | `int64_t` |
| a shape or a list of dimensions | `at::IntArrayRef`, written as a tuple |
| `keepdim`, `is_causal` | `bool` |
| `gelu`'s approximation | a mode string |

Keyword arguments are matched against the C++ parameter names. An operation
with two signatures (`mean` over everything, or over named dimensions) takes
the one the call fills. That is enough for a whole transformer block
(`layer_norm`, `linear`, reshapes, transposes,
`scaled_dot_product_attention`, `gelu`) to be one region
(`examples/46_gpt2`).

A region hands back one tensor or a fixed tuple of them.
`tuple[torch.Tensor, torch.Tensor]` becomes a `std::tuple`, which pybind11
gives Python as an ordinary tuple. A region may write through a parameter with
`narrow` and `copy_`, which carries `WriteMemory` and so cannot appear in a
`@ppy.pure` function.

### Dispatch and guards

The region still calls through the dispatcher, so autograd, device
selection, and backend keys behave identically. A tensor subclass or
`__torch_function__` override fails the guard and the Python body runs.

### Building and shipping

- Building the region needs a C++ compiler and `ninja`; `toolchain_ready()`
  reports what is missing. CUDA is used when available.
- A built artifact carries its regions. The extension is copied beside the
  manifest and recorded under `regions`, and `ppy_runtime` loads it with no
  compiler in the process. So a warm `ppy run`, a built launcher, and a
  `.ppy` served by `import ppy` all get the region, under any launcher
  (`examples/31_torchrun`).
- A region library that has gone missing falls back to the Python body; the
  artifact still works.

### What it is worth

- About 20% on small CPU tensors.
- On an accelerator, it is worth whatever the Python between operators costs:
  nothing on a shape whose kernels are long, and about 20% on GPT-2 XL at a
  short sequence, where 480 dispatches per forward pass are a fifth of the
  wall clock (`examples/46_gpt2`).

A region does no fusion. It removes the interpreter between operators and
keeps the kernel boundaries.

### Fused tensor arithmetic

The curated arithmetic and reductions (`add`, `sub`, `mul`, `div`, `pow`,
`neg`, `abs`, `sum`, `prod`, `mean`, `max`, `min`) are the same tensor
operations NumPy's are. An expression tree of them over tensors fuses into the
same kind of kernel, which runs over an exact CPU `float64` contiguous tensor
outside autograd. A tensor that records its history, or lives elsewhere, runs
torch. `matmul` and the other dispatcher-sensitive operations stay with the
dispatcher.

## JAX

- A `@jax.jit` function (or one marked `@ppy.jax`) whose inputs carry
  `ppy.Shape` and `ppy.DType` exports to StableHLO at build time via
  `jax.export`, so the trace is not repeated at startup. Shapes may be
  symbolic, so one artifact serves every batch size.
- Export executes project code at build time, so it is governed by
  `[tool.ppy] build-execution` and **off by default** (`"deny"`).
- At runtime the artifact executes through PJRT; a mismatch falls back to the
  ordinary jitted call.
- `jax.numpy` spells the shared tensor operations as NumPy does, and the
  plugin names them (`jax.numpy.add` is `tensor.add`). An eager call outside
  an exported region still runs on the Python path.
- The plugin also models the Flax (linen) and optax surface: layer
  constructors and activations, `Module.init`/`apply` resolved through a
  class's external MRO (`class Mlp(nn.Module)` gets them from a base only the
  plugin knows), optimizers and `tx.update`. A Flax training loop checks
  under `strict = true` with nothing extra. `examples/29_flax` converts one
  whose three paths train identically.

## Pydantic

- Models are typed from their fields; the constructor signature and the
  validated output shape are kept distinct.
- Field constraints become integer-range refinements the checker propagates,
  in both spellings: `Annotated[int, Field(ge=0, le=100)]` and
  `count: int = Field(ge=0, le=100)` (`ge`/`gt`/`le`/`lt`, `conint`).
- Schema-building code execution is policy-gated like JAX export.

## Uvicorn (and FastAPI on top of it)

- `uvicorn.run(app)` with a statically resolvable application skips the
  per-worker re-import by module string.
- The reloader is told to watch `*.ppy` alongside `*.py`.
- FastAPI is an ASGI framework and rides the same serving path, so this
  plugin models its surface too: `FastAPI()`/`APIRouter()`, route decorators,
  dependency markers (`Depends`, `Query`, ...), and the in-process
  `TestClient` with its responses. A FastAPI project checks under
  `strict = true` with nothing extra installed; the plugin activates when
  the project imports the library.
- Route handlers stay as their author wrote them. `@app.get` is an unvouched
  decorator and FastAPI reads `__annotations__` at import to build
  validation, so the conversion policy refuses to touch them. The pydantic
  plugin still types the models they exchange. `examples/27_uvicorn`
  converts a FastAPI service whose three paths return byte-identical
  responses.

## SciPy

- The plugin covers a curated surface and does not reimplement SciPy.
  `scipy.special` (one- and two-argument functions over scalars and arrays),
  `scipy.fft`, `scipy.linalg` (dense solves, factorizations, norms,
  determinants), and `scipy.sparse` (the CSR/CSC/COO constructors and their
  methods) are typed and named as operations of the `special`, `fft`,
  `linalg`, and `sparse` dialects. A backend that has those lowers them, and
  every other runs SciPy.
- A one-argument special function over arrays is also `tensor.unary` of that
  function, so `special.erf(x) * 2.0` fuses into one loop with the arithmetic
  around it.
- `scipy.optimize`, `scipy.integrate`, and `scipy.stats` drive Python
  callbacks; their calls carry that effect and stay Python calls.

## pandas

### Typing

`DataFrame`, `Series`, `Index`, and grouped frames are typed by what they
are. Selection, boolean filters, `assign`, arithmetic and comparison
operators, `isna`/`notna`/`fillna`, `astype`, `sort_values`, the
aggregations, grouped aggregation, `merge`/`join`, and `concat` are named as
`columnar` operations, the same ones PyArrow's compute names.

### Fused Series expressions

An expression tree of Series arithmetic, comparison, `fillna`,
`isna`/`notna`, and the boolean operators fuses into one loop (a
`columnar.map`). It is the same kernel PyArrow's compute gets, with nothing
materialized between the operations. At run time:

- An Arrow-backed Series (`pd.ArrowDtype`) is its Arrow array, read in place
  with its validity bits.
- A NumPy-backed `float64` Series goes to the kernel's twin under NumPy's
  null model: a NaN is the null `fillna` fills and `isna` finds, a bool mask
  is a byte per row, and `!=` is IEEE's.
- The answer is a Series over the callers' index with the same backing,
  written straight into the array it wraps.

Series whose indexes are not one index, a mix of backings, or a nullable
extension dtype run pandas. The index alignment, the copy-or-view rule, and
the dtype are pandas' own semantics and are not approximated (spec 59).

### Lending Arrow arrays

`ppy_runtime.arrow.exported(array)` lends a PyArrow array to native code as
the Arrow C Data Interface's `ArrowArray` struct (the buffers shared, no
`PyObject` in the ABI) and releases it, once, when the borrow ends.
`arrow.import` in the IR reads such a struct.

### What stays pandas

What the model does not capture exactly keeps the pandas implementation: the
index, nullable dtypes, `NA` against `NaN`, categoricals, time zones,
extension dtypes, duplicate column names, copy-or-view. A frame is never
treated as a 2-D tensor. `apply`/`map`/`transform` carry the callback effect;
`read_*`/`to_*` carry IO.

## PyArrow

- `pyarrow.compute` over `float64` and `bool` arrays converges onto the
  columnar dialect: `pc.add` is `columnar.add`, `pc.greater` is
  `columnar.greater`, `pc.if_else` is `columnar.select`, `pc.fill_null` is
  `columnar.fill_null`. A maximal expression tree of them becomes one kernel:
  columnar IR over the arrays' own values and validity buffers, lowered to one
  loop that keeps Arrow's null semantics.
- The kernel reads an `Array` where it lies (no copy, no `PyObject` in the
  ABI) and answers an `Array` built over the buffers it filled. A chunked
  array, another type, or a bitmap sliced inside a byte runs Arrow's own
  compute.
- `Array`, `ChunkedArray`, `Table`, `RecordBatch`, `Schema`, `Field`,
  `DataType`, `Buffer`, and `Scalar` are typed as Arrow, with their
  representation-level attributes (`null_count`, `offset`, `buffers`,
  `chunks`, `num_rows`, `schema`). The curated `pyarrow.compute` surface
  (cast, filter, take, sort, arithmetic, comparison, boolean, aggregation,
  null handling) is named as `columnar` operations shared with pandas.
- `pyarrow.parquet`, `pyarrow.dataset`, and `pyarrow.csv` carry IO.
  `to_numpy` says what it is: a zero-copy view only for a fixed-width array
  without nulls.

## Writing against the interface

### Hooks

A plugin extends `ppy_compiler.plugins.Plugin` (interface 2). Every hook has
a no-op default, so a plugin implements what it knows and the compiler calls
the rest without probing. The hooks:

- `external_types`, `attribute_type`, `instance_attribute`, `subscript`,
  `call`, `operator`, `call_alias`, `decorator_semantics`, `adjust_call`,
  `stage`
- `tensor_operation`: the shared tensor operation a call converges onto,
  whatever its own lowering is
- the IR hooks `register_dialects`, `register_passes`, `register_patterns`,
  `register_lowerings`

`fingerprint()` names the library's version and enters every cache key.

What a `call` answers about lowering is a typed spec a backend reads, never
backend code: `IntrinsicSpec`, `DialectOperationSpec`, `DirectCallSpec`,
`GraphRegionSpec`, `FallbackSpec`, or `RejectSpec` (the bare `Lowering`
kinds still work).

### Registering a plugin

An external plugin is a Python package with an entry point:

```toml
[project.entry-points."ppy.plugins"]
foo = "foo.ppy_plugin:create_plugin"
```

Discovery reads the entry points without importing anything. The plugin is
imported and loaded only for a project that names it and does not disable
it:

```toml
[tool.ppy.plugins.foo]
enabled = true
```

### Errors

- Two plugins claiming one module are reported (`E1901`) rather than settled
  by registration order.
- A plugin written against another interface version is refused with the
  reason.
- A plugin pass that leaves the IR invalid is named in the error (`E1902`).

### Plugins and backends

A plugin models a library: types, effects, how an operation lowers, dialects
and passes for the IR. It never emits code. A backend makes code from the IR
and never types a call. An accelerator's package may carry both, one entry
point in `ppy.plugins` and one in `ppy.backends`; the second is covered in
[Backends](backends.md).
