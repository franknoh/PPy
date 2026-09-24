# Backends

A backend turns the canonical IR into something else: LLVM IR and objects,
C, PTX, StableHLO, or (from a package of its own) code for an accelerator the
compiler has never heard of. This page is the contract between the compiler
and a backend, builtin or installed, and covers what it takes to write one.
The interface itself is documented from the source in
[Backend API](../api/backends.md).

## Where a backend stands

```
source → frontend → analysis → typed AST → canonical IR
                                               │  shared canonicalization
                                               │  shared optimization
                                               ▼
                                     the backend boundary
                                               │  the backend's passes
                                               │  the backend's validation
                                               ▼
                                     emit, or build
```

### Above the boundary

Everything above the boundary is one road, `driver/ir_pipeline.py`:

- `canonical_ir_modules(bundle)` is the canonical IR of every module in a
  project after the shared passes.
- `optimize_shared_ir(module, level)` runs those passes over one module.

The LLVM backend, the C backend, the device backends, `ppy emit ir`,
`ppy inspect --stage`, and an external backend all take what those two hand
back. None of them re-reads the Python AST or the checker's tables, and none
of them may. A backend that needs to know something about the program reads
it off the IR: the types, the attributes (`ppy.qualname`, `ppy.export`, the
effects), the locations. That is why the frontend writes all of it there.

### Below the boundary

Below the boundary the backend is on its own: its passes, its validation,
its emission. Nothing it does reaches back up.

## Plugins and backends

They are two extensions, kept apart.

- A [plugin](plugins.md) models a library. It types the calls and attributes
  of the modules it claims, declares their effects, says how each recognized
  operation lowers as a backend-neutral spec, and may register dialects,
  patterns, lowerings, and passes for the IR. A plugin never emits.
- A backend makes code from the IR. A backend never types a call.

An accelerator vendor may ship both (a plugin for its library's operations, a
backend for its device) as two entry points in two groups.

## The interface

A backend is a class extending `ppy_compiler.backend.Backend`, with a
`name`, an `api_version`, and these methods, every one with a default:

| method | what |
|---|---|
| `fingerprint()` | what identifies its code generation for the cache: its version, its SDK's, anything whose change makes an old artifact wrong. |
| `emit_formats()` | the `EmitFormat`s `ppy emit` may ask it for: a name, a file suffix, whether the output is bytes. |
| `register_passes(passes)` | its own passes, through a `BackendPassRegistrar`: `passes.add(MyPass)`, which hangs them at the `backend` stage and can reach no other. |
| `validate(module, context)` | refuse IR it cannot take, with `BackendValidationError`: the backend, what, the capability it lacks, the location. |
| `emit(module, format, context)` | one module as text, or bytes for a binary format; for a `module`-scoped format, which is the default. |
| `emit_program(modules, format, context)` | every module at once, one artifact, for a format declared `scope="program"`. |
| `build(modules, output, context)` | every module into a directory; `BuildResult` says what was written. |
| `toolchain_status()` | whether it can work here, and what `ppy doctor` should print. |

### The API version

`BACKEND_API_VERSION` numbers the interface, independently of the
compiler's version. An external backend declares the version it implements
**as a literal of its own**:

```python
class MyBackend(Backend):
    api_version = 1
```

Do not write `api_version = BACKEND_API_VERSION`. A package that spells the
constant is rewritten by every compiler it is imported into. A backend
written against version 1 would call itself version 2 the moment a version-2
compiler imported it, and the number would catch nothing.

- The base class declares no version at all. A backend that declares none is
  refused with what to write instead: an inherited version is not a declared
  one.
- A backend declaring a version this compiler does not speak is refused with
  both numbers.
- A backend package with a base class of its own may declare once there for
  all of them.

The compiler's own backends ship with the interface and are upgraded with
it, so they track the constant; nothing installed should.

### Toolchain checks

An `EmitFormat` checks `toolchain_status()` before analysis by default. A
format that only writes source can declare `requires_toolchain=False` to emit
without an installed SDK.

- This applies to both module and program scope; validation and passes still
  run.
- `ppy build` always checks the backend's toolchain, regardless of its
  formats' emission requirements.
- Packages using this field must require `ppy-lang>=0.3.2`. Existing backend
  API version 1 implementations continue to load with the default behavior.

### `BackendContext`

The `BackendContext` a backend receives with the IR carries:

- the project root and configuration
- the backend's own table from `pyproject.toml` (`backend_config`)
- the optimization level
- the `target` that table names
- the dialect registry the IR was verified against
- the plugin fingerprints
- the entry file for a build of one
- the artifact identity per module (the cache key below)
- `note()` for remarks the driver prints

## Discovery and loading

An external backend is a distribution with an entry point in the
`ppy.backends` group, the same way a plugin is one in `ppy.plugins`:

```toml
[project.entry-points."ppy.backends"]
toy = "ppy_toy:create_backend"
```

Discovery (`available_backends()`) reads the entry points without importing
anything. The package is imported when its backend is selected:

- by `ppy build --backend toy`
- by `ppy emit` asking for a format the builtin backends do not have
- by `ppy doctor`

Its factory is then called with the backend's table from
`[tool.ppy.backends.toy]`.

### Load errors

What can go wrong is reported as what it is, `E1903`:

- no backend of that name (the message lists the ones there are)
- a name two distributions register (the name is unusable until one is gone;
  it is never settled by which came first)
- a package that fails to import or a factory that raises (the error is the
  message)
- a factory answering something that is not a `Backend`
- an interface version other than this compiler's

One broken package never breaks the others: `ppy doctor` prints it as
`unusable` with the reason and goes on. A backend registering a builtin's
name is reported and ignored; the builtin is used.

### Declaring formats

`ppy emit <format>` has to know which backend owns a format before it can
load one. A distribution may say so in a second, optional entry-point group,
`ppy.backend-formats`, whose entry names are formats and whose values are the
backend's name. Discovery reads it without importing anything, and the one
backend it names is then imported and asked.

A format no distribution declares is still found, by loading the installed
backends and asking each. That is correct, but it imports every accelerator
SDK on the machine to answer one question, so declaring the group is worth
it.

The group is a hint and never the authority. The backend it names is asked
all the same, and a format it does not actually emit is an error. Two
distributions declaring one format leaves neither owning it.

### Builtin backends

The builtin backends (`llvm`, `python`, `c`, `nvvm`, `stablehlo`, and `ir`
for the canonical IR itself) stand in the same registry, with their formats,
fingerprints, and toolchain status behind the same interface.

Their emission and builds run on the driver's own roads. Several of their
formats are whole-program or flag-shaped (`--standalone`, `--header-only`,
`linked-ir`), and the LLVM road keeps a lowering cache. So `emit` and `build`
are not what the driver calls on them. An installed backend is called
through the interface and nothing else.

## Passes, validation, order

For one module, in order:

1. canonical IR generation, then the plugins' `after-ir-generation` passes;
2. canonicalization, then `after-canonicalization`;
3. `before-optimization`, the shared optimization, `after-optimization`;
4. tensor and parallel lowering, then `before-backend`;
5. the `backend` stage: the passes the backend hung with `passes.add(...)`;
6. the backend's `validate`;
7. the backend's `emit` or `build`.

### The `backend` stage

A backend reaches the `backend` stage and no other. It is handed a
`BackendPassRegistrar` and not the `PassManager`. The stages before `backend`
are the shared pipeline's and the plugins'. A pass of one backend's running
among them would be deciding for every other backend what the canonical IR
is. A backend that asks for another stage is refused by name (`E1802`),
rather than quietly shaping the IR everything else receives.

### Verification and validation

The module is verified before the first pass and after the last. When a
plugin or a backend contributed a pass, it is also verified after every pass,
so the one that leaves the IR invalid is named rather than handed on:
`E1902` for a plugin's, `E1904` for a backend's.

`validate` is the backend's last word before it writes anything. A backend
must refuse there what it cannot take, with the operation, type, or function
named and the location when the IR has one (`E1802`), and never emit it
wrongly.

There is no fallback on the `ppy build --backend` and `ppy emit` roads. A
program a backend cannot take is an error; the backend does not produce a
quieter artifact.

## Configuration

```toml
[tool.ppy.backends.toy]
target = "rngd"
sdk-version = "1.4"
```

A backend's table is handed to that backend and to no other. A table for a
backend that is not installed is not an error, so a `pyproject.toml` can
carry settings for a backend a colleague has.

`target` is the one key the driver reads itself, into
`BackendContext.target` and the cache key. The rest means what the backend
says it means. `ppy build --target` is the LLVM road's triple and is refused
for a backend of its own, which builds for what this `target` names.

## Emit scope

An `EmitFormat` says whether it is written per module or per program:

```python
EmitFormat("toy", ".toy")  # scope="module", the default
EmitFormat("toy-image", ".img", binary=True, scope="program")
```

### Module scope

A **module-scoped** format is asked for once per module, through `emit`,
and one artifact comes back for each.

- One module goes to standard output or to `-o FILE`.
- A target that resolves to more than one module needs `-o DIR`, and one file
  per module is written into it, named for the module with the format's
  suffix.

Two artifacts are never written end to end into one file. Two object files,
two firmware images, or two device programs do not become one by
concatenation, and the ambiguity is refused (`E1002`) with the names of the
files `-o DIR` would write.

### Program scope

A **program-scoped** format is asked for once, through `emit_program`, with
every module at once, and answers the single artifact. Use it for a linked
image, a package, or an archive. A whole-program artifact is the backend's to
make from every module; the driver cannot assemble one from per-module
answers.

Either way the answer's type must match the format: text for a text
format, `bytes` for one declared `binary`, or the backend is refused
(`E1802`).

## The cache key

The identity the driver computes for one module's artifact from a backend
(`BackendContext.identity[module]`, `driver.pipeline.backend_identity`) is
the module's key combined with the backend's identity:

| part | contents |
|---|---|
| the module's key | its source digest, the compiler version and fingerprint, its dependencies' public ABI, the directives, the optimization level, the project options, the plugin fingerprints |
| the backend | its name, the distribution it came from and that distribution's version, the interface version it declares, its `fingerprint()`, the digest of its configuration table, its `target`, the IR schema version |

A new SDK under the backend, a changed setting, another target: each is a
different key, and an old artifact is never reused for it.

### Why the distribution version is in the key

The distribution's version is in the key whether or not the backend's author
put it in `fingerprint()`. Two releases of one package carry the same module
and class names and the same interface version and can generate entirely
different code. So `ppy-toy 0.1` and `ppy-toy 0.2` address different
artifacts by construction.

`fingerprint()` is for what the package version does not cover: the SDK under
it, a firmware revision, the version of a code generator it calls. The
builtin LLVM road's keys carry the same kind of thing (the `llvmlite` under
the objects is part of them), so a cached object does not outlive the LLVM
that made it.

## Building

`ppy build TARGET --backend NAME`:

1. loads the backend with its table
2. checks its toolchain
3. runs the shared passes and the backend's
4. validates
5. calls `build` with every module and a directory to write into: `-o`, or
   `<cache>/backends/NAME`

`BuildResult.outputs` is what the driver lists.

### Building from `.ppyir`

`ppy build TARGET.ppyir --backend NAME` is the same below the boundary
without the frontend above it. The file is canonical IR that has already been
through the shared passes, so the backend's own passes run over it, then its
validation, then its `build`. Its artifact identity is the module's encoded
bytes with everything that identifies the compiler and the backend, since
there is no source bundle behind it.

### LLVM-only options

The LLVM road's options belong to the LLVM road. These are refused for
another backend (`E1002`) rather than accepted and ignored:

`--unsafe`, `--sanitize`, `--pgo`, `--prover`, `--host-cpu`, `--standalone`,
`--python-extension`, `--library`, `--report-opt`, `--report-opt-json`,
`--target`

Each refusal says what the option does and, for `--target`, where to name a
target that does reach the backend. `--warm` builds the artifact `ppy run` and
`import ppy` take, which is the LLVM backend's, and is refused for any other
backend. `-o` and `-O` reach every backend.

## A minimal backend

`ppy_toy/__init__.py`, the whole package:

```python
from pathlib import Path

from ppy_compiler.backend import (
    Backend,
    BackendValidationError,
    BuildResult,
    EmitFormat,
    ToolchainStatus,
)
from ppy_compiler.ir import FunctionPass, print_function


class CountOps(FunctionPass):
    name = "toy-count-ops"

    def run_on_function(self, function, ctx):
        function.attributes["toy.ops"] = sum(len(b.operations) for b in function.body.blocks)
        return True


class ToyBackend(Backend):
    name = "toy"
    api_version = 1  # the version this package implements, as a literal

    def fingerprint(self):
        return f"toy:{self.options.get('sdk-version', '0')}"

    def emit_formats(self):
        return (EmitFormat("toy", ".toy", description="each function, counted"),)

    def register_passes(self, passes):
        passes.add(CountOps)

    def validate(self, module, context):
        for function in module.functions.values():
            for block in function.body.blocks:
                for op in block.operations:
                    if op.dialect == "gpu":
                        raise BackendValidationError(
                            self.name,
                            f"{op.name} in {function.name}",
                            capability="device kernels",
                            location=op.location,
                        )

    def emit(self, module, format, context):
        return "".join(
            f"// {f.name}: {f.attributes['toy.ops']} operations\n" + print_function(f) + "\n"
            for f in module.functions.values()
            if not f.is_declaration
        )

    def build(self, modules, output, context):
        written = []
        for name, module in modules.items():
            path = output / f"{name}.toy"
            path.write_text(self.emit(module, "toy", context), encoding="utf-8")
            written.append(path)
        return BuildResult(outputs=tuple(written))

    def toolchain_status(self):
        return ToolchainStatus(True, f"toy sdk {self.options.get('sdk-version', '0')}")


def create_backend(options):
    return ToyBackend(options)
```

`pyproject.toml` of that package:

```toml
[project]
name = "ppy-toy"
version = "0.1"
dependencies = ["ppy-lang"]

[project.entry-points."ppy.backends"]
toy = "ppy_toy:create_backend"

# Optional, and worth declaring: which formats this backend owns, so that
# `ppy emit toy` imports this package and no other installed backend.
[project.entry-points."ppy.backend-formats"]
toy = "toy"
```

Installed beside the compiler:

```bash
ppy emit toy foo.ppy                    # to stdout
ppy emit toy src/ -o build/toy/         # one .toy per module
ppy build foo.ppy --backend toy -o out/ # what build() wrote, listed
ppy doctor                              # toy  available, toy sdk 0 [ppy-toy 0.1 (ppy_toy:create_backend)]
```

The package imports only `ppy_compiler.backend` and `ppy_compiler.ir`;
nothing under `backend/llvm` or the driver.

## Limitations

- A module-scoped format receives modules one at a time, each through the
  shared passes on its own. A program-scoped format and `build` receive them
  together, and may link them with `ppy_compiler.ir.linker.link`, which is
  what `ppy emit linked-ir` uses.
- The `ppy run` road (the JIT, the guarded bindings, the fallback to Python)
  is the LLVM backend's and is not opened to an installed backend. A backend
  builds and emits; running what it built is its own runtime's business.
- The Python backend and the LLVM backend keep their `--backend` spellings
  and their behavior.
