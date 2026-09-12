# Backends

A backend turns the canonical IR into something else: LLVM IR and objects,
C, PTX, StableHLO, or -- from a package of its own -- code for an
accelerator the compiler has never heard of. This page is the contract
between the compiler and a backend, builtin or installed, and what it
takes to write one. The interface itself is documented from the source in
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

Everything above the boundary is one road, `driver/ir_pipeline.py`:
`canonical_ir_modules(bundle)` is the canonical IR of every module in a
project after the shared passes, `optimize_shared_ir(module, level)` those
passes over one module. The LLVM backend, the C backend, the device
backends, `ppy emit ir`, `ppy inspect --stage`, and an external backend
all take what those two hand back; none of them re-reads the Python AST or
the checker's tables, and none of them may. A backend that needs to know
something about the program reads it off the IR -- the types, the
attributes (`ppy.qualname`, `ppy.export`, the effects), the locations --
which is why the frontend writes all of it there.

Below the boundary the backend is on its own: its passes, its validation,
its emission. Nothing it does reaches back up.

## Plugins and backends

They are two extensions, kept apart. A [plugin](plugins.md) models a
library: it types the calls and attributes of the modules it claims,
declares their effects, says how each recognized operation lowers as a
backend-neutral spec, and may register dialects, patterns, lowerings, and
passes for the IR. A backend makes code from the IR. A plugin never emits;
a backend never types a call. An accelerator vendor may ship both -- a
plugin for its library's operations, a backend for its device -- as two
entry points in two groups.

## The interface

A backend is a class extending `ppy_compiler.backend.Backend`, with a
`name`, an `api_version`, and these methods, every one with a default:

| method | what |
|---|---|
| `fingerprint()` | what identifies its code generation for the cache: its version, its SDK's, anything whose change makes an old artifact wrong. |
| `emit_formats()` | the `EmitFormat`s `ppy emit` may ask it for: a name, a file suffix, whether the output is bytes. |
| `register_passes(manager)` | its own passes, hung at the `backend` stage of the shared `PassManager`. |
| `validate(module, context)` | refuse IR it cannot take, with `BackendValidationError`: the backend, what, the capability it lacks, the location. |
| `emit(module, format, context)` | one module as text, or bytes for a binary format. |
| `build(modules, output, context)` | every module into a directory; `BuildResult` says what was written. |
| `toolchain_status()` | whether it can work here, and what `ppy doctor` should print. |

`BACKEND_API_VERSION` numbers the interface, independently of the
compiler's version. A backend declares the version it was written against
(`api_version`, inherited as the current one); one declaring another is
refused at load time with both numbers in the message, never loaded and
hoped about.

The `BackendContext` a backend receives with the IR carries the project
root and configuration, the backend's own table from `pyproject.toml`
(`backend_config`), the optimization level, the `target` that table names,
the dialect registry the IR was verified against, the plugin fingerprints,
the entry file for a build of one, the artifact identity per module (the
cache key below), and `note()` for remarks the driver prints.

## Discovery and loading

An external backend is a distribution with an entry point in the
`ppy.backends` group, the same way a plugin is one in `ppy.plugins`:

```toml
[project.entry-points."ppy.backends"]
toy = "ppy_toy:create_backend"
```

Discovery (`available_backends()`) reads the entry points without
importing anything; the package is imported when its backend is selected
-- by `ppy build --backend toy`, by `ppy emit` asking for a format the
builtin backends do not have, or by `ppy doctor` -- and its factory is
called with the backend's table from `[tool.ppy.backends.toy]`. What can
go wrong is reported as what it is, `E1903`: no backend of that name (the
message lists the ones there are), a name two distributions register (the
name is unusable until one is gone; it is never settled by which came
first), a package that fails to import or a factory that raises (the
error is the message), a factory answering something that is not a
`Backend`, or an interface version other than this compiler's. One broken
package never breaks the others: `ppy doctor` prints it as `unusable`
with the reason and goes on. A backend registering a builtin's name is
reported and ignored; the builtin is used.

The builtin backends -- `llvm`, `python`, `c`, `nvvm`, `stablehlo`, and
`ir` for the canonical IR itself -- stand in the same registry, with their
formats, fingerprints, and toolchain status behind the same interface.
Their emission and builds run on the driver's own roads (several of their
formats are whole-program or flag-shaped: `--standalone`, `--header-only`,
`linked-ir`, and the LLVM road keeps a lowering cache), so `emit` and
`build` are not what the driver calls on them; an installed backend is
called through the interface and nothing else.

## Passes, validation, order

For one module, in order:

1. canonical IR generation, then the plugins' `after-ir-generation` passes;
2. canonicalization, then `after-canonicalization`;
3. `before-optimization`, the shared optimization, `after-optimization`;
4. tensor and parallel lowering, then `before-backend`;
5. the `backend` stage: the backend's `register_passes`;
6. the backend's `validate`;
7. the backend's `emit` or `build`.

The module is verified before the first pass and after the last, and when
a plugin or a backend contributed a pass, after every pass, so the one
that leaves the IR invalid is named -- `E1902` for a plugin's, `E1904` for
a backend's -- rather than handed on. `validate` is the backend's last
word before it writes anything: a backend must refuse there what it
cannot take, with the operation, type, or function named and the location
when the IR has one (`E1802`), and never emit it wrongly. There is no
fallback on the `ppy build --backend` and `ppy emit` roads; a program a
backend cannot take is an error, not a quieter artifact.

## Configuration

```toml
[tool.ppy.backends.toy]
target = "rngd"
sdk-version = "1.4"
```

A backend's table is handed to that backend and to no other; a table for a
backend that is not installed is not an error, so a `pyproject.toml` can
carry settings for a backend a colleague has. `target` is the one key the
driver reads itself, into `BackendContext.target` and the cache key; the
rest means what the backend says it means.

## The cache key

The identity the driver computes for one module's artifact from a backend
(`BackendContext.identity[module]`, `driver.pipeline.backend_identity`) is
the module's key -- its source digest, the compiler version and
fingerprint, its dependencies' public ABI, the directives, the
optimization level, the project options, and the plugin fingerprints --
with the backend's name, the backend's `fingerprint()`, the digest of its
configuration table, its `target`, and the IR schema version. A new SDK
under the backend, a changed setting, another target: each is a different
key, and an old artifact is never reused for it. The builtin LLVM road's
keys carry the same kind of thing -- the `llvmlite` under the objects is
part of them -- so a cached object does not outlive the LLVM that made it.

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

    def fingerprint(self):
        return f"toy:{self.options.get('sdk-version', '0')}"

    def emit_formats(self):
        return (EmitFormat("toy", ".toy", description="each function, counted"),)

    def register_passes(self, manager):
        manager.register_stage_pass("backend", CountOps)

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
```

Installed beside the compiler:

```bash
ppy emit toy foo.ppy                    # to stdout
ppy emit toy src/ -o build/toy/         # one .toy per module
ppy build foo.ppy --backend toy -o out/ # what build() wrote, listed
ppy doctor                              # toy  available, toy sdk 0 [ppy-toy 0.1 (ppy_toy:create_backend)]
```

Everything the package imports is `ppy_compiler.backend` and
`ppy_compiler.ir`; nothing under `backend/llvm` or the driver.

## What is not there

A backend receives modules one at a time, each through the shared passes
on its own; a whole-program view (`ppy emit linked-ir`'s) is the linker's,
`ppy_compiler.ir.linker.link`, which a backend may call on the modules it
is given. The `ppy run` road -- the JIT, the guarded bindings, the
fallback to Python -- is the LLVM backend's and is not opened to an
installed backend: a backend builds and emits; running what it built is
its own runtime's business. The Python backend and the LLVM backend keep
their `--backend` spellings and their behavior.
