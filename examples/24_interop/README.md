# Interop

A plain `.py` file importing a `.ppy` module, with no build step. `import
ppy` installs a `sys.meta_path` finder; `import geometry` then loads
`geometry.ppy` — as source with no compiler installed, from its native
build with one.

## The hook is explicit

```python
import ppy

import geometry

print("area     :", geometry.area(3.0, 4.0))
print("hook     :", ppy.is_installed())
```

Without `import ppy` the module is invisible and the import raises
`ModuleNotFoundError`; the hook is never implicit. If `geometry.py` and
`geometry.ppy` both exist, the `.ppy` wins and a `PPyAmbiguousModuleWarning`
says so; `ppy check` rejects the ambiguity outright (`E1003`), which is why
`ppy convert --in-place` removes the `.py` it replaces. `ppy convert`
inserts `import ppy` ahead of first-party imports, so a converted entry
point can import its siblings.

## Native when it can be, source when it cannot

With the compiler installed, the first process to import `geometry.ppy`
builds it into the project's `.ppy-cache` — the artifact `ppy run` would
build — and binds its functions through the prebuilt binder. Every later
process finds the build. A module that does not check clean, or needs the
in-process JIT, loads as Python source with one line on stderr saying why.
`PPY_IMPORT=python` turns the native path off for a process;
`[tool.ppy] native-import = false` turns it off for a project.

This is also how a program under someone else's launcher gets native
kernels: `torchrun`, `accelerate launch`, or a scheduler starts ordinary
interpreters, and each one's `import ppy` finds the same build
([torchrun](../31_torchrun/README.md)).

## Run it

```bash
python consumer.py
```

<!-- outputs:start -->
## What it prints

**`python consumer.py`**

```text
area     : 12.0
perimeter: 14.0
loaded   : geometry.ppy
hook     : True
```

<!-- outputs:end -->

Read on: [Migrating a real project](../../docs/internals/migrating.md) ·
[A multi-module project](../26_project/README.md)

`geometry.ppy` is hand-written; there is no `.py` source and no conversion
step.
