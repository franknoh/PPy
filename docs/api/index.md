# API

Three packages, installed together as `ppy-lang`. The pages in this section
are generated at build time from the docstrings in the source, and the
surface is what each module's `__all__` says it is.

| package | what | when you import it |
|---|---|---|
| [`ppy`](ppy.md) | the directives, the markers, typed input, and the namespaces a program writes against (`native`, `simd`, `atomic`, `concurrent`, `parallel`, `aio`, `cuda`, `xla` …), each a reference implementation under CPython | every `.ppy` program, and every `.py` that imports a `.ppy` module |
| [`ppy_runtime`](runtime.md) | what a built artifact needs at launch: the manifest, the bindings, the launcher, and the runtimes native code calls into; never imports the compiler | the launcher does; a program reaches for `ppy_runtime.arrow` or `launch.main` at most |
| [Plugin API](plugins.md) | `ppy_compiler.plugins.base`: the class a library plugin extends and the lowering specs a `call` answers with | when writing a library plugin |
| [IR](ir.md) | `ppy_compiler.ir`: the canonical IR's model, types, dialects, passes, patterns, verifier, linker, and codec | when a plugin registers a dialect or a pass, or a tool reads `.ppyir` |
| [Backend API](backends.md) | `ppy_compiler.backend`: the class a code-generating backend extends, the context and formats it is handed, and the registry that finds one | when writing a backend for another target |

The rest of `ppy_compiler` — analysis, lowering, the builtin backends, the
driver — is the implementation behind the CLI, not public API. What is stable is in
[Compatibility](../reference/compatibility.md).
