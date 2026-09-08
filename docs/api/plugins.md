# Plugin API

The library plugin interface, version 2. A plugin extends the compiler's
semantics through explicit hooks and nothing else: it types the calls and
attributes of the modules it claims, declares their effects, says how each
recognized operation lowers — as a backend-neutral spec, never as backend
code — and may register dialects, passes, patterns, and lowerings for the
IR. Every hook has a no-op default, so the compiler calls them directly.
What each builtin plugin does is in [Plugins](../internals/plugins.md).

::: ppy_compiler.plugins.base
