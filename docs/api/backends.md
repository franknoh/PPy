# Backend API

The backend interface, version 1 (`BACKEND_API_VERSION`). A backend
consumes the canonical IR after the shared passes and answers with text,
bytes, or built artifacts; it may hang passes at the `backend` stage, must
refuse in `validate` what it cannot take, and says whether its toolchain
is here. An external backend is a package with an entry point in the
`ppy.backends` group; the compiler finds it without importing it and loads
it when it is asked for. How the pieces fit, and a whole example package,
is in [Backends](../internals/backends.md).

What a backend package imports is this module and [`ppy_compiler.ir`](ir.md).

::: ppy_compiler.backend.base

## The registry

::: ppy_compiler.backend.registry
