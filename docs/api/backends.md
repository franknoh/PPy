# Backend API

This is the backend interface, version 1 (`BACKEND_API_VERSION`).

A backend takes the canonical IR after the shared passes and returns text,
bytes, or built artifacts. It:

- may hang passes at the `backend` stage
- must refuse in `validate` what it cannot take
- says whether its toolchain is present

An external backend is a package with an entry point in the `ppy.backends`
group. The compiler finds it without importing it, and loads it when it is
asked for. [Backends](../internals/backends.md) explains how the pieces fit
and walks through a whole example package.

A backend package imports this module and [`ppy_compiler.ir`](ir.md).

::: ppy_compiler.backend.base

## The registry

::: ppy_compiler.backend.registry
