# ASGI application

Uvicorn stays the server; what changes is how the application is resolved.

## Provenance

Hand-written. `service.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- The application is resolved statically instead of re-imported by module string
  on every worker start.
- The development reloader is told to watch `.ppy` sources, which it otherwise
  never sees because it watches `*.py` by default.
- `Scope` and the receive/send callables are an explicit `Any` boundary, not an
  inferred one.

## Run it

```bash
ppy service.ppy   # runs the handler directly, no server
```
