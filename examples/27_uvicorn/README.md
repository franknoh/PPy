# Serving over Uvicorn

One folder, one serving story: Uvicorn is the server, and what runs on it
ranges from a raw ASGI callable to a FastAPI application — FastAPI is an ASGI
framework, so both go through the same `uvicorn.run` and the same plugin.

## Provenance

Mixed, deliberately:

- `service.ppy` is hand-written; there is no `.py` source.
- `api.ppy` is **Generated, not hand-written**: exactly what
  `ppy convert api.py` writes, and `verify_conversions.py` regenerates it to
  prove that.

## What it shows

- The uvicorn plugin resolves the application statically instead of
  re-importing it by module string per worker, and tells the reloader to
  watch `.ppy` sources.
- In `api.ppy`, the helpers gain types and `@ppy.pure` (`describe` reads
  pydantic model fields through the plugin), while the FastAPI route handlers
  keep exactly the signatures their author wrote: `@app.get` is a decorator
  nobody can vouch for, and FastAPI reads `__annotations__` at import time to
  build validation — the annotation-materialization policy doing its job.
- `TestClient` exercises the app in-process, so the output is deterministic
  and all three paths return byte-identical responses.

## Limits

There is no FastAPI plugin yet, so under `strict = true` every `FastAPI()`
and `TestClient` call is an unknown signature (`E1306`); this folder sets
`strict = false`. A plugin would lift this.

## Run it

```bash
ppy service.ppy   # raw ASGI handler, no server
ppy api.ppy       # FastAPI routes through TestClient
ppy run api.ppy   # the same, native-eligible helpers through LLVM
```
