# FastAPI

A FastAPI service converted from ordinary Python, exercised in-process
through `TestClient` so the output is deterministic.

## Provenance

Generated, not hand-written. `service.ppy` is exactly what
`ppy convert service.py` writes; `verify_conversions.py` regenerates it to
prove that.

## What it shows

- The helpers gain types and `@ppy.pure`; `describe` reads pydantic model
  fields through the plugin.
- The route handlers are left exactly as written: `@app.get` is a decorator
  nobody can vouch for, and FastAPI reads `__annotations__` at import time to
  build validation — so the conversion must hand it the same signatures the
  author wrote. This is the annotation-materialization policy doing its job,
  not a gap.
- All three execution paths return byte-identical responses, matching the
  original `.py`.

## Limits

There is no FastAPI plugin yet, so under `strict = true` every
`FastAPI()`/`TestClient` call is an unknown signature (`E1306`). The project
sets `strict = false` and the checker passes; a plugin would lift this.

## Run it

```bash
python service.ppy    # plain CPython
ppy service.ppy       # optimized backend
ppy run service.ppy   # native-eligible helpers through LLVM
```
