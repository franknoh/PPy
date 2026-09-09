# Serving over Uvicorn

One folder, one serving story: `service.ppy` is a raw ASGI callable,
`api.ppy` a FastAPI application converted from `api.py` with no hand
editing. Both run on Uvicorn through the same plugin, and the whole folder
checks under `strict = true` — route handlers included, as their author
wrote them.

## The helpers gain types; the routes keep theirs

```python
@ppy.pure
def describe(item: Item) -> str:
    return item.name + " x" + str(item.count)


@app.post("/items")
def create_item(item: Item):
    return {"label": describe(item), "cost": total_cost(item.price, item.count)}
```

`describe` and `total_cost` got their types from the call sites and
`@ppy.pure` from the checker; `describe` reads pydantic fields through the
pydantic plugin. `create_item` was left alone: `@app.get` is a decorator
nobody can vouch for, and FastAPI reads `__annotations__` at import to build
its validation, so the conversion policy refuses to touch the signature.

## What the plugin models

`FastAPI()`, `APIRouter()`, the route decorators, dependency markers
(`Depends`, `Query`, …), and the in-process `TestClient` with its responses
all have signatures under strict mode, so `client.get("/").json()`
type-checks. `uvicorn.run(app)` with a statically resolvable application
skips the per-worker re-import by module string, and the reloader is told
to watch `.ppy` alongside `.py`. `TestClient` exercises the app in-process,
which is why the output is deterministic and all three paths return
byte-identical responses.

## Run it

```bash
ppy service.ppy
ppy api.ppy
ppy run api.ppy
```

<!-- outputs:start -->
## What it prints

**`ppy service.ppy`**

```text
{"hello": "/", "count": 3} 200 404
```

**`ppy api.ppy`**

```text
{'service': 'inventory'}
{'item_id': 7, 'q': 'fast'}
{'label': 'bolt x12', 'cost': 6.0}
```

**`ppy run api.ppy`**

```text
{'service': 'inventory'}
{'item_id': 7, 'q': 'fast'}
{'label': 'bolt x12', 'cost': 6.0}
```

<!-- outputs:end -->

Read on: [Pydantic](../06_pydantic/README.md) ·
[Plugins: Uvicorn and FastAPI](../../docs/internals/plugins.md)

`service.ppy` is hand-written. `api.ppy` is Generated, not hand-written:
exactly what `ppy convert api.py` writes, and `examples/verify_conversions.py`
checks that on every run.
