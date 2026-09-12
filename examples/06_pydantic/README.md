# Pydantic

Models are typed by the plugin and still validated by pydantic at run time.

## Two shapes, kept apart

A pydantic model has a constructor shape and a validated output shape, and
the plugin distinguishes them. `User(id="123", name="ada")` is legal —
pydantic coerces the string — and `user.id + 1` is `int` arithmetic, because
the checker types the field from the model, not from the argument.
Validation still runs exactly as pydantic wrote it.

## Constraints become facts

```python
class Pixel(BaseModel):
    r: Annotated[int, Field(ge=0, le=255)]
    g: Annotated[int, Field(ge=0, le=255)]
    b: Annotated[int, Field(ge=0, le=255)]


@ppy.pure
def luminance(pixel: Pixel) -> float:
    return 0.2126 * pixel.r + 0.7152 * pixel.g + 0.0722 * pixel.b
```

`Field(ge=0, le=255)` is read as an integer range as well as a validator —
the same refinement `ppy.Range(0, 255)` gives — so inside `luminance` the
three fields are known to fit a byte and the arithmetic needs no overflow
guard. Both spellings count, `Annotated[int, Field(...)]` and
`count: int = Field(ge=0, le=100)`, and so does `conint`.

Schema building is code the model runs at import. It falls under the same
policy as JAX export: `[tool.ppy] build-execution` decides whether a build
may execute project code, and the default is `deny`.

## Run it

```bash
python  pydantic_models.ppy
ppy     pydantic_models.ppy
ppy run pydantic_models.ppy
```

<!-- outputs:start -->
## What it prints

**`python  pydantic_models.ppy`**, **`ppy     pydantic_models.ppy`**, **`ppy run pydantic_models.ppy`**

```text
145.7586 124 ada
```

<!-- outputs:end -->

Read on: [Plugins: Pydantic](../../docs/internals/plugins.md) ·
[Uvicorn and FastAPI](../27_uvicorn/README.md)

`pydantic_models.ppy` is hand-written; there is no `.py` source and no
conversion step.
