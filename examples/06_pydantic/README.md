# Pydantic

Models keep their runtime validation.

## Provenance

Hand-written. `pydantic_models.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- The plugin describes field types to the checker without bypassing validation.

## Run it

```bash
python  pydantic_models.ppy
ppy     pydantic_models.ppy
ppy run pydantic_models.ppy
```

<!-- outputs:start -->
## What it prints

**`python  pydantic_models.ppy`**

```text
145.7586 124 ada
```

**`ppy     pydantic_models.ppy`**

```text
145.7586 124 ada
```

**`ppy run pydantic_models.ppy`**

```text
145.7586 124 ada
```

<!-- outputs:end -->
