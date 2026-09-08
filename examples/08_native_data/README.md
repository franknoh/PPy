# Native data

Which Python values have a native representation.

## Provenance

Hand-written. `native_data.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- Scalars, fixed tuples, and all-scalar classes flatten into machine values.
- Everything else stays boxed, and `ppy explain` says which and why.

## Run it

```bash
python  native_data.ppy
ppy     native_data.ppy
ppy run native_data.ppy
```

<!-- outputs:start -->
## What it prints

**`python  native_data.ppy`**

```text
32.0
15
1000000000000000000000000000001
14.0 (2.0, 3.0)
383
```

**`ppy     native_data.ppy`**

```text
32.0
15
1000000000000000000000000000001
14.0 (2.0, 3.0)
383
```

**`ppy run native_data.ppy`**

```text
32.0
15
1000000000000000000000000000001
14.0 (2.0, 3.0)
383
```

<!-- outputs:end -->
