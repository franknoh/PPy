# Narrowing

Every form the checker understands.

## Provenance

Hand-written. `narrowing.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- `is None`, `isinstance`, `match` patterns, boolean operators, and walrus bindings all narrow.
- A `match` case sees the subject with earlier cases subtracted; a guarded case rules nothing out.

## Run it

```bash
python  narrowing.ppy
ppy     narrowing.ppy
ppy run narrowing.ppy
```

<!-- outputs:start -->
## What it prints

**`python  narrowing.ppy`**

```text
3 0
True False False
3 -1 -1
4 7
none int:42 str:HI
3 0
```

**`ppy     narrowing.ppy`**

```text
3 0
True False False
3 -1 -1
4 7
none int:42 str:HI
3 0
```

**`ppy run narrowing.ppy`**

```text
3 0
True False False
3 -1 -1
4 7
none int:42 str:HI
3 0
```

<!-- outputs:end -->
