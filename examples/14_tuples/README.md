# Tuples

Fixed tuples flatten into scalar atoms.

## Provenance

Hand-written. `tuples.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- A tuple of known length and scalar elements is passed and returned unboxed.
- A homogeneous or oversized tuple stays boxed.

## Run it

```bash
python  tuples.ppy
ppy     tuples.ppy
ppy run tuples.ppy
```

<!-- outputs:start -->
## What it prints

**`python  tuples.ppy`**

```text
(2.0, 3.0)
25.0
(3, 2) (-4, 3)
```

**`ppy     tuples.ppy`**

```text
(2.0, 3.0)
25.0
(3, 2) (-4, 3)
```

**`ppy run tuples.ppy`**

```text
(2.0, 3.0)
25.0
(3, 2) (-4, 3)
```

<!-- outputs:end -->
