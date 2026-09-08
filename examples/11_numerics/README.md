# Numerics

Where PPY refuses to differ from CPython.

## Provenance

Hand-written. `numerics.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- Floor division rounds toward negative infinity and the remainder takes the divisor's sign.
- An expression that overflows a machine word hands the call back to CPython rather than wrapping.

## Run it

```bash
python  numerics.ppy
ppy     numerics.ppy
ppy run numerics.ppy
```

<!-- outputs:start -->
## What it prints

**`python  numerics.ppy`**

```text
1000000
2432902008176640000
265252859812191058636308480000000
-4 3
1 -1
```

**`ppy     numerics.ppy`**

```text
1000000
2432902008176640000
265252859812191058636308480000000
-4 3
1 -1
```

**`ppy run numerics.ppy`**

```text
1000000
2432902008176640000
265252859812191058636308480000000
-4 3
1 -1
```

<!-- outputs:end -->
