# Containers

Element types, aliasing, and local mutation.

## Provenance

Hand-written. `containers.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- An empty container gets its element type from what is first put into it.
- Filling a container the function allocated is permitted inside `@ppy.pure`.
- Mutating a parameter, or sharing a local before mutating it, is not.

## Run it

```bash
python  containers.ppy
ppy     containers.ppy
ppy run containers.ppy
```

<!-- outputs:start -->
## What it prints

**`python  containers.ppy`**

```text
[(1, 1), (2, 2), (3, 3)]
[0, 1, 4, 9, 16, 25]
3
[1, 2, 3, 4, 5]
```

**`ppy     containers.ppy`**

```text
[(1, 1), (2, 2), (3, 3)]
[0, 1, 4, 9, 16, 25]
3
[1, 2, 3, 4, 5]
```

**`ppy run containers.ppy`**

```text
[(1, 1), (2, 2), (3, 3)]
[0, 1, 4, 9, 16, 25]
3
[1, 2, 3, 4, 5]
```

<!-- outputs:end -->
