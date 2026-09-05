# Interop

A plain `.py` file importing a `.ppy` module, with no compiler and no build step.

```python
import ppy        # installs the sys.meta_path finder
import geometry   # loads geometry.ppy as ordinary Python source
```

## Provenance

Hand-written. `geometry.ppy` is written directly; there is no `.py`
source and no conversion step involved.

## What it shows

- Importing `ppy` installs the hook; without it the module is invisible and the
  import raises `ModuleNotFoundError`. The hook is never implicit.
- If `foo.py` and `foo.ppy` both exist, the `.ppy` wins and a
  `PPyAmbiguousModuleWarning` is raised.
- `ppy convert` always inserts `import ppy`, so a converted entry point can
  import its siblings.
- With the compiler installed the import is native: `geometry.ppy` is built
  once into `.ppy-cache` and its functions are bound; `PPY_IMPORT=python`
  keeps it source. Without the compiler, source it is.

## Run it

```bash
python consumer.py
```
