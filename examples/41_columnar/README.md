# Columnar expressions

Expressions over pandas Series converge onto the columnar dialect of the IR
and fuse into one kernel over the columns' memory, nulls included.

## Provenance

Hand-written. `frames.ppy` is written directly; there is no `.py` source
and no conversion step involved.

## What it shows

- `s * t + s.fillna(0.0)` and `(s > t) & t.notna()` are expression trees
  the pandas plugin recognizes; each becomes one fused loop over the
  columns, with the nulls carried as a validity mask rather than a NaN
  convention, so the answer is what pandas would compute.
- A NumPy-backed Series keeps its index and its NaNs; an Arrow-backed one
  (`pd.ArrowDtype`) keeps its nulls, and the same kernel serves both
  through the Arrow C data interface.
- Anything the plugin does not recognize -- a Series of strings, an index
  that does not align, an unknown method -- falls back to pandas itself,
  never to a different answer.
- The example needs pandas (`uv sync --group pandas`); the runners skip it
  where pandas is missing and say so.

## Run it

```bash
python  frames.ppy
ppy run frames.ppy
ppy inspect frames.ppy --stage columnar   # the columnar dialect before it lowers to loops
```
