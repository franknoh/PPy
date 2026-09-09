# A pandas expression as one kernel, nulls included

`s * t + s.fillna(0.0)` is four pandas operations and three intermediate
Series. The pandas plugin recognizes the expression tree, converges it onto
the columnar dialect of the IR, and `lower-tensor` writes one loop over the
columns' memory — with the nulls carried as a validity mask, not a NaN
convention, so the answer is what pandas computes. The same kernel serves
a NumPy-backed Series and an Arrow-backed one.

## Two expression trees, two fused loops

```python
@ppy.pure
def blend(s: pd.Series, t: pd.Series) -> pd.Series:
    return s * t + s.fillna(0.0)


@ppy.pure
def above(s: pd.Series, t: pd.Series) -> pd.Series:
    return (s > t) & t.notna()
```

Arithmetic, comparison, `fillna`, `isna`/`notna`, and the boolean operators
are the surface pandas and PyArrow share, and the plugin names them as the
same `columnar` operations PyArrow's compute lowers to. A NaN in a
`float64` Series stays the value it is, behind a bitmap of ones; an
Arrow-backed Series (`pd.ArrowDtype`) is read in place with its nulls
through the Arrow C Data Interface. The answer is a Series over the callers'
index with the same backing.

## What stays with pandas

A Series of strings, indexes that are not one index, a nullable extension
dtype, a mix of backings, an unknown method — anything the model does not
capture exactly runs pandas itself, never an approximation. Index
alignment, copy-or-view, and dtype rules are pandas' own semantics, and a
frame is never treated as a 2-D tensor. The example needs pandas
(`uv sync --group pandas`); the runners skip it where pandas is missing and
say so.

## Run it

```bash
python  frames.ppy
ppy run frames.ppy
ppy inspect frames.ppy --stage columnar
```

<!-- outputs:start -->
## What it prints

**`python  frames.ppy`**

```text
[3.0, nan, nan, 6.0, 8.75]
[False, False, False, True, False]
17.75 1
```

**`ppy run frames.ppy`**

```text
[3.0, nan, nan, 6.0, 8.75]
[False, False, False, True, False]
17.75 1
```

**`ppy inspect frames.ppy --stage columnar`**

*(prints nothing; exits 0)*

<!-- outputs:end -->

## Read on

- [Plugins: pandas and PyArrow](../../docs/internals/plugins.md) — what is modeled and what is not.
- [The IR: the columnar and arrow dialects](../../docs/internals/ir.md) — the operations and their lowering.

`frames.ppy` is hand-written; there is no `.py` source and no conversion step.
