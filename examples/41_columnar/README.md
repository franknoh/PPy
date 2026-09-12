# Columnar expressions

Expressions over pandas Series converge onto the columnar dialect of the IR
and fuse into one kernel over the columns' memory, nulls included.

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
same `columnar` operations PyArrow's compute lowers to. Each tree becomes
one loop -- a `columnar.map` -- that reads each input once and writes the
answer once, with nothing materialized between the operators. The nulls are
whatever the Series' backing makes them: a NumPy-backed `float64` Series
runs under NumPy's convention, where a NaN is the null `fillna` fills and
`isna` finds and a bool mask is a byte per row, and the answer is written
straight into the array the result Series wraps; an Arrow-backed Series
(`pd.ArrowDtype`) is read in place with its validity bits through the Arrow
C Data Interface, and the answer carries its own. Either way the Series
comes back over the callers' index with the same backing.

## What stays with pandas

A Series of strings, indexes that are not one index, a nullable extension
dtype, a mix of backings, an unknown method — anything the model does not
capture exactly runs pandas itself, never an approximation. The fused loop
is one thread; a `polars` plan runs across all of them. Index
alignment, copy-or-view, and dtype rules are pandas' own semantics, and a
frame is never treated as a 2-D tensor. The example needs pandas
(`uv sync --group pandas`); the runners skip it where pandas is missing and
say so.

## Compared with pandas and polars

The two expressions over eight million rows, a NaN in every seventh row of
`s` and every eleventh of `t`, in [`compare/`](compare/):
[`frames_bench.ppy`](compare/frames_bench.ppy),
[`frames_pandas.py`](compare/frames_pandas.py),
[`frames_polars.py`](compare/frames_polars.py). Milliseconds, best of five
calls, over five processes.

**PPY** is the pandas expression in a function; **pandas** is the same
expression with no function around it, one pass and one temporary per
operator (numexpr under it, where installed); **polars** is the expression
rewritten in its own vocabulary, with the NaNs as nulls, planned and run
across threads:

```python
@ppy.pure
def blend(s: pd.Series, t: pd.Series) -> pd.Series:
    return s * t + s.fillna(0.0)
```

```python
def blend(frame):
    mixed = pl.col("s") * pl.col("t") + pl.col("s").fill_null(0.0)
    return frame.select(mixed.alias("out"))["out"]
```

<!-- compare:start -->
| | PPY | pandas | polars |
|---|---:|---:|---:|
| blend | 22.47 ± 1.63 | 31.09 ± 1.11 | **13.24 ± 0.42** |
| above | **4.65 ± 0.37** | 6.78 ± 0.19 | 4.91 ± 0.19 |
<!-- compare:end -->

The fused loop reads `s` and `t` once and writes the answer once, on one
thread, into the array the result Series wraps; pandas makes a temporary
per operator, three passes over 64 MB each, and numexpr under it splits
the arithmetic across threads to make that up. polars plans the expression
and runs it across every core, which is what its row is. The point of the
PPY column is the source: it is the pandas line, unchanged, with pandas'
own NaN convention, where the polars column is a different program.

Intel Core Ultra 9 386H (16 threads); pandas 3.0.5 with numexpr 2.14.2,
polars 1.44.2 on CPython 3.12.13, PPY with pandas 3.0.5 on CPython 3.14.5,
from a checkout on a native filesystem.

## Run it

```bash
python  frames.ppy
ppy run frames.ppy
ppy inspect frames.ppy --stage columnar
```

<!-- outputs:start -->
## What it prints

**`python  frames.ppy`**, **`ppy run frames.ppy`**

```text
[3.0, nan, nan, 6.0, 8.75]
[False, False, False, True, False]
17.75 1
```

**`ppy inspect frames.ppy --stage columnar`**

*(prints nothing; exits 0)*

<!-- outputs:end -->

Read on: [Plugins: pandas and PyArrow](../../docs/internals/plugins.md) ·
[The IR: the columnar and arrow dialects](../../docs/internals/ir.md)

`frames.ppy` is hand-written; there is no `.py` source and no conversion step.
