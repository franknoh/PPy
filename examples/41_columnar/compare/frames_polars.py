"""The same expressions in polars: lazy expressions the engine fuses and runs across threads."""

import time

import numpy as np
import polars as pl


def blend(frame):
    mixed = pl.col("s") * pl.col("t") + pl.col("s").fill_null(0.0)
    return frame.select(mixed.alias("out"))["out"]


def above(frame):
    mask = (pl.col("s") > pl.col("t")) & pl.col("t").is_not_null()
    return frame.select(mask.alias("out"))["out"]


def timed(label, run):
    best = 1e9
    answer = None
    for _ in range(5):
        started = time.perf_counter()
        answer = run()
        best = min(best, (time.perf_counter() - started) * 1000.0)
    print(f"# {label}: {best:.2f} ms")
    return answer


def main():
    n = 8_000_000
    left = np.linspace(0.0, 10.0, n)
    right = np.linspace(5.0, 0.0, n)
    left[::7] = np.nan
    right[::11] = np.nan
    # polars keeps nulls apart from NaN: the missing values arrive as nulls.
    frame = pl.DataFrame(
        {"s": pl.Series(left).fill_nan(None), "t": pl.Series(right).fill_nan(None)}
    )
    blend(frame)
    above(frame)
    mixed = timed("blend", lambda: blend(frame))
    mask = timed("above", lambda: above(frame))
    print(f"{float(mixed.sum()):.3f} {int(mask.sum())} {int(mixed.null_count())}")


main()
