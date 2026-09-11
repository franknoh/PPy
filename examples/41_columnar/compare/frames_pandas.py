"""The same expressions as pandas evaluates them: one pass and one temporary per operator."""

import time

import numpy as np
import pandas as pd


def blend(s, t):
    return s * t + s.fillna(0.0)


def above(s, t):
    return (s > t) & t.notna()


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
    s = pd.Series(left, name="s")
    t = pd.Series(right, name="t")
    mixed = timed("blend", lambda: blend(s, t))
    mask = timed("above", lambda: above(s, t))
    print(f"{float(mixed.sum()):.3f} {int(mask.sum())} {int(mixed.isna().sum())}")


main()
