"""The same expression under numexpr: one string, evaluated in chunks across threads."""

import time

import numexpr as ne
import numpy as np


def timed(label, run):
    best = 1e9
    answer = None
    for _ in range(5):
        started = time.perf_counter()
        answer = run()
        best = min(best, time.perf_counter() - started)
    print(f"# {label}: {best * 1000:.2f} ms")
    return answer


def main():
    x = np.linspace(0.0, 10.0, 8_000_000)
    y = np.linspace(1.0, 5.0, 8_000_000)
    names = {"x": x, "y": y}
    expression = "(x * y + x) * (y - x) + x * 0.5 - y * 0.25"
    ne.evaluate(expression, local_dict=names)
    out = timed("fused", lambda: ne.evaluate(expression, local_dict=names))
    print(f"{float(out[7]):.12f} {float(out[-1]):.12f}")
    total = lambda: float(ne.evaluate("sum(x * x)", local_dict=names))  # noqa: E731
    print(f"{timed('sum of squares', total):.3f}")


main()
