# PPY — Pretty Python

A source-compatible, statically analyzable subset of Python. A `.ppy` file *is*
valid Python: it runs under plain CPython with no compiler involved. The
compiler adds static checking, an optimized Python backend, and an LLVM native
backend — and all three must produce the same answer.

```bash
uv sync --group all
uv run ppy doctor
```

```python
# collatz.ppy
import ppy


@ppy.pure
@ppy.opt(3)
def longest(limit: int) -> int:
    best: int = 0
    for start in range(1, limit):
        n: int = start
        steps: int = 0
        while n != 1:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            steps += 1
        best = max(best, steps)
    return best


print(longest(300000))
```

```bash
uv run python  collatz.ppy   # plain CPython, no compiler   1262 ms
uv run ppy     collatz.ppy   # optimized Python backend     1305 ms
uv run ppy run collatz.ppy   # LLVM native                   112 ms
```

Turn ordinary Python into it:

```bash
uv run ppy convert src/ --in-place
```

## Docs

- [docs/guide.md](docs/guide.md) — what it does, what it costs, how the pieces fit
- [docs/cli.md](docs/cli.md) — every command and option
- [examples/README.md](examples/README.md) — 27 worked examples

## Checks

```bash
uv run pytest -q
```

```bash
uv run python examples/run_all.py             # every example x 3 paths, compared
uv run python examples/verify_conversions.py  # every .ppy regenerated from its .py
uv run python examples/lint_all.py            # pylint over every example
```
