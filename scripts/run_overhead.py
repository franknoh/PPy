"""Is `ppy run` ever slower than `python`? Every example, timed both ways.

For each example program this reports the median wall time of `python
prog.ppy`, of `ppy run prog.ppy` from a cold project cache (the first run:
analysis, lowering, and LLVM compilation included), and of `ppy run` warm
(the cache from the run before). A warm run slower than `python` by more than
the noise margin is flagged; a cold run is reported, not judged, since it
pays for compilation once.

    uv run python scripts/run_overhead.py            # every example
    uv run python scripts/run_overhead.py 48 strings # the ones whose path matches

Run it alone on a quiet machine: it measures wall time, one process at a
time. Not a CI gate; the decision logic has cheap tests of its own.
"""

from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(EXAMPLES))

from run_all import _missing_libraries  # noqa: E402  pylint: disable=wrong-import-position

#: A warm `ppy run` within this much of `python` counts as the same time:
#: 5% of the Python time, and never less than 30 ms, which is what two runs of
#: the same `python` command differ by on a loaded laptop.
MARGIN_FRACTION = 0.05
MARGIN_FLOOR = 0.030

RUNS = int(os.environ.get("PPY_OVERHEAD_RUNS", "5"))


def _entries(only: list[str]) -> list[Path]:
    entries = (
        sorted(EXAMPLES.glob("[0-9]*/[a-z]*.ppy"))
        + sorted(EXAMPLES.glob("[0-9]*/[0-9]*/[a-z]*.ppy"))
        + sorted(EXAMPLES.glob("*/src/app.ppy"))
    )
    if only:
        entries = [e for e in entries if any(token in str(e) for token in only)]
    return entries


def _cache(entry: Path) -> Path:
    """The project cache `ppy run` keeps beside the nearest `pyproject.toml`."""
    for folder in [entry.parent, *entry.parents]:
        if (folder / "pyproject.toml").is_file():
            return folder / ".ppy-cache"
        if folder == ROOT:
            break
    return entry.parent / ".ppy-cache"


def _once(command: list[str], cwd: Path) -> float | None:
    started = time.perf_counter()
    done = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL
    )
    elapsed = time.perf_counter() - started
    return elapsed if done.returncode == 0 else None


def _median(command: list[str], cwd: Path, runs: int, before=None) -> float | None:  # type: ignore[no-untyped-def]
    times = []
    for _ in range(runs):
        if before is not None:
            before()
        elapsed = _once(command, cwd)
        if elapsed is None:
            return None
        times.append(elapsed)
    return statistics.median(times)


def margin(python: float) -> float:
    """How much slower a warm `ppy run` may be and still count as no slower."""
    return max(python * MARGIN_FRACTION, MARGIN_FLOOR)


def main() -> int:
    slower = 0
    print(f"{'example':44} {'python':>9} {'cold':>9} {'warm':>9}  verdict")
    for entry in _entries(sys.argv[1:]):
        if _missing_libraries(entry.parent):
            continue
        cwd, name = entry.parent, entry.name
        cache = _cache(entry)
        rel = str(entry.relative_to(EXAMPLES))
        python = _median([sys.executable, name], cwd, RUNS)
        if python is None:
            print(f"{rel:44} {'fails':>9}")
            continue
        run = [sys.executable, "-m", "ppy_compiler", "run", name]
        cold = _median(run, cwd, max(1, RUNS // 2), before=lambda: shutil.rmtree(cache, True))
        warm = _median(run, cwd, RUNS)
        if cold is None or warm is None:
            print(f"{rel:44} {python:9.3f} {'fails':>9}")
            continue
        verdict = "ok" if warm <= python + margin(python) else "SLOWER"
        slower += verdict != "ok"
        print(f"{rel:44} {python:9.3f} {cold:9.3f} {warm:9.3f}  {verdict}")
    print(f"\n{slower} example(s) slower warm than python beyond the margin")
    return 1 if slower else 0


if __name__ == "__main__":
    raise SystemExit(main())
