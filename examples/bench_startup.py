"""Startup and build latency, measured in the categories that matter.

Kernel speed and startup cost are different budgets; reporting one total
hides regressions in both. This script separates them the way the compiler
thinks about them: build once, launch many times.

Not a CI gate: timings are for humans and for tracking direction.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

KERNEL = """import time

import ppy


@ppy.pure
@ppy.opt(3)
def longest(limit: int) -> int:
    best: int = 0
    for start in range(1, limit):
        n: int = start
        steps: int = 0
        while n != 1:
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
            steps += 1
        best = max(best, steps)
    return best


def main() -> None:
    t0 = time.perf_counter()
    result = longest(300000)
    print(result, f"# kernel {(time.perf_counter() - t0) * 1000.0:.1f} ms")


main()
"""


def _wall(command: list[str], cwd: Path, runs: int = 3) -> float:
    best = float("inf")
    for _ in range(runs):
        started = time.perf_counter()
        done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - started
        if done.returncode != 0:
            raise SystemExit(f"{' '.join(command)} failed:\n{done.stderr}")
        best = min(best, elapsed)
    return best * 1000.0


def main() -> int:
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        (root / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
        (root / "collatz.ppy").write_text(KERNEL, encoding="utf-8")
        compiler = [sys.executable, "-m", "ppy_compiler"]

        started = time.perf_counter()
        done = subprocess.run(
            [*compiler, "build", "collatz.ppy", "-o", "dist"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        cold_build = (time.perf_counter() - started) * 1000.0
        if done.returncode != 0:
            raise SystemExit(f"build failed:\n{done.stderr}")
        warm_build = _wall([*compiler, "build", "collatz.ppy", "-o", "dist2"], root, runs=2)

        launcher = _wall(["dist/collatz"], root)
        prebuilt = _wall(
            [*compiler, "run", "--prebuilt", "dist/ppy-bindings.json", "collatz.ppy"], root
        )
        jit = _wall([*compiler, "run", "collatz.ppy"], root, runs=2)

        print(f"cold build            {cold_build:8.0f} ms")
        print(f"warm build            {warm_build:8.0f} ms")
        print(f"launcher wall         {launcher:8.0f} ms   (kernel excluded: see program output)")
        print(f"ppy run --prebuilt    {prebuilt:8.0f} ms")
        print(f"ppy run (JIT)         {jit:8.0f} ms")
        shutil.rmtree(root / "dist2", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
