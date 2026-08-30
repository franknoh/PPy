"""Python/native boundary costs, measured one category at a time.

The profitability model (`should_lower_native`) is built on these numbers;
this script keeps them honest on the machine in front of you. Not a CI
gate.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

PROGRAM = """import array
import time

import ppy
from ppy import Buffer


@ppy.pure
def tiny(x: int, y: int) -> int:
    return x + y


@ppy.native
@ppy.pure
def tiny_native(x: int, y: int) -> int:
    return x + y


@ppy.pure
@ppy.opt(3)
def loop100(n: int) -> int:
    out: int = 0
    for i in range(n):
        out += i
    return out


@ppy.pure
def summed(xs: Buffer[int]) -> int:
    return sum(xs)


def rate(label: str, call, rounds: int) -> None:
    started = time.perf_counter()
    for _ in range(rounds):
        call()
    elapsed = time.perf_counter() - started
    print(f"{label:<28s} {elapsed / rounds * 1e9:9.0f} ns/call")


def main() -> None:
    xs = array.array("q", range(100))
    overflowing: list[int] = [1 << 100]
    big: int = overflowing[0]
    rate("tiny, kept in Python", lambda: tiny(2, 3), 200000)
    rate("tiny, forced native", lambda: tiny_native(2, 3), 200000)
    rate("native loop, n=100", lambda: loop100(100), 200000)
    rate("borrowed buffer, n=100", lambda: summed(xs), 200000)
    rate("guard failure -> fallback", lambda: tiny_native(big, 3), 200000)


main()
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        (root / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
        (root / "boundary.ppy").write_text(PROGRAM, encoding="utf-8")
        done = subprocess.run(
            [sys.executable, "-m", "ppy_compiler", "run", "boundary.ppy"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            raise SystemExit(f"run failed:\n{done.stderr}")
        print(done.stdout, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
