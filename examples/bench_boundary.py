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
from collections.abc import Callable

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


# Module-level aliases and per-iteration-varying arguments: both defeat the
# optimizer, which happily folds a small pure call with constant arguments
# into its answer -- and a folded call measures nothing.
plain = tiny
native = tiny_native
looped = loop100
buffered = summed
values = array.array("q", range(100))
overflowing: list[int] = [1 << 100]
big: int = overflowing[0]


def drive_plain(i: int) -> None:
    plain(i, 3)


def drive_native(i: int) -> None:
    native(i, 3)


def drive_loop(i: int) -> None:
    looped(100)


def drive_buffer(i: int) -> None:
    buffered(values)


def drive_guard(i: int) -> None:
    native(big, i)


def rate(label: str, call: Callable[[int], None], rounds: int) -> None:
    started = time.perf_counter()
    for i in range(rounds):
        call(i)
    elapsed = time.perf_counter() - started
    print(f"{label:<28s} {elapsed / rounds * 1e9:9.0f} ns/call")


def main() -> None:
    rate("tiny, kept in Python", drive_plain, 200000)
    rate("tiny, forced native", drive_native, 200000)
    rate("native loop, n=100", drive_loop, 200000)
    rate("borrowed buffer, n=100", drive_buffer, 200000)
    rate("guard failure -> fallback", drive_guard, 200000)


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
