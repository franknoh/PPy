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


@ppy.native
@ppy.pure
def tiny_default(x: int, y: int = 3) -> int:
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


@ppy.native
def scaled(xs: list[int]) -> int:
    for i in range(len(xs)):
        xs[i] = xs[i] * 3 % 1000
    return len(xs)


@ppy.native
def weighed(d: dict[int, int]) -> int:
    s = 0
    for k, v in d.items():
        s += k * v
    return s


class Link:
    def __init__(self, value: int) -> None:
        self.value = value
        self.next: Link | None = None


@ppy.native
def chained(head: Link) -> int:
    s = 0
    node: Link | None = head
    while node is not None:
        s += node.value
        node = node.next
    return s


@ppy.native
def filled(xs: list[int]) -> None:
    for i in range(len(xs)):
        xs[i] = i


# Module-level aliases and per-iteration-varying arguments: both defeat the
# optimizer, which happily folds a small pure call with constant arguments
# into its answer -- and a folded call measures nothing.
plain = tiny
native = tiny_native
defaulted = tiny_default
looped = loop100
buffered = summed
values = array.array("q", range(100))
overflowing: list[int] = [1 << 100]
big: int = overflowing[0]
listed: list[int] = list(range(100))
mapped: dict[int, int] = {i: i for i in range(100)}
links: list[Link] = [Link(i) for i in range(10)]
for left, right in zip(links, links[1:]):
    left.next = right
scale = scaled
weigh = weighed
chain = chained
fill = filled


def drive_plain(i: int) -> None:
    plain(i, 3)


def drive_native(i: int) -> None:
    native(i, 3)


def drive_keyword(i: int) -> None:
    native(i, y=3)


def drive_default(i: int) -> None:
    defaulted(i)


def drive_loop(i: int) -> None:
    looped(100)


def drive_buffer(i: int) -> None:
    buffered(values)


def drive_guard(i: int) -> None:
    native(big, i)


def drive_list(i: int) -> None:
    scale(listed)


def drive_dict(i: int) -> None:
    weigh(mapped)


def drive_objects(i: int) -> None:
    chain(links[0])


def drive_none(i: int) -> None:
    fill(listed)


def rate(label: str, call: Callable[[int], None], rounds: int) -> None:
    started = time.perf_counter()
    for i in range(rounds):
        call(i)
    elapsed = time.perf_counter() - started
    print(f"{label:<28s} {elapsed / rounds * 1e9:9.0f} ns/call")


def main() -> None:
    rate("tiny, kept in Python", drive_plain, 200000)
    rate("tiny, forced native", drive_native, 200000)
    rate("tiny, by keyword", drive_keyword, 200000)
    rate("tiny, default left out", drive_default, 200000)
    rate("native loop, n=100", drive_loop, 200000)
    rate("borrowed buffer, n=100", drive_buffer, 200000)
    rate("guard failure -> fallback", drive_guard, 200000)
    rate("list[int] written, n=100", drive_list, 50000)
    rate("dict[int, int] read, n=100", drive_dict, 50000)
    rate("10 linked objects", drive_objects, 50000)
    rate("returns None, n=100", drive_none, 50000)


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
