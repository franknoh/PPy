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


@ppy.native
def total(xs: list[int]) -> int:
    s = 0
    for x in xs:
        s += x
    return s


@ppy.native
def grid_sum(g: list[list[int]]) -> int:
    s = 0
    for row in g:
        for x in row:
            s += x
    return s


@ppy.native
def lengths(words: list[str]) -> int:
    s = 0
    for w in words:
        s += len(w)
    return s


@ppy.native
def members(s: set[int], n: int) -> int:
    found = 0
    for i in range(n):
        if i in s:
            found += 1
    return found


@ppy.native
def lookups(d: dict[str, int], keys: list[str]) -> int:
    s = 0
    for k in keys:
        s += d[k]
    return s


@ppy.native
def truths(flags: list[bool]) -> int:
    s = 0
    for f in flags:
        if f:
            s += 1
    return s


@ppy.native
def touch_one(xs: list[int], i: int) -> None:
    xs[i] = xs[i] + 1


@ppy.native
def tally(xs: list[int], out: list[int]) -> None:
    s = 0
    for x in xs:
        s += x * x % 7
    out[0] = s


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
grid: list[list[int]] = [list(range(10)) for _ in range(10)]
words: list[str] = [f"word{i}" for i in range(100)]
numbers: set[int] = set(range(0, 200, 2))
names: dict[str, int] = {w: i for i, w in enumerate(words)}
flags: list[bool] = [i % 3 == 0 for i in range(100)]
sink: list[int] = [0]
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


def drive_total(i: int) -> None:
    total(listed)


def drive_grid(i: int) -> None:
    grid_sum(grid)


def drive_words(i: int) -> None:
    lengths(words)


def drive_set(i: int) -> None:
    members(numbers, 100)


def drive_names(i: int) -> None:
    lookups(names, words)


def drive_flags(i: int) -> None:
    truths(flags)


def drive_touch(i: int) -> None:
    touch_one(listed, 7)


def drive_tally(i: int) -> None:
    tally(listed, sink)


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
    rate("list[int] read, n=100", drive_total, 50000)
    rate("list[list[int]] read, 10x10", drive_grid, 50000)
    rate("list[str] read, n=100", drive_words, 50000)
    rate("set[int] read, 100 lookups", drive_set, 50000)
    rate("dict[str, int] read, n=100", drive_names, 50000)
    rate("list[bool] read, n=100", drive_flags, 50000)
    rate("one element of 100 written", drive_touch, 50000)
    rate("reads 100, writes 1 (None)", drive_tally, 50000)


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
        if "--python" in sys.argv:
            python = subprocess.run(
                [sys.executable, "boundary.ppy"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            ppy_rows = done.stdout.splitlines()
            for ours, theirs in zip(ppy_rows, python.stdout.splitlines()):
                label = ours[:28]
                print(f"{label} {ours[28:].split()[0]:>9s} {theirs[28:].split()[0]:>9s}")
        else:
            print(done.stdout, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
