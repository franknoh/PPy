"""Writes across the Python boundary, copied by the generated wrapper.

When Python calls a native function that takes a `list`, a `dict`, a `set`,
or an object, the generated CPython-ABI wrapper copies each argument in and,
after a call that writes through a parameter, every container and object that
came in back into the caller's object (`backend/llvm/crossing.c`). These hold
`ppy run` to CPython where the copies could disagree with it: the same list
passed twice, shared rows, a row the call took out of its list, a dict's
values, objects reached through fields, a cycle; and the rest of what the
wrapper does in C: a result of `None`, output held while the call runs, and
an exception that ends a call with containers in it. The last line of each
program says which functions answered natively (`ppy.native.compiled`), which
CPython says False of.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler

pytestmark = [
    pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed"),
    pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH"),
]

ALIASES = """
import ppy


@ppy.native
def bump(g: list[list[int]], n: int) -> None:
    for i in range(n):
        for j in range(len(g[i])):
            g[i][j] += i * 10 + j


@ppy.native
def add_into(a: list[int], b: list[int]) -> int:
    for i in range(len(a)):
        a[i] += b[i]
    return len(a)


@ppy.native
def take_row(g: list[list[int]]) -> list[int]:
    row = g[0]
    row.append(len(g))
    g.pop(0)
    g.append([7, 7])
    return row


@ppy.native
def tags(d: dict[str, list[int]], n: int) -> int:
    total = 0
    for key in d:
        for i in range(n):
            d[key].append(i)
            total += len(key)
    return total


@ppy.native
def counts(d: dict[int, int], n: int) -> None:
    for i in range(n):
        d[i % 3] = d.get(i % 3, 0) + i
    d.pop(2, 0)


@ppy.native
def sort_all(xs: list[int], ys: list[float]) -> None:
    xs.sort()
    xs.reverse()
    xs.append(len(xs))
    ys.sort()
    ys.insert(0, -1.5)


@ppy.native
def reorder(pairs: list[tuple[int, float]], names: list[str]) -> list[str]:
    pairs.sort()
    names.reverse()
    names.append(names[0] + "!")
    return names


@ppy.native
def members(s: set[int], n: int) -> int:
    for i in range(n):
        if i % 2 == 0:
            s.add(i)
    return len(s)


@ppy.native
def same_set(s: set[int], n: int) -> int:
    total = 0
    for i in range(n):
        if i in s:
            total += i
    return total


def main() -> None:
    shared = [[0] * 3] * 3
    bump(shared, 3)
    print(shared, shared[0] is shared[1])
    fresh = [[1, 2], [3, 4]]
    rows = [fresh[0], fresh[0], fresh[1]]
    bump(rows, 3)
    print(rows, fresh, rows[0] is fresh[0], rows[1] is rows[0])
    a = [1, 2, 3]
    print(add_into(a, a), a)
    first = [1, 2]
    g = [first, [3]]
    kept = g
    got = take_row(g)
    print(got is first, first, g, kept is g, g[1])
    inner = [5]
    d = {"a": inner, "bb": inner, "c": [1]}
    print(tags(d, 2), d, d["a"] is inner, d["a"] is d["bb"])
    walked = {0: 1, 1: 2, 2: 3, 9: 9}
    view = walked
    counts(walked, 7)
    print(walked, list(walked), view is walked)
    xs = [3, 1, 2]
    ys = [2.5, -0.5]
    alias = xs
    sort_all(xs, ys)
    print(xs, ys, alias is xs)
    pairs = [(2, 0.5), (1, 9.0), (2, -1.0)]
    names = ["x", "y"]
    out = reorder(pairs, names)
    print(pairs, names, out is names)
    s = {5, 3, 1}
    print(members(s, 6), sorted(s))
    big = {100, 3, 200, 7, 1}
    before = list(big)
    print(same_set(big, 10), list(big) == before)
    print(ppy.native.compiled(bump), ppy.native.compiled(take_row), ppy.native.compiled(members))


main()
"""

OBJECTS = """
from dataclasses import dataclass

import ppy


class Node:
    def __init__(self, value: int) -> None:
        self.value = value
        self.next: Node | None = None
        self.seen: list[int] = []

    @ppy.native
    def push(self, k: int) -> int:
        for i in range(k):
            self.seen.append(i * self.value)
        return len(self.seen)

    @staticmethod
    @ppy.native
    def twice(n: int) -> int:
        total = 0
        for i in range(n):
            total += 2
        return total


@dataclass
class Cell:
    x: int
    y: float


@ppy.native
def walk(head: Node, step: int) -> int:
    node: Node | None = head
    count = 0
    while node is not None and count < 10:
        node.value += step
        node.seen.append(count)
        node = node.next
        count += 1
    return count


@ppy.native
def relink(head: Node) -> Node:
    second = head.next
    if second is not None:
        head.next = second.next
        second.next = head
        return second
    return head


@ppy.native
def through(holder: list[Node], k: int) -> None:
    for node in holder:
        for i in range(k):
            node.value += i
        if node.next is not None:
            node.next.value = -node.value


@ppy.native
def cells(xs: list[Cell], dx: int) -> list[Cell]:
    for i in range(len(xs)):
        if i % 2 == 0:
            xs[i] = Cell(xs[i].x + dx, xs[i].y * 2.0)
    return xs


def main() -> None:
    a = Node(1)
    b = Node(2)
    c = Node(3)
    a.next = b
    b.next = c
    c.next = a  # a cycle
    print(walk(a, 5), a.value, b.value, c.value, a.seen, c.next is a)
    head = relink(a)
    print(head is b, b.next is a, a.next is c, c.next is a, head.value)
    nodes = [a, b, a]
    through(nodes, 3)
    print(a.value, b.value, c.value, nodes[0] is nodes[2])
    keep = Cell(1, 0.5)
    xs = [keep, Cell(2, 1.5), keep]
    out = cells(xs, 10)
    print(out is xs, xs, xs[1] is not keep)
    print(a.push(3), a.seen, Node.twice(4), b.push(1))
    print(ppy.native.compiled(walk), ppy.native.compiled(Node.push), ppy.native.compiled(a.push))


main()
"""

EFFECTS = """
import sys

import ppy


def report(xs: list[int]) -> None:
    for x in xs:
        print("item", x)
    xs.append(len(xs))


def count_up(n: int) -> None:
    total = 0
    for i in range(n):
        total += i
        if i % 4 == 0:
            print("at", i, file=sys.stderr)
    print("total", total)


def checked(n: int) -> int:
    total = 0
    for i in range(n):
        print("step", i)
        total += i
    if total > 5:
        raise ValueError(f"too big: {total}")
    return total


def flushed(n: int) -> int:
    total = 0
    for i in range(n):
        print("flushed", i, flush=True)
        total += i
    if n > 3:
        raise KeyError(n)
    return total


@ppy.native
def look(d: dict[int, list[int]], k: int) -> int:
    s = 0
    for key in d:
        for v in d[key]:
            s += v * key
    if s > k:
        raise ValueError("over")
    return s


def main() -> None:
    xs = [1, 2]
    print(report(xs), xs)
    count_up(9)
    print(checked(3))
    try:
        checked(5)
    except ValueError as error:
        print("caught", error)
    print(flushed(2))
    try:
        flushed(4)
    except KeyError as error:
        print("caught key", error)
    d = {1: [1, 2], 2: [3, 4]}
    for k in (100, 5, 100, 1):
        try:
            print(look(d, k))
        except ValueError as error:
            print("over", k, error)
    print(ppy.native.compiled(report), ppy.native.compiled(count_up), ppy.native.compiled(look))


main()
"""


def _run(tmp_path: Path, *args: str) -> list[str]:
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    done = subprocess.run(
        [sys.executable, *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    assert done.returncode == 0, done.stderr
    lines = [line for line in done.stderr.splitlines() if not line.startswith("compiling")]
    return [*done.stdout.splitlines(), "--", *lines]


@pytest.mark.parametrize(
    ("source", "native"),
    [
        (ALIASES, "True True True"),
        (OBJECTS, "True True True"),
        (EFFECTS, "True True True"),
    ],
    ids=["aliases", "objects", "effects"],
)
def test_writes_reach_the_callers_objects(tmp_path: Path, source: str, native: str):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "prog.ppy").write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    python = _run(tmp_path, "prog.ppy")
    ran = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    split = python.index("--")
    assert ran[: split - 1] == python[: split - 1]
    assert ran[split - 1] == native
    assert ran[split:] == python[split:]
