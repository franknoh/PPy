"""Project objects at the Python boundary under `ppy run`.

An instance of a project class crosses whole: the objects and collections its
fields hold go with it, each once, so a cycle and a shared object stay what
they were. Written fields come back to the caller's objects, and an object
native code hands back is the caller's own where it came from one, or a new
instance of its class, made without running `__init__` again.
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

LINKED = """
from dataclasses import dataclass


class Node:
    def __init__(self, value: int, label: str) -> None:
        self.value = value
        self.label = label
        self.next: Node | None = None


@dataclass
class Point:
    x: int
    y: int


def total(head: Node | None) -> int:
    s = 0
    node = head
    while node is not None:
        s += node.value
        node = node.next
    return s


def bump(head: Node) -> None:
    node: Node | None = head
    while node is not None:
        node.value += 1
        node.label = node.label + "!"
        node = node.next


def build(n: int) -> Node:
    head = Node(0, "a")
    for i in range(1, n):
        fresh = Node(i, "b")
        fresh.next = head
        head = fresh
    return head


def shift(points: list[Point], dx: int) -> list[Point]:
    out: list[Point] = []
    for p in points:
        out.append(Point(p.x + dx, p.y))
    return out


def main() -> None:
    a = Node(1, "x")
    b = Node(2, "y")
    a.next = b
    print(total(a), total(None))
    bump(a)
    print(a.value, a.label, b.value, b.label, a.next is b)
    h = build(5)
    print(total(h), h.value, h.label, type(h).__name__)
    print(shift([Point(1, 2), Point(3, 4)], 10))
    print(hasattr(total, "__ppy_native__"), hasattr(bump, "__ppy_native__"), hasattr(build, "__ppy_native__"), hasattr(shift, "__ppy_native__"))


main()
"""

EDGES = """
from dataclasses import dataclass


class Node:
    def __init__(self, value: int) -> None:
        self.value = value
        self.next: Node | None = None
        self.tags: list[int] = []


class Special(Node):
    def __init__(self, value: int, extra: float) -> None:
        super().__init__(value)
        self.extra = extra


@dataclass(frozen=True)
class Frozen:
    name: str
    size: int


class Holder:
    def __init__(self, item: Frozen, pair: tuple[int, float]) -> None:
        self.item = item
        self.pair = pair
        self.flag = True


def walk(head: Node) -> int:
    s = 0
    node: Node | None = head
    seen = 0
    while node is not None and seen < 10:
        s += node.value + len(node.tags)
        node.tags.append(seen)
        node = node.next
        seen += 1
    return s


def pick(nodes: list[Node], k: int) -> Node:
    best = nodes[0]
    for n in nodes:
        if n.value * k > best.value * k:
            best = n
    return best


def grow(h: Holder, times: int) -> Holder:
    out = h
    while out.pair[0] > 100:
        out = out
    for i in range(times):
        out = Holder(Frozen(out.item.name + "x", out.item.size + i), (out.pair[0] + 1, out.pair[1] * 2.0))
        out.flag = not h.flag
    return out


def main() -> None:
    a = Node(1)
    b = Special(2, 0.5)
    a.next = b
    b.next = a  # a cycle
    print(walk(a), a.tags, b.tags, a.next is b, b.next is a, type(b).__name__, b.extra)
    xs = [Node(3), a, Node(-1)]
    best = pick(xs, 1)
    ys: list[Node] = [b, a]
    print(best is xs[0], pick(xs, -1) is xs[2], pick(ys, 1) is b)
    h = Holder(Frozen("n", 1), (1, 1.5))
    g = grow(h, 3)
    print(g.item, g.pair, g.flag, h.item, grow(h, 0) is h)
    print([hasattr(f, "__ppy_native__") for f in (walk, pick, grow)])


main()
"""


def _run(tmp_path: Path, *args: str) -> list[str]:
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    done = subprocess.run(
        [sys.executable, *args], cwd=tmp_path, capture_output=True, text=True, check=False, env=env
    )
    assert done.returncode == 0, done.stderr
    return [line for line in done.stdout.splitlines() if not line.startswith("compiling")]


@pytest.mark.parametrize(
    ("source", "native"),
    [(LINKED, "True True True True"), (EDGES, "[True, True, False]")],
    ids=["linked", "edges"],
)
def test_objects_cross_the_boundary(tmp_path: Path, source: str, native: str):
    """The answers are CPython's, and the functions that take or return objects
    answer natively (the last line: CPython says False of each)."""
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "prog.ppy").write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    python = _run(tmp_path, "prog.ppy")
    ran = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert ran[:-1] == python[:-1]
    assert ran[-1] == native
