"""Methods that write objects, called from Python under `ppy run`.

A field the class never annotates takes its type from everything the program
stores into it: `self.left = None` in `__init__` and `node.left = Node(k)`
elsewhere make `left` a `Node | None`, and `self.queue = []` with
`self.queue.append(item)` a `list` of what is appended. A method that edits
the structure in place then runs natively: the objects cross whole, and its
writes, new objects linked in included, come back to the caller's objects,
which stay the same objects. A write through a local that aliases a field
(`node = self.head`), or through an object a call handed back, comes back
as one through the parameter does.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.analysis import types as T
from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler
from ppy_compiler.driver.pipeline import analyze_paths, open_project

requires_native = pytest.mark.skipif(
    not llvm_available() or c_compiler() is None, reason="needs llvmlite and a C compiler"
)

TREES = """
import ppy


class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None


class Tree:
    def __init__(self):
        self.root = None
        self.count = 0

    @ppy.native
    def insert(self, key: int) -> None:
        self.count += 1
        new = Node(key)
        if self.root is None:
            self.root = new
            return
        node = self.root
        while True:
            if key < node.key:
                if node.left is None:
                    node.left = new
                    new.parent = node
                    return
                node = node.left
            else:
                if node.right is None:
                    node.right = new
                    new.parent = node
                    return
                node = node.right

    def find(self, key: int):
        node = self.root
        while node is not None and node.key != key:
            node = node.left if key < node.key else node.right
        return node

    @ppy.native
    def rotate_left(self, key: int) -> bool:
        x = self.find(key)
        if x is None or x.right is None:
            return False
        y = x.right
        x.right = y.left
        if y.left is not None:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        return True


class Ring:
    def __init__(self):
        self.head = None
        self.tail = None

    @ppy.native
    def push(self, key: int) -> None:
        n = Node(key)
        n.left = self.tail
        if self.tail is not None:
            self.tail.right = n
        else:
            self.head = n
        self.tail = n

    @ppy.native
    def reverse(self) -> None:
        node = self.head
        self.head, self.tail = self.tail, self.head
        while node is not None:
            node.left, node.right = node.right, node.left
            node = node.left


def shape(n):
    if n is None:
        return "."
    return "(" + shape(n.left) + str(n.key) + shape(n.right) + ")"


def linked(t):
    ok = True
    pending = [t.root]
    while pending:
        n = pending.pop()
        if n is None:
            continue
        for c in (n.left, n.right):
            if c is not None:
                ok = ok and c.parent is n
                pending.append(c)
    return ok


def main():
    t = Tree()
    for k in [50, 30, 70, 20, 40, 60, 80, 35]:
        t.insert(k)
    root = t.root
    kept = t.find(35)
    print(t.rotate_left(50), shape(t.root), t.count, linked(t))
    print(t.root.left is root, root.parent is t.root, t.find(35) is kept)
    print(t.rotate_left(30), t.rotate_left(20), shape(t.root), linked(t))
    r = Ring()
    for k in range(5):
        r.push(k)
    first = r.head
    r.reverse()
    keys = []
    n = r.head
    while n is not None:
        keys.append(n.key)
        n = n.right
    print(keys, r.tail is first, r.head.left is None, r.tail.right is None)
    print([ppy.native.compiled(f) for f in (t.insert, t.rotate_left, r.push, r.reverse)])


main()
"""

ALIASES = """
import ppy


class Node:
    def __init__(self, value: int) -> None:
        self.value = value
        self.next: Node | None = None


class Chain:
    def __init__(self) -> None:
        self.head: Node | None = None

    def last(self) -> Node | None:
        node = self.head
        if node is None:
            return None
        while node.next is not None:
            node = node.next
        return node

    @ppy.native
    def bump_all(self, k: int) -> None:
        node = self.head
        while node is not None:
            node.value += k
            node = node.next

    @ppy.native
    def bump_last(self, k: int) -> None:
        node = self.last()
        if node is not None:
            node.value += k


def tail(n: Node) -> Node:
    while n.next is not None:
        n = n.next
    return n


@ppy.native
def bump_tail(c: Chain, k: int) -> None:
    head = c.head
    if head is not None:
        tail(head).value += k


@ppy.native
def extend(c: Chain, k: int) -> None:
    head = c.head
    if head is not None:
        tail(head).next = Node(k)


def values(c: Chain) -> list[int]:
    out: list[int] = []
    node = c.head
    while node is not None:
        out.append(node.value)
        node = node.next
    return out


def main() -> None:
    c = Chain()
    for i in range(5):
        n = Node(i)
        n.next = c.head
        c.head = n
    nodes = []
    node = c.head
    while node is not None:
        nodes.append(node)
        node = node.next
    c.bump_all(10)
    print(values(c))
    c.bump_last(100)
    bump_tail(c, 1000)
    extend(c, 7)
    print(values(c), all(a is b for a, b in zip(nodes, [c.head, c.head.next])))
    fns = (c.bump_all, c.bump_last, bump_tail, extend)
    print([ppy.native.compiled(f) for f in fns])


main()
"""


def _project(tmp_path: Path, source: str, strict: bool) -> Path:
    flag = "true" if strict else "false"
    (tmp_path / "pyproject.toml").write_text(f"[tool.ppy]\nstrict = {flag}\n", encoding="utf-8")
    program = tmp_path / "prog.ppy"
    program.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return program


def _run(tmp_path: Path, *args: str) -> list[str]:
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    done = subprocess.run(
        [sys.executable, *args], cwd=tmp_path, capture_output=True, text=True, check=False, env=env
    )
    assert done.returncode == 0, done.stderr
    return [line for line in done.stdout.splitlines() if not line.startswith("compiling")]


@requires_native
@pytest.mark.parametrize(
    ("source", "strict"), [(TREES, False), (ALIASES, True)], ids=["trees", "aliases"]
)
def test_structure_edits_come_back_as_cpython_makes_them(tmp_path: Path, source: str, strict: bool):
    """Every line is CPython's, identities included, and the methods that
    relink the structure ran natively (the last line: CPython says False)."""
    _project(tmp_path, source, strict)
    python = _run(tmp_path, "prog.ppy")
    ran = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert ran[:-1] == python[:-1]
    assert python[-1] == "[False, False, False, False]"
    assert ran[-1] == "[True, True, True, True]"


def _fields(tmp_path: Path, source: str) -> dict[str, dict[str, str]]:
    program = _project(tmp_path, source, strict=False)
    project = open_project(program, config_overrides={"strict": False})
    bundle = analyze_paths(project, [program], backend="llvm")
    return {
        info.name: {name: str(T.strip_literal(t)) for name, t in info.fields.items()}
        for info in bundle.symbols.classes.values()
    }


def test_unannotated_fields_take_what_the_program_stores(tmp_path: Path):
    fields = _fields(
        tmp_path,
        """
        import heapq


        class Node:
            def __init__(self, key: int) -> None:
                self.key = key
                self.next = None
                self.up = self.down = self


        class Special(Node):
            pass


        class Graph:
            def __init__(self, n: int) -> None:
                self.adj = [[] for _ in range(n)]
                self.weights = {}
                self.heap = []
                self.seen = set()
                self.head = None

            def add(self, u: int, v: int, w: float) -> None:
                self.adj[u].append(v)
                self.weights[(u, v)] = w
                heapq.heappush(self.heap, (w, u))
                self.seen.add(u)


        def link(g: Graph, a: Node, k: int) -> None:
            a.next = Node(k)
            g.head = Special(k)
            g.head.next = a
        """,
    )
    assert set(fields["Node"]["next"].split(" | ")) == {"prog.Node", "NoneType"}
    assert fields["Node"]["up"] == "prog.Node"
    assert set(fields["Graph"]["head"].split(" | ")) == {"NoneType", "prog.Special"}
    assert fields["Graph"]["adj"] == "list[list[int]]"
    assert fields["Graph"]["weights"] == "dict[tuple[int, int], float]"
    assert fields["Graph"]["heap"] == "list[tuple[float, int]]"
    assert fields["Graph"]["seen"] == "set[int]"


def test_a_field_set_from_a_subclass_is_the_base(tmp_path: Path):
    fields = _fields(
        tmp_path,
        """
        class Shape:
            def __init__(self) -> None:
                self.next = None


        class Square(Shape):
            pass


        def chain(a: Shape, b: Square, c: Shape) -> None:
            a.next = b
            b.next = c
        """,
    )
    assert set(fields["Shape"]["next"].split(" | ")) == {"prog.Shape", "NoneType"}


def test_an_annotated_field_keeps_its_annotation(tmp_path: Path):
    fields = _fields(
        tmp_path,
        """
        class Box:
            def __init__(self) -> None:
                self.value: float = 0.0
                self.items: list[int] = []


        def fill(b: Box) -> None:
            b.value = 3
            b.items.append(1)
        """,
    )
    assert fields["Box"] == {"value": "float", "items": "list[int]"}


def test_a_field_stored_only_from_unknown_values_stays_unknown(tmp_path: Path):
    """A value the checker could not type is no evidence: the field keeps
    what the class itself says, and a function that needs more stays in
    Python."""
    fields = _fields(
        tmp_path,
        """
        class Cell:
            def __init__(self) -> None:
                self.parent = None
                self.queue = []


        def attach(c: Cell, other) -> None:
            c.parent = other
            c.queue.append(other)
        """,
    )
    assert fields["Cell"] == {"parent": "NoneType", "queue": "list[Never]"}


PROTOCOL = """
from typing import Protocol

import ppy


class Comparable(Protocol):
    def __lt__(self, other: "Comparable") -> bool: ...


def smallest[T: Comparable](xs: list[T]) -> T:
    best = xs[0]
    for x in xs:
        if x < best:
            best = x
    return best
"""

POINTS = """
import ppy


class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __lt__(self, other: "Point") -> bool:
        return (self.x, self.y) < (other.x, other.y)

    @ppy.native
    def shift(self, k: int) -> None:
        for _ in range(k):
            self.x += 1
            self.y -= 1


def main() -> None:
    p = Point(1, 2)
    p.shift(3)
    print(p.x, p.y, p < Point(5, 0), ppy.native.compiled(p.shift))


main()
"""


@requires_native
def test_a_protocol_a_class_only_satisfies_does_not_keep_it_in_python(tmp_path: Path):
    """The checker puts a project Protocol a class covers into its MRO, even
    one defined in another module; it gives the class nothing, so native code
    holds the class as it would without it."""
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "order.py").write_text(textwrap.dedent(PROTOCOL).lstrip("\n"), encoding="utf-8")
    source = "import order\n" + textwrap.dedent(POINTS).lstrip("\n")
    (tmp_path / "prog.ppy").write_text(source, encoding="utf-8")
    python = _run(tmp_path, "prog.ppy")
    ran = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert python == ["4 -1 True False"]
    assert ran == ["4 -1 True True"]


SWAPS = """
import ppy


class Node:
    def __init__(self, key: int) -> None:
        self.key: int = key
        self.prev: Node | None = None
        self.next: Node | None = None


class Ring:
    def __init__(self) -> None:
        self.head: Node | None = None
        self.tail: Node | None = None

    def push(self, key: int) -> None:
        n = Node(key)
        n.prev = self.tail
        if self.tail is not None:
            self.tail.next = n
        else:
            self.head = n
        self.tail = n

    def reverse(self) -> None:
        node = self.head
        self.head, self.tail = self.tail, self.head
        while node is not None:
            node.prev, node.next = node.next, node.prev
            node = node.prev

    def total(self) -> int:
        s = 0
        node = self.head
        weight = 1
        while node is not None:
            s += node.key * weight
            weight += 1
            node = node.next if weight % 7 != 0 else node.next
        return s


def main() -> None:
    n: int = ppy.scan[int]()
    r = Ring()
    for i in range(n):
        r.push(i * 3 - 7)
    r.reverse()
    a = r.total()
    r.reverse()
    print(a, r.total())


main()
"""


def _sanitizes(compiler: str, directory: Path) -> bool:
    probe = directory / "probe.c"
    probe.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    done = subprocess.run(
        [compiler, "-fsanitize=address", "-x", "c", str(probe), "-o", str(directory / "probe")],
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


@pytest.mark.parametrize("unsafe", [False, True])
def test_swapped_links_are_freed_once(tmp_path: Path, unsafe: bool):
    """Fields swapped by tuple assignment, and objects chosen by a conditional
    expression, hold their references right: the emitted C runs clean under
    AddressSanitizer with leak detection, cycles included."""
    compiler = c_compiler()
    if not llvm_available() or compiler is None or not _sanitizes(compiler, tmp_path):
        pytest.skip("no C compiler with AddressSanitizer")
    _project(tmp_path, SWAPS, strict=True)
    flags = ["--unsafe"] if unsafe else []
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "ppy_compiler",
            "emit",
            "c",
            "--standalone",
            *flags,
            "prog.ppy",
            "-o",
            "prog.c",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert done.returncode == 0, done.stderr
    binary = tmp_path / "prog"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-g",
            "-O1",
            "-fsanitize=address,undefined",
            str(tmp_path / "prog.c"),
            "-lm",
            "-o",
            str(binary),
        ],
        check=True,
    )
    ran = subprocess.run(
        [str(binary)],
        input="9\n",
        capture_output=True,
        text=True,
        check=False,
        env={"ASAN_OPTIONS": "detect_leaks=1"},
    )
    assert ran.returncode == 0, ran.stderr
    expected = subprocess.run(
        [sys.executable, "prog.ppy"],
        cwd=tmp_path,
        input="9\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.stdout.strip() == expected.stdout.strip()
