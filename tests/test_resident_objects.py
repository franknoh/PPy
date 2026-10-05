"""Resident objects at the Python boundary under `ppy run` (`crossing.c`).

An object of a project class that crosses again and again keeps its native
record between calls: the second crossing admits it to the process's world,
and later calls hand native code that record instead of copying the object
in and back. Python stays the truth: a write Python makes to a resident
object (an attribute set or deleted, `vars()`, `setattr`) marks it stale and
the next call reads it again; a field native code stores is set on the
Python object once the call answers; a class that gains a data descriptor
ends residency. Every program here prints what CPython prints, with the
world and without it (`PPY_RESIDENT=0`).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.analysis.symbols import _rewrites_identity
from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler

native = pytest.mark.skipif(
    not llvm_available() or c_compiler() is None, reason="llvmlite or a C compiler is missing"
)

STACK = '''
import copy
import gc
import pickle

import ppy


class Node:
    def __init__(self, value: int, label: str) -> None:
        self.value = value
        self.label = label
        self.next: Node | None = None


class Stack:
    def __init__(self) -> None:
        self.head: Node | None = None
        self.size = 0

    @ppy.native
    def push(self, value: int) -> None:
        node = Node(value, "n" + str(value))
        node.next = self.head
        self.head = node
        self.size += 1

    @ppy.native
    def find(self, value: int) -> bool:
        node = self.head
        while node is not None:
            if node.value == value:
                return True
            node = node.next
        return False

    @ppy.native
    def total(self) -> int:
        s = 0
        node = self.head
        while node is not None:
            s += node.value
            node = node.next
        return s

    @ppy.native
    def labels(self) -> str:
        out = ""
        node = self.head
        while node is not None:
            out += node.label
            node = node.next
        return out

    @ppy.native
    def bump(self, by: int) -> None:
        node = self.head
        while node is not None:
            node.value += by
            node = node.next

    @ppy.native
    def pop(self) -> int:
        node = self.head
        if node is None:
            raise IndexError("pop from an empty stack")
        self.head = node.next
        self.size -= 1
        return node.value

    @ppy.native
    def shrink(self, k: int) -> int:
        self.size -= 1
        node = self.head
        if node is not None:
            node.value += 1000
            node.label = node.label + "?"
        if k > 5:
            raise ValueError("too far")
        return self.size

    @ppy.native
    def reverse(self) -> None:
        prev: Node | None = None
        node = self.head
        while node is not None:
            after = node.next
            node.next = prev
            prev = node
            node = after
        self.head = prev

    @ppy.native
    def nth(self, k: int) -> Node | None:
        node = self.head
        i = 0
        while node is not None:
            if i == k:
                return node
            i += 1
            node = node.next
        return None


def show(s: Stack) -> str:
    out = []
    node = s.head
    while node is not None:
        out.append(f"{node.value}:{node.label}")
        node = node.next
    return " ".join(out) + f" (size {s.size})"


def main() -> None:
    s = Stack()
    for i in range(6):
        s.push(i)
    print(show(s), s.total(), s.find(3), s.labels())
    first = s.head
    assert first is not None
    first.value = 100
    print(s.total(), s.find(100), s.find(5))
    first.label = "zz"
    print(s.labels())
    extra = Node(7, "py")
    extra.next = first.next
    first.next = extra
    print(s.total(), show(s))
    s.bump(10)
    print(show(s), extra.value, first.value)
    third = s.nth(2)
    print(third is extra, s.nth(0) is first, s.nth(99))
    s.reverse()
    print(show(s), s.head is not first, s.nth(6) is first)
    print(s.pop(), s.pop(), show(s))
    vars(s.head)["value"] = -5
    setattr(s.head.next, "value", -6)
    print(s.total(), show(s))
    c = copy.copy(s)
    c.push(55)
    print(show(c), show(s), c.head.next is s.head)
    d = copy.deepcopy(s)
    d.bump(1000)
    print(show(d), show(s))
    p = pickle.loads(pickle.dumps(s))
    p.bump(1)
    print(show(p), show(s))
    victim = s.nth(1)
    del victim.value
    try:
        print(s.total())
    except AttributeError as e:
        print("AttributeError", e)
    victim.value = 1
    print(s.total())
    vars(victim)["value"] = "x"
    try:
        print(s.total())
    except TypeError as e:
        print("TypeError", e)
    victim.value = 2
    print(s.total())
    e = Stack()
    e.push(1)
    print(e.pop())
    try:
        e.pop()
    except IndexError as err:
        print("IndexError", err)
    print(show(e))
    for k in range(200):
        t = Stack()
        for i in range(k % 7):
            t.push(i)
        t.bump(1)
        if t.total() != sum(range(1, k % 7 + 1)):
            print("wrong", k)
    gc.collect()
    print(s.total(), show(s))
    a = Stack()
    b = Stack()
    for i in range(3):
        a.push(i)
    b.push(9)
    assert b.head is not None
    b.head.next = a.head
    b.size = 4
    a.bump(1)
    print(show(a), show(b), b.total())
    b.bump(100)
    print(show(a), show(b), a.total())
    print(s.total(), s.nth(0) is s.head)
    for k in (9, 1, 9):
        try:
            print(s.shrink(k), show(s))
        except ValueError as err:
            print("ValueError", err, show(s))
    print(s.total(), s.labels())
    ENDING


main()
'''

#: The class gains a property named like a field: residency ends, and the
#: property is what Python reads from then on.
PROPERTY = """with ppy.dynamic:
        setattr(Node, "value", property(lambda self: 42))
    try:
        print(s.total())
    except Exception as err:
        print(type(err).__name__)"""


def _program(tmp_path: Path, source: str) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    (tmp_path / "prog.ppy").write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")


def _run(tmp_path: Path, *args: str, **env: str) -> tuple[list[str], str]:
    environment = {k: v for k, v in os.environ.items() if k not in {"PPY_LOWERING", "PPY_RESIDENT"}}
    environment.update(env)
    done = subprocess.run(
        [sys.executable, *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert done.returncode == 0, done.stderr
    lines = [line for line in done.stdout.splitlines() if not line.startswith("compiling")]
    return lines, done.stderr


def _report(stderr: str) -> dict[str, int]:
    """What `PPY_RESIDENT_REPORT` printed at exit, by name."""
    for line in stderr.splitlines():
        if line.startswith("resident: "):
            words = line.removeprefix("resident: ").replace(",", "").split()
            return {
                "live": int(words[0]),
                "stale": int(words[2]),
                "entries": int(words[4]),
                "enabled": int(words[7]),
                "admitted": int(words[8]),
                "calls": int(words[10]),
            }
    raise AssertionError(f"no resident report in:\n{stderr}")


def _three_ways(tmp_path: Path, source: str) -> dict[str, int]:
    """The program under CPython, under `ppy run` with resident objects and
    without: one output. The world's report at exit."""
    _program(tmp_path, source)
    python, _ = _run(tmp_path, "prog.ppy")
    resident, stderr = _run(
        tmp_path, "-m", "ppy_compiler", "run", "prog.ppy", PPY_RESIDENT_REPORT="1"
    )
    copied, _ = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy", PPY_RESIDENT="0")
    assert resident == python
    assert copied == python
    return _report(stderr)


@native
@pytest.mark.parametrize("ending", ["pass", PROPERTY], ids=["kept", "property"])
def test_resident_objects_match_cpython(tmp_path: Path, ending: str):
    """Python's writes between native calls, native writes read by Python,
    identities, copies, pickles, a deleted attribute, a value of another
    type, a native raise, many short-lived objects, two roots sharing nodes:
    all as CPython does them. With the property, residency ends."""
    report = _three_ways(tmp_path, STACK.replace("ENDING", ending))
    assert report["admitted"] > 20
    assert report["calls"] > 20
    assert report["enabled"] == (ending == "pass")


DROPPED = """
import gc
import weakref

import ppy


class Node:
    def __init__(self, value: int) -> None:
        self.value = value
        self.next: Node | None = None


class Chain:
    def __init__(self) -> None:
        self.head: Node | None = None

    @ppy.native
    def add(self, value: int) -> None:
        node = Node(value)
        node.next = self.head
        self.head = node

    @ppy.native
    def total(self) -> int:
        s = 0
        node = self.head
        while node is not None:
            s += node.value
            node = node.next
        return s


def round_trip(n: int) -> None:
    c = Chain()
    for i in range(n):
        c.add(i)
    print(c.total(), c.total())
    head = weakref.ref(c.head)
    whole = weakref.ref(c)
    del c
    print(head() is None, whole() is None)


for n in (1, 5, 300):
    round_trip(n)
kept = Chain()
for i in range(100):
    kept.add(i)
    kept.total()
gc.collect()
print(kept.total())
"""


@native
def test_resident_objects_die_with_their_python_objects(tmp_path: Path):
    """The world refers to its objects weakly and never holds their dicts: an
    object Python lets go of dies when CPython's would."""
    report = _three_ways(tmp_path, DROPPED)
    assert report["enabled"] == 1
    # What is still alive at exit: `kept` and its hundred nodes.
    assert report["live"] == 101


def test_identity_rewrites_are_found():
    """`x.__class__ = C`, `x.__dict__ = d`, and `setattr` of a computed name
    keep a program's objects copied; a constant name does not."""

    def rewrites(source: str) -> bool:
        return _rewrites_identity(ast.parse(textwrap.dedent(source)))

    assert rewrites("x.__class__ = C")
    assert rewrites("x.__dict__ = {}")
    assert rewrites("del x.__dict__")
    assert rewrites("setattr(x, name, 1)")
    assert rewrites("setattr(x, '__class__', C)")
    assert rewrites("object.__setattr__(x, name, 1)")
    assert rewrites("self.__setattr__(name, 1)")
    assert rewrites("exec(code)")
    assert not rewrites("setattr(x, 'value', 1)")
    assert not rewrites("monkeypatch.setattr(module, name, 1)")
    assert not rewrites("x.value = 1\nvars(x)['value'] = 2")
