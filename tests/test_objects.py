"""Classes in native code: objects held by handle, and generic classes.

A class whose methods write its instances, or whose fields hold other objects
or collections, is an object class: native code holds an instance by handle,
counts its references, and frees it with what it holds. `Tree | None` is a
handle that may be null. A generic class (`class Stack[T]`) is instantiated
per type argument, its methods with it. Each program agrees on every path, goes
native, and its emitted C and C++ run clean under AddressSanitizer with leak
detection.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler, standalone_toolchain_status

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
_standalone, _standalone_detail = standalone_toolchain_status()
requires_standalone = pytest.mark.skipif(
    not (llvm_available() and _standalone), reason=f"no standalone toolchain: {_standalone_detail}"
)
_CXX = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)

PROGRAM = """
from dataclasses import dataclass

import ppy
from ppy import HashMap, Vec


@dataclass
class Tree:
    key: int
    left: "Tree | None" = None
    right: "Tree | None" = None


def insert(root: Tree | None, key: int) -> Tree:
    if root is None:
        return Tree(key)
    if key < root.key:
        root.left = insert(root.left, key)
    elif key > root.key:
        root.right = insert(root.right, key)
    return root


def smallest(root: Tree) -> int:
    node = root
    while node.left is not None:
        node = node.left
    return node.key


def remove(root: Tree | None, key: int) -> Tree | None:
    if root is None:
        return None
    if key < root.key:
        root.left = remove(root.left, key)
        return root
    if key > root.key:
        root.right = remove(root.right, key)
        return root
    if root.left is None:
        return root.right
    if root.right is None:
        return root.left
    successor: int = smallest(root.right)
    root.key = successor
    root.right = remove(root.right, successor)
    return root


def walk(root: Tree | None, out: Vec[int]) -> None:
    if root is None:
        return
    walk(root.left, out)
    out.push(root.key)
    walk(root.right, out)


def height(root: Tree | None) -> int:
    if root is None:
        return 0
    return 1 + max(height(root.left), height(root.right))


@dataclass
class Cell:
    value: int
    next: "Cell | None" = None


def reverse(head: Cell | None) -> Cell | None:
    previous: Cell | None = None
    current = head
    while current is not None:
        following = current.next
        current.next = previous
        previous = current
        current = following
    return previous


class Stack[T]:
    def __init__(self) -> None:
        self.items: Vec[T] = Vec[T]()
        self.pushed: int = 0

    def push(self, value: T) -> None:
        self.items.push(value)
        self.pushed += 1

    def pop(self) -> T:
        return self.items.pop()

    def __len__(self) -> int:
        return len(self.items)


class Pair[A, B]:
    def __init__(self, first: A, second: B) -> None:
        self.first: A = first
        self.second: B = second

    def swapped(self) -> "Pair[B, A]":
        return Pair[B, A](self.second, self.first)


def trees(n: int) -> int:
    root: Tree | None = None
    for i in range(n):
        root = insert(root, (i * 37) % 101)
    for i in range(0, n, 3):
        root = remove(root, (i * 37) % 101)
    keys = Vec[int]()
    walk(root, keys)
    acc: int = 0
    for key in keys:
        acc = (acc * 31 + key) % 1000003
    return acc * 100 + height(root)


def cells(n: int) -> int:
    head: Cell | None = None
    for i in range(n):
        head = Cell(i, head)
    head = reverse(head)
    acc: int = 0
    node = head
    while node is not None:
        acc = acc * 7 + node.value
        acc %= 1000003
        node = node.next
    return acc


def stacks(n: int) -> float:
    ints = Stack[int]()
    floats = Stack[float]()
    for i in range(n):
        ints.push(i)
        floats.push(i / 4)
    total: float = 0.0
    while len(ints) > 2:
        total += ints.pop() + floats.pop()
    pair = Pair[int, float](3, 0.5)
    other = pair.swapped()
    return total + ints.pushed + other.first + other.second


def registry(n: int) -> int:
    groups = HashMap[int, Vec[Tree]]()
    same: int = 0
    for i in range(n):
        key: int = i % 5
        if key not in groups:
            groups[key] = Vec[Tree]()
        groups[key].push(Tree(i))
    first = groups[0][0]
    alias = first
    if alias is first:
        same = 1
    alias.key = 999
    return groups[0][0].key + len(groups) + same


def main() -> None:
    n: int = ppy.scan[int]()
    print(trees(n), cells(n), stacks(n), registry(n))


main()
"""

EXPECTED = "36746708 600502 2274.75 1005"
NATIVE = [
    "insert",
    "smallest",
    "remove",
    "walk",
    "height",
    "reverse",
    "trees",
    "cells",
    "stacks",
    "registry",
]


def _write(tmp_path: Path, source: str) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    program = tmp_path / "prog.ppy"
    program.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return program


def _run(tmp_path: Path, *args: str, text: str = "") -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    return subprocess.run(
        [sys.executable, *args],
        cwd=tmp_path,
        input=text,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@requires_llvm
def test_objects_agree_on_every_path_and_go_native(tmp_path: Path):
    _write(tmp_path, PROGRAM)
    for args in (
        ["prog.ppy"],
        ["-m", "ppy_compiler", "prog.ppy"],
        ["-m", "ppy_compiler", "run", "prog.ppy"],
    ):
        done = _run(tmp_path, *args, text="60\n")
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == EXPECTED, args
    for function in NATIVE:
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


@requires_standalone
def test_a_standalone_binary_holds_objects(tmp_path: Path):
    program = _write(tmp_path, PROGRAM)
    built = _run(
        tmp_path, "-m", "ppy_compiler", "build", "--standalone", program.name, "-o", "dist"
    )
    assert built.returncode == 0, built.stderr
    ran = subprocess.run(
        [str(tmp_path / "dist" / "prog")], input="60\n", capture_output=True, text=True, check=False
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == EXPECTED


def _sanitizes(compiler: str, directory: Path) -> bool:
    probe = directory / "probe.c"
    probe.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    done = subprocess.run(
        [compiler, "-fsanitize=address", "-x", "c", str(probe), "-o", str(directory / "probe")],
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


@requires_llvm
@pytest.mark.parametrize("unsafe", [False, True])
@pytest.mark.parametrize("language", ["c", "cpp"])
def test_emitted_objects_are_freed_once(tmp_path: Path, language: str, unsafe: bool):
    compiler = c_compiler() if language == "c" else _CXX
    if compiler is None or not _sanitizes(compiler, tmp_path):
        pytest.skip(f"no {language} compiler with AddressSanitizer")
    program = _write(tmp_path, PROGRAM)
    emitted = tmp_path / f"prog.{language}"
    flags = ["--unsafe"] if unsafe else []
    done = _run(
        tmp_path,
        "-m",
        "ppy_compiler",
        "emit",
        language,
        "--standalone",
        *flags,
        program.name,
        "-o",
        emitted.name,
    )
    assert done.returncode == 0, done.stderr
    standard = "-std=c11" if language == "c" else "-std=c++17"
    binary = tmp_path / "prog"
    subprocess.run(
        [
            compiler,
            standard,
            "-g",
            "-O1",
            "-fsanitize=address,undefined",
            str(emitted),
            "-lm",
            "-o",
            str(binary),
        ],
        check=True,
    )
    ran = subprocess.run(
        [str(binary)],
        input="60\n",
        capture_output=True,
        text=True,
        check=False,
        env={"ASAN_OPTIONS": "detect_leaks=1"},
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == EXPECTED


def test_the_checker_types_generic_classes(tmp_path: Path):
    _write(
        tmp_path,
        """
        from ppy import Vec


        class Box[T]:
            def __init__(self, first: T) -> None:
                self.items: Vec[T] = Vec[T]()
                self.items.push(first)

            def add(self, value: T) -> None:
                self.items.push(value)

            def first(self) -> T:
                return self.items[0]


        def fine() -> float:
            box = Box[float](1.5)
            box.add(2)
            return box.first()


        def bad() -> str:
            box = Box[int](1)
            box.add(2.5)
            text: str = box.first()
            return text


        def wrong_arity() -> int:
            box = Box[int, float](1)
            return 0
        """,
    )
    checked = _run(tmp_path, "-m", "ppy_compiler", "check", "prog.ppy")
    shown = (checked.stdout + checked.stderr).splitlines()
    errors = [line for line in shown if line.startswith("error[")]
    assert errors[:3] == [
        "error[E1301]: `add` parameter `value` expects `int`, got `Literal[2.5]`",
        "error[E1301]: cannot assign `int` to a variable declared `str`",
        "error[E1301]: `Box` takes 1 type argument(s), not 2",
    ]
