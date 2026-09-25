"""The collections runtime's memory, and what a guard says with no Python under it.

Every handle is on its thread's list, so a collector finds the cycles that
reference counts leave (a doubly linked list, a tree whose nodes point at
their parents) and frees them, and a native call that fails and falls back
under `ppy run` leaves nothing behind. A standalone binary, and emitted C
and C++, print the line CPython's traceback would end with where a guard
fails, and exit with status 1 as CPython does.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler, standalone_toolchain_status
from ppy_runtime.aio import compiler

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(compiler() is None, reason="no C compiler to build the runtime")
_standalone, _standalone_detail = standalone_toolchain_status()
requires_standalone = pytest.mark.skipif(
    not (llvm_available() and _standalone), reason=f"no standalone toolchain: {_standalone_detail}"
)
_CXX = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)

CYCLES = """
import gc
from dataclasses import dataclass

import ppy


@dataclass
class Node:
    value: int
    prev: "Node | None" = None
    next: "Node | None" = None


@dataclass
class Tree:
    key: int
    parent: "Tree | None" = None
    left: "Tree | None" = None
    right: "Tree | None" = None


def chain(n: int) -> int:
    head = Node(0)
    tail = head
    for i in range(1, n):
        node = Node(i, tail)
        tail.next = node
        tail = node
    total: int = 0
    current = tail
    while current is not None:
        total += current.value
        current = current.prev
    return total


def insert(root: Tree, key: int) -> None:
    node = root
    placed: bool = False
    while not placed:
        if key < node.key:
            left = node.left
            if left is None:
                node.left = Tree(key, node)
                placed = True
            else:
                node = left
        else:
            right = node.right
            if right is None:
                node.right = Tree(key, node)
                placed = True
            else:
                node = right


def depth(n: int) -> int:
    root = Tree(n // 2)
    for i in range(n):
        insert(root, (i * 37) % n)
    node = root
    going: bool = True
    while going:
        left = node.left
        if left is None:
            going = False
        else:
            node = left
    steps: int = 0
    up = node.parent
    while up is not None:
        steps += 1
        up = up.parent
    return steps


def pairs(n: int) -> int:
    total: int = 0
    for i in range(n):
        a = Node(i)
        b = Node(i + 1, a)
        a.next = b
        total += a.value + b.value
    gc.collect()
    return total


def main() -> None:
    n: int = ppy.scan[int]()
    total: int = 0
    for _ in range(30):
        total += chain(n) + depth(n) + pairs(n)
    print(total)


main()
"""

CYCLES_EXPECTED = "1797030"
CYCLES_NATIVE = ["chain", "insert", "depth", "pairs"]


def _write(tmp_path: Path, source: str, strict: bool = True) -> Path:
    setting = "true" if strict else "false"
    (tmp_path / "pyproject.toml").write_text(f"[tool.ppy]\nstrict = {setting}\n", encoding="utf-8")
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


@requires_cc
def test_the_collector_frees_a_cycle_and_keeps_what_is_held():
    from ppy_runtime.collections import library_path

    path = library_path()
    assert path is not None
    lib = ctypes.CDLL(str(path))
    handle, word = ctypes.c_void_p, ctypes.c_int64
    for name, result, arguments in [
        ("ppy_seq_new", handle, [word, word, word, word]),
        ("ppy_seq_at", handle, [handle, word]),
        ("ppy_coll_release", None, [handle]),
        ("ppy_coll_retain", None, [handle]),
        ("ppy_coll_collect", word, []),
        ("ppy_coll_live_handles", word, []),
    ]:
        function = getattr(lib, name)
        function.restype = result
        function.argtypes = arguments

    def made() -> int:
        return lib.ppy_seq_new(1, 2, 0, 0b11)

    def link(holder: int, field: int, held: int) -> None:
        words = ctypes.cast(lib.ppy_seq_at(holder, 0), ctypes.POINTER(ctypes.c_int64))
        lib.ppy_coll_retain(held)
        words[field] = held

    before = lib.ppy_coll_live_handles()
    a, b = made(), made()
    link(a, 0, b)
    link(b, 0, a)
    kept, leaf = made(), lib.ppy_seq_new(3, 1, 0, 0)
    link(kept, 0, kept)
    link(kept, 1, leaf)
    lib.ppy_coll_release(leaf)
    lib.ppy_coll_release(a)
    lib.ppy_coll_release(b)
    assert lib.ppy_coll_live_handles() == before + 4
    # `a` and `b` hold only each other; `kept` is still named from here.
    assert lib.ppy_coll_collect() == 2
    assert lib.ppy_coll_live_handles() == before + 2
    lib.ppy_coll_release(kept)
    assert lib.ppy_coll_collect() == 1
    assert lib.ppy_coll_live_handles() == before


@requires_llvm
def test_cycles_agree_on_every_path_and_go_native(tmp_path: Path):
    _write(tmp_path, CYCLES)
    for args in (
        ["prog.ppy"],
        ["-m", "ppy_compiler", "prog.ppy"],
        ["-m", "ppy_compiler", "run", "prog.ppy"],
    ):
        done = _run(tmp_path, *args, text="120\n")
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == CYCLES_EXPECTED, args
    for function in CYCLES_NATIVE:
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


def _sanitizes(compiler_path: str, directory: Path) -> bool:
    probe = directory / "probe.c"
    probe.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    done = subprocess.run(
        [
            compiler_path,
            "-fsanitize=address",
            "-x",
            "c",
            str(probe),
            "-o",
            str(directory / "probe"),
        ],
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


@requires_llvm
@pytest.mark.parametrize("language", ["c", "cpp"])
def test_emitted_cycles_are_collected(tmp_path: Path, language: str):
    compiler_path = c_compiler() if language == "c" else _CXX
    if compiler_path is None or not _sanitizes(compiler_path, tmp_path):
        pytest.skip(f"no {language} compiler with AddressSanitizer")
    program = _write(tmp_path, CYCLES)
    emitted = tmp_path / f"prog.{language}"
    done = _run(
        tmp_path, "-m", "ppy_compiler", "emit", language, "--standalone", program.name,
        "-o", emitted.name,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    standard = "-std=c11" if language == "c" else "-std=c++17"
    binary = tmp_path / "prog"
    sanitize = "-fsanitize=address,undefined"
    subprocess.run(
        [compiler_path, standard, "-g", "-O1", sanitize, str(emitted), "-lm", "-o", str(binary)],
        check=True,
    )
    ran = subprocess.run(
        [str(binary)],
        input="120\n",
        capture_output=True,
        text=True,
        check=False,
        env={"ASAN_OPTIONS": "detect_leaks=1"},
    )
    # A parent pointer or a `prev` link is a cycle: leak detection passes
    # only because the collector frees them.
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == CYCLES_EXPECTED


FALLBACK = """
from ppy import Vec


def rows(n: int, pick: int) -> int:
    grid = Vec[Vec[int]]()
    for i in range(n):
        row = Vec[int](3)
        row[0] = i
        grid.push(row)
    return grid[pick][0]


def main() -> None:
    print(rows(10, 4))
    try:
        print(rows(10, 12))
    except IndexError as error:
        print("IndexError", error)
    print(rows(5, 1))


main()
"""

_LIVE = """
import ctypes
from ppy_compiler.driver.cli import main

code = main(["run", "prog.ppy"])
found = []
with open("/proc/self/maps") as maps:
    paths = sorted({line.split()[-1] for line in maps if line.rstrip().endswith(".so")})
for path in paths:
    try:
        counter = ctypes.CDLL(path).ppy_coll_live_handles
    except (OSError, AttributeError):
        continue
    counter.restype = ctypes.c_int64
    found.append(counter())
print("exit", code, "live", found)
"""


@requires_llvm
@pytest.mark.skipif(not Path("/proc/self/maps").exists(), reason="reads the loaded libraries")
def test_a_failed_call_leaves_no_handles_behind(tmp_path: Path):
    _write(tmp_path, FALLBACK)
    (tmp_path / "live.py").write_text(_LIVE, encoding="utf-8")
    explained = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.rows")
    assert "llvm backend: native" in explained.stdout, explained.stdout
    done = _run(tmp_path, "live.py")
    assert done.returncode == 0, done.stderr
    lines = done.stdout.strip().splitlines()
    assert lines[:3] == ["4", "IndexError index 12 is out of range for length 10", "1"]
    # The second call made eleven handles and fell back; the third freed them.
    assert lines[-1].startswith("exit 0 live [") and lines[-1] != "exit 0 live []"
    assert set(lines[-1].partition("[")[2].rstrip("]").split(", ")) == {"0"}, lines[-1]


FAILURES = """
from dataclasses import dataclass

import ppy
from ppy import Deque, HashMap, Heap, LinkedList, TreeSet, Vec


@dataclass
class Node:
    value: int
    next: "Node | None" = None


def fail(case: int, n: int) -> int:
    v = Vec[int]()
    v.push(1)
    if case == 0:
        return v[n]
    if case == 1:
        m = HashMap[tuple[int, int], int]()
        m[(1, 2)] = 3
        return m[(n, -n)]
    if case == 2:
        d = Deque[int]()
        return d.pop_front()
    if case == 3:
        node = Node(n)
        return node.next.value
    if case == 4:
        return n // (n - n)
    if case == 5:
        return n % (n - n)
    if case == 6:
        return int(n / (n - n))
    if case == 7:
        return int(1.5 / float(n - n))
    if case == 8:
        return n << -n
    if case == 9:
        h = Heap[int]()
        return h.peek()
    if case == 10:
        t = TreeSet[int]()
        t.add(1)
        return t.floor(-n)
    if case == 11:
        chain = LinkedList[int]()
        chain.push_back(1)
        return chain.value(n)
    if case == 12:
        m = HashMap[int, int]()
        m[1] = 1
        for k in m:
            m[k + n] = 1
        return 0
    if case == 13:
        w = Vec[int](-n)
        return len(w)
    if case == 14:
        t = TreeSet[int]()
        return t.min()
    if case == 15:
        b = ppy.buffer[int](3)
        return b[n]
    if case == 16:
        b = ppy.buffer[int](3)
        b[n] = 1
        return b[0]
    if case == 17:
        b = ppy.buffer[int](3)
        b[2] = 7
        return b[-1]
    if case == 18:
        m = HashMap[int, int]()
        return m.pop(n)
    return 0


def main() -> None:
    case: int = ppy.scan[int]()
    n: int = ppy.scan[int]()
    print(fail(case, n))


main()
"""

CASES = 19


def _agree(tmp_path: Path, program: Path, binary: Path) -> None:
    for case in range(CASES):
        text = f"{case} 5\n"
        python = _run(tmp_path, program.name, text=text)
        native = subprocess.run(
            [str(binary)], input=text, capture_output=True, text=True, check=False
        )
        assert native.returncode == python.returncode, (case, native.stderr)
        assert native.stdout == python.stdout, case
        said = python.stderr.strip().splitlines()[-1:] if python.stderr.strip() else []
        assert native.stderr.strip().splitlines() == said, (case, native.stderr)


@requires_standalone
def test_a_standalone_binary_says_what_cpython_raises(tmp_path: Path):
    program = _write(tmp_path, FAILURES, strict=False)
    built = _run(
        tmp_path, "-m", "ppy_compiler", "build", "--standalone", program.name, "-o", "dist"
    )
    assert built.returncode == 0, built.stderr
    _agree(tmp_path, program, tmp_path / "dist" / "prog")


@requires_llvm
@pytest.mark.parametrize("unsafe", [False, True])
@pytest.mark.parametrize("language", ["c", "cpp"])
def test_emitted_source_says_what_cpython_raises(tmp_path: Path, language: str, unsafe: bool):
    compiler_path = c_compiler() if language == "c" else _CXX
    if compiler_path is None:
        pytest.skip(f"no {language} compiler on PATH")
    program = _write(tmp_path, FAILURES, strict=False)
    source = tmp_path / f"prog.{language}"
    flags = ["--unsafe"] if unsafe else []
    done = _run(
        tmp_path, "-m", "ppy_compiler", "emit", language, "--standalone", *flags, program.name,
        "-o", source.name,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    standard = "-std=c11" if language == "c" else "-std=c++17"
    binary = tmp_path / "prog"
    subprocess.run(
        [compiler_path, standard, "-O1", str(source), "-lm", "-o", str(binary)], check=True
    )
    if unsafe:
        # Wrapping arithmetic has no overflow to report; the rest still agree.
        for case in (0, 1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18):
            text = f"{case} 5\n"
            python = _run(tmp_path, program.name, text=text)
            native = subprocess.run(
                [str(binary)], input=text, capture_output=True, text=True, check=False
            )
            assert native.returncode == python.returncode, (case, native.stderr)
            said = python.stderr.strip().splitlines()[-1:] if python.stderr.strip() else []
            assert native.stderr.strip().splitlines() == said, (case, native.stderr)
        return
    _agree(tmp_path, program, binary)
