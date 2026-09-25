"""Objects a collection orders, hashes, and compares by their class's own methods.

A `Heap`, a sort, and a tree's keys order instances by `__lt__`; a hash map's
keys hash and compare by `__hash__` and `__eq__` (written with `other: object`,
as Python spells it), or by identity where the class defines neither. The
runtime calls the compiled methods back. Each program is held to the
reference classes under `ppy`, `ppy run`, a standalone binary, and emitted C
and C++ under AddressSanitizer, with `ppy explain` confirming every function
went native. A method that fails a guard while the runtime calls it hands
the call back to Python, which raises what CPython raises.
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
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")
_standalone, _standalone_detail = standalone_toolchain_status()
requires_standalone = pytest.mark.skipif(
    not (llvm_available() and _standalone), reason=f"no standalone toolchain: {_standalone_detail}"
)
_CXX = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)

ORDERED = """
from ppy import Heap, MaxHeap, TreeMap, TreeSet, Vec


class Task:
    def __init__(self, cost: int, name: int) -> None:
        self.cost: int = cost
        self.name: int = name

    def __lt__(self, other: "Task") -> bool:
        return self.cost < other.cost


def fold(total: int, value: int) -> int:
    return (total * 31 + value) % 1000003


def heaps(n: int) -> int:
    h = Heap[Task]()
    for i in range(n):
        h.push(Task((i * 7) % 11, i))
    total = 0
    total = fold(total, h.pushpop(Task(3, 999)).name)
    total = fold(total, h.replace(Task(0, 555)).name)
    while h:
        total = fold(total, h.pop().name)
    most = MaxHeap[Task]()
    for i in range(n):
        most.push(Task((i * 5) % 9, i))
    total = fold(total, most.peek().name)
    v = Vec[Task]()
    for i in range(n):
        v.push(Task((i * 3) % 8, i))
    built = Heap[Task](v)
    for task in built.to_sorted():
        total = fold(total, task.name)
    return total


def sorts(n: int) -> int:
    v = Vec[Task]()
    for i in range(n):
        v.push(Task((i * 5) % 7, i))
    v.sort()
    total = 0
    for task in v:
        total = fold(total, task.name)
    v.sort(reverse=True)
    total = fold(total, v[0].name)
    for task in sorted(v):
        total = fold(total, task.name)
    return total


def trees(n: int) -> int:
    t = TreeMap[Task, int]()
    for i in range(n):
        t[Task((i * 3) % 13, i)] = i
    total = len(t) * 1000 + t[Task(4, -1)]
    total = fold(total, t.floor(Task(6, 0)).name)
    total = fold(total, t.ceiling(Task(7, 0)).name)
    total = fold(total, t.min().name)
    for key in t.between(Task(2, 0), Task(5, 0)):
        total = fold(total, key.cost)
    total = fold(total, t.max().cost)
    s = TreeSet[Task]()
    for i in range(n):
        s.add(Task(i % 6, i))
    total = fold(total, len(s))
    total = fold(total, 1 if Task(3, 0) in s else 0)
    s.discard(Task(3, 0))
    total = fold(total, 1 if Task(3, 0) in s else 0)
    total = fold(total, s.pop_max().cost)
    return total


def main() -> None:
    print(heaps(40), sorts(40), trees(40))


main()
"""

HASHED = """
from ppy import HashMap, HashSet


class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x: int = x
        self.y: int = y

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return self.x * 31 + self.y


class Token:
    def __init__(self, n: int) -> None:
        self.n: int = n

    def bump(self) -> None:
        self.n += 1


def maps(n: int) -> int:
    m = HashMap[Point, int]()
    for i in range(n):
        p = Point(i % 5, i % 3)
        m[p] = m.get(p, 0) + i
    total = len(m) * 100000 + m[Point(2, 2)]
    total += m.pop(Point(0, 0))
    total += m.setdefault(Point(9, 9), 7)
    total += 1 if Point(1, 1) in m else 0
    total += 1 if Point(0, 0) in m else 0
    copied = m.copy()
    copied[Point(1, 1)] = -1
    total += m[Point(1, 1)] + copied[Point(1, 1)]
    for key in m:
        total += key.x * 10 + key.y
    return total


def sets(n: int) -> int:
    a = HashSet[Point]()
    b = HashSet[Point]()
    for i in range(n):
        a.add(Point(i % 4, 0))
        b.add(Point(i % 6, 0))
    union = a | b
    both = a & b
    only = b - a
    total = len(union) * 100 + len(both) * 10 + len(only)
    a.update(b)
    total += len(a)
    a.remove(Point(5, 0))
    a.discard(Point(42, 0))
    total += len(a)
    return total


def identity(n: int) -> int:
    tokens = HashMap[Token, int]()
    kept = Token(7)
    kept.bump()
    tokens[kept] = 1
    for i in range(n):
        tokens[Token(i)] = i
    total = len(tokens)
    total += tokens[kept]
    total += 1 if Token(7) in tokens else 0
    return total


def main() -> None:
    print(maps(30), sets(30), identity(30))


main()
"""

GUARDED = """
from ppy import Heap


class Share:
    def __init__(self, total: int, parts: int) -> None:
        self.total: int = total
        self.parts: int = parts

    def __lt__(self, other: "Share") -> bool:
        return self.total // self.parts < other.total // other.parts


def cheapest(n: int, zero_at: int) -> int:
    h = Heap[Share]()
    for i in range(n):
        h.push(Share(i * 10 + 3, 0 if i == zero_at else i % 4 + 1))
    return h.pop().total


def main() -> None:
    print(cheapest(20, -1))
    try:
        print(cheapest(20, 7))
    except ZeroDivisionError as error:
        print(type(error).__name__, error)


main()
"""

OVERRIDDEN = """
from ppy import Heap


class Base:
    def __init__(self, rank: int) -> None:
        self.rank: int = rank

    def __lt__(self, other: "Base") -> bool:
        return self.rank < other.rank


class Reversed(Base):
    def __lt__(self, other: "Base") -> bool:
        return self.rank > other.rank


def first(n: int) -> int:
    h = Heap[Base]()
    for i in range(n):
        h.push(Base(i % 5))
    return h.pop().rank


def main() -> None:
    print(first(10))


main()
"""

PROGRAMS = {
    "ordered": (ORDERED, "", ["heaps", "sorts", "trees"]),
    "hashed": (HASHED, "", ["maps", "sets", "identity"]),
}


def _write(tmp_path: Path, source: str) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    program = tmp_path / "prog.ppy"
    program.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return program


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    return subprocess.run(
        [sys.executable, *args], cwd=tmp_path, capture_output=True, text=True, check=False, env=env
    )


def _output(done: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(line for line in done.stdout.splitlines() if not line.startswith("compiling"))


def _expected(tmp_path: Path, source: str) -> str:
    _write(tmp_path, source)
    done = _run(tmp_path, "prog.ppy")
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@requires_llvm
@requires_cc
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_every_path_agrees_and_each_function_goes_native(tmp_path: Path, name: str):
    source, _, natives = PROGRAMS[name]
    expected = _expected(tmp_path, source)
    for args in (["-m", "ppy_compiler", "prog.ppy"], ["-m", "ppy_compiler", "run", "prog.ppy"]):
        done = _run(tmp_path, *args)
        assert done.returncode == 0, done.stderr
        assert _output(done).strip() == expected, args
    for function in natives:
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


@requires_standalone
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_a_standalone_binary_agrees(tmp_path: Path, name: str):
    source = PROGRAMS[name][0]
    expected = _expected(tmp_path, source)
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "--standalone", "prog.ppy", "-o", "dist")
    assert built.returncode == 0, built.stderr
    ran = subprocess.run(
        [str(tmp_path / "dist" / "prog")], capture_output=True, text=True, check=False
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == expected


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
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_emitted_source_frees_everything_once(
    tmp_path: Path, name: str, language: str, unsafe: bool
):
    compiler = c_compiler() if language == "c" else _CXX
    if compiler is None or not _sanitizes(compiler, tmp_path):
        pytest.skip(f"no {language} compiler with AddressSanitizer")
    source = PROGRAMS[name][0]
    expected = _expected(tmp_path, source)
    emitted = tmp_path / f"prog.{language}"
    flags = ["--unsafe"] if unsafe else []
    done = _run(
        tmp_path, "-m", "ppy_compiler", "emit", language, "--standalone", *flags, "prog.ppy",
        "-o", emitted.name,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    standard = "-std=c11" if language == "c" else "-std=c++17"
    binary = tmp_path / "prog"
    subprocess.run(
        [compiler, standard, "-g", "-O1", "-fsanitize=address,undefined", str(emitted), "-lm",
         "-o", str(binary)],
        check=True,
    )  # fmt: skip
    ran = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        check=False,
        env={"ASAN_OPTIONS": "detect_leaks=1"},
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == expected


@requires_llvm
@requires_cc
def test_a_method_that_fails_while_called_back_hands_the_call_to_python(tmp_path: Path):
    """`__lt__` divides by zero partway through the heap's pushes: native code
    cannot raise, so the call runs again as Python, which does."""
    expected = _expected(tmp_path, GUARDED)
    assert expected.splitlines()[1].startswith("ZeroDivisionError")
    for args in (["-m", "ppy_compiler", "prog.ppy"], ["-m", "ppy_compiler", "run", "prog.ppy"]):
        done = _run(tmp_path, *args)
        assert done.returncode == 0, done.stderr
        assert _output(done).strip() == expected, args
    explained = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.cheapest")
    assert "llvm backend: native" in explained.stdout, explained.stdout


@requires_standalone
def test_a_standalone_binary_says_what_the_method_raised(tmp_path: Path):
    source = GUARDED.replace("    print(cheapest(20, -1))\n", "")
    source = source.replace(
        "    try:\n        print(cheapest(20, 7))\n"
        "    except ZeroDivisionError as error:\n        print(type(error).__name__, error)\n",
        "    print(cheapest(20, 7))\n",
    )
    _write(tmp_path, source)
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "--standalone", "prog.ppy", "-o", "dist")
    assert built.returncode == 0, built.stderr
    ran = subprocess.run(
        [str(tmp_path / "dist" / "prog")], capture_output=True, text=True, check=False
    )
    assert ran.returncode == 1
    assert ran.stderr.strip().splitlines()[-1].startswith("ZeroDivisionError")


@requires_llvm
@requires_cc
def test_a_method_a_subclass_overrides_keeps_the_collection_in_python(tmp_path: Path):
    """A callback cannot choose a method by the object's class: the heap stays
    in Python, where each comparison finds its own `__lt__`."""
    expected = _expected(tmp_path, OVERRIDDEN)
    done = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert done.returncode == 0, done.stderr
    assert _output(done).strip() == expected
    explained = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.first")
    assert "overridden below" in explained.stdout, explained.stdout


def test_the_reference_classes_take_the_same_keys(tmp_path: Path):
    """What the checker takes as a key, the reference classes take, and a tree's
    keys are one where neither is less, as in any ordered map."""
    import ppy  # pylint: disable=import-outside-toplevel

    class Ranked:
        def __init__(self, rank: int) -> None:
            self.rank = rank

        def __lt__(self, other: Ranked) -> bool:
            return self.rank < other.rank

    tree = ppy.TreeMap[Ranked, int]()
    tree[Ranked(1)] = 10
    tree[Ranked(1)] = 20
    assert len(tree) == 1 and tree[Ranked(1)] == 20

    class Unhashable:
        def __eq__(self, other: object) -> bool:
            return self is other

    with pytest.raises(TypeError, match="hashable class"):
        ppy.HashMap[Unhashable, int]()
    with pytest.raises(TypeError, match="__lt__"):
        ppy.TreeSet[Unhashable]()
