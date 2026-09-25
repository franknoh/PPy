"""Classes in native code, past the basics: inheritance, operators, inference.

A subclass is held as one record, its bases' fields first; a method a
subclass overrides dispatches on the class the object was made as, which a
tag in its header records. `super()`, `isinstance`, operator methods
(`__add__`, `__lt__`, `__getitem__`, `__contains__`), `field(default_factory=...)`,
in-place writes to a field of a value-class element, and reference fields
read off temporaries all lower. A generic class or a collection made without
its type arguments takes them from its constructor's arguments or from where
it goes. Each program agrees on every path, goes native, and its emitted C
and C++ run clean under AddressSanitizer with leak detection.
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

SHAPES = """
import ppy
from ppy import Vec


class Shape:
    def __init__(self, scale: int) -> None:
        self.scale: int = scale
        self.hits: int = 0

    def area(self) -> int:
        return 0

    def describe(self) -> int:
        self.hits += 1
        return self.area() * 10 + self.scale


class Square(Shape):
    def __init__(self, scale: int, side: int) -> None:
        super().__init__(scale)
        self.side: int = side

    def area(self) -> int:
        return self.side * self.side


class Rect(Square):
    def __init__(self, scale: int, side: int, other: int) -> None:
        super().__init__(scale, side)
        self.other: int = other

    def area(self) -> int:
        return self.side * self.other + super().area()


def total(shapes: Vec[Shape]) -> int:
    acc: int = 0
    for s in shapes:
        acc += s.describe()
        if isinstance(s, Square):
            acc += s.side
    return acc


def run(n: int) -> int:
    shapes = Vec[Shape]()
    for i in range(n):
        if i % 3 == 0:
            shapes.push(Shape(i))
        elif i % 3 == 1:
            shapes.push(Square(i, i + 1))
        else:
            shapes.push(Rect(i, i, 2))
    return total(shapes)


def main() -> None:
    print(run(ppy.scan[int]()))


main()
"""

EXPRESSIONS = """
from dataclasses import dataclass

import ppy
from ppy import Vec


@dataclass
class Expr:
    weight: int = 1

    def eval(self) -> int:
        return 0

    def size(self) -> int:
        return self.weight


@dataclass
class Num(Expr):
    value: int = 0

    def eval(self) -> int:
        return self.value


@dataclass
class Add(Expr):
    left: Expr | None = None
    right: Expr | None = None

    def eval(self) -> int:
        total: int = 0
        if self.left is not None:
            total += self.left.eval()
        if self.right is not None:
            total += self.right.eval()
        return total

    def size(self) -> int:
        n: int = self.weight
        if self.left is not None:
            n += self.left.size()
        if self.right is not None:
            n += self.right.size()
        return n


@dataclass
class Mul(Add):
    def eval(self) -> int:
        a: int = 1
        if self.left is not None:
            a = self.left.eval()
        b: int = 1
        if self.right is not None:
            b = self.right.eval()
        return (a * b) % 1000003


def build(depth: int, seed: int) -> Expr:
    if depth == 0:
        return Num(1, seed % 10)
    if seed % 2 == 0:
        return Add(1, build(depth - 1, seed * 3 + 1), build(depth - 1, seed + 7))
    return Mul(2, build(depth - 1, seed + 5), build(depth - 1, seed * 7 + 2))


def count_adds(e: Expr | None) -> int:
    if e is None:
        return 0
    if isinstance(e, Mul):
        return count_adds(e.left) + count_adds(e.right)
    if isinstance(e, Add):
        return 1 + count_adds(e.left) + count_adds(e.right)
    return 0


def run(n: int) -> int:
    trees = Vec[Expr]()
    for i in range(n):
        trees.push(build(6, i))
    acc: int = 0
    for t in trees:
        acc = (acc * 31 + t.eval() + t.size() + count_adds(t)) % 1000003
    return acc


def main() -> None:
    print(run(ppy.scan[int]()))


main()
"""

INFERENCE = """
import ppy
from ppy import HashMap, Vec


class Pair[A: float, B: float]:
    def __init__(self, first: A, second: B) -> None:
        self.first: A = first
        self.second: B = second

    def total(self) -> float:
        return self.first + self.second


class Stack[T]:
    def __init__(self) -> None:
        self.items: Vec[T] = Vec[T]()

    def push(self, value: T) -> None:
        self.items.push(value)

    def pop(self) -> T:
        return self.items.pop()

    def __len__(self) -> int:
        return len(self.items)


def fill(n: int) -> Stack[int]:
    s: Stack[int] = Stack()
    for i in range(n):
        s.push(i * i)
    return s


def counts(n: int) -> HashMap[int, int]:
    m: HashMap[int, int] = HashMap()
    for i in range(n):
        m[i % 7] = m.get(i % 7, 0) + 1
    return m


def run(n: int) -> float:
    p = Pair(3, 0.25)
    v: Vec[int] = Vec()
    for i in range(n):
        v.push(i)
    s = fill(n)
    total: float = p.total() + len(v)
    while s:
        total += s.pop()
    m = counts(n)
    total += m[3]
    return total


def main() -> None:
    print(run(ppy.scan[int]()))


main()
"""

OPERATORS = """
import ppy
from ppy import Vec


class Money:
    def __init__(self, cents: int) -> None:
        self.cents: int = cents
        self.touched: int = 0

    def bump(self) -> None:
        self.touched += 1

    def __add__(self, other: "Money") -> "Money":
        return Money(self.cents + other.cents)

    def __sub__(self, other: "Money") -> "Money":
        return Money(self.cents - other.cents)

    def __mul__(self, k: int) -> "Money":
        return Money(self.cents * k)

    def __neg__(self) -> "Money":
        return Money(-self.cents)

    def __eq__(self, other: "Money") -> bool:
        return self.cents == other.cents

    def __lt__(self, other: "Money") -> bool:
        return self.cents < other.cents

    def __le__(self, other: "Money") -> bool:
        return self.cents <= other.cents


class Grid:
    def __init__(self, width: int) -> None:
        self.width: int = width
        self.cells: Vec[int] = Vec[int](width * width)

    def __getitem__(self, at: int) -> int:
        return self.cells[at]

    def __setitem__(self, at: int, value: int) -> None:
        self.cells[at] = value

    def __contains__(self, value: int) -> bool:
        for cell in self.cells:
            if cell == value:
                return True
        return False

    def __len__(self) -> int:
        return len(self.cells)


def money(n: int) -> int:
    total = Money(0)
    best = Money(0)
    for i in range(n):
        m = Money(i * 37 % 101)
        total = total + m * 2 - Money(1)
        if best < m:
            best = m
        if m == Money(5):
            total = total + -m
        if m > best or m >= best:
            total = total + Money(1000)
    return total.cents * 1000 + best.cents


def grid(n: int) -> int:
    g = Grid(n)
    for i in range(len(g)):
        g[i] = i * i % 17
        g[i] += 1
    hits: int = 0
    for v in range(20):
        if v in g:
            hits += 1
        if v not in g:
            hits += 100
    return hits * 1000 + g[n + 1]


def main() -> None:
    n: int = ppy.scan[int]()
    print(money(n), grid(n))


main()
"""

FIELDS = """
from dataclasses import dataclass, field

import ppy
from ppy import HashMap, Vec


@dataclass
class Point:
    x: int
    y: float


@dataclass
class Counter:
    hits: int = 0

    def hit(self) -> None:
        self.hits += 1


@dataclass
class Bag:
    name: int
    items: Vec[int] = field(default_factory=Vec[int])
    seen: HashMap[int, int] = field(default_factory=HashMap[int, int])
    counter: Counter = field(default_factory=Counter)
    limit: int = field(default=7)

    def add(self, x: int) -> None:
        self.items.push(x)
        self.seen[x % self.limit] = x
        self.counter.hit()


@dataclass
class Node:
    value: int
    child: "Node | None" = None


def shift(n: int) -> float:
    points = Vec[Point]()
    for i in range(n):
        points.push(Point(i, i / 2))
    for i in range(n):
        points[i].x = points[i].x * 2 + 1
        points[i].y += 0.5
    table = HashMap[int, Point]()
    table[3] = Point(1, 1.0)
    table[3].x += 10
    total: float = 0.0
    for p in points:
        total += p.x + p.y
    return total + table[3].x


def bags(n: int) -> int:
    held = Vec[Bag]()
    for i in range(3):
        held.push(Bag(i))
    for i in range(n):
        held[i % 3].add(i)
    total: int = 0
    for b in held:
        total += len(b.items) * 100 + len(b.seen) * 10 + b.counter.hits + b.name
    return total


def make(n: int) -> Node:
    return Node(n, Node(n * 2, Node(n * 3)))


def rows(n: int) -> Vec[Vec[int]]:
    out = Vec[Vec[int]]()
    for i in range(n):
        row = Vec[int]()
        row.push(i)
        out.push(row)
    return out


def temporaries(n: int) -> int:
    total: int = 0
    for i in range(n):
        c = make(i).child
        if c is not None:
            total += c.value
        total += len(rows(i + 1)[i])
        grand = make(i).child
        if grand is not None and grand.child is not None:
            total += grand.child.value
    return total


def main() -> None:
    n: int = ppy.scan[int]()
    print(shift(n), bags(n), temporaries(n))


main()
"""

CASES = {
    "shapes": (SHAPES, "10\n", "2235", ["total", "run", "Rect.area", "Square.__init__"]),
    "expressions": (
        EXPRESSIONS,
        "20\n",
        "704631",
        ["run", "build", "count_adds", "Add.eval", "Mul.eval", "Add.size"],
    ),
    "inference": (INFERENCE, "20\n", "2496.25", ["run", "fill", "counts"]),
    "operators": (OPERATORS, "30\n", "8870097 1109010", ["money", "grid"]),
    "fields": (FIELDS, "10\n", "138.5 1113 235", ["shift", "bags", "temporaries"]),
}


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
@pytest.mark.parametrize("case", sorted(CASES))
def test_classes_agree_on_every_path_and_go_native(tmp_path: Path, case: str):
    source, given, expected, native = CASES[case]
    _write(tmp_path, source)
    checked = _run(tmp_path, "-m", "ppy_compiler", "check", "prog.ppy")
    assert "no errors" in checked.stdout + checked.stderr, checked.stdout + checked.stderr
    for args in (
        ["prog.ppy"],
        ["-m", "ppy_compiler", "prog.ppy"],
        ["-m", "ppy_compiler", "run", "prog.ppy"],
    ):
        done = _run(tmp_path, *args, text=given)
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == expected, args
    for function in native:
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


@requires_standalone
@pytest.mark.parametrize("case", sorted(CASES))
def test_a_standalone_binary_holds_the_classes(tmp_path: Path, case: str):
    source, given, expected, _ = CASES[case]
    program = _write(tmp_path, source)
    built = _run(
        tmp_path, "-m", "ppy_compiler", "build", "--standalone", program.name, "-o", "dist"
    )
    assert built.returncode == 0, built.stderr
    ran = subprocess.run(
        [str(tmp_path / "dist" / "prog")], input=given, capture_output=True, text=True, check=False
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
@pytest.mark.parametrize("case", sorted(CASES))
def test_emitted_classes_are_freed_once(tmp_path: Path, case: str, language: str, unsafe: bool):
    compiler = c_compiler() if language == "c" else _CXX
    if compiler is None or not _sanitizes(compiler, tmp_path):
        pytest.skip(f"no {language} compiler with AddressSanitizer")
    source, given, expected, _ = CASES[case]
    program = _write(tmp_path, source)
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
        input=given,
        capture_output=True,
        text=True,
        check=False,
        env={"ASAN_OPTIONS": "detect_leaks=1"},
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == expected


@requires_llvm
def test_a_copy_written_through_its_element_stays_in_python(tmp_path: Path):
    _write(
        tmp_path,
        """
        from dataclasses import dataclass

        from ppy import Vec


        @dataclass
        class Point:
            x: int
            y: float


        def alias(n: int) -> int:
            points = Vec[Point]()
            points.push(Point(n, 0.0))
            p = points[0]
            points[0].x = 99
            return p.x + points[0].x


        print(alias(3))
        """,
    )
    for args in (["prog.ppy"], ["-m", "ppy_compiler", "run", "prog.ppy"]):
        done = _run(tmp_path, *args)
        assert done.stdout.strip().splitlines()[-1] == "198", done.stderr
    explained = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.alias")
    assert "copied out and written in place" in explained.stdout


def test_the_checker_infers_type_arguments(tmp_path: Path):
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


        def inferred() -> None:
            box = Box(1)
            box.add(2.5)


        def expected() -> None:
            box: Box[int] = Box(1)
            box.add(2.5)


        def unknown() -> int:
            v = Vec()
            return len(v)


        def floats() -> float:
            v: Vec[float] = Vec()
            return 0.0
        """,
    )
    checked = _run(tmp_path, "-m", "ppy_compiler", "check", "prog.ppy")
    shown = (checked.stdout + checked.stderr).splitlines()
    errors = [line for line in shown if line.startswith("error[")]
    assert errors[:4] == [
        "error[E1301]: `add` parameter `value` expects `int`, got `Literal[2.5]`",
        "error[E1301]: `add` parameter `value` expects `int`, got `Literal[2.5]`",
        "error[E1305]: `Vec()` takes its type from where it goes, and nothing says it here: "
        "write `Vec[int]()`, or annotate the target",
        "error[E1305]: a `Vec` of floats is made with its type written out, `Vec[float]()`, "
        "so CPython stores `3` as `3.0` too",
    ]
