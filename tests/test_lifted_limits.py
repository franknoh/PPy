"""What used to keep a function in Python, now native: CPython's answers kept.

Exception classes with fields and `__init__`, `raise ... from`; a generator
held in a name and stepped by `next` and `for`; a set of ints or of tuples
of ints walked in CPython's own order; dict and set keys that are floats or
bools; a dataclass's generated `==`, `order=True`, and `__repr__`, and
`str()` of an object; `str.format` and `%` of a literal. Each program is held
to CPython under `ppy`, `ppy run`, a standalone binary, and emitted C and C++
under AddressSanitizer with leak detection, with `ppy explain` confirming the
listed functions went native.
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

EXCEPTIONS = """
class ParseError(ValueError):
    def __init__(self, line: int, text: str) -> None:
        super().__init__(f"line {line}: {text}")
        self.line: int = line
        self.text: str = text


class Plain(RuntimeError):
    def __init__(self, code: int, where: str) -> None:
        self.code: int = code
        self.where: str = where


def parse(n: int) -> int:
    total = 0
    for i in range(n):
        try:
            if i % 7 == 3:
                raise ParseError(i, "bad token")
            total += i
        except ParseError as e:
            total += e.line * 1000 + len(e.text) + len(str(e))
    return total


def plain(n: int) -> str:
    try:
        raise Plain(n, "here")
    except Plain as e:
        return f"{e.code} {e.where} {e}"


def chained(n: int) -> int:
    try:
        try:
            return 10 // (n - n)
        except ZeroDivisionError as inner:
            raise ParseError(n, "divide") from inner
    except ValueError as outer:
        return len(str(outer))


def uncaught(n: int) -> int:
    if n > 2:
        raise ParseError(n, "too big") from None
    return n


def main() -> None:
    print(parse(30), plain(5), chained(4), uncaught(1))
    try:
        uncaught(9)
    except ParseError as e:
        print("caught", e, e.line)


main()
"""

GENERATORS = """
from collections.abc import Iterator


def countdown(n: int) -> Iterator[int]:
    while n > 0:
        yield n
        n -= 1


def squares(n: int) -> Iterator[int]:
    for i in range(n):
        yield i * i


def header_then_rows(n: int) -> int:
    it = squares(n)
    first = next(it)
    second = next(it, -1)
    total = first * 100 + second
    for row in it:
        total += row
    last = next(it, 77)
    return total * 1000 + last


def partial(n: int) -> int:
    it = countdown(n)
    got = 0
    for x in it:
        if x < 5:
            break
        got += x
    rest = 0
    for y in it:
        rest += y
    return got * 100 + rest


def short(n: int) -> int:
    it = (i + 1 for i in range(n))
    a = next(it, -1)
    b = next(it, -2)
    c = next(it, -3)
    return a * 100 + b * 10 + c


def stop(n: int) -> int:
    it = countdown(n)
    a = next(it)
    b = next(it)
    return a + b


def main() -> None:
    print(header_then_rows(6), header_then_rows(1), partial(9), short(2), short(0), stop(5))
    try:
        stop(1)
    except StopIteration:
        print("stopped")


main()
"""

WORDS = """
from collections.abc import Iterator


def words(n: int) -> Iterator[str]:
    parts = [f"w{i}" for i in range(n)]
    for p in parts:
        yield p + "!"


def joined(n: int) -> str:
    it = words(n)
    out = ""
    for w in it:
        if len(out) > 8:
            break
        out += w
    for w in it:
        out += "|" + w
    return out


def main() -> None:
    print(joined(6), joined(1))


main()
"""

SETS = """
def build(n: int) -> set[int]:
    s: set[int] = set()
    for i in range(n):
        s.add(i * 7919 % 1000 - 300)
    for i in range(0, n, 3):
        s.discard(i * 7919 % 1000 - 300)
    return s


def pairs(n: int) -> set[tuple[int, int]]:
    return {(i % 7, i * 31 % 11) for i in range(n)}


def algebra(n: int) -> str:
    a = build(n)
    b = build(n - 15)
    c = set(a)
    c |= b
    c -= {1, 2, 3, 700}
    c ^= b
    c &= a
    popped = [c.pop() for _ in range(3)]
    return f"{a | b}\\n{a & b}\\n{a - b}\\n{a ^ b}\\n{c}\\n{popped}\\n{list(a)}"


def shown(n: int) -> str:
    e: set[int] = set()
    d = {i * 1000003: i for i in range(n)}
    walked = 0
    for x in build(n):
        walked = walked * 31 % 1000003 + x
    lit = {5, 3, 1000, 17, -4}
    return f"{pairs(40)} {e} {set(d)} {lit} {walked}"


def main() -> None:
    print(algebra(60))
    print(shown(20))
    empty: set[int] = set()
    try:
        empty.pop()
    except KeyError as err:
        print("caught", err)


main()
"""

KEYS = """
def floats(n: int) -> str:
    d: dict[float, int] = {}
    for i in range(n):
        d[i * 0.5] = i
    d[-0.0] = 100
    z: dict[float, int] = {-0.0: 1}
    z[0.0] = 2
    s: set[float] = {0.0, -0.0, 1.5}
    return f"{d} {z} {d[0.0]} {2.0 in d} {len(s)} {-0.0 in s}"


def bools(n: int) -> str:
    b: dict[bool, int] = {True: 1, False: 0}
    b[n > 2] += 5
    t: dict[tuple[bool, int], int] = {(True, 1): 1}
    t[(False, n)] = 2
    return f"{b} {t} {(False, n) in t}"


def counted(xs: list[float]) -> int:
    d: dict[float, int] = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return len(d)


def main() -> None:
    print(floats(8))
    print(bools(3))
    print(counted([1.0, 2.0, -0.0, 0.0, 1.0]))


main()
"""

DATACLASSES = """
from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: float
    name: str


@dataclass(order=True)
class Version:
    major: int
    minor: int
    tag: str


@dataclass
class Money:
    cents: int
    rate: float


class Account:
    def __init__(self, cents: int) -> None:
        self.cents: int = cents
        self.log: list[int] = []

    def add(self, n: int) -> None:
        self.cents += n

    def __repr__(self) -> str:
        return f"Account({self.cents})"

    def __str__(self) -> str:
        return f"${self.cents // 100}.{self.cents % 100:02d}"


def compared(n: int) -> str:
    p = Point(n, 2.5, "a'b")
    q = Point(n, 2.5, "a'b")
    v = Version(1, n, "rc")
    w = Version(1, 10, "a")
    return f"{p == q} {p != q} {p == Point(n + 1, 2.5, 'x')} {v < w} {v >= w} {w > v}"


def shown(n: int) -> str:
    p = Point(n, 2.5, "a'b")
    m = Money(n, 0.5)
    a = Account(n * 100 + 5)
    return f"{p} {p!r} {repr(p)} {m} {m == Money(n, 0.5)} {a} {a!r} {str(a)} {[a]}"


def main() -> None:
    print(compared(2))
    print(shown(3))
    print(Point(1, 1.0, ""), Account(250))


main()
"""

FORMATTING = """
def show(n: int, x: float, name: str, flag: bool) -> str:
    a = "%-6s|%+5d|% d|%#o|%#X|%e|%g|%.3s|%5.1f%%" % (name, n, n, n, n, x, x, name, x)
    b = "{:>8}|{:<5}|{:^7}|{:+.2e}|{!r:>9}|{}|{:b}".format(name, n, flag, x, name, flag, n)
    c = "{0:08.2f} {1} {0} {k}".format(x, n, k=name)
    d = "%s %s %r %d" % (flag, x, x, flag)
    return a + "\\n" + b + "\\n" + c + "\\n" + d


def main() -> None:
    print(show(42, 3.14159, "ppy", True))
    print(show(-7, -1e-9, "longer name", False))
    print("%s has %d" % ("it", 3), "{} and {}".format(1.5, "x"))


main()
"""

PROGRAMS = {
    "exceptions": (EXCEPTIONS, ["parse", "plain", "chained", "uncaught"]),
    "generators": (GENERATORS, ["header_then_rows", "partial", "short", "stop"]),
    "words": (WORDS, ["joined"]),
    "sets": (SETS, ["build", "pairs", "algebra", "shown"]),
    "keys": (KEYS, ["floats", "bools", "counted"]),
    "dataclasses": (DATACLASSES, ["compared", "shown"]),
    "formatting": (FORMATTING, ["show"]),
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
    source, natives = PROGRAMS[name]
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


FROM_PYTHON = """
def walk(s: set[int]) -> list[int]:
    out: list[int] = []
    for x in s:
        out.append(x * 2)
    return out


def main() -> None:
    given = {5, 1000, 33, -7, 2 ** 40}
    given.discard(33)
    print(walk(given))
    nan = float("nan")
    print(walk({nan_free for nan_free in range(9, 0, -2)}), nan != nan)


main()
"""


@requires_llvm
@requires_cc
def test_a_set_from_python_walks_in_its_own_order(tmp_path: Path):
    """A set Python made has an order native code does not know: the walk
    falls back, and `ppy run` still prints what CPython prints."""
    expected = _expected(tmp_path, FROM_PYTHON)
    done = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert done.returncode == 0, done.stderr
    assert _output(done).strip() == expected
    explained = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.walk")
    assert "llvm backend: native" in explained.stdout, explained.stdout


NAN_KEYS = """
def count(xs: list[float]) -> int:
    d: dict[float, int] = {}
    for x in xs:
        if x in d:
            d[x] += 1
        else:
            d[x] = 1
    return len(d)


def main() -> None:
    nan = float("nan")
    print(count([1.0, -0.0, 0.0]), count([nan, nan, 1.0]), count([float("nan"), float("nan")]))


main()
"""


@requires_llvm
@requires_cc
def test_a_nan_key_falls_back_to_python(tmp_path: Path):
    """A NaN key is found only by its own object: native code hands the call
    to Python, which counts as CPython does."""
    expected = _expected(tmp_path, NAN_KEYS)
    assert expected == "2 2 2"
    done = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert done.returncode == 0, done.stderr
    assert _output(done).strip() == expected
