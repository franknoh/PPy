"""Small expressions, more `for` loops, exact floats, `e.args`, and generators with frames, native.

`isinstance` folds from the checker's types; chained comparisons evaluate
each operand once; `in` tests a tuple, a literal set, or a `range` without
building one; `**` and `pow` of ints loop with checked products. `for` walks
a `range` with any step, a string, a tuple, and a list lent as a buffer,
alone or under `enumerate`, `zip`, and `reversed`. A generator that is
returned, passed on, held and stepped, or an `__iter__` method has a frame of
its own. Each program is held to CPython under `ppy`, `ppy run`, a standalone
binary, and emitted C and C++ (safe and `--unsafe`) under AddressSanitizer
with leak detection, with `ppy explain` confirming each function went native.
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

EXPRESSIONS = """
from dataclasses import dataclass


@dataclass
class Node:
    value: int
    next: "Node | None" = None


def kinds(n: int, q: float, s: str, xs: list[int], d: dict[int, int], flag: bool) -> int:
    t = (n, n)
    score = 0
    if isinstance(n, int):
        score += 1
    if isinstance(q, (int, float)):
        score += 2
    if isinstance(s, str) and not isinstance(s, (int, list)):
        score += 4
    if isinstance(xs, list) and not isinstance(xs, tuple):
        score += 8
    if isinstance(d, dict) and not isinstance(d, set):
        score += 16
    if isinstance(flag, int) and isinstance(flag, bool):
        score += 32
    if isinstance(t, tuple):
        score += 64
    if isinstance(n, (float, type(None))):
        score += 128
    return score


def nodes(head: Node | None) -> int:
    count = 0
    node = head
    while not isinstance(node, type(None)):
        if isinstance(node, Node):
            count += node.value
        node = node.next
    if isinstance(head, (Node, type(None))):
        count += 1000
    return count


def bump(xs: list[int]) -> int:
    xs.append(len(xs))
    return len(xs)


def chains(a: int, b: int, n: int, z: int, ch: str, x: float) -> int:
    c = 0
    if 0 <= a < n and 0 <= b < n:
        c += 1
    if a < b <= n < 100:
        c += 2
    if b < a < 1 // z:
        c += 4
    if "a" <= ch <= "z":
        c += 8
    if 0.0 < x < 1.0 != x:
        c += 16
    calls: list[int] = []
    if 0 < bump(calls) < 5 > bump(calls):
        c += 32
    c += len(calls) * 100
    if a == a == a < b:
        c += 1000
    return c


def members(n: int, cell: str, ch: str, x: float, step: int) -> int:
    c = 0
    if n % 10 not in (1, 3, 7, 9):
        c += 1
    if cell not in ("", ch):
        c += 2
    if n in range(3, 50, 7):
        c += 4
    if ch in "aeiou":
        c += 8
    if x in (0.5, 1.5, 2.5):
        c += 16
    if n in {2, 4, 6, 8, 13}:
        c += 32
    if n in range(40, -10, -3):
        c += 64
    if n in range(0, 100, step):
        c += 128
    if n not in range(n):
        c += 256
    if True in (0, 1) and 1.0 in (1, 2):
        c += 512
    t = (n, n + 1, n * 2)
    if n + 1 in t and 0 not in t:
        c += 1024
    try:
        if n in range(0, 10, step - step):
            c += 99999
    except ValueError:
        c += 2048
    return c


def powers(a: int, b: int, m: int) -> int:
    total = a ** b + pow(a, b, m) + 2 ** 10 + (-3) ** 3 + pow(-a, b, -m) + pow(a, 0, 1)
    total += 7 ** 0 + 0 ** 0 + 1 ** 62 + (-1) ** 63 + 3 ** 39
    try:
        total += pow(a, b, 0)
    except ValueError:
        total += 1
    return total


def float_powers(a: float, b: int) -> float:
    return a ** b + pow(2, 0.5) + 2 ** -1 + pow(a, 2)


def main() -> None:
    one = [1]
    none: list[int] = []
    print(kinds(3, 2.5, "x", one, {1: 2}, True), kinds(-1, 2.0, "", none, {3: 4}, False))
    print(nodes(Node(1, Node(2, Node(3)))), nodes(None))
    print(chains(1, 2, 5, 0, "q", 0.5), chains(3, 1, 2, 1, "Q", 1.0), chains(-1, 7, 7, -1, "a", 0.0))
    print(members(13, "", "q", 1.5, 2), members(40, "x", "a", 3.0, -1), members(1, "q", "q", 0.5, 7))
    print(powers(3, 5, 7), powers(-2, 7, 5), powers(10, 18, 1000000007))
    print(float_powers(1.5, 3), float_powers(-2.0, 2))


main()
"""

LOOPS = """
def counted(n: int, step: int, word: str, xs: list[int]) -> int:
    t = 0
    for i, v in enumerate(range(n)):
        t += i * v
    for i, ch in enumerate(word):
        t += i * ord(ch)
    for i, ch in enumerate(word, 1):
        if ch == "é":
            continue
        t += i
    for r, c in zip(range(n), [n] * n):
        t += r * c
    for a, b in zip(word, word[1:]):
        if a == b:
            t += 1000
    for j in range(0, n, step):
        t += j
    for j in range(n, -n, -step):
        t += j
    for j in reversed(range(n // 2)):
        t = t * 2 + j
    for j in reversed(range(1, n, 3)):
        t += j * 7
    for j in reversed(range(n, 0, -2)):
        t += j * 11
    for i, x in enumerate(xs):
        if x < 0:
            break
        t += i * x
    for x, y in zip(xs, reversed(xs)):
        t += x - y
    return t


def tupled(a: int, b: int, c: int) -> int:
    t = 0
    for v in (a, b, c):
        t = t * 3 + v
    for x, y in ((a, b), (b, c)):
        t += x * y
    for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        t += dx * 10 + dy
    p = (a, b, c)
    for i, v in enumerate(p, start=1):
        t += i * v
    for v in reversed(p):
        t = t * 2 - v
    for f in (0.5, 1.5):
        t += int(f * 4)
    for flag in (True, False):
        if flag:
            t += 1
    return t


def stepped(n: int, step: int) -> int:
    t = 0
    for i in range(n, 0, step):
        t += i
        i = 100
    return t


def chained(n: int) -> int:
    a = b = n * 2
    a += 1
    xs = ys = [1, 2]
    xs.append(n)
    i = j = k = 0
    for m in range(n):
        i = j = i + m
    return a * 1000 + b + len(ys) * 10 + i + j + k


def picked(row: int, col: int, vertical: bool, word: str) -> int:
    total = 0
    for i, ch in enumerate(word):
        r, c = (row + i, col) if vertical else (row, col + i)
        total += r * 10 + c + ord(ch)
    return total


def main() -> None:
    xs = [3, 1, -4, 1]
    ys: list[int] = []
    zs = [2, 2]
    print(counted(9, 2, "hello", xs), counted(0, 5, "", ys), counted(7, 3, "aébbc", zs))
    print(tupled(2, 3, 4), tupled(-1, 0, 1))
    try:
        print(stepped(10, 0))
    except ValueError as e:
        print("caught", e)
    print(stepped(10, -3), stepped(-5, 1))
    print(chained(4), chained(0))
    print(picked(1, 2, True, "hey"), picked(3, 4, False, "ab"))


main()
"""

EXCEPTION_ARGS = """
def check(n: int) -> int:
    if n < 0:
        raise ValueError("negative value")
    if n == 0:
        raise ValueError()
    if n > 100:
        raise KeyError("big")
    return n


def read(xs: list[int], n: int) -> str:
    out = ""
    try:
        check(n)
        out += str(xs[n])
    except ValueError as e:
        if len(e.args):
            out += f"{e.args[0]}"
        out += f"|{e.args}|{len(e.args)}"
    except IndexError as e:
        out += str(e.args[0]) + " " + str(e.args)
    except KeyError as e:
        out += "key"
    return out


def main() -> None:
    xs = [1, 2, 3]
    for n in (-1, 0, 1, 7):
        print(read(xs, n))
    print(read(xs, 200))


main()
"""

GENERATORS = """
from collections.abc import Iterator


def count_up(n: int) -> Iterator[int]:
    i = 0
    while i < n:
        yield i
        i += 1


def evens(n: int) -> Iterator[int]:
    for v in count_up(n):
        if v % 2 == 0:
            yield v


def total(it: Iterator[int]) -> int:
    t = 0
    for v in it:
        t += v
    return t


def make(n: int) -> Iterator[int]:
    return count_up(n)


def stepped(n: int) -> int:
    it = count_up(n)
    t = 0
    for k in range(3):
        if k % 2 == 0:
            t += next(it, -1) * 10
        else:
            t += next(it, -1)
    for v in it:
        t += v * 100
    return t


def main() -> None:
    print(total(count_up(5)), total(evens(9)), total(make(4)), stepped(6), stepped(1))


main()
"""

ITERATORS = """
from collections.abc import Iterator
from dataclasses import dataclass


class Node:
    def __init__(self, value: int) -> None:
        self.value = value
        self.left: Node | None = None
        self.right: Node | None = None

    def __iter__(self) -> Iterator[int]:
        if self.left:
            yield from self.left
        yield self.value
        if self.right:
            yield from self.right


class Link:
    def __init__(self, data: str, next_node: "Link | None") -> None:
        self.data = data
        self.next_node = next_node


class Chain:
    def __init__(self) -> None:
        self.head: Link | None = None

    def push(self, data: str) -> None:
        self.head = Link(data, self.head)

    def __iter__(self) -> Iterator[str]:
        node = self.head
        while node:
            yield node.data
            node = node.next_node


def words(text: str) -> Iterator[str]:
    parts = text.split(" ")
    seen: list[str] = []
    for p in parts:
        if p not in seen:
            seen.append(p)
            yield p.upper()


def insert(root: Node, v: int) -> None:
    if v < root.value:
        if root.left is None:
            root.left = Node(v)
        else:
            insert(root.left, v)
    elif root.right is None:
        root.right = Node(v)
    else:
        insert(root.right, v)


def tree_sum(values: list[int]) -> int:
    root = Node(values[0])
    for i in range(1, len(values)):
        insert(root, values[i])
    total = 0
    for v in root:
        total = total * 3 + v
    return total


def chain_text(n: int) -> str:
    c = Chain()
    for i in range(n):
        c.push(str(i))
    out = ""
    for s in c:
        out += s + ","
    return out


def first_word(text: str) -> str:
    for w in words(text):
        return w
    return "-"


def abandon(text: str) -> int:
    count = 0
    for w in words(text):
        count += len(w)
        if count > 4:
            break
    return count


def main() -> None:
    xs = [5, 3, 8, 1, 4, 9, 7]
    print(tree_sum(xs), chain_text(4), first_word("hi there hi"), abandon("ab cd ef ab gh"))


main()
"""

HELD_GENERATORS = """
from collections.abc import Iterator


class Bag:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, s: str) -> None:
        self.items.append(s)

    def __iter__(self) -> Iterator[str]:
        copy = [s + "*" for s in self.items]
        for s in copy:
            yield s


def pairs(n: int) -> Iterator[tuple[int, int]]:
    for i in range(n):
        yield i, i * i


def safe(xs: list[int]) -> Iterator[int]:
    for x in xs:
        try:
            yield 100 // x
        except ZeroDivisionError:
            yield -1


def head(it: Iterator[str]) -> str:
    return next(it, "none")


def rest(it: Iterator[str]) -> int:
    n = 0
    for s in it:
        n += len(s)
    return n


def use(k: int) -> str:
    b = Bag()
    for i in range(k):
        b.add("w" + str(i))
    it = iter(b)
    first = head(it)
    total = rest(it)
    early = ""
    for s in b:
        early = s
        break
    return first + "|" + str(total) + "|" + early


def sums(n: int) -> int:
    t = 0
    for a, b in pairs(n):
        t += a * 10 + b
    zs = [0, 5, 0, 20]
    for v in safe(zs):
        t += v
    return t


def main() -> None:
    print(use(3), use(0), sums(4))


main()
"""

RETURNS_INSIDE = """
def pick(words: list[str], n: int) -> str:
    keep = [w + "!" for w in words]
    for w in (x.upper() for x in keep if len(x) > n):
        return w + keep[0]
    return "-"


def main() -> None:
    ws = ["ab", "cde", "f"]
    print(pick(ws, 2), pick(ws, 9))


main()
"""

RECURSIVE = """
from collections.abc import Iterator


class Node:
    def __init__(self, data: int) -> None:
        self.data = data
        self.left: Node | None = None
        self.right: Node | None = None


def preorder(root: Node | None) -> Iterator[int]:
    if root:
        yield root.data
        yield from preorder(root.left)
        yield from preorder(root.right)


def build() -> Node:
    tree = Node(1)
    tree.left = Node(2)
    tree.right = Node(3)
    tree.left.left = Node(4)
    tree.left.right = Node(5)
    return tree


def walk() -> list[int]:
    return list(preorder(build()))


def main() -> None:
    print(walk(), sum(preorder(build())))


main()
"""

IDENTITY = """
def sieve(num: int) -> list[int]:
    marks = [True] * (num + 1)
    marks[0] = marks[1] = False
    primes: list[int] = []
    for start in range(2, num + 1):
        if marks[start] is True:
            primes.append(start)
            for i in range(start * start, num + 1, start):
                if marks[i] is not False:
                    marks[i] = False
    return primes


def both(a: bool, b: bool) -> int:
    return (a is b) + 2 * (a is not True) + 4 * (b is False)


def main() -> None:
    print(sieve(30), both(True, True), both(False, True), both(True, False))


main()
"""

FOR_ELSE = """
def first_factor(n: int) -> int:
    for d in range(2, n):
        if n % d == 0:
            break
    else:
        return n
    return d


def find(words: list[str], want: str) -> int:
    for i, w in enumerate(words):
        if w == want:
            break
    else:
        i = -1
    return i


def letters(s: str) -> str:
    out = ""
    for ch in s:
        if ch == "!":
            break
        out += ch
    else:
        out += "?"
    return out


def main() -> None:
    ws = ["a", "b", "c"]
    print(first_factor(15), first_factor(13), find(ws, "b"), find(ws, "z"), letters("hi!x"), letters("yo"))


main()
"""

PROGRAMS = {
    "for_else": (FOR_ELSE, ["first_factor", "find", "letters"]),
    "identity": (IDENTITY, ["sieve", "both"]),
    "recursive": (RECURSIVE, ["preorder", "build", "walk"]),
    "expressions": (EXPRESSIONS, ["kinds", "nodes", "chains", "members", "powers", "float_powers"]),
    "loops": (LOOPS, ["counted", "tupled", "stepped", "chained", "picked"]),
    "exception_args": (EXCEPTION_ARGS, ["check", "read"]),
    "generators": (GENERATORS, ["count_up", "evens", "total", "make", "stepped"]),
    "iterators": (
        ITERATORS,
        [
            "Node.__iter__",
            "Chain.__iter__",
            "words",
            "tree_sum",
            "chain_text",
            "first_word",
            "abandon",
        ],
    ),
    "held_generators": (
        HELD_GENERATORS,
        ["Bag.__iter__", "pairs", "safe", "head", "rest", "use", "sums"],
    ),
    "returns_inside": (RETURNS_INSIDE, ["pick"]),
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
    expected = _expected(tmp_path, PROGRAMS[name][0])
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
    expected = _expected(tmp_path, PROGRAMS[name][0])
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


EXACT = """
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float


def cross_product(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def ident(x: float) -> float:
    return x


def scaled(x: float) -> float:
    return x * 2


def hidden(x: float) -> float:
    return x / 2 + 1


def via(x: float) -> float:
    return ident(x) + 1


def length(x: float, y: float) -> float:
    return (x * x + y * y) ** 0.5


def shown(x: float) -> str:
    return f"{x}"


def main() -> None:
    print(cross_product((0, 0), (1, 1), (2, 2)), cross_product((0.0, 0.0), (1.5, 1.0), (2.0, 2.0)))
    print(ident(3), ident(3.0), scaled(4), hidden(4), via(2), length(3, 4), shown(5))
    p = Point(1, 2)
    print(p.x, p)


main()
"""


@requires_llvm
@requires_cc
def test_an_int_given_for_a_float_stays_an_int_where_it_would_show(tmp_path: Path):
    """`f(0)` of a native `f(x: float)` that returns `x` is `0`, as in CPython:
    the boundary keeps such a call in Python. A float parameter whose int-ness
    never shows (`x / 2`) still takes the int, converted."""
    expected = _expected(tmp_path, EXACT)
    assert expected.splitlines()[0] == "0 1.0"
    for args in (["-m", "ppy_compiler", "prog.ppy"], ["-m", "ppy_compiler", "run", "prog.ppy"]):
        done = _run(tmp_path, *args)
        assert done.returncode == 0, done.stderr
        assert _output(done).strip() == expected, args
    from ppy_compiler.driver.pipeline import analyze_paths, open_project
    from ppy_compiler.lowering.intness import ModuleIntness

    path = tmp_path / "prog.ppy"
    bundle = analyze_paths(open_project(path), [path], backend="llvm")
    module = bundle.analysis.modules["prog"]
    intness = ModuleIntness(module.functions, module.node_types)
    assert intness.exact("prog.cross_product") == {"a", "b", "c"}
    assert intness.exact("prog.ident") == {"x"}
    assert intness.exact("prog.via") == {"x"}
    assert intness.exact("prog.hidden") == frozenset()
    assert intness.exact("prog.length") == frozenset()


BOOL_EXACT = """
from collections.abc import Callable


def shows(n: int) -> int:
    print(n)
    return n + 1


def asks(n: int) -> int:
    return 1 if isinstance(n, bool) else 0


def counts(n: int, m: int) -> int:
    return (n + 1) * -m + n.bit_length()


def passes(n: int) -> int:
    return shows(n)


def taken(n: int) -> int:
    return 1 if isinstance(n, bool) else 0


def main() -> None:
    f: Callable[[int], int] = taken
    print(f(True), shows(True), asks(1), counts(True, False), passes(2))


main()
"""


def test_int_parameters_that_show_a_bool(tmp_path: Path):
    """`print(n)` and `isinstance(n, bool)` show whether an `int` parameter
    was given a `bool`; arithmetic does not. Only a function called by name
    has every native caller checked."""
    (tmp_path / "prog.ppy").write_text(BOOL_EXACT, encoding="utf-8")
    from ppy_compiler.driver.pipeline import analyze_paths, open_project
    from ppy_compiler.lowering.intness import ModuleIntness

    path = tmp_path / "prog.ppy"
    bundle = analyze_paths(open_project(path), [path], backend="llvm")
    module = bundle.analysis.modules["prog"]
    intness = ModuleIntness(module.functions, module.node_types, module.symbols.module.tree)
    assert intness.bool_exact("prog.shows") == {"n"}
    assert intness.bool_exact("prog.asks") == {"n"}
    assert intness.bool_exact("prog.passes") == {"n"}
    assert intness.bool_exact("prog.counts") == frozenset()
    assert intness.exact("prog.shows") == frozenset()
    assert intness.called_directly("asks")
    assert not intness.called_directly("taken")


def test_locals_only_ever_given_a_real_int_or_float():
    import ast

    from ppy_compiler.analysis import types as T
    from ppy_compiler.lowering.intness import exact_locals

    source = textwrap.dedent(
        """
        def f(x: float, n: int, flag: bool) -> None:
            t = (2, 1e308 * x)
            m, g = t
            h: float = 3
            k = 0.0
            k += n
            r = 0.5
            for i in range(3):
                r = r * i
            a = len("ab") * 10 + n
            a *= n - 8
            b = flag & flag
            c: int = flag
            d = 0
            d = d + flag
        """
    )
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    typed = {"n": T.INT, "flag": T.BOOL, "a": T.INT, "d": T.INT}

    def type_of(node: ast.expr) -> T.Type:
        if isinstance(node, ast.Constant):
            return T.INT if type(node.value) is int else T.FLOAT
        if isinstance(node, ast.Name):
            return typed.get(node.id, T.FLOAT)
        return T.INT if isinstance(node, (ast.Call, ast.BinOp)) else T.UNKNOWN

    params = ["x", "n", "flag"]
    floats = exact_locals(function, params, frozenset(), "float", type_of)
    ints = exact_locals(function, params, frozenset({"n"}), "int", type_of)
    assert {"g", "k", "r"} <= floats  # `0.5 * i` is a float
    assert not floats & {"h", "i", "x", "m"}
    assert {"a", "m", "d", "n", "h"} <= ints  # `h` holds the int 3
    assert not ints & {"b", "c", "flag", "i"}


def test_the_python_binding_refuses_an_int_for_an_exact_float_and_a_bool_for_an_int():
    from ppy_runtime.abi import NativeParam
    from ppy_runtime.binding import GuardFailed, _expander_for

    exact = _expander_for(NativeParam("x", "float", exact=True))
    loose = _expander_for(NativeParam("x", "float"))
    whole = _expander_for(NativeParam("n", "int"))
    atoms: list = []
    loose(3, atoms, [])
    assert atoms == [3.0]
    for expand, value in ((exact, 3), (whole, True), (loose, 1 << 60)):
        with pytest.raises(GuardFailed):
            expand(value, [], [])


WIDENED = """
from collections.abc import Sequence


def total(xs: Sequence[float]) -> float:
    t = 0.0
    for x in xs:
        t = t + x
    return t


def biggest(xs: Sequence[float]) -> float:
    best = xs[0]
    for x in xs:
        if x > best:
            best = x
    return best


def mean(xs: Sequence[float]) -> float:
    t = 0.0
    for x in xs:
        t += x
    return t / len(xs)


def main() -> None:
    ints = [3, 1, 4]
    floats = [1.5, 2.5]
    print(total(ints), biggest(ints), mean(ints), total(floats), biggest(floats), mean(floats))


main()
"""


@requires_llvm
@requires_cc
def test_a_list_of_ints_for_a_sequence_of_floats_keeps_its_ints(tmp_path: Path):
    """`biggest([3, 1, 4])` of a `Sequence[float]` is `4` in CPython: the element
    is returned, so the boundary refuses the ints there; `total` and `mean`
    make floats of them either way, and take them, converted."""
    expected = _expected(tmp_path, WIDENED)
    assert expected == "8.0 4 2.6666666666666665 4.0 2.5 2.0"
    for args in (["-m", "ppy_compiler", "prog.ppy"], ["-m", "ppy_compiler", "run", "prog.ppy"]):
        done = _run(tmp_path, *args)
        assert done.returncode == 0, done.stderr
        assert _output(done).strip() == expected, args
    for function in ("total", "biggest", "mean"):
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


BOOL_STORES = """
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Box:
    v: int


def local(flag: bool) -> int:
    x: int = flag
    return x


def rebound(flag: bool, n: int) -> int:
    y = n
    y = flag
    return y


def returned(n: int) -> int:
    return n > 2


def listed(flag: bool) -> list[int]:
    xs: list[int] = [1]
    xs.append(flag)
    return xs


def keyed(flag: bool) -> int:
    d: dict[str, int] = {}
    d["a"] = flag
    return len(d)


def boxed(flag: bool) -> int:
    return Box(flag).v


def through(f: Callable[[int], int], flag: bool) -> int:
    return f(flag)


def arithmetic(flag: bool, n: int) -> int:
    z: int = flag + n
    w: int = int(flag)
    z += flag
    return z * 2 - w


def main() -> None:
    print(local(True), rebound(True, 3), returned(5), listed(False))
    print(keyed(True), boxed(True), through(returned, True), arithmetic(True, 2))


main()
"""


def test_a_bool_stored_where_an_int_is_declared_stays_in_python(tmp_path: Path):
    """Each store of a `bool` into an `int` slot keeps its function in Python
    (native code would hold `True` as 1); arithmetic on `bool`s does not."""
    (tmp_path / "prog.ppy").write_text(BOOL_STORES, encoding="utf-8")
    import json

    done = _run(tmp_path, "-m", "ppy_compiler", "explain", "--summary", "--json", "prog.ppy")
    assert done.returncode == 0, done.stderr
    reasons = {f["qualname"]: f["reason"] for f in json.loads(done.stdout)["functions"]}
    stores = "stores a `bool` where an `int` is declared"
    for name in ("local", "rebound", "returned", "listed", "keyed", "boxed", "through"):
        assert stores in reasons[f"prog.{name}"], (name, reasons[f"prog.{name}"])
    assert stores not in reasons["prog.arithmetic"]
    expected = _expected(tmp_path, BOOL_STORES)
    done = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert done.returncode == 0, done.stderr
    assert _output(done).strip() == expected
