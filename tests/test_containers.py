"""Python's own `list`, `dict`, and `set`, native with no rewrite.

A function that builds and uses plain containers lowers over the collections
runtime: displays, comprehensions, `[x] * n`, the constructors, a list's
methods counted from either end, a dict's and a set's methods, indexing and
slicing, `del`, `in`, `+`, the set algebra, `sorted`, `min`, `max`, `sum`,
`any`, `all`, `enumerate`, `zip`, `reversed`, and `repr` in a standalone
build. Each program is held to CPython under `ppy`, `ppy run`, a standalone
binary, and emitted C and C++ under AddressSanitizer, with `ppy explain`
confirming each function went native. Containers cross the Python boundary
by copy, and a set is never walked where its order would show.
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

BIG = """
from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int


def lists(n: int) -> int:
    xs: list[int] = []
    for i in range(n):
        xs.append((i * 37) % 11)
    xs.insert(0, 99)
    xs.insert(-1, 77)
    xs.insert(1000, 55)
    total = xs[-1] * 100 + xs[0] + xs.pop() + xs.pop(0) + xs.pop(-2)
    xs.extend([5, 6, 7])
    xs.remove(5)
    total += xs.index(6) * 1000 + xs.count(3)
    ys = xs[1:5] + xs[-3:] + xs[::-2]
    ys.sort()
    ys.sort(reverse=True)
    ys.sort(key=lambda v: (v % 3, v))
    total += sum(ys) + len(ys) + ys[0] + max(ys) - min(ys)
    del ys[0]
    del ys[-1]
    ys.reverse()
    zs = ys.copy()
    zs[0] = -5
    zs[-1] = 42
    total += 1 if 42 in zs else 0
    total += 1 if 1000 in zs else 0
    total += 1 if zs == ys else 2
    zs.clear()
    return total * 3 + len(zs) + len(ys)


def nested(n: int) -> int:
    grid = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            grid[i][j] = i * j
    row = grid[2]
    row.append(7)
    flat = [v for r in grid for v in r if v % 2 == 0]
    words = ["a", "bb", "ccc"]
    lengths = [len(w) for w in words]
    pairs = [(i, i * i) for i in range(n)]
    points = [Point(i, -i) for i in range(n)]
    return sum(flat) + len(grid[2]) + sum(lengths) + pairs[-1][1] + points[3].y


def dicts(n: int) -> int:
    d: dict[str, int] = {}
    for i in range(n):
        key = "k" + str(i % 5)
        d[key] = d.get(key, 0) + i
    e = {k: v * 2 for k, v in d.items() if v > 10}
    total = len(e) + d["k1"] + d.setdefault("zz", 3) + d.pop("k0") + d.pop("nope", 11)
    for k, v in d.items():
        total += len(k) * v
    for k in d:
        total += d[k]
    total += sum(d.values()) + len(list(d.keys()))
    other = dict(d)
    other["k2"] = -1
    del other["k3"]
    total += 1 if "k3" in other else 0
    total += 1 if "k3" in d else 0
    g: dict[int, list[int]] = {}
    for i in range(n):
        g.setdefault(i % 3, []).append(i)
    total += len(g[1]) + sum(g[2])
    counts = {i: i * i for i in range(n)}
    total += counts[4] + (1 if d == other else 0)
    return total


def sets(n: int) -> int:
    s: set[int] = set()
    for i in range(n):
        s.add(i % 7)
    t = {i for i in range(3, 12)}
    u = s | t
    v = s & t
    w = s - t
    total = len(u) * 100 + len(v) * 10 + len(w)
    s.discard(3)
    s.remove(4)
    total += 1 if 3 in s else 0
    total += sum(sorted(s)) + max(t) + min(t)
    total += len(set([1, 2, 2, 3]))
    return total


def main() -> None:
    print(lists(20), nested(6), dicts(30), sets(25))


main()
"""

PARAMS = """
def histogram(words: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return counts


def top(counts: dict[str, int]) -> str:
    best = ""
    most = -1
    for w, c in counts.items():
        if c > most:
            best, most = w, c
    return best


def evens(n: int) -> list[int]:
    return [i for i in range(n) if i % 2 == 0]


def grow(xs: list[int], n: int) -> None:
    for i in range(n):
        xs.append(i)


def pipeline(n: int) -> str:
    words = ["a", "b", "a", "c", "a", "b"]
    h = histogram(words)
    e = evens(n)
    grow(e, 3)
    return top(h) + str(len(e))


def shout(word: str, times: int) -> str:
    out = ""
    for _ in range(times):
        out += word.upper()
    return out


def main() -> None:
    print(pipeline(10), shout("ab", 2))


main()
"""

ITERS = """
def walks(n: int) -> int:
    xs = [i * 3 % 7 for i in range(n)]
    ys = [float(i) / 2 for i in range(n)]
    total = 0
    for i, x in enumerate(xs):
        total += i * x
    for x, y in zip(xs, ys):
        total += int(x * y)
    for x in reversed(xs):
        total = total * 3 % 1000003 + x
    total += sum(xs) + max(xs) + min(xs) + int(sum(ys))
    total += len(sorted(xs, reverse=True))
    return total


def words(n: int) -> str:
    parts = [str(i) for i in range(n)]
    d = {p: len(p) for p in parts}
    keys = list(d.keys())
    vals = list(d.values())
    return ",".join(parts) + "|" + str(len(keys)) + str(sum(vals)) + ("ok" if "5" in d and "x" not in d else "no")


def flags(n: int) -> int:
    seen = [False] * n
    count = 0
    for i in range(0, n, 3):
        seen[i] = True
    for s in seen:
        if s:
            count += 1
    return count + (1 if any(seen) else 0) + (1 if all(seen) else 0)


def generated(n: int) -> float:
    rows = [[j for j in range(i % 4)] for i in range(n)]
    total = sum(1 for r in rows if r) + max(len(r) * 2 for r in rows)
    again = [[len(r) + k for r in rows] for k in range(3)]
    total += sum(again[2]) + len(again)
    return total + min(x - 3 for x in range(n)) + sum(x * 0.1 for x in range(n))


def main() -> None:
    print(walks(20), words(12), flags(10), generated(9))


main()
"""

ALIAS = """
def aliases(n: int) -> int:
    a = [1, 2, 3]
    b = a
    b.append(n)
    grid = [[0] * 3 for _ in range(3)]
    row = grid[1]
    row[2] = 9
    grid[0].append(5)
    table: dict[str, list[int]] = {}
    table["x"] = a
    table["x"].append(7)
    i = 100
    squares = [i * i for i in range(4)]
    kept = [[1, 2]] * 2
    kept[0].append(3)
    return len(a) * 1000 + grid[1][2] * 100 + len(grid[0]) + a[-1] + i + sum(squares) + len(kept[1])


def main() -> None:
    print(aliases(4))


main()
"""

SHOW = """
from dataclasses import dataclass


@dataclass
class P:
    x: int
    y: float


def main() -> None:
    xs = [1, 2, 3]
    ys = [0.5, 1e20, -0.0]
    names = ["a", "it's", 'q"']
    d = {"k": [1, 2], "j": []}
    pairs = [(1, 2.5), (3, 4.0)]
    grid = [[1], [2, 3]]
    flags = [True, False]
    points = [P(1, 2.0)]
    single = [(7,)]
    print(xs, ys, names)
    print(d)
    print(pairs, grid, flags, points, single)
    print(f"xs={xs} d={d}")
    print([], {})


main()
"""

CAUGHT = """
def lookups(n: int) -> int:
    d = {i: i * i for i in range(n)}
    xs = [1, 2, 3]
    total = 0
    for k in range(n + 3):
        try:
            total += d[k]
        except KeyError:
            total += 1000
    try:
        total += xs[7]
    except IndexError:
        total += 7
    try:
        xs.remove(99)
    except ValueError:
        total += 99
    try:
        xs.index(42)
    except ValueError:
        total += 42
    empty: list[int] = []
    try:
        total += empty.pop()
    except IndexError:
        total += 5
    return total


def main() -> None:
    print(lookups(10))


main()
"""

PROGRAMS = {
    "big": (BIG, ["lists", "nested", "dicts", "sets"]),
    "params": (PARAMS, ["histogram", "top", "evens", "grow", "pipeline", "shout"]),
    "iters": (ITERS, ["walks", "words", "flags", "generated"]),
    "alias": (ALIAS, ["aliases"]),
    "caught": (CAUGHT, ["lookups"]),
}

#: Programs a standalone build prints from natively, where `ppy run` keeps
#: printing in Python.
STANDALONE = {**PROGRAMS, "show": (SHOW, [])}


def _write(tmp_path: Path, source: str) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    program = tmp_path / "prog.ppy"
    program.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return program


def _run(tmp_path: Path, *args: str, text: str = "") -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    return subprocess.run(
        [sys.executable, *args], cwd=tmp_path, input=text, capture_output=True, text=True,
        check=False, env=env,
    )  # fmt: skip


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
@pytest.mark.parametrize("name", sorted(STANDALONE))
def test_a_standalone_binary_agrees(tmp_path: Path, name: str):
    expected = _expected(tmp_path, STANDALONE[name][0])
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
@pytest.mark.parametrize("name", sorted(STANDALONE))
def test_emitted_source_frees_everything_once(
    tmp_path: Path, name: str, language: str, unsafe: bool
):
    compiler = c_compiler() if language == "c" else _CXX
    if compiler is None or not _sanitizes(compiler, tmp_path):
        pytest.skip(f"no {language} compiler with AddressSanitizer")
    expected = _expected(tmp_path, STANDALONE[name][0])
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


GUARDS = """
import ppy


def index(n: int) -> int:
    xs = [1, 2, 3]
    return xs[n]


def key(n: int) -> int:
    d = {1: 2}
    return d[n]


def removed(n: int) -> int:
    xs = [1, 2, 3]
    xs.remove(n)
    return len(xs)


def popped(n: int) -> int:
    xs: list[int] = []
    for i in range(n):
        xs.append(i)
    return xs.pop()


def main() -> None:
    which = ppy.input[int]()
    n = ppy.input[int]()
    if which == 0:
        print(index(n))
    elif which == 1:
        print(key(n))
    elif which == 2:
        print(removed(n))
    else:
        print(popped(n))


main()
"""

#: (function, argument) -> what a standalone binary prints last, and its status.
GUARDED = {
    (0, -1): ("3", 0),
    (0, 5): ("IndexError: list index out of range", 1),
    (0, -4): ("IndexError: list index out of range", 1),
    (1, 3): ("KeyError: 3", 1),
    (2, 9): ("ValueError: list.remove(x): x not in list", 1),
    (3, 0): ("IndexError: pop from empty list", 1),
}


@requires_standalone
def test_a_standalone_binary_says_what_cpython_raises(tmp_path: Path):
    _write(tmp_path, GUARDS)
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "--standalone", "prog.ppy", "-o", "dist")
    assert built.returncode == 0, built.stderr
    for (which, argument), (last, status) in GUARDED.items():
        text = f"{which}\n{argument}\n"
        expected = _run(tmp_path, "prog.ppy", text=text)
        shown = (expected.stdout + expected.stderr).strip().splitlines()[-1]
        assert shown == last, (which, argument, shown)
        ran = subprocess.run(
            [str(tmp_path / "dist" / "prog")], input=text, capture_output=True, text=True,
            check=False,
        )  # fmt: skip
        assert ran.returncode == status, (which, argument, ran.stderr)
        assert (ran.stdout + ran.stderr).strip().splitlines()[-1] == last


BOUNDARY = """
def histogram(words: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for w in words:
        for ch in w:
            counts[ch] = counts.get(ch, 0) + 1
    return counts


def top(counts: dict[str, int]) -> str:
    best = ""
    most = -1
    for w, c in counts.items():
        score = c
        for ch in w:
            if ch == "q":
                score += 1
        if score > most:
            best, most = w, score
    return best


def once(words: list[str]) -> int:
    total = 0
    for w in words:
        total += len(w)
    return total


def grow(xs: list[int], n: int) -> None:
    for i in range(n):
        xs.append(i)


def same(xs: list[list[int]]) -> list[int]:
    for row in xs:
        total = 0
        for v in row:
            total += v * v + 1
        row.append(total % 10 + len(row))
    return xs[0]


def kept(s: set[int], n: int) -> int:
    t = {i for i in range(n)}
    return len(s & t)
"""


@requires_llvm
@requires_cc
def test_python_calls_container_functions_natively(tmp_path: Path):
    """Python's containers cross by copy: strings included where the body does
    more than one pass over them, a write through an argument reaches the
    caller's object, a returned argument is the caller's
    own, and an argument of another type runs the Python body."""
    from ppy_compiler.backend.llvm import _collect
    from ppy_compiler.backend.llvm.jit import JitEngine
    from ppy_compiler.backend.llvm.runtime import bind
    from ppy_compiler.driver.pipeline import analyze_paths, open_project
    from ppy_runtime.collections import library_path

    path = _write(tmp_path, BOUNDARY)
    bundle = analyze_paths(open_project(path), [path], backend="llvm")
    module = _collect(bundle)["prog"]
    runtime = library_path()
    assert runtime is not None
    engine = JitEngine(opt_level=2).open()
    engine.load_library(str(runtime))
    engine.add(module.ir)
    engine.finalize()
    fell: list[str] = []

    def native(name: str):  # type: ignore[no-untyped-def]
        lowered = module.functions[f"prog.{name}"]
        assert lowered.exposed, (name, lowered.exposure_reason)
        signature = lowered.boundary or lowered.signature

        def fallback(*arguments: object) -> None:
            del arguments
            fell.append(name)

        return bind(signature, engine.address(signature.symbol), fallback)

    histogram = native("histogram")
    counted = histogram.wrapper(["xy", "é", "x"])
    assert counted == {"x": 2, "y": 1, "é": 1} and type(counted) is dict
    assert histogram.calls == 1
    assert native("top").wrapper({"q": 3, "r": 5, "": 4}) == "r"
    # One pass over a list of strings is cheaper in Python: borrowing each
    # string is cheap, holding it natively costs what `len` saves.
    once = module.functions["prog.once"]
    assert not once.exposed and "copying the collections" in once.exposure_reason
    xs = [1, 2]
    grow = native("grow")
    assert grow.wrapper(xs, 2) is None and xs == [1, 2, 0, 1] and grow.calls == 1
    inner = [7]
    rows = [inner, [8, 9]]
    assert native("same").wrapper(rows) is inner
    assert rows == [[7, 1], [8, 9, 9]] and rows[0] is inner
    assert native("kept").wrapper({1, 5, 9}, 6) == 2
    assert histogram.wrapper(["x", 3]) is None
    assert fell == ["histogram"] and histogram.fallbacks == 1


@requires_llvm
@requires_cc
def test_a_set_of_strings_walked_where_its_order_shows_stays_in_python(tmp_path: Path):
    """A set of ints walks in CPython's order natively; a string's hash changes
    from one process to the next, so a set of strings stays in Python."""
    source = """
    def named(n: int) -> str:
        s = {str(i * 7 % 5) for i in range(n)}
        out = ""
        for x in s:
            out += x
        return out


    def walked(n: int) -> int:
        s = {i * 7 % 5 for i in range(n)}
        total = 0
        for x in s:
            total = total * 10 + x
        return total


    def ordered(n: int) -> int:
        s = {i * 7 % 5 for i in range(n)}
        return sum(sorted(s)) + max(s) + len(s)


    def main() -> None:
        print(sorted(named(9)), walked(9), ordered(9))


    main()
    """
    expected = _expected(tmp_path, source)
    done = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert _output(done).strip() == expected
    named = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.named")
    assert "hash order" in named.stdout, named.stdout
    walked = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.walked")
    assert "llvm backend: native" in walked.stdout, walked.stdout
    ordered = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.ordered")
    assert "llvm backend: native" in ordered.stdout, ordered.stdout


@requires_llvm
@requires_cc
def test_get_without_a_default_binds_a_number_or_none(tmp_path: Path):
    """`v = d.get(k)` of a dict of numbers binds `v` as a number or `None`,
    natively; `d.get(k)` used any other way (`or`, a dict of strings) keeps its
    function in Python, with a reason, and the rest of the module lowers (it
    raised `IndexError` in the lowering before)."""
    source = """
    def look(d: dict[int, int], k: int) -> int:
        v = d.get(k)
        if v is None:
            return -1
        return v


    def named(d: dict[str, str], k: str) -> str:
        return d.get(k) or "?"


    def total(d: dict[int, int]) -> int:
        s = 0
        for k in range(10):
            s += d.get(k, 0)
        return s


    def main() -> None:
        d = {1: 10, 2: 20}
        print(look(d, 1), look(d, 3), named({"a": "b"}, "a"), named({}, "x"), total(d))


    main()
    """
    expected = _expected(tmp_path, source)
    for args in (["-m", "ppy_compiler", "prog.ppy"], ["-m", "ppy_compiler", "run", "prog.ppy"]):
        done = _run(tmp_path, *args)
        assert done.returncode == 0, done.stderr
        assert _output(done).strip() == expected, args
    look = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.look")
    assert "llvm backend: native" in look.stdout, look.stdout
    named = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.named")
    assert "llvm backend: boxed" in named.stdout, named.stdout
    total = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.total")
    assert "llvm backend: native" in total.stdout, total.stdout
    summary = _run(tmp_path, "-m", "ppy_compiler", "explain", "--summary")
    assert summary.returncode == 0, summary.stderr
