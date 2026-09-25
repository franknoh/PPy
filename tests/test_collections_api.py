"""The collections' wider API: one meaning on every path, and a Python boundary.

`extend`, `insert`, slices, `==`, `+`, `sort(key=...)`, the map views and
`setdefault`, the set algebra, a tree's `between`, a heap's `pushpop`, and
`enumerate`, `zip`, `reversed`, `sorted`, `min`, `max`, `sum` over a
collection: each program here is held to the reference classes under `ppy`,
`ppy run`, a standalone binary, and emitted C and C++ under AddressSanitizer,
with `ppy explain` confirming every function went native.

A native function that takes or returns collections is called from Python
through a boundary that copies them: the calls are counted, identity is kept
(a returned argument is the caller's object), and writes come back.
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

SEQUENCES = """
from dataclasses import dataclass

from ppy import Deque, LinkedList, Vec


@dataclass(order=True)
class Job:
    rank: int
    size: float


def weight(job: Job) -> float:
    return job.size * 2.0 - job.rank


def edits(n: int) -> int:
    v = Vec[int]([5, 3, 8])
    v.extend(v)
    v.extend(range(n, 0, -3))
    v.extend([1, 2])
    v.insert(0, 42)
    v.insert(len(v), 7)
    first: int = v.pop(0)
    last: int = v.pop(len(v) - 1)
    v.remove(8)
    code: int = first * 100 + last + v.index(3) * 7 + v.count(5) * 1000
    if 2 in v and 99 not in v:
        code += 1
    return code * 10000 + len(v)


def slices(n: int) -> int:
    v = Vec[int](range(n))
    total: int = 0
    for x in v[1:4]:
        total = total * 3 + x
    for x in v[::-2]:
        total = total * 3 + x
    tail = v[-3:]
    empty = v[5:1]
    wide = v[-100:100:3]
    return total + len(tail) * 1000 + len(empty) + sum(wide) * 7


def compare(n: int) -> int:
    a = Vec[int](range(n))
    b = a.copy()
    c = a + b
    code: int = 0
    if a == b:
        code += 1
    b.push(1)
    if a != b:
        code += 2
    rows = Vec[Vec[int]](2)
    rows[0].push(1)
    shallow = rows.copy()
    shallow[0].push(5)
    if rows == shallow and len(rows[0]) == 2:
        code += 4
    return code * 1000 + len(c)


def sorting(n: int) -> int:
    v = Vec[int]()
    for i in range(n):
        v.push((i * 7) % 5)
    v.sort(reverse=True)
    out: int = 0
    for x in v:
        out = out * 5 + x
    jobs = Vec[Job]()
    for i in range(n):
        jobs.push(Job(i % 3, float(i % 4)))
    jobs.sort(key=lambda j: j.size)
    order: int = 0
    for job in jobs:
        order = (order * 7 + job.rank) % 1000003
    jobs.sort(key=weight, reverse=True)
    for job in jobs:
        order = (order * 5 + job.rank) % 1000003
    words = Vec[tuple[int, int]]()
    for i in range(n):
        words.push((i % 4, n - i))
    words.sort(key=lambda w: (w[0], -w[1]))
    head = words[0]
    return ((out * 1000003 + order) * 10 + head[0]) * 100 + head[1]


def walks(n: int) -> float:
    v = Vec[float]()
    for i in range(n):
        v.push(i / 4)
    total: float = 0.0
    for i, x in enumerate(v, 1):
        total += i * x
    for x in reversed(v):
        total = total * 0.5 + x
    w = Vec[int](range(3))
    for a, b in zip(v, w):
        total += a * b
    for i, x in enumerate(sorted(w, reverse=True)):
        total += i - x
    return total + min(v) + max(v) + sum(v) + min(w) + max(w) + sum(w)


def fsum() -> float:
    v = Vec[float]([1e16, 1.0, -1e16, 0.1, 0.2])
    return sum(v)


def queues(n: int) -> int:
    d = Deque[int](range(5))
    d.rotate(2)
    d.rotate(-7)
    d.extendleft([10, 11])
    d.extend(range(3))
    d.insert(2, 99)
    d.remove(99)
    code: int = 0
    for x in d:
        code = code * 3 + x
    for x in reversed(d):
        code = code * 2 + x
    e = d.copy()
    e.push_back(1)
    joined = d + e
    if d != e and 11 in d:
        code += 1
    return code + len(joined) + d.index(10) + d.count(1) + n


def links(n: int) -> int:
    l = LinkedList[int](range(n))
    l.extend(l)
    m = LinkedList[int](range(n))
    m.extend(range(n))
    code: int = 0
    for x in reversed(l):
        code = code * 3 + x
    if l == m and (n - 1) in l:
        code += 1
    return code


def main() -> None:
    print(edits(20), slices(12), compare(6))
    print(sorting(11))
    print(walks(10), fsum(), queues(4), links(4))


main()
"""

MAPS = """
from ppy import HashMap, HashSet, TreeMap, TreeSet, Vec


def views(n: int) -> int:
    m = HashMap[int, float]()
    for i in range(n):
        m[(i * 7) % 11] = i / 2
    m.pop(3)
    code: float = 0.0
    for k, v in m.items():
        code = code * 1.5 + k - v
    for k in m.keys():
        code += k
    for v in m.values():
        code -= v
    for k in reversed(m):
        code = code * 0.5 + k
    for i, (k, v) in enumerate(m.items()):
        code += i * k * v
    return int(code * 1000) + int(sum(m.values())) + max(m.keys()) + min(m)


def defaults(n: int) -> int:
    groups = HashMap[int, Vec[int]]()
    for i in range(n):
        groups.setdefault(i % 4, Vec[int]()).push(i)
    spare = groups.get(9, Vec[int]())
    spare.push(100)
    kept = groups.get(1, Vec[int]())
    kept.push(-1)
    taken = groups.pop(2, Vec[int]())
    missing = groups.pop(7, Vec[int]())
    counts = HashMap[int, int]()
    for i in range(n):
        counts[i % 3] = counts.get(i % 3, 0) + i
    code: int = len(spare) * 1000 + len(groups[1]) * 100 + len(taken) * 10 + len(missing)
    code = code * 100 + counts.setdefault(5, 7) + counts.pop(0, -1) + counts.pop(0, -1)
    return code


def merging(n: int) -> int:
    a = HashMap[int, int]()
    b = HashMap[int, int]()
    for i in range(n):
        a[i] = i
        b[i + n // 2] = -i
    c = a.copy()
    c.update(b)
    code: int = 0
    for k, v in c.items():
        code = (code * 31 + k * 7 + v) % 1000003
    if a != c and a == a.copy():
        code += 1
    grid = HashMap[tuple[int, int], Vec[int]]()
    grid[(0, 1)] = Vec[int]([1, 2])
    other = grid.copy()
    other.update(grid)
    if grid == other:
        code += 10
    return code


def sets(n: int) -> int:
    a = HashSet[int](range(n))
    b = HashSet[int]([3, 5, 100, 2, 3])
    code: int = 0
    for part in (a | b):
        code = (code * 7 + part) % 1000003
    for part in (b | a):
        code = (code * 7 + part) % 1000003
    for part in a & b:
        code = code * 3 + part
    for part in a - b:
        code = code * 3 + part
    for part in (a ^ b):
        code = (code * 5 + part) % 1000003
    u = a.union(b)
    i = b.intersection(a)
    d = b.difference(a)
    x = a.symmetric_difference(b)
    code += len(u) + len(i) * 10 + len(d) * 100 + len(x) * 1000
    if HashSet[int]([3, 5]).issubset(a) and a.issuperset(HashSet[int]([1])):
        code += 7
    if not a.isdisjoint(b) and HashSet[int]([-1]).isdisjoint(a):
        code += 70
    b.update(a)
    if b == (a | HashSet[int]([100])):
        code += 700
    return code


def trees(n: int) -> int:
    t = TreeSet[int]()
    for i in range(n):
        t.add((i * 37) % 101)
    code: int = 0
    for k in t.between(10, 60):
        code = code * 3 + k
    for k in reversed(t):
        code = (code * 7 + k) % 1000003
    low: int = t.pop_min()
    high: int = t.pop_max()
    m = TreeMap[int, float]()
    for i in range(n):
        m[(i * 13) % 17] = i * 0.25
    k1, v1 = m.pop_min()
    k2, v2 = m.pop_max()
    for k, v in m.items():
        code = (code + k * int(v * 4)) % 1000003
    for k in reversed(m):
        code = (code * 3 + k) % 1000003
    s = TreeSet[int](t)
    s.update(TreeSet[int]([500, -5]))
    both = s & t
    return code + low + high * 1000 + k1 + int(v1) + k2 * 10 + int(v2 * 4) + len(both) + s.min() + s.max()


def main() -> None:
    print(views(15), defaults(13), merging(10))
    print(sets(8), trees(40))


main()
"""

HEAPS = """
from dataclasses import dataclass

from ppy import Heap, MaxHeap, Vec


@dataclass(order=True)
class Task:
    due: int
    cost: float


def heaps(n: int) -> int:
    h = Heap[int](range(n, 0, -1))
    code: int = h.pushpop(0) * 1000 + h.pushpop(50)
    code = code * 100 + h.replace(7)
    ordered = h.to_sorted()
    for x in ordered:
        code = (code * 3 + x) % 1000003
    top = MaxHeap[int]([4, 9, 1, 9, 3])
    down = top.to_sorted()
    code = code * 10 + down[0] + down[len(down) - 1]
    copied = top.copy()
    copied.push(100)
    return code + len(top) * 1000 + copied.peek()


def tasks(n: int) -> float:
    work = Heap[Task]()
    for i in range(n):
        work.push(Task((i * 5) % 7, i * 0.5))
    first = work.pushpop(Task(-1, 9.0))
    second = work.replace(Task(3, 1.5))
    total: float = first.cost + second.cost * 10
    while work:
        task = work.pop()
        total = total * 0.9 + task.due + task.cost
    return total


def rows(n: int) -> int:
    grid = Heap[tuple[int, int]]()
    for i in range(n):
        grid.push((i % 3, -i))
    kept = grid.to_sorted()
    code: int = 0
    for a, b in kept:
        code = code * 5 + a - b
    return code


def main() -> None:
    print(heaps(12), tasks(9), rows(6))


main()
"""

GUARDS = """
from ppy import HashMap, Vec


def nan_same(n: int) -> int:
    big: float = 1e308 + n
    odd: float = big * 10.0 - big * 10.0
    v = Vec[float]()
    for i in range(n):
        v.push(odd if i == 1 else i)
    w = v.copy()
    code: int = 1 if v == w else 0
    return code * 10 + v.index(odd) + v.count(odd) * 100


def missing(n: int) -> int:
    v = Vec[int](range(n))
    v.remove(n + 1)
    return len(v)


def empty_sum(n: int) -> float:
    v = Vec[float]()
    for i in range(n):
        v.push(i * 0.5)
    return sum(v)


def empty_min(n: int) -> int:
    v = Vec[int](range(n))
    return min(v)


def changed(n: int) -> int:
    m = HashMap[int, int]()
    for i in range(n):
        m[i] = i
    for k in m.keys():
        if k == 1:
            m[100] = 1
    return len(m)


def bad_slice(n: int) -> int:
    v = Vec[int](range(n))
    return len(v[::n - n])


def main() -> None:
    print(nan_same(3), empty_sum(0), empty_sum(3))
    for call in range(4):
        try:
            if call == 0:
                print(missing(3))
            elif call == 1:
                print(empty_min(0))
            elif call == 2:
                print(changed(3))
            else:
                print(bad_slice(2))
        except (ValueError, RuntimeError) as error:
            print(type(error).__name__)


main()
"""

BOUNDARY = """
from ppy import Deque, HashMap, Heap, TreeSet, Vec


def total(v: Vec[int]) -> int:
    s: int = 0
    for x in v:
        s += x
    return s


def fill(v: Vec[Vec[int]], n: int) -> None:
    for i in range(n):
        v[i % len(v)].push(i)


def grown(v: Vec[float], n: int) -> Vec[float]:
    for i in range(n):
        v.push(i * 0.5)
    return v


def counts(values: Vec[int]) -> HashMap[int, int]:
    found = HashMap[int, int]()
    for x in values:
        found[x % 3] = found.get(x % 3, 0) + 1
    return found


def first_row(rows: Vec[Vec[int]]) -> Vec[int]:
    for row in rows:
        row.push(9)
    return rows[0]


def shuffle(q: Deque[tuple[int, float]], h: Heap[int], t: TreeSet[int]) -> int:
    q.rotate(1)
    h.push(-5)
    t.add(42)
    return len(q) + h.peek() + t.max()


def main() -> None:
    v = Vec[int]([1, 2, 3])
    print(total(v), total(Vec[int]()))
    rows = Vec[Vec[int]](3)
    fill(rows, 7)
    print(rows)
    f = Vec[float]([1.0])
    g = grown(f, 3)
    print(g is f, f)
    print(counts(Vec[int](range(10))))
    same = Vec[int]([7])
    pair = Vec[Vec[int]]()
    pair.push(same)
    pair.push(same)
    head = first_row(pair)
    print(head is same, same, pair)
    q = Deque[tuple[int, float]]([(1, 0.5), (2, 1.5)])
    h = Heap[int]([3, 1, 2])
    t = TreeSet[int]([5, 1])
    print(shuffle(q, h, t), list(q), h.to_sorted(), list(t))


main()
"""

PROGRAMS = {
    "sequences": (
        SEQUENCES,
        "72150014 19894 7012\n48319645344847011\n20.3251953125 1.3 290504388 8365",
        ["edits", "slices", "compare", "sorting", "walks", "fsum", "queues", "links"],
    ),
    "maps": (
        MAPS,
        "981284 143036 5850\n9052 882468",
        ["views", "defaults", "merging", "sets", "trees"],
    ),
    "heaps": (HEAPS, "1362860 39.454276886 10089", ["heaps", "tasks", "rows"]),
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


@requires_llvm
@requires_cc
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_every_path_agrees_and_each_function_goes_native(tmp_path: Path, name: str):
    source, expected, natives = PROGRAMS[name]
    _write(tmp_path, source)
    for args in (
        ["prog.ppy"],
        ["-m", "ppy_compiler", "prog.ppy"],
        ["-m", "ppy_compiler", "run", "prog.ppy"],
    ):
        done = _run(tmp_path, *args)
        assert done.returncode == 0, done.stderr
        assert _output(done).strip() == expected, args
    for function in natives:
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


@requires_standalone
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_a_standalone_binary_agrees(tmp_path: Path, name: str):
    source, expected, _ = PROGRAMS[name]
    program = _write(tmp_path, source)
    built = _run(
        tmp_path, "-m", "ppy_compiler", "build", "--standalone", program.name, "-o", "dist"
    )
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
    source, expected, _ = PROGRAMS[name]
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
        capture_output=True,
        text=True,
        check=False,
        env={"ASAN_OPTIONS": "detect_leaks=1"},
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == expected


@requires_llvm
@requires_cc
def test_what_python_raises_native_code_hands_back(tmp_path: Path):
    """A NaN compared, a missing element, an empty `min`, a map changed while
    walked, a zero slice step: native code hands each call back to Python,
    which raises what CPython raises."""
    _write(tmp_path, GUARDS)
    expected = "111 0 1.5\nValueError\nValueError\nRuntimeError\nValueError"
    for args in (
        ["prog.ppy"],
        ["-m", "ppy_compiler", "prog.ppy"],
        ["-m", "ppy_compiler", "run", "prog.ppy"],
    ):
        done = _run(tmp_path, *args)
        assert done.returncode == 0, done.stderr
        assert _output(done).strip() == expected, args
    for function in ("nan_same", "missing", "empty_sum", "empty_min", "changed", "bad_slice"):
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


BOUNDARY_OUTPUT = """6 0
Vec([Vec([0, 3, 6]), Vec([1, 4]), Vec([2, 5])])
True Vec([1.0, 0.0, 0.5, 1.0])
HashMap({0: 4, 1: 3, 2: 3})
True Vec([7, 9, 9]) Vec([Vec([7, 9, 9]), Vec([7, 9, 9])])
39 [(2, 1.5), (1, 0.5)] Vec([-5, 1, 2, 3]) [1, 5, 42]"""


@requires_llvm
@requires_cc
def test_python_calls_collection_functions_natively(tmp_path: Path):
    """Each function is called from Python through the boundary and counted as a
    native call; a returned argument is the caller's object, a write through an
    element of an argument reaches the caller, and an argument of another type
    runs the Python body."""
    from ppy import HashMap, Vec
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
    fallbacks: list[str] = []

    def native(name: str):  # type: ignore[no-untyped-def]
        lowered = module.functions[f"prog.{name}"]
        assert lowered.exposed, (name, lowered.exposure_reason)

        def fallback(*arguments: object) -> None:
            del arguments
            fallbacks.append(name)

        return bind(lowered.signature, engine.address(lowered.signature.symbol), fallback)

    total = native("total")
    assert total.wrapper(Vec[int]([1, 2, 3])) == 6
    assert total.wrapper(Vec[float]([1.0])) is None
    assert fallbacks == ["total"] and total.calls == 1 and total.fallbacks == 1

    rows = Vec[Vec[int]](2)
    fill = native("fill")
    assert fill.wrapper(rows, 5) is None
    assert repr(rows) == "Vec([Vec([0, 2, 4]), Vec([1, 3])])" and fill.calls == 1

    counted = native("counts").wrapper(Vec[int](range(7)))
    assert type(counted) is HashMap[int, int]
    assert repr(counted) == "HashMap({0: 3, 1: 2, 2: 2})"

    same = Vec[int]([7])
    pair = Vec[Vec[int]]()
    pair.push(same)
    pair.push(same)
    first = native("first_row")
    assert first.wrapper(pair) is same
    assert repr(pair) == "Vec([Vec([7, 9, 9]), Vec([7, 9, 9])])"
    assert first.calls == 1 and first.fallbacks == 0

    grown = Vec[float]([1.0])
    assert native("grown").wrapper(grown, 2) is grown
    assert repr(grown) == "Vec([1.0, 0.0, 0.5])"


@requires_llvm
@requires_cc
def test_the_boundary_program_agrees_under_run_and_build(tmp_path: Path):
    _write(tmp_path, BOUNDARY)
    plain = _run(tmp_path, "prog.ppy")
    assert plain.stdout.strip() == BOUNDARY_OUTPUT
    ran = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert ran.returncode == 0, ran.stderr
    assert "W2004" not in ran.stderr
    assert _output(ran).strip() == BOUNDARY_OUTPUT
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "prog.ppy", "-o", "out")
    assert built.returncode == 0, built.stderr
    launched = subprocess.run(
        [str(tmp_path / "out" / "prog")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert launched.returncode == 0, launched.stderr
    assert launched.stdout.strip() == BOUNDARY_OUTPUT


def test_the_reference_classes_answer_as_the_builtins_do():
    from ppy import Deque, HashMap, HashSet, Heap, LinkedList, MaxHeap, TreeMap, TreeSet, Vec

    v = Vec[int]([3, 1, 2])
    v.extend(v)
    assert list(v) == [3, 1, 2, 3, 1, 2]
    assert list(v[1:4]) == [1, 2, 3] and list(v[::-2]) == [2, 3, 1]
    assert v.index(2) == 2 and v.count(3) == 2 and 2 in v
    v.sort(key=lambda x: -x)
    assert list(v) == [3, 3, 2, 2, 1, 1]
    assert v == Vec[int]([3, 3, 2, 2, 1, 1]) and v != Vec[int]()
    assert v != Deque[int](v)
    with pytest.raises(ValueError):
        v.remove(7)
    nan = float("nan")
    w = Vec[float]([nan])
    assert w.index(nan) == 0 and nan in w and w.count(nan) == 1
    h = Heap[int]([5, 3, 8, 1])
    assert list(h.to_sorted()) == [1, 3, 5, 8]
    assert h.pushpop(0) == 0 and h.replace(10) == 1 and h.pop() == 3
    assert list(MaxHeap[int]([1, 5, 3]).to_sorted()) == [5, 3, 1]
    m = HashMap[int, float]()
    m[1] = 2
    assert list(m.items()) == [(1, 2.0)] and m.setdefault(5, 6) == 6.0
    assert m.pop(9, 0) == 0.0 and list(reversed(m)) == [5, 1]
    a, b = HashSet[int]([1, 2, 3]), HashSet[int]([2, 3, 4])
    assert list(a | b) == [1, 2, 3, 4] and list(a ^ b) == [1, 4] and list(a - b) == [1]
    assert HashSet[int]([2]).issubset(a) and not a.isdisjoint(b)
    t = TreeSet[int]([5, 1, 3, 9])
    assert list(t.between(2, 9)) == [3, 5] and t.pop_min() == 1 and t.pop_max() == 9
    tm = TreeMap[int, float]()
    tm[2] = 1
    tm[1] = 5
    assert tm.pop_min() == (1, 5.0)
    d = Deque[int]([1, 2, 3])
    d.rotate(1)
    d.extendleft([7, 8])
    assert list(d) == [8, 7, 3, 1, 2]
    assert list(reversed(LinkedList[int]([1, 2]))) == [2, 1]
    # A bare `HashSet()`, whose type the checker infers, combines like any other.
    assert list(HashSet() | HashSet[int]([100])) == [100]
    with pytest.raises(TypeError):
        HashSet[int]([1]) | TreeSet[int]([2])  # pylint: disable=expression-not-assigned


STANDALONE_FAILURE = """
import ppy
from ppy import Vec


def main() -> None:
    n: int = ppy.input[int]()
    v = Vec[int](range(n))
    {body}
    print(len(v))


main()
"""


@requires_standalone
@pytest.mark.parametrize(
    ("body", "raised"),
    [
        ("v.remove(n + 5)", "ValueError: 8 is not in the Vec"),
        ("v.insert(n + 2, 1)", "IndexError: insert at 5 is out of range for length 3"),
        ("v.pop(n + 1)", "IndexError: index 4 is out of range for length 3"),
        ("print(min(Vec[int]()))", "ValueError: min() iterable argument is empty"),
        ("print(len(v[:: n - n]))", "ValueError: slice step cannot be zero"),
    ],
)
def test_a_standalone_binary_says_what_cpython_raises(tmp_path: Path, body: str, raised: str):
    _write(tmp_path, STANDALONE_FAILURE.format(body=body))
    plain = subprocess.run(
        [sys.executable, "prog.ppy"],
        cwd=tmp_path,
        input="3\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert plain.stderr.strip().splitlines()[-1] == raised
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "--standalone", "prog.ppy", "-o", "dist")
    assert built.returncode == 0, built.stderr
    ran = subprocess.run(
        [str(tmp_path / "dist" / "prog")], input="3\n", capture_output=True, text=True, check=False
    )
    assert ran.returncode == 1
    assert ran.stderr.strip().splitlines()[-1] == raised
