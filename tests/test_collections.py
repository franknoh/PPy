"""The `ppy` collections: one meaning on every path.

The reference in `ppy._collections` is what the types mean. Native code holds
each as a handle into the C runtime in `ppy_runtime.collections`, and the
programs here are held to the reference under `ppy`, `ppy run`, a standalone
binary, and emitted C and C++, with `ppy explain` confirming the functions
went native rather than agreeing by staying in Python.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import ppy
from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler, standalone_toolchain_status

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")
_standalone, _standalone_detail = standalone_toolchain_status()
requires_standalone = pytest.mark.skipif(
    not (llvm_available() and _standalone), reason=f"no standalone toolchain: {_standalone_detail}"
)
_CXX = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)

PROGRAM = """
from ppy import Deque, Heap, MaxHeap, Vec


def total(v: Vec[int]) -> int:
    s: int = 0
    for x in v:
        s += x
    return s


def fill(v: Vec[int], n: int) -> None:
    for i in range(n):
        v.push((i * 37) % 11)


def sorted_code(n: int) -> int:
    v = Vec[int]()
    fill(v, n)
    if n > 50:
        return total(v)
    v.sort()
    acc: int = 0
    for x in v:
        acc = (acc * 7 + x) % 1000003
    v.reverse()
    return acc + v.last() * 100 + v.pop() + len(v)


def empty_pop(n: int) -> int:
    v = Vec[int]()
    if n > 0:
        v.push(n)
    return v.pop()


def grows_while_iterating() -> int:
    v = Vec[int]()
    v.push(1)
    count: int = 0
    for x in v:
        if x < 5:
            v.push(x + 1)
        count += 1
    return count


def at(n: int) -> int:
    v = Vec[int](3)
    v[1] = 7
    return v[n]


def breadth(n: int) -> int:
    dist = Vec[int](n)
    for i in range(n):
        dist[i] = -1
    dist[0] = 0
    q = Deque[int]()
    q.push_back(0)
    while q:
        u: int = q.pop_front()
        for k in range(2):
            w: int = (u + 1) % n if k == 0 else (u * 7 + 3) % n
            if dist[w] < 0:
                dist[w] = dist[u] + 1
                q.push_back(w)
    return total(dist)


def heaps(n: int) -> float:
    low = Heap[float]()
    high = MaxHeap[int]()
    for i in range(n):
        low.push((i * 7) % 11)
        high.push((i * 5) % 13)
    best: float = 0.0
    while len(low) > 1:
        best += low.pop() * 2.0 - low.peek()
    if not low:
        return -1.0
    return best + high.pop() + high.peek()


def floats(n: int) -> float:
    d = Deque[float]()
    d.push_front(1)
    d.push_back(2.5)
    for i in range(n):
        d.push_front(d.back() / 2.0)
        d.pop_back()
    d[0] = d[0] + 0.25
    return d.front() + d[len(d) - 1]


def main() -> None:
    print(sorted_code(20), sorted_code(60), grows_while_iterating(), at(1))
    print(breadth(1000), heaps(30), floats(3), Vec[float](2)[1])
    for probe in (0, 4):
        try:
            print(empty_pop(probe))
        except IndexError as error:
            print("IndexError", error)
    for probe in (-1, 3):
        try:
            print(at(probe))
        except IndexError as error:
            print("IndexError", error)


main()
"""

NATIVE = [
    "total",
    "fill",
    "sorted_code",
    "empty_pop",
    "grows_while_iterating",
    "at",
    "breadth",
    "heaps",
    "floats",
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


def test_the_reference_holds_its_element_type_and_its_index_rules():
    v = ppy.Vec[float](2)
    v.push(3)
    assert [repr(x) for x in v] == ["0.0", "0.0", "3.0"]
    assert ppy.Vec[float] is ppy.Vec[float]
    with pytest.raises(IndexError):
        _ = v[-1]
    with pytest.raises(IndexError):
        ppy.Deque[int]().pop_front()
    h = ppy.MaxHeap[int]()
    for x in (3, 1, 4, 1, 5):
        h.push(x)
    assert [h.pop() for _ in range(5)] == [5, 4, 3, 1, 1]
    with pytest.raises(TypeError):
        ppy.Vec[bytes]()
    with pytest.raises(ValueError):
        ppy.Vec[int](-1)


@requires_llvm
def test_collections_agree_on_every_path_and_go_native(tmp_path: Path):
    _write(tmp_path, PROGRAM)
    outputs = [
        _run(tmp_path, "prog.ppy"),
        _run(tmp_path, "-m", "ppy_compiler", "prog.ppy"),
        _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy"),
    ]
    for done in outputs:
        assert done.returncode == 0, done.stderr
    assert outputs[0].stdout == outputs[1].stdout == outputs[2].stdout
    assert outputs[0].stdout.splitlines()[2:] == [
        "IndexError pop from an empty Vec",
        "4",
        "IndexError index -1 is out of range for length 3",
        "IndexError index 3 is out of range for length 3",
    ]
    for name in NATIVE:
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{name}")
        assert "llvm backend: native" in explained.stdout, (name, explained.stdout)


STANDALONE = """
import ppy
from ppy import Deque, Heap, Vec


def fill(v: Vec[int], n: int) -> None:
    for i in range(n):
        v.push((i * 2654435761) % 1000003)


def main() -> None:
    n: int = ppy.scan[int]()
    v = Vec[int]()
    fill(v, n)
    v.sort()
    q = Deque[float]()
    h = Heap[int]()
    for x in v:
        q.push_front(x / 2)
        h.push(-x)
    print(v[0], v.last(), len(v), q.front(), q.back(), -h.pop(), v.pop())


main()
"""


@requires_standalone
def test_a_standalone_binary_holds_collections(tmp_path: Path):
    program = _write(tmp_path, STANDALONE)
    built = _run(
        tmp_path, "-m", "ppy_compiler", "build", "--standalone", program.name, "-o", "dist"
    )
    assert built.returncode == 0, built.stderr
    binary = str(tmp_path / "dist" / "prog")
    expected = _run(tmp_path, program.name, text="1000\n")
    native = subprocess.run([binary], input="1000\n", capture_output=True, text=True, check=False)
    assert native.returncode == 0, native.stderr
    assert native.stdout == expected.stdout
    empty = subprocess.run([binary], input="0\n", capture_output=True, text=True, check=False)
    python = _run(tmp_path, program.name, text="0\n")
    # No Python to fall back to: the binary says what CPython says, and stops as it does.
    assert empty.returncode == python.returncode == 1
    assert empty.stderr.strip() == python.stderr.strip().splitlines()[-1]


@requires_llvm
@pytest.mark.parametrize("keyed", [False, True])
@pytest.mark.parametrize("unsafe", [False, True])
@pytest.mark.parametrize("language", ["c", "cpp"])
def test_emitted_source_carries_the_collections(
    tmp_path: Path, language: str, unsafe: bool, keyed: bool
):
    compiler = c_compiler() if language == "c" else _CXX
    if compiler is None:
        pytest.skip(f"no {language} compiler on PATH")
    program = _write(tmp_path, KEYED_STANDALONE if keyed else STANDALONE)
    source = tmp_path / f"prog.{language}"
    flags = ["--unsafe"] if unsafe else []
    emitted = _run(
        tmp_path,
        "-m",
        "ppy_compiler",
        "emit",
        language,
        "--standalone",
        *flags,
        program.name,
        "-o",
        source.name,
    )
    assert emitted.returncode == 0, emitted.stderr
    assert "ppy_coll_" in source.read_text(encoding="utf-8")
    standard = "-std=c11" if language == "c" else "-std=c++17"
    binary = tmp_path / "prog"
    compiled = subprocess.run(
        [compiler, standard, "-O2", "-Wall", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    expected = _run(tmp_path, program.name, text="1000\n")
    native = subprocess.run(
        [str(binary)], input="1000\n", capture_output=True, text=True, check=False
    )
    assert native.stdout == expected.stdout


def test_the_checker_names_each_misuse(tmp_path: Path):
    _write(
        tmp_path,
        """
        from ppy import Deque, Heap, Vec


        def a() -> int:
            v = Vec[bytes]()
            return len(v)


        def b() -> int:
            h = Heap[int]()
            return h[0]


        def c() -> int:
            h = Heap[int]()
            total: int = 0
            for x in h:
                total += x
            return total


        def d() -> int:
            v = Vec[int]()
            v.push(1.5)
            v[0] = 2.5
            return v.pop()


        def e() -> int:
            q = Deque[int](3)
            return len(q)


        def f() -> int:
            q = Deque[int]()
            return q.missing()
        """,
    )
    checked = _run(tmp_path, "-m", "ppy_compiler", "check", "prog.ppy")
    shown = (checked.stdout + checked.stderr).splitlines()
    errors = [line for line in shown if line.startswith("error[")]
    assert errors[:7] == [
        (
            "error[E1305]: a `ppy.Vec` holds numbers, strings, tuples of numbers, "
            "dataclasses, or collections, not `bytes`"
        ),
        "error[E1301]: a `ppy.Heap` is read by `peek` and `pop`",
        "error[E1302]: a `ppy.Heap` is read by `peek` and `pop`, not iterated",
        "error[E1301]: argument 1 expects `int`, got `Literal[1.5]`",
        "error[E1301]: a `float` does not fit an `int` element",
        "error[E1302]: a `ppy.Deque` takes an iterable, not `Literal[3]`",
        "error[E1202]: `ppy.Deque[int]` has no attribute `missing`",
    ]


KEYED = """
from ppy import HashMap, HashSet, LinkedList, TreeMap, TreeSet


def josephus(n: int, k: int) -> int:
    # Remove every k-th person from a circle; the last one standing.
    people = LinkedList[int]()
    for i in range(n):
        people.push_back(i)
    node: int = people.head()
    while len(people) > 1:
        for _ in range(k - 1):
            node = people.next(node)
            if node == -1:
                node = people.head()
        following: int = people.next(node)
        people.remove(node)
        node = following if following != -1 else people.head()
    return people.front()


def counts(n: int) -> int:
    seen = HashMap[int, int]()
    for i in range(n):
        key: int = (i * i) % 97
        seen[key] = seen.get(key, 0) + 1
    unique = HashSet[int]()
    for key in seen:
        if seen[key] > 1:
            unique.add(key)
    unique.discard(0)
    total: int = 0
    for key in unique:
        total = total * 31 + key
    return total % 1000003 + len(seen) + (1 if 4 in unique else 0)


def nearest(n: int) -> int:
    marks = TreeSet[int]()
    prices = TreeMap[int, float]()
    for i in range(n):
        marks.add((i * 37) % 101)
        prices[(i * 13) % 50] = i / 4
    acc: int = marks.floor(50) + marks.ceiling(51) + marks.lower(marks.min() + 1)
    acc += marks.higher(10) + marks.max()
    for key in marks:
        acc = (acc * 3 + key) % 1000003
    popped: float = prices.pop(prices.min())
    return acc + int(popped * 4) + len(prices) + (1 if 13 in prices else 0)


print(josephus(41, 3), counts(500), nearest(60))


from ppy import HashMap, HashSet, LinkedList, TreeMap, TreeSet


def grow_while_walking(n: int) -> int:
    m = HashMap[int, int]()
    for i in range(n):
        m[i] = i
    total: int = 0
    for key in m:
        total += key
        if key == 2:
            m[100] = 1
    return total


def update_while_walking(n: int) -> int:
    m = HashMap[int, int]()
    for i in range(n):
        m[i] = i
    for key in m:
        m[key] = m[key] * 2
    return m[n - 1] + m.get(n + 5, -1) + (1 if n not in m else 0)


def missing(n: int) -> float:
    m = TreeMap[int, float]()
    m[1] = 0.5
    return m[n]


def below(n: int) -> int:
    s = TreeSet[int]()
    s.add(5)
    s.add(9)
    return s.lower(n)


def stale(n: int) -> int:
    xs = LinkedList[int]()
    a: int = xs.push_back(1)
    xs.push_back(2)
    xs.remove(a)
    return xs.value(n)


def remove_while_walking() -> int:
    xs = LinkedList[int]()
    for i in range(6):
        xs.push_back(i * 10)
    seen: int = 0
    node: int = xs.head()
    for x in xs:
        seen = seen * 10 + x // 10
        if x == 20:
            xs.remove(xs.head())
    return seen * 100 + len(xs) + node


def sets(n: int) -> int:
    s = HashSet[int]()
    for i in range(n):
        s.add(i % 7)
    s.remove(3)
    s.discard(3)
    t = TreeSet[int]()
    for key in s:
        t.add(key * 3)
    t.remove(0)
    return len(s) * 1000 + t.floor(10) * 10 + t.min()


def check() -> None:
    print(update_while_walking(10), remove_while_walking(), sets(20))
    for call in range(4):
        try:
            if call == 0:
                print(grow_while_walking(5))
            elif call == 1:
                print(missing(2))
            elif call == 2:
                print(below(5))
            else:
                print(stale(0))
        except (RuntimeError, KeyError, IndexError) as error:
            print(type(error).__name__, error)


check()
"""

KEYED_NATIVE = [
    "josephus",
    "counts",
    "nearest",
    "grow_while_walking",
    "update_while_walking",
    "missing",
    "below",
    "stale",
    "remove_while_walking",
    "sets",
]


@requires_llvm
def test_linked_lists_maps_and_trees_agree_on_every_path(tmp_path: Path):
    _write(tmp_path, KEYED)
    outputs = [
        _run(tmp_path, "prog.ppy"),
        _run(tmp_path, "-m", "ppy_compiler", "prog.ppy"),
        _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy"),
    ]
    for done in outputs:
        assert done.returncode == 0, done.stderr
    assert outputs[0].stdout == outputs[1].stdout == outputs[2].stdout
    assert outputs[0].stdout.splitlines() == [
        "30 729427 388552",
        "18 1234505 6063",
        "RuntimeError HashMap changed during iteration",
        "KeyError 2",
        "KeyError 'no key below 5'",
        "IndexError node 0 is not in the list",
    ]
    for name in KEYED_NATIVE:
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{name}")
        assert "llvm backend: native" in explained.stdout, (name, explained.stdout)


KEYED_STANDALONE = """
import ppy
from ppy import HashMap, LinkedList, TreeMap


def main() -> None:
    n: int = ppy.scan[int]()
    counts = HashMap[int, int]()
    order = TreeMap[int, float]()
    chain = LinkedList[int]()
    for i in range(n):
        key: int = (i * 2654435761) % 1009
        counts[key] = counts.get(key, 0) + 1
        order[key % 97] = i / 8
        if i % 3 == 0:
            chain.push_front(key)
        else:
            chain.push_back(key)
    walk: int = 0
    for x in chain:
        walk = (walk * 31 + x) % 1000003
    total: float = 0.0
    for key in order:
        total += order[key]
    print(len(counts), counts[0], walk, order.floor(50), order.ceiling(50), total, chain.front())


main()
"""


@requires_standalone
def test_a_standalone_binary_holds_maps_trees_and_lists(tmp_path: Path):
    program = _write(tmp_path, KEYED_STANDALONE)
    built = _run(
        tmp_path, "-m", "ppy_compiler", "build", "--standalone", program.name, "-o", "dist"
    )
    assert built.returncode == 0, built.stderr
    expected = _run(tmp_path, program.name, text="5000\n")
    native = subprocess.run(
        [str(tmp_path / "dist" / "prog")],
        input="5000\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert native.returncode == 0, native.stderr
    assert native.stdout == expected.stdout == "1009 5 46427 50 50 59986.625 263\n"


def test_the_checker_names_misuse_of_keyed_collections(tmp_path: Path):
    _write(
        tmp_path,
        """
        from ppy import HashMap, HashSet, TreeMap, TreeSet


        def a() -> int:
            m = HashMap[float, int]()
            return len(m)


        def b() -> int:
            s = HashSet[int]()
            return s[0]


        def c() -> int:
            m = TreeMap[int]()
            return len(m)


        def d() -> float:
            m = TreeMap[int, int]()
            m[1] = 2.5
            return m.get(1, 0)


        def e() -> int:
            s = TreeSet[int]()
            return s.floor(1.5)
        """,
    )
    checked = _run(tmp_path, "-m", "ppy_compiler", "check", "prog.ppy")
    shown = (checked.stdout + checked.stderr).splitlines()
    errors = [line for line in shown if line.startswith("error[")]
    assert errors == [
        "error[E1305]: a `ppy.HashMap` key is an `int`, a `str`, a tuple of `int`, or a hashable "
        "instance, not `float`",
        "error[E1301]: a `ppy.HashSet` is read by its methods",
        "error[E1305]: a `ppy.TreeMap` takes a key and a value type",
        "error[E1301]: a `float` does not fit an `int` element",
        "error[E1301]: argument 1 expects `int`, got `Literal[1.5]`",
    ]
