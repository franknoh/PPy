"""`ppy.Vec`, `ppy.Deque`, `ppy.Heap`, and `ppy.MaxHeap`: one meaning on every path.

The reference in `ppy._collections` is what the types mean. Native code holds
each as a handle into the C runtime in `ppy_runtime.collections`, and the
programs here are held to the reference under `ppy`, `ppy run`, a standalone
binary, and emitted C and C++, with `ppy explain` confirming the functions
went native rather than agreeing by staying in Python.
"""

from __future__ import annotations

import ctypes
import heapq
import os
import random
import shutil
import subprocess
import sys
import textwrap
from collections import deque
from pathlib import Path

import pytest

import ppy
from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler, standalone_toolchain_status
from ppy_runtime import collections as runtime

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
        ppy.Vec[str]()
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
    assert empty.returncode == 70, "an empty pop has no Python to fall back to"


@requires_llvm
@pytest.mark.parametrize("unsafe", [False, True])
@pytest.mark.parametrize("language", ["c", "cpp"])
def test_emitted_source_carries_the_collections(tmp_path: Path, language: str, unsafe: bool):
    compiler = c_compiler() if language == "c" else _CXX
    if compiler is None:
        pytest.skip(f"no {language} compiler on PATH")
    program = _write(tmp_path, STANDALONE)
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
    assert "ppy_coll_push_back_i64" in source.read_text(encoding="utf-8")
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
            v = Vec[str]()
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
        "error[E1305]: a `ppy.Vec` holds `int` or `float`, not `str`",
        "error[E1301]: a `ppy.Heap` is read by `peek` and `pop`",
        "error[E1302]: a `ppy.Heap` is read by `peek` and `pop`, not iterated",
        "error[E1301]: argument 1 expects `int`, got `Literal[1.5]`",
        "error[E1301]: a `float` does not fit an `int` element",
        "error[E1305]: `ppy.Deque[T]()` takes no arguments",
        "error[E1202]: `ppy.Deque[int]` has no attribute `missing`",
    ]


def _library() -> ctypes.CDLL:
    path = runtime.library_path()
    assert path is not None
    lib = ctypes.CDLL(str(path))
    lib.ppy_coll_new.restype = ctypes.c_void_p
    lib.ppy_coll_len.restype = ctypes.c_int64
    for suffix, kind in (("i64", ctypes.c_int64), ("f64", ctypes.c_double)):
        for name in ("get", "pop_back", "pop_front"):
            getattr(lib, f"ppy_coll_{name}_{suffix}").restype = kind
        for order in ("min", "max"):
            getattr(lib, f"ppy_heap_pop_{order}_{suffix}").restype = kind
    return lib


@requires_cc
def test_the_runtime_matches_deque_heapq_and_a_stable_sort():
    lib = _library()
    rng = random.Random(3)
    handle = ctypes.c_void_p
    for _ in range(100):
        made = handle(lib.ppy_coll_new(ctypes.c_int64(0)))
        reference: deque[int] = deque()
        for _ in range(300):
            op, value = rng.randrange(5), rng.randrange(-1000, 1000)
            if op == 0:
                lib.ppy_coll_push_back_i64(made, ctypes.c_int64(value))
                reference.append(value)
            elif op == 1:
                lib.ppy_coll_push_front_i64(made, ctypes.c_int64(value))
                reference.appendleft(value)
            elif op == 2 and reference:
                assert lib.ppy_coll_pop_back_i64(made) == reference.pop()
            elif op == 3 and reference:
                assert lib.ppy_coll_pop_front_i64(made) == reference.popleft()
            assert lib.ppy_coll_len(made) == len(reference)
        items = [lib.ppy_coll_get_i64(made, ctypes.c_int64(i)) for i in range(len(reference))]
        assert items == list(reference)
        lib.ppy_coll_sort_i64(made)
        items = [lib.ppy_coll_get_i64(made, ctypes.c_int64(i)) for i in range(len(reference))]
        assert items == sorted(reference)
        lib.ppy_coll_free(made)
    for order, sign in (("min", 1.0), ("max", -1.0)):
        made = handle(lib.ppy_coll_new(ctypes.c_int64(0)))
        heap: list[float] = []
        for _ in range(2000):
            if rng.random() < 0.6 or not heap:
                value = rng.uniform(-5, 5)
                getattr(lib, f"ppy_heap_push_{order}_f64")(made, ctypes.c_double(value))
                heapq.heappush(heap, sign * value)
            else:
                popped = getattr(lib, f"ppy_heap_pop_{order}_f64")(made)
                assert popped == sign * heapq.heappop(heap)
        lib.ppy_coll_free(made)
    # Stable: 0.0 and -0.0 compare equal and keep their order, as `sorted` keeps it.
    made = handle(lib.ppy_coll_new(ctypes.c_int64(0)))
    values = [0.0, -0.0, 1.0, -0.0, 0.0, -1.0]
    for value in values:
        lib.ppy_coll_push_back_f64(made, ctypes.c_double(value))
    lib.ppy_coll_sort_f64(made)
    items = [lib.ppy_coll_get_f64(made, ctypes.c_int64(i)) for i in range(len(values))]
    assert [repr(x) for x in items] == [repr(x) for x in sorted(values)]
    lib.ppy_coll_free(made)
