"""Collections of anything: nested, of tuples, of dataclasses, keyed by tuples.

`Vec[Vec[int]]`, `HashMap[tuple[int, int], int]`, `Vec[Vec[Edge]]` with a
dataclass `Edge`, collections returned from functions and passed as
temporaries, and generic functions that take, make, and return collections of
their type parameter, one native instance per type. Each program agrees on every path and goes native, and its
emitted C and C++ run under AddressSanitizer with leak detection, which is
what holds the reference counting to account: no collection is used after
it is freed, and none is left unfreed.
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

NESTED = """
from dataclasses import dataclass

from ppy import Deque, HashMap, Heap, MaxHeap, TreeMap, Vec


@dataclass(order=True)
class Edge:
    cost: int
    to: int


def graph(n: int) -> Vec[Vec[int]]:
    adj = Vec[Vec[int]](n)
    for u in range(n):
        adj[u].push((u * 3 + 1) % n)
        adj[u].push((u * 5 + 2) % n)
    return adj


def bfs(n: int) -> int:
    adj = graph(n)
    dist = Vec[int](n)
    for i in range(n):
        dist[i] = -1
    dist[0] = 0
    q = Deque[int]()
    q.push_back(0)
    while q:
        u: int = q.pop_front()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.push_back(v)
    total: int = 0
    for d in dist:
        total += d
    return total


def dijkstra(n: int) -> int:
    edges = Vec[Vec[Edge]](n)
    for u in range(n):
        edges[u].push(Edge((u * 7) % 13 + 1, (u + 1) % n))
        edges[u].push(Edge((u * 11) % 17 + 1, (u * 2 + 3) % n))
    best = HashMap[int, int]()
    frontier = Heap[tuple[int, int]]()
    frontier.push((0, 0))
    while frontier:
        d, u = frontier.pop()
        if u in best:
            continue
        best[u] = d
        for e in edges[u]:
            if e.to not in best:
                frontier.push((d + e.cost, e.to))
    total: int = 0
    for u in best:
        total += best[u]
    return total


def grid(w: int) -> int:
    seen = HashMap[tuple[int, int], int]()
    order = TreeMap[tuple[int, int], float]()
    for x in range(w):
        for y in range(w):
            if (x * y) % 3 == 1:
                seen[(x, y)] = x + y
                order[(y, x)] = x / 2
    top = MaxHeap[tuple[int, float]]()
    for key in order:
        top.push((key[0] + key[1], order[key]))
    a, b = top.pop()
    lo = order.floor((2, 100))
    return len(seen) * 1000 + a * 10 + int(b) + lo[0] + lo[1]


def aliases(n: int) -> int:
    rows = Vec[Vec[int]]()
    for i in range(n):
        row = Vec[int]()
        row.push(i)
        rows.push(row)
        row.push(i * 10)
    first = rows[0]
    rows.clear()
    first.push(99)
    last = Vec[int]()
    last.push(len(first))
    return first[2] + len(rows) + last.pop()


def main() -> None:
    print(bfs(50), dijkstra(40), grid(12), aliases(5))


main()
"""

OWNERSHIP = """
import ppy
from ppy import Deque, HashMap, LinkedList, TreeMap, Vec


def row(n: int) -> Vec[int]:
    made = Vec[int]()
    for i in range(n):
        made.push(i)
    return made


def total(v: Vec[int]) -> int:
    s: int = 0
    for x in v:
        s += x
    return s


def churn(n: int) -> int:
    buckets = HashMap[int, Vec[int]]()
    ordered = TreeMap[int, Vec[int]]()
    chain = LinkedList[Vec[int]]()
    queue = Deque[Vec[int]]()
    acc: int = 0
    for i in range(n):
        key: int = i % 17
        if key not in buckets:
            buckets[key] = Vec[int]()
        buckets[key].push(i)
        ordered[i % 5] = row(i % 7)
        node: int = chain.push_back(row(3))
        if i % 4 == 0:
            acc += total(chain.remove(node))
        queue.push_back(row(i % 3))
        if len(queue) > 3:
            acc += len(queue.pop_front())
        acc += total(row(2))
        scratch = Vec[int](3)
        scratch = row(4)
        acc += len(scratch)
    kept = buckets[3]
    buckets.pop(3)
    buckets.clear()
    kept.push(1)
    acc += total(kept) + len(ordered.pop(0))
    ordered.clear()
    chain.clear()
    grid = Vec[Vec[int]](n)
    for i in range(n):
        grid[i] = row(i % 4)
        grid[i].push(i)
    last = grid.pop()
    acc += total(last) + len(grid[0])
    return acc


def main() -> None:
    n: int = ppy.scan[int]()
    print(churn(n))


main()
"""

GENERIC = """
from ppy import HashMap, Heap, Vec


def merged[T: int | float](a: Vec[T], b: Vec[T]) -> Vec[T]:
    out = Vec[T]()
    for x in a:
        out.push(x)
    for x in b:
        out.push(x)
    out.sort()
    return out


def smallest[T: int | float](values: Vec[T], k: int) -> Vec[T]:
    heap = Heap[T]()
    for x in values:
        heap.push(x)
    taken = Vec[T]()
    while heap and len(taken) < k:
        taken.push(heap.pop())
    return taken


def group[K](keys: Vec[K]) -> HashMap[K, int]:
    counts = HashMap[K, int]()
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return counts


def run(n: int) -> float:
    a = Vec[int]()
    b = Vec[float]()
    pairs = Vec[tuple[int, int]]()
    for i in range(n):
        a.push((i * 7) % 11)
        b.push(i / 3)
        pairs.push((i % 3, i % 2))
    both = merged(a, a)
    top = smallest(b, 3)
    counted = group(pairs)
    total: float = 0.0
    for x in both:
        total += x
    for y in top:
        total += y
    return total + len(counted) * 100 + counted[(0, 0)]


def main() -> None:
    print(run(20))


main()
"""

PROGRAMS = {
    "nested": (NESTED, "", "181 1072 32238 102", ["graph", "bfs", "dijkstra", "grid", "aliases"]),
    "ownership": (OWNERSHIP, "200\n", "2714", ["row", "total", "churn"]),
    "generic": (GENERIC, "", "801.0", ["run"]),
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
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_every_path_agrees_and_each_function_goes_native(tmp_path: Path, name: str):
    source, text, expected, natives = PROGRAMS[name]
    _write(tmp_path, source)
    for args in (
        ["prog.ppy"],
        ["-m", "ppy_compiler", "prog.ppy"],
        ["-m", "ppy_compiler", "run", "prog.ppy"],
    ):
        done = _run(tmp_path, *args, text=text)
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == expected, args
    for function in natives:
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


@requires_standalone
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_a_standalone_binary_agrees(tmp_path: Path, name: str):
    source, text, expected, _ = PROGRAMS[name]
    program = _write(tmp_path, source)
    built = _run(
        tmp_path, "-m", "ppy_compiler", "build", "--standalone", program.name, "-o", "dist"
    )
    assert built.returncode == 0, built.stderr
    ran = subprocess.run(
        [str(tmp_path / "dist" / "prog")], input=text, capture_output=True, text=True, check=False
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
    source, text, expected, _ = PROGRAMS[name]
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
        input=text,
        capture_output=True,
        text=True,
        check=False,
        env={"ASAN_OPTIONS": "detect_leaks=1"},
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == expected
