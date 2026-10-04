"""Containers a native call only reads, across the Python boundary.

A call that writes through no parameter reads its containers in place of
copies (`backend/llvm/crossing.c`): its lists are laid out in an arena the
call owns, the strings in them are the Python strings' own bytes, borrowed,
and a dict's index is made at its first lookup. A call that writes copies
back only what came in with a parameter it writes, and of a list of numbers
only the elements it changed. These hold `ppy run` to CPython where that
could show: a row or a string handed back, non-ASCII text, a string with no
UTF-8, a string a cached callee keeps past the call, a read parameter that is
also the written one, and writes that leave a value as it was. The last line
of each program says which functions answered natively
(`ppy.native.compiled`), which CPython says False of.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler

pytestmark = [
    pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed"),
    pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH"),
]

READS = """
import functools

import ppy


@ppy.native
def row_of(g: list[list[int]], i: int) -> list[int]:
    best = g[0]
    for row in g:
        if row[i] > best[i]:
            best = row
    return best


@ppy.native
def picked(g: list[list[int]], n: int) -> list[list[int]]:
    out: list[list[int]] = []
    for i in range(n):
        out.append(g[i % len(g)])
    return out


@ppy.native
def longest(words: list[str]) -> list[str]:
    best: list[str] = []
    size = -1
    for w in words:
        if len(w) > size:
            size = len(w)
            best = [w]
        elif len(w) == size:
            best.append(w)
    return best


@ppy.native
def letters(words: list[str]) -> int:
    total = 0
    for w in words:
        for c in w:
            if c != "a":
                total += 1
    return total


@ppy.native
def scores(d: dict[str, int], keys: list[str]) -> int:
    total = 0
    for k in keys:
        if k in d:
            total += d[k]
    for k, v in d.items():
        total += len(k) * v
    return total


@ppy.native
def spread(d: dict[int, list[int]]) -> int:
    total = 0
    for k in d:
        for v in d[k]:
            total += k * v
    return total


@ppy.native
def hits(s: set[str], words: list[str]) -> int:
    n = 0
    for w in words:
        if w in s:
            n += 1
    return n


@ppy.native
def flags(xs: list[bool]) -> int:
    n = 0
    for x in xs:
        if x:
            n += 1
    return n


@functools.cache
def weight(w: str) -> int:
    total = 0
    for c in w:
        total += ord(c)
    return total


@ppy.native
def weights(words: list[str]) -> int:
    total = 0
    for w in words:
        total += weight(w)
    return total


def main() -> None:
    g = [[1, 5], [9, 2], [3, 8]]
    print(row_of(g, 0) is g[1], row_of(g, 1) is g[2], row_of(g, 1))
    out = picked(g, 5)
    print(out, out[0] is g[0], out[3] is g[0], out[4] is g[1])
    words = ["kiwi", "fig", "plum", "pear", "é" * 4, "日本語"]
    best = longest(words)
    print(best, best[0] is words[0], best[-1] is words[4])
    print(letters(words), letters(["aaa", "", "naïve"]))
    print(letters(["ok", "bad\\ud800"]))
    d = {"one": 1, "two": 2, "three": 3, "ünï": 4}
    print(scores(d, ["two", "four", "ünï", "one"]))
    print(spread({1: [1, 2], 2: [3], 5: []}), spread({}))
    s = {"a", "b", "ç"}
    print(hits(s, ["a", "x", "ç", "b", "b"]))
    print(flags([True, False, True, True]), flags([]))
    for _ in range(3):
        made = [("w" + str(i)) * 2 for i in range(4)]
        print(weights(made))
        del made
    print(weight("w0w0"), weight("w3w3"), weights(["w1w1"]))
    print(ppy.native.compiled(row_of), ppy.native.compiled(longest), ppy.native.compiled(scores))


main()
"""

WRITES = """
import ppy


@ppy.native
def copy_first(src: list[int], dst: list[int]) -> None:
    for i in range(len(dst)):
        dst[i] = src[0] + i


@ppy.native
def mark(g: list[list[int]], row: list[int]) -> int:
    row[0] = -1
    total = 0
    for r in g:
        total += r[0]
    return total


@ppy.native
def same_values(xs: list[int], ys: list[float]) -> None:
    for i in range(len(xs)):
        xs[i] = xs[i] * 1
        ys[i] = ys[i] + 0.0
    xs[0] = 100


@ppy.native
def grow(xs: list[int], seen: list[int]) -> int:
    total = 0
    for x in seen:
        total += x
    xs.append(total)
    xs[0] = total
    return len(xs)


@ppy.native
def flip(bits: list[bool], names: list[str]) -> int:
    n = 0
    for i in range(len(bits)):
        bits[i] = not bits[i]
        n += len(names[i % len(names)])
    return n


def main() -> None:
    a = [1, 2, 3]
    copy_first(a, a)
    print(a)
    b = [5, 6]
    c = [0, 0, 0]
    copy_first(b, c)
    print(b, c)
    g = [[1, 2], [3, 4]]
    print(mark(g, g[1]), g)
    shared = [7, 8]
    rows = [shared, shared]
    print(mark(rows, shared), rows, rows[0] is rows[1])
    xs = [10**3, 2, 3]
    big = xs[0]
    ys = [0.5, -0.0, 2.0]
    neg = ys[1]
    same_values(xs, ys)
    print(xs, ys, xs[1] is not None, ys[1] is neg)
    ws = [4, 5]
    print(grow(ws, ws), ws)
    print(grow([1], [2, 3]))
    bits = [True, False, False]
    names = ["x", "yy"]
    print(flip(bits, names), bits, names)
    print(big == 1000, ppy.native.compiled(copy_first), ppy.native.compiled(mark))


main()
"""


def _run(tmp_path: Path, *args: str) -> list[str]:
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    done = subprocess.run(
        [sys.executable, *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    assert done.returncode == 0, done.stderr
    lines = [line for line in done.stderr.splitlines() if not line.startswith("compiling")]
    return [*done.stdout.splitlines(), "--", *lines]


@pytest.mark.parametrize(
    ("source", "native"),
    [(READS, "True True True"), (WRITES, "True True True")],
    ids=["reads", "writes"],
)
def test_the_boundary_agrees_with_cpython(tmp_path: Path, source: str, native: str):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "prog.ppy").write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    python = _run(tmp_path, "prog.ppy")
    ran = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    split = python.index("--")
    assert ran[: split - 1] == python[: split - 1]
    assert ran[split - 1].endswith(native)
    assert ran[split:] == python[split:]
