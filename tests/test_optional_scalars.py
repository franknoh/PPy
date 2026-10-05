"""Numbers and strings that may be `None`, natively.

`int | None`, `float | None`, and `bool | None` are a number and a flag: two
slots in a local, two atoms as a parameter or a result, two words in a field
or an element. `str | None` is a string's handle, null for `None`. Each
program here puts them in every place a value lives -- parameters, results,
fields, list and dict elements, locals -- and uses them the ways CPython
lets: `is None`, truth, `==`, `or`, `in`, `isinstance`, printing, `str()`
and f-strings, `d.get(k)`, `any`/`all`, loops, and narrowing after a test.
A field narrowed by the checker and then reset by a call raises CPython's
`TypeError` for the arithmetic that meets `None`.

Each program is held to CPython under `ppy`, `ppy run`, a standalone binary,
and emitted C and C++ (safe and `--unsafe`) under AddressSanitizer with leak
detection, with `ppy explain` confirming the functions go native. The
boundary converts `None` to a clear flag (or a null handle) and back: the
generated C wrapper and the ctypes binding are each called directly, and an
argument of another type (a `bool` for an `int | None`) asks for the Python
body.
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

NUMBERS = """
class Node:
    def __init__(self, key: int, label: int | None = None) -> None:
        self.key = key
        self.label = label
        self.weight: float | None = None
        self.seen: bool | None = None
        self.left: Node | None = None
        self.right: Node | None = None

    def reset(self) -> None:
        self.label = None


def find(xs: list[int], t: int) -> int | None:
    for i in range(len(xs)):
        if xs[i] == t:
            return i
    return None


def bump(x: int | None, d: int) -> int:
    if x is None:
        return d
    return x + d


def halve(x: float | None) -> float | None:
    if x is None:
        return None
    return x / 2.0


def flip(b: bool | None) -> bool | None:
    if b is None:
        return None
    return not b


def describe(x: int | None) -> str:
    kind = "int" if isinstance(x, int) else "none"
    return f"{kind}:{x}|" + str(x)


def insert(root: Node, key: int, label: int | None) -> None:
    node = root
    while True:
        if key < node.key:
            if node.left is None:
                node.left = Node(key, label)
                return
            node = node.left
        else:
            if node.right is None:
                node.right = Node(key, label)
                return
            node = node.right


def labels(root: Node | None, out: list[int | None]) -> None:
    if root is None:
        return
    labels(root.left, out)
    out.append(root.label)
    labels(root.right, out)


def floor_label(root: Node, key: int) -> int | None:
    best: int | None = None
    node: Node | None = root
    while node is not None:
        if node.key <= key:
            best = node.label
            node = node.right
        else:
            node = node.left
    return best


def tally(xs: list[int | None]) -> int:
    total = 0
    for x in xs:
        if x:
            total += x
        elif x is None:
            total += 1000
    return total


def fill(xs: list[int | None], v: int) -> int:
    count = 0
    for i in range(len(xs)):
        x = xs[i]
        if x is None:
            xs[i] = v + i
            count += 1
        else:
            xs[i] = x * 2
    return count


def scores(d: dict[str, int | None]) -> int:
    t = 0
    for k in d:
        v = d.get(k)
        t += v or -1
    d["new"] = None
    return t


def unsound(n: Node) -> int:
    if n.label is not None:
        n.reset()
        return n.label + 1
    return 0


def compare(a: int | None, b: int | None) -> bool:
    if a is not None and b is not None:
        return a < b
    return a == b


def main() -> None:
    xs = [3, 4, 5]
    print(find(xs, 4), find(xs, 9), bump(find(xs, 5), 1), bump(None, 7))
    print(halve(None), halve(3.0), flip(None), flip(False))
    print(describe(None), describe(-4))
    root = Node(50, 5)
    for k in [30, 70, 20, 40, 60, 80]:
        insert(root, k, k // 10 if k % 20 else None)
    out: list[int | None] = []
    labels(root, out)
    print(out, len(out), out.count(None), None in out, 4 in out, any(out), all(out))
    print(floor_label(root, 65), floor_label(root, 55), floor_label(root, 10))
    empty: list[int | None] = []
    print(tally(out), tally(empty), tally([0, None]))
    print(fill(out, 100), out)
    d: dict[str, int | None] = {"a": 1, "b": None, "c": 0}
    print(scores(d), d)
    root.weight = 2.5
    root.seen = False
    print(root.label, root.weight, root.seen, root.left is not None)
    if root.label == 5 and root.seen is not None and not root.seen:
        print("five", root.label != 4, root.label in (5, None), root.weight)
    print(compare(1, 2), compare(None, None), compare(None, 3))
    try:
        print(unsound(root))
    except TypeError as e:
        print("TypeError:", e)
    print(root.label, [root.label, None], [root.weight])


main()
"""

STRINGS = """
LETTERS = "abcdefghijklmnopqrstuvwxyz"


class Person:
    def __init__(self, name: str, nick: str | None = None) -> None:
        self.name = name
        self.nick = nick

    def shown(self) -> str:
        if self.nick is None:
            return self.name
        return self.name + " (" + self.nick + ")"


class TrieNode:
    def __init__(self) -> None:
        self.nodes: dict[str, TrieNode] = {}
        self.word: str | None = None

    def insert(self, word: str) -> None:
        cur = self
        for c in word:
            nxt = cur.nodes.get(c)
            if nxt is None:
                nxt = TrieNode()
                cur.nodes[c] = nxt
            cur = nxt
        cur.word = word

    def find(self, word: str) -> str | None:
        cur = self
        for c in word:
            nxt = cur.nodes.get(c)
            if nxt is None:
                return None
            cur = nxt
        return cur.word


def first_long(words: list[str], n: int) -> str | None:
    for w in words:
        if len(w) > n:
            return w
    return None


def greet(name: str | None) -> str:
    if name is None:
        return "hello, stranger"
    return "hello, " + name


def encrypt(text: str, key: int, alphabet: str | None = None) -> str:
    alpha = alphabet or LETTERS
    out = ""
    for c in text:
        i = alpha.find(c)
        if i < 0:
            out += c
        else:
            out += alpha[(i + key) % len(alpha)]
    return out


def decrypt(text: str, key: int, alphabet: str | None = None) -> str:
    return encrypt(text, -key, alphabet)


def board(w: int, h: int) -> list[list[str | None]]:
    b: list[list[str | None]] = [[None] * w for _ in range(h)]
    b[0][1] = "x"
    b[h - 1][0] = "y"
    return b


def parents(edges: list[tuple[int, int]], names: list[str]) -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for n in names:
        found[n] = None
    for a, b in edges:
        found[names[b]] = names[a]
    return found


def main() -> None:
    words = ["a", "abc", "hello", "hi"]
    print(first_long(words, 2), first_long(words, 9))
    print(greet(None), greet("bob"))
    p = Person("ann")
    q = Person("bob", "bobby")
    print(p.shown(), q.shown(), p.nick, q.nick)
    print(f"{p.nick}|{q.nick}|{q.nick!r}", str(p.nick))
    q.nick = None
    p.nick = "annie"
    print(p.nick == "annie", q.nick == "x", p.nick != q.nick, q.nick is None, not q.nick)
    names: list[str | None] = ["x", None, "y"]
    names.append(None)
    names[0] = None
    print(names, len(names), names[2], None in names, "y" in names, names.count(None))
    best = None
    for w in words:
        if best is None or len(w) > len(best):
            best = w
    print(best)
    s = encrypt("hello world", 3)
    print(s, decrypt(s, 3), encrypt("abc", 1, "cba"), decrypt("abc", 1, None))
    t = TrieNode()
    for w in ["car", "cat", "dog"]:
        t.insert(w)
    found = t.find("cat")
    missing = t.find("ca")
    print(found, missing, t.find("cow"), found or "-", missing or "-")
    print(board(3, 2))
    tree = parents([(0, 1), (0, 2), (1, 3)], ["r", "a", "b", "c"])
    print(tree, tree["r"], tree["c"])


main()
"""

BOUNDARY = """
class Node:
    def __init__(self, label: int | None = None) -> None:
        self.label = label
        self.weight: float | None = None
        self.tag: str | None = None


def first_at_least(xs: list[int], low: int) -> int | None:
    for x in xs:
        if x >= low:
            return x
    return None


def triangle(n: int | None) -> int:
    if n is None:
        return -1
    total = 0
    for i in range(n):
        total += i
    return total


def mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    total = 0.0
    for x in xs:
        total += x
    return total / len(xs)


def walk(n: Node, steps: int) -> int:
    total = 0
    for i in range(steps):
        if n.label is not None:
            total += n.label * i
    n.weight = 2.5 if n.label else None
    n.tag = "seen" if n.label is not None else None
    return total


def fill(xs: list[int | None]) -> int:
    count = 0
    for i in range(len(xs)):
        if xs[i] is None:
            xs[i] = i * 10
            count += 1
    xs.append(None)
    return count


def longest(words: list[str], skip: str | None) -> str | None:
    best: str | None = None
    for w in words:
        if skip is not None and w == skip:
            continue
        if best is None or len(w) > len(best):
            best = w
    return best


print(first_at_least([1, 5, 9], 4), first_at_least([1, 2], 4))
print(triangle(None), triangle(5))
print(mean([]), mean([1.0, 2.0]))
n = Node(5)
print(walk(n, 10), n.label, n.weight, n.tag)
m = Node()
print(walk(m, 10), m.label, m.weight, m.tag)
xs = [1, None, 3, None]
print(fill(xs), xs)
print(longest(["a", "abc", "ab"], None), longest(["a", "abc", "ab"], "abc"), longest([], "x"))
try:
    print(triangle(True))
except TypeError as e:
    print("TypeError", e)
"""

PROGRAMS = {
    "numbers": (
        NUMBERS,
        [
            "find",
            "bump",
            "halve",
            "flip",
            "describe",
            "insert",
            "labels",
            "floor_label",
            "tally",
            "fill",
            "scores",
            "unsound",
            "compare",
            "main",
            "Node.__init__",
        ],
    ),
    "strings": (
        STRINGS,
        [
            "Person.shown",
            "TrieNode.insert",
            "TrieNode.find",
            "first_long",
            "greet",
            "encrypt",
            "decrypt",
            "board",
            "parents",
            "main",
        ],
    ),
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
    expected = _expected(tmp_path, PROGRAMS[name][0])
    compiler = c_compiler() if language == "c" else _CXX
    if compiler is None or not _sanitizes(compiler, tmp_path):
        pytest.skip(f"no {language} compiler with AddressSanitizer")
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
def test_python_calls_through_the_boundary(tmp_path: Path):
    """Module-level code is Python: each call crosses, `None` in and out."""
    expected = _expected(tmp_path, BOUNDARY)
    done = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert done.returncode == 0, done.stderr
    assert _output(done).strip() == expected


class Node:
    """The Python side of `BOUNDARY`'s `Node`, for the wrappers to make and fill."""

    def __init__(self, label: int | None = None) -> None:
        self.label = label
        self.weight: float | None = None
        self.tag: str | None = None


def _module(tmp_path: Path):  # type: ignore[no-untyped-def]
    from ppy_compiler.backend.llvm import _collect
    from ppy_compiler.backend.llvm.jit import JitEngine
    from ppy_compiler.driver.pipeline import analyze_paths, open_project
    from ppy_runtime.collections import library_path

    path = _write(tmp_path, BOUNDARY)
    module = _collect(analyze_paths(open_project(path), [path], backend="llvm"))["prog"]
    runtime = library_path()
    assert runtime is not None
    engine = JitEngine(opt_level=2).open()
    engine.load_library(str(runtime))
    engine.add(module.ir)
    engine.finalize()
    return path, module, engine


@requires_llvm
@requires_cc
def test_the_generated_wrapper_takes_and_gives_none(tmp_path: Path):
    """The C wrapper: `None` is a clear flag or a null handle both ways, a field
    or an element that may be `None` crosses with its object or its list, and
    anything else (a `bool` for an `int | None`, an `int` for a `float | None`
    the body shows) asks for the Python body."""
    from ppy_compiler.backend.llvm.wrapper_build import build_wrappers, wrapper_toolchain

    ready, detail = wrapper_toolchain()
    if not ready:
        pytest.skip(detail)
    path, module, engine = _module(tmp_path)
    built = build_wrappers(
        "prog",
        {q: lowered.boundary or lowered.signature for q, lowered in module.functions.items()},
        path.parent / ".ppy-cache",
    )
    if not built.ok:
        pytest.skip(built.reason)
    assert built.attach_runtime()

    def entry(name: str):  # type: ignore[no-untyped-def]
        lowered = module.functions[f"prog.{name}"]
        signature = lowered.boundary or lowered.signature
        found = built.bind(
            f"prog.{name}", engine.address(signature.symbol), (), None, lambda: (Node,)
        )
        assert found is not None, name
        return found

    first = entry("first_at_least")
    assert first([1, 5, 9], 4) == 5 and first([1, 2], 4) is None
    triangle = entry("triangle")
    assert triangle(None) == -1 and triangle(5) == 10
    assert triangle(True) is NotImplemented and triangle(2.0) is NotImplemented
    mean = entry("mean")
    assert mean([]) is None and mean([1.0, 2.0]) == 1.5
    longest = entry("longest")
    assert longest(["a", "abc", "ab"], None) == "abc"
    assert longest(["a", "abc", "ab"], "abc") == "ab"
    assert longest([], "x") is None and longest(["a"], 3) is NotImplemented
    xs = [1, None, 3, None]
    assert entry("fill")(xs) == 2 and xs == [1, 10, 3, 30, None]
    walk = entry("walk")
    node = Node(5)
    assert walk(node, 10) == 225 and (node.weight, node.tag) == (2.5, "seen")
    empty = Node()
    assert walk(empty, 10) == 0 and (empty.label, empty.weight, empty.tag) == (None, None, None)
    assert walk(Node(True), 10) is NotImplemented  # type: ignore[arg-type]


@requires_llvm
@requires_cc
def test_the_ctypes_binding_takes_and_gives_none(tmp_path: Path):
    """The binding without a generated wrapper: the same conversions, in Python."""
    from ppy_compiler.backend.llvm.runtime import bind

    _path, module, engine = _module(tmp_path)
    fell: list[str] = []

    def entry(name: str):  # type: ignore[no-untyped-def]
        lowered = module.functions[f"prog.{name}"]
        signature = lowered.boundary or lowered.signature

        def fallback(*arguments: object) -> None:
            del arguments
            fell.append(name)

        return bind(signature, engine.address(signature.symbol), fallback)

    first = entry("first_at_least")
    assert first.wrapper([1, 5, 9], 4) == 5 and first.wrapper([1, 2], 4) is None
    triangle = entry("triangle")
    assert triangle.wrapper(None) == -1 and triangle.wrapper(5) == 10
    assert triangle.wrapper(True) is None and fell == ["triangle"]
    mean = entry("mean")
    assert mean.wrapper([]) is None and mean.wrapper([1.0, 2.0]) == 1.5
    longest = entry("longest")
    assert longest.wrapper(["a", "abc", "ab"], None) == "abc"
    assert longest.wrapper([], "x") is None
    xs = [1, None, 3, None]
    fill = entry("fill")
    assert fill.wrapper(xs) == 2 and xs == [1, 10, 3, 30, None]
    assert first.calls == 2 and triangle.calls == 2 and longest.calls == 2 and fill.calls == 1
    assert fell == ["triangle"]


RESIDENT = """
import ppy


class Cell:
    def __init__(self, label: int | None, tag: str | None) -> None:
        self.label = label
        self.tag = tag
        self.weight: float | None = None
        self.seen: bool | None = None
        self.next: Cell | None = None

    @ppy.native
    def step(self, k: int) -> int:
        total = 0
        node = self
        while node is not None:
            if node.label is None:
                node.label = k * 2
            elif node.label > 20:
                node.label = None
            else:
                total += node.label
                node.label += k
            node.weight = None if node.weight is not None else 0.5 * total
            node.seen = node.label is not None
            if node.tag is None or len(node.tag) > 3:
                node.tag = "n"
            else:
                node.tag = node.tag + "x"
            node = node.next
        return total


def show(c: Cell) -> None:
    print(c.label, c.tag, c.weight, c.seen)


a = Cell(3, None)
b = Cell(None, "q")
a.next = b
for i in range(30):
    print(a.step(i % 5))
    show(a)
    show(b)
    if i % 7 == 1:
        a.label = None
    if i % 7 == 2:
        vars(b)["tag"] = None
    if i % 7 == 3:
        setattr(b, "weight", 1.25)
    if i % 7 == 4:
        b.label = 9
    if i % 11 == 5:
        vars(a)["label"] = True
        print(a.step(1))
        vars(a)["label"] = None
    if i % 13 == 6:
        a.weight = 3
        print(a.step(2))
        a.weight = None
show(a)
show(b)
"""


@requires_llvm
@requires_cc
def test_resident_objects_keep_fields_that_may_be_none(tmp_path: Path):
    """Objects whose fields may be `None` stay resident between native calls
    (`crossing.c`): native code sets a field to `None` and back, Python does
    too between the calls (an attribute, `vars()`, `setattr`), and a value of
    another type for a while runs that call in Python. One output, with the
    world and without it."""
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    (tmp_path / "prog.ppy").write_text(RESIDENT.lstrip("\n"), encoding="utf-8")
    expected = _run(tmp_path, "prog.ppy")
    assert expected.returncode == 0, expected.stderr
    env = {k: v for k, v in os.environ.items() if k not in {"PPY_LOWERING", "PPY_RESIDENT"}}
    for extra in ({"PPY_RESIDENT_REPORT": "1"}, {"PPY_RESIDENT": "0"}):
        done = subprocess.run(
            [sys.executable, "-m", "ppy_compiler", "run", "prog.ppy"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            env={**env, **extra},
        )
        assert done.returncode == 0, done.stderr
        assert _output(done).strip() == expected.stdout.strip(), extra
        if "PPY_RESIDENT_REPORT" in extra:
            report = next(
                line for line in done.stderr.splitlines() if line.startswith("resident: ")
            )
            words = report.removeprefix("resident: ").replace(",", "").split()
            assert int(words[7]) == 1 and int(words[8]) > 5 and int(words[10]) > 20, report
