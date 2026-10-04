"""More evidence for parameter types under `--no-strict` (analysis.call_inference).

Beyond plain calls with typed arguments: values `argparse` parses, `int`
and `float` calls of one parameter, decorators that keep the signature,
operators on a class's instances, parameters declared `list`, defaults that
are not literals, doctest loops, and a body's own use of a parameter nothing
calls with a type. Every inferred type is still checked at the Python
boundary, so each path prints what CPython prints.
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
from ppy_compiler.driver.pipeline import analyze_paths, open_project

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")
_standalone, _standalone_detail = standalone_toolchain_status()
requires_standalone = pytest.mark.skipif(
    not (llvm_available() and _standalone), reason=f"no standalone toolchain: {_standalone_detail}"
)
_CXX = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)


def _write(tmp_path: Path, source: str) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    program = tmp_path / "prog.py"
    program.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return program


def _run(tmp_path: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    return subprocess.run(
        [sys.executable, *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        input=stdin,
    )


def _output(done: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(line for line in done.stdout.splitlines() if not line.startswith("compiling"))


def _params(path: Path, qualname: str) -> dict[str, str]:
    bundle = analyze_paths(open_project(path), [path], backend="python")
    info = bundle.symbols.functions[qualname]
    return {p.name: str(p.type) for p in info.params if p.name != "self"}


def _agrees(tmp_path: Path, source: str, natives: list[str], stdin: str | None = None) -> None:
    """CPython, `ppy`, and `ppy run` print the same, and the functions
    named go native."""
    _write(tmp_path, source)
    reference = _run(tmp_path, "prog.py", stdin=stdin)
    assert reference.returncode == 0, reference.stderr
    for args in (["-m", "ppy_compiler", "prog.py"], ["-m", "ppy_compiler", "run", "prog.py"]):
        done = _run(tmp_path, *args, stdin=stdin)
        assert done.returncode == 0, (args, done.stderr)
        assert _output(done).strip() == reference.stdout.strip(), (args, done.stderr)
    for function in natives:
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


#: Values whose type the program states without annotations: `input()`
#: conversions and an `argparse` parser's `type=`.
PARSED = """
import argparse


def square(n):
    return n * n


def halve(x):
    return x / 2


def width(parts):
    total = 0
    for part in parts:
        total += len(part)
    return total


def summed(xs):
    total = 0
    for x in xs:
        total += x
    return total


def scaled(k, ratio, name, loud):
    return f"{name}:{k * ratio}:{loud}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--ratio", type=float, default="0.5")
    parser.add_argument("--name", default="run")
    parser.add_argument("--loud", action="store_true")
    parser.add_argument("--maybe", type=int)
    args = parser.parse_args()
    print(scaled(args.k, args.ratio, args.name, args.loud), args.maybe)


if __name__ == "__main__":
    n = int(input())
    print(square(n), halve(float(input())))
    print(width(input().split()), summed(list(map(int, input().split()))))
    main()
"""


def test_input_and_argparse_values_type_the_calls_they_feed(tmp_path: Path):
    path = _write(tmp_path, PARSED)
    assert _params(path, "prog.square") == {"n": "int"}
    assert _params(path, "prog.halve") == {"x": "float"}
    assert _params(path, "prog.width") == {"parts": "list[str]"}
    assert _params(path, "prog.summed") == {"xs": "list[int]"}
    # `--maybe` may be None, and is passed nowhere; the rest are always there.
    assert _params(path, "prog.scaled") == {
        "k": "int",
        "ratio": "float",
        "name": "str",
        "loud": "bool",
    }


@requires_llvm
@requires_cc
def test_input_and_argparse_programs_agree(tmp_path: Path):
    _agrees(tmp_path, PARSED, ["summed"], stdin="7\n2.5\nab cde f\n1 2 3\n")


def test_an_argparse_option_that_may_be_none_or_reconfigured_is_not_taken(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        import argparse


        def f(k):
            return k + 1


        def g(k):
            return k + 1


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--k", type=int)
            args = parser.parse_args()
            print(f(args.k))
            other = argparse.ArgumentParser()
            other.add_argument("--k", type=int, default=1)
            other.set_defaults(k="x")
            more = other.parse_args()
            print(g(more.k))


        main()
        """,
    )
    assert _params(path, "prog.f") == {"k": "<unknown>"}
    assert _params(path, "prog.g") == {"k": "<unknown>"}


#: One parameter called with an `int` and a `float`: a `float`, whose int-ness
#: shows in what the function returns, so an `int` runs the Python body.
MIXED = """
\"\"\"
>>> triple(2), triple(2.5), triple("ab")
(6, 7.5, 'ababab')
\"\"\"


def triple(x):
    return x * 3


def main():
    print(triple(4), triple(1.5), triple(-0.25))


if __name__ == "__main__":
    main()
    import doctest

    print(doctest.testmod().failed)
"""


def test_int_and_float_calls_join_to_a_float(tmp_path: Path):
    path = _write(tmp_path, MIXED)
    assert _params(path, "prog.triple") == {"x": "float"}


@requires_llvm
@requires_cc
def test_int_and_float_calls_keep_an_int_an_int(tmp_path: Path):
    _agrees(tmp_path, MIXED, ["triple"])
    assert _run(tmp_path, "-m", "ppy_compiler", "prog.py").stdout.splitlines()[-2:] == [
        "12 4.5 -0.75",
        "0",
    ]


#: Decorators that keep the parameters: a project decorator that wraps with
#: `functools.wraps` and passes `*args, **kwargs` through, `lru_cache`, and
#: a pytest mark whose cases are literals or a module constant.
DECORATED = """
import functools

import pytest


def logged(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)

    return inner


def doubled(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, *args, **kwargs)

    return inner


@logged
def tri(n):
    total = 0
    for i in range(n):
        total += i
    return total


@doubled
def pair(a, b):
    return a + b


@functools.lru_cache(maxsize=None)
def ways(n):
    if n < 2:
        return 1
    return ways(n - 1) + ways(n - 2)


CASES = [(1, "a"), (2, "bb")]


@pytest.mark.parametrize("count, text", CASES)
def test_lengths(count, text):
    assert len(text) == count


@pytest.mark.parametrize("value", [1.5, 2.0])
def test_halves(value):
    assert value / 2 < value


def main():
    print(tri(5), ways(20), pair(3))


if __name__ == "__main__":
    main()
"""


def test_signature_keeping_decorators_are_inferred(tmp_path: Path):
    path = _write(tmp_path, DECORATED)
    assert _params(path, "prog.tri") == {"n": "int"}
    assert _params(path, "prog.ways") == {"n": "int"}
    assert _params(path, "prog.test_lengths") == {"count": "int", "text": "str"}
    assert _params(path, "prog.test_halves") == {"value": "float"}
    # Its wrapper passes the arguments twice: the calls say nothing of `b`.
    assert _params(path, "prog.pair") == {"a": "<unknown>", "b": "<unknown>"}


@requires_llvm
@requires_cc
def test_decorated_programs_agree(tmp_path: Path):
    source = DECORATED.replace("import pytest\n", "").split("CASES =", maxsplit=1)[0]
    source += "\nprint(tri(5), ways(20))\n"
    # Inference types `tri`'s parameter; the name holds `logged`'s wrapper,
    # which only Python runs.
    _agrees(tmp_path, source, ["ways"])
    explained = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.tri")
    assert "decorated by `@logged`" in explained.stdout, explained.stdout


#: Decorators nobody vouches for that change what a call does: every call by
#: the name, from Python, from a function that goes native, at module level,
#: and recursively, reaches the decorator's object.
ACTING = """
import functools


def doubled(fn):
    def wrapper(n):
        return 2 * fn(n)

    return wrapper


def logged(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print("calling", fn.__name__)
        return fn(*args, **kwargs)

    return wrapper


def memo(fn):
    seen = {}

    def wrapper(n):
        if n not in seen:
            print("miss", n)
            seen[n] = fn(n)
        return seen[n]

    return wrapper


@doubled
def square(n):
    return n * n


@logged
def work(n: int) -> int:
    t = 0
    for i in range(n):
        t += i
    return t


@memo
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def loop(n: int) -> int:
    total = 0
    for i in range(n):
        total += square(i)
        work(i)
    return total


def main():
    print(square(3), work(7), fib(12))
    print(loop(5))


main()
print(square(4), fib(13))
"""


@requires_llvm
@requires_cc
def test_decorators_that_act_run_on_every_call(tmp_path: Path):
    _agrees(tmp_path, ACTING, [])
    explained = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.square")
    assert "decorated by `@doubled`" in explained.stdout, explained.stdout


#: Operators on a class's instances are calls of its dunders.
OPERATORS = """
\"\"\"
>>> v = Vec(1, 2)
>>> v[0], v[1], (v * 2.5).x, v == Vec(1, 2), v == 3
Traceback (most recent call last):
...
AttributeError: 'int' object has no attribute 'x'
>>> (Vec(1, 2) + Vec(3, 4)).y, v < Vec(1, 3)
(6, True)
\"\"\"


class Vec:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y)

    def __mul__(self, k):
        return Vec(self.x * k, self.y * k)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.x < other.x or (self.x == other.x and self.y < other.y)

    def __getitem__(self, i):
        return self.x if i == 0 else self.y


class Grid:
    def __init__(self):
        self.cells = {}

    def __setitem__(self, key, value):
        self.cells[key] = value

    def __contains__(self, key):
        return key in self.cells


def main():
    a, b = Vec(1, 2), Vec(3, 4)
    c = a + b
    d = c * 3
    print(c.x, c.y, d.x, d[1], a == b, a == Vec(1, 2), a < b)
    vs = [Vec(3, 1), Vec(1, 5), Vec(1, 2)]
    vs.sort()
    print([(w.x, w.y) for w in vs])
    g = Grid()
    g[1] = "a"
    for g[2] in ["b"]:
        pass
    print(1 in g, 3 in g)


if __name__ == "__main__":
    main()
    import doctest

    print(doctest.testmod().failed)
"""


def test_operators_type_the_dunders_they_call(tmp_path: Path):
    path = _write(tmp_path, OPERATORS)
    assert _params(path, "prog.Vec.__add__") == {"other": "prog.Vec"}
    assert _params(path, "prog.Vec.__mul__") == {"k": "int"}
    assert _params(path, "prog.Vec.__eq__") == {"other": "prog.Vec"}
    assert _params(path, "prog.Vec.__lt__") == {"other": "prog.Vec"}
    assert _params(path, "prog.Vec.__getitem__") == {"i": "int"}
    assert _params(path, "prog.Grid.__contains__") == {"key": "int"}
    # `for g[2] in ...` stores a value no use types.
    assert _params(path, "prog.Grid.__setitem__") == {"key": "<unknown>", "value": "<unknown>"}


def test_a_comparison_is_inferred_only_as_its_own_class(tmp_path: Path):
    """The runtime calls `__eq__` and `__lt__` itself with two of the class's
    objects (sorting, `in` on a list); an `int` from `p == 3` is not taken."""
    path = _write(
        tmp_path,
        """
        class P:
            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                return self.x == other


        def main():
            print(P(3) == 3)


        main()
        """,
    )
    assert _params(path, "prog.P.__eq__") == {"other": "<unknown>"}


@requires_llvm
@requires_cc
def test_operator_programs_agree(tmp_path: Path):
    _agrees(tmp_path, OPERATORS, ["Vec.__add__", "Vec.__mul__"])


#: A parameter declared `list` takes the element type its calls pass.
LISTED = """
\"\"\"
>>> mean([1, 2, 4]), mean([0.5])
(2.3333333333333335, 0.5)
\"\"\"


def mean(xs: list):
    total = 0.0
    for v in xs:
        total += v
    return total / len(xs)


def first_word(words: list):
    return words[0].upper()


def broken(xs: list):
    return xs[0] + "!"


def main():
    print(mean([1.0, 2.0, 4.0]), mean([3.0]), first_word(["ab", "c"]))
    print(broken(["a"]), broken([1.5]) if False else "")


if __name__ == "__main__":
    main()
    import doctest

    print(doctest.testmod().failed)
"""


def test_a_list_parameter_takes_its_element_type_from_the_calls(tmp_path: Path):
    path = _write(tmp_path, LISTED)
    assert _params(path, "prog.mean") == {"xs": "list[float]"}
    assert _params(path, "prog.first_word") == {"words": "list[str]"}
    # `list[str] | list[float]`: the declaration stays.
    assert _params(path, "prog.broken") == {"xs": "list[Any]"}


@requires_llvm
@requires_cc
def test_list_parameter_programs_agree(tmp_path: Path):
    _agrees(tmp_path, LISTED, ["mean"])


def test_an_element_type_the_checker_rejects_goes_back_to_the_declaration(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        def total(xs: list) -> int:
            out = 0
            for x in xs:
                out += x
            return out / 1


        def main():
            print(total([1, 2]))


        main()
        """,
    )
    assert _params(path, "prog.total") == {"xs": "list[Any]"}


#: What the body does with a parameter nothing calls with a type, defaults
#: that are not literals, and loops in doctests.
USED = """
\"\"\"
>>> pair_sum({1: [2, 3], 2: []})
5
>>> for value in [3, 4]:
...     print(bump(value))
4
5
\"\"\"
import sys

STEP = 2


def countdown(n):
    total = 0
    for i in range(n, 0, -1):
        total = total * 3 + i
    return total


def initials(name):
    out = []
    for part in name.split():
        out.append(part[:1].upper())
    return out


def maybe(n):
    if n is None:
        return 0
    return sum(range(n))


def offset(x, base=len("four"), step=STEP):
    return x + base * step


def pair_sum(graph):
    total = 0
    for key in graph:
        for value in graph[key]:
            total += value
    return total


def bump(v):
    return v + 1


if __name__ == "__main__":
    me = sys.modules[__name__]
    print(getattr(me, "countdown")(5), getattr(me, "countdown")(True))
    print(getattr(me, "initials")("ada king lovelace"), getattr(me, "initials")(b"x y"))
    try:
        getattr(me, "countdown")(2.5)
    except TypeError as e:
        print("TypeError", e)
    print(getattr(me, "maybe")(4), offset(1), getattr(me, "offset")(1.5, 2.5))
    import doctest

    print(doctest.testmod().failed)
"""


def test_a_body_types_a_parameter_nothing_calls_with_a_type(tmp_path: Path):
    path = _write(tmp_path, USED)
    assert _params(path, "prog.countdown") == {"n": "int"}
    assert _params(path, "prog.initials") == {"name": "str"}
    # Tested against `None`: no one type.
    assert _params(path, "prog.maybe") == {"n": "<unknown>"}


def test_defaults_and_doctest_loops_give_types(tmp_path: Path):
    path = _write(tmp_path, USED)
    assert _params(path, "prog.offset") == {"x": "int", "base": "int", "step": "int"}
    assert _params(path, "prog.pair_sum") == {"graph": "dict[int, list[int]]"}
    assert _params(path, "prog.bump") == {"v": "int"}


@requires_llvm
@requires_cc
def test_body_typed_programs_agree(tmp_path: Path):
    _agrees(tmp_path, USED, ["countdown", "offset"])


@requires_llvm
@requires_cc
def test_explain_says_where_the_new_evidence_came_from(tmp_path: Path):
    _write(tmp_path, USED)
    done = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.countdown")
    assert "n: int, from its use in `range` (prog.py:" in done.stdout, done.stdout
    _write(tmp_path, DECORATED)
    done = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.test_lengths")
    assert "count: int, from 2 parametrize cases (prog.py:" in done.stdout, done.stdout


#: Inferred functions only, and a `main`: a standalone build and emitted C
#: hold it whole.
NATIVE_ONLY = """
class Vec:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y)

    def __lt__(self, other):
        return self.x < other.x or (self.x == other.x and self.y < other.y)


def half(x):
    return x / 2 + 1


def total(xs: list, k):
    out = 0
    for x in xs:
        out += x * k
    return out


def main():
    print(half(4), half(1.5))
    v = Vec(1, 2) + Vec(3, 4)
    print(v.x, v.y, Vec(1, 2) < Vec(1, 3))
    print(total([1, 2, 3], 2), total([7], 5))


main()
"""


def _expected(tmp_path: Path, source: str) -> str:
    _write(tmp_path, source)
    done = _run(tmp_path, "prog.py")
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@requires_standalone
def test_a_standalone_binary_of_the_new_inferences_agrees(tmp_path: Path):
    expected = _expected(tmp_path, NATIVE_ONLY)
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "--standalone", "prog.py", "-o", "dist")
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
def test_emitted_source_of_the_new_inferences_frees_everything_once(
    tmp_path: Path, language: str, unsafe: bool
):
    compiler = c_compiler() if language == "c" else _CXX
    if compiler is None or not _sanitizes(compiler, tmp_path):
        pytest.skip(f"no {language} compiler with AddressSanitizer")
    expected = _expected(tmp_path, NATIVE_ONLY)
    emitted = tmp_path / f"prog.{language}"
    flags = ["--unsafe"] if unsafe else []
    done = _run(
        tmp_path, "-m", "ppy_compiler", "emit", language, "--standalone", *flags, "prog.py",
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


#: A result declared `float` that the body gives as an `int`: CPython hands
#: back the `int`, so native code must not make it a `float`. Refining
#: `files: list` to `list[int]` made this function native and showed it
#: (greedy_methods/optimal_merge_pattern).
FLOAT_RESULT = """
\"\"\"
>>> merge_cost([8, 8, 8, 8, 8]), halves([3, 4])
(96, (1.5, 3))
\"\"\"


def merge_cost(files: list) -> float:
    cost = 0
    while len(files) > 1:
        temp = 0
        for _ in range(2):
            i = files.index(min(files))
            temp += files[i]
            files.pop(i)
        files.append(temp)
        cost += temp
    return cost


def halves(xs: list[int]) -> tuple[float, float]:
    return xs[0] / 2, xs[1] - 1


if __name__ == "__main__":
    print(merge_cost([2, 3, 4]), halves([5, 6]))
    import doctest

    print(doctest.testmod().failed)
"""


@requires_llvm
@requires_cc
def test_an_int_returned_for_a_declared_float_stays_an_int(tmp_path: Path):
    _agrees(tmp_path, FLOAT_RESULT, [])
    explained = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.merge_cost")
    assert "returns an `int` where `float` is declared" in explained.stdout, explained.stdout


#: A local bound to `None` first and to an object later is the object or
#: `None`: `prev = None` before a loop that relinks a list, `root = None`
#: before the calls that grow a tree. A module that binds `head = None` and
#: then a node is not naming a type.
NONE_FIRST = """
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.left = None
        self.right = None


def reverse(head: Node) -> Node | None:
    prev = None
    node = head
    while node is not None:
        nxt = node.next
        node.next = prev
        prev = node
        node = nxt
    return prev


def insert(node: Node | None, key: int) -> Node:
    if node is None:
        return Node(key)
    if key < node.value:
        node.left = insert(node.left, key)
    else:
        node.right = insert(node.right, key)
    return node


def build(count: int) -> int:
    root = None
    for k in range(count):
        root = insert(root, (k * 7919) % count)
    return root.value


def main():
    head = Node(1)
    head.next = Node(2)
    head.next.next = Node(3)
    back = reverse(head)
    print(back.value, back.next.value, build(200))


main()
chain = None
for i in range(5):
    cell = Node(i)
    cell.next = chain
    chain = cell
print(chain.value, chain.next.value)
"""


@requires_llvm
@requires_cc
def test_a_local_bound_to_none_then_an_object_goes_native(tmp_path: Path):
    _agrees(tmp_path, NONE_FIRST, ["reverse", "build"])


#: Settled module globals read by a method a native function calls, by a
#: nested function, and by a nested function whose enclosing one stays in
#: Python (its entry reads the global from the module).
GLOBALS_READ = """
PRIMES: list[int] = [2, 3, 5, 7, 11, 13]


class Counter:
    def __init__(self) -> None:
        self.hits = 0

    def count(self, n: int) -> int:
        total = 0
        for i in range(n):
            for p in PRIMES:
                if i % p == 0:
                    total += 1
        self.hits += total
        return total


def outer(n: int) -> int:
    def inner(k: int) -> int:
        t = 0
        for i in range(k):
            for p in PRIMES:
                t += i % p
        return t

    return inner(n) + inner(n // 2)


def kept(n: int) -> int:
    import sys  # stays in Python

    def inner(k: int) -> int:
        t = 0
        for p in PRIMES:
            t += k % p
        return t

    return inner(n) + len(sys.argv)


def main() -> None:
    c = Counter()
    print(c.count(1000), outer(100), c.hits)


main()
print(kept(50))
PRIMES.append(17)
print(outer(40), kept(51))
"""


@requires_llvm
@requires_cc
def test_settled_globals_reach_methods_and_nested_functions(tmp_path: Path):
    _agrees(
        tmp_path,
        GLOBALS_READ,
        ["main", "outer", "Counter.count", "outer.<locals>.inner", "kept.<locals>.inner"],
    )


def test_a_reason_never_names_an_implicit_global():
    from ppy_compiler.backend.llvm.lowering import Unsupported

    reason = str(Unsupported("`m.f` expects a `list[int]`, not `__global_m_PRIMES`"))
    assert "__global_" not in reason and "a module global passed on" in reason


#: `max(a, b)` and `min(a, b, c)` of objects ordered by `__lt__` or `__gt__`:
#: the first of equals wins, as in CPython, and the uses type `other`. A
#: doctest operator whose operand is a constructor call types it too.
ORDERED = """
class Vector:
    \"\"\"
    >>> a = Vector(1, 2)
    >>> a < Vector(2, 2)
    True
    \"\"\"

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __lt__(self, other):
        return self.norm() < other.norm()

    def norm(self):
        return self.x * self.x + self.y * self.y


class Ranked:
    def __init__(self, value, tag):
        self.value = value
        self.tag = tag

    def __gt__(self, other):
        return self.value > other.value

    def __lt__(self, other):
        return self.value < other.value


def walk(steps, seed):
    farthest = Vector(0, 0)
    nearest = Vector(100, 100)
    state = seed
    for _ in range(steps):
        state = (state * 1103515245 + 12345) % 2147483648
        here = Vector(state % 7 - 3, state // 7 % 7 - 3)
        farthest = max(farthest, here)
        nearest = min(here, nearest)
    return farthest.norm() * 1000 + nearest.norm()


def pick(n):
    total = 0
    for i in range(n):
        a = Ranked(i % 3, 1)
        b = Ranked(i % 2, 2)
        c = Ranked((i * 7) % 4, 3)
        hi = max(a, b)
        lo = min(a, b, c)
        top = max(c, a, b)
        total += hi.tag * 100 + lo.tag * 10 + top.tag + hi.value
    return total


print(walk(500, 7), pick(60))
"""


def test_a_doctest_operand_made_by_a_constructor_is_evidence(tmp_path: Path):
    path = _write(tmp_path, ORDERED.split("class Ranked", maxsplit=1)[0])
    assert _params(path, "prog.Vector.__lt__") == {"other": "prog.Vector"}


@requires_llvm
@requires_cc
def test_max_and_min_of_objects_go_native(tmp_path: Path):
    _agrees(tmp_path, ORDERED, ["walk", "pick"])
