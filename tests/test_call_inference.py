"""Parameter types from call sites under `--no-strict` (analysis.call_inference).

An unannotated parameter of any function or method takes the type the
project's calls pass, a default value, or a doctest's literal. Inferred types
are guarded: a Python caller passing something else runs the Python body, so
every path prints what CPython prints.
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


#: Public functions called from the program, a method family, defaults, and
#: a call from Python with another type after the native ones.
CALLED = """
\"\"\"
>>> count_divisors(12), count_divisors(True), scale("ab", 2)
(6, 1, 'abab')
>>> window_sum((5, 6, 7), 1), total_area(True)
(13, 0)
>>> count_divisors("x")
Traceback (most recent call last):
...
TypeError: '<=' not supported between instances of 'int' and 'str'
\"\"\"


def count_divisors(n):
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 2 if i * i != n else 1
        i += 1
    return count


def window_sum(xs, lo=0, hi=-1):
    if hi < 0:
        hi = len(xs)
    total = 0
    for i in range(lo, hi):
        total += xs[i]
    return total


def scale(x, factor):
    return x * factor


class Shape:
    def __init__(self, size):
        self.size = size

    def area(self, k):
        return self.size * k


class Square(Shape):
    def area(self, k):
        return self.size * self.size * k


def total_area(n):
    shapes = [Shape(2), Square(3)]
    out = 0
    for i in range(n):
        out += shapes[i % 2].area(i)
    return out


def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def solution():
    print(count_divisors(28), count_divisors(36))
    print(window_sum([1, 2, 3, 4]), window_sum([1, 2, 3, 4], 1, 3))
    print(scale(3, 2.5), scale(4, 0.5))
    print(total_area(5), fib(15))


if __name__ == "__main__":
    solution()
    # The doctests call with other types, through Python: the native
    # entries refuse those arguments and the Python bodies run.
    import doctest

    print(doctest.testmod().failed)
"""

#: Functions called only from their doctests and from `__main__` with literals.
DOCTESTED = """
def digit_sum(n):
    \"\"\"
    >>> digit_sum(1234)
    10
    >>> digit_sum("12")
    Traceback (most recent call last):
    ...
    TypeError: '>' not supported between instances of 'str' and 'int'
    \"\"\"
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total


class Stack:
    \"\"\"
    >>> s = Stack()
    >>> s.push(3)
    >>> s.push(4)
    >>> s.peek_plus(1)
    5
    \"\"\"

    def __init__(self):
        self.items = []

    def push(self, value):
        self.items.append(value)

    def peek_plus(self, extra):
        return self.items[-1] + extra


if __name__ == "__main__":
    import doctest

    print(doctest.testmod().failed)
    print(digit_sum(987654321))
"""

PROGRAMS = {
    "called": (CALLED, ["count_divisors", "window_sum", "scale", "fib"]),
    "doctested": (DOCTESTED, ["digit_sum"]),
}


def _write(tmp_path: Path, source: str) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    program = tmp_path / "prog.py"
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
    done = _run(tmp_path, "prog.py")
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _params(path: Path, qualname: str) -> dict[str, str]:
    bundle = analyze_paths(open_project(path), [path], backend="python")
    info = bundle.symbols.functions[qualname]
    return {p.name: str(p.type) for p in info.params}


def test_public_functions_take_their_types_from_every_call(tmp_path: Path):
    path = _write(tmp_path, CALLED)
    assert _params(path, "prog.count_divisors") == {"n": "int"}
    # `lo=0` with `1` passed, `hi=-1` with `3` passed.
    assert _params(path, "prog.window_sum") == {"xs": "list[int]", "lo": "int", "hi": "int"}
    # `3` with `2.5`: a float, as a declared float takes an int.
    assert _params(path, "prog.scale") == {"x": "int", "factor": "float"}


def test_a_method_family_takes_evidence_together(tmp_path: Path):
    """`shapes[i].area(i)` with a `Shape` may run `Square.area`."""
    path = _write(tmp_path, CALLED)
    assert _params(path, "prog.Shape.area")["k"] == "int"
    assert _params(path, "prog.Square.area")["k"] == "int"
    assert _params(path, "prog.Shape.__init__")["size"] == "int"


def test_doctest_calls_type_a_function_nothing_else_calls(tmp_path: Path):
    path = _write(tmp_path, DOCTESTED)
    assert _params(path, "prog.Stack.push")["value"] == "int"
    assert _params(path, "prog.Stack.peek_plus")["extra"] == "int"
    # The `__main__` call is a call; the raising doctest is not evidence.
    assert _params(path, "prog.digit_sum") == {"n": "int"}


def test_strict_mode_still_asks_for_annotations(tmp_path: Path):
    path = _write(tmp_path, CALLED)
    project = open_project(path, config_overrides={"strict": True})
    bundle = analyze_paths(project, [path], backend="python")
    assert "E1201" in {d.code for d in bundle.diagnostics.sorted()}
    assert str(bundle.symbols.functions["prog.count_divisors"].params[0].type) == "<unknown>"


def test_a_function_used_as_a_sort_key_is_not_inferred(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        def weight(x):
            return -x


        def main():
            print(weight(3), sorted([3, 1, 2], key=weight))


        main()
        """,
    )
    assert _params(path, "prog.weight") == {"x": "<unknown>"}


def test_a_call_with_a_splat_keeps_the_parameters_unknown(tmp_path: Path):
    path = _write(
        tmp_path,
        """
        def add(a, b):
            return a + b


        def main():
            args = (1, 2)
            print(add(1, 2), add(*args))


        main()
        """,
    )
    assert _params(path, "prog.add") == {"a": "<unknown>", "b": "<unknown>"}


def test_an_inference_the_checker_rejects_is_taken_back(tmp_path: Path):
    """`add(p, 2)` makes `y` an `int`, and `x + y` with `x` a `Point` is
    then an error the checker reports, on a call CPython never makes. The
    source did not say `int`, so the inference is taken back: nothing is
    reported, and the function stays in Python."""
    path = _write(
        tmp_path,
        """
        class Point:
            def __init__(self, x):
                self.x = x


        def add(x, y, flag):
            if flag:
                return x + y
            return y


        def main():
            print(add(Point(1), 2, False))


        main()
        """,
    )
    bundle = analyze_paths(open_project(path), [path], backend="python")
    assert [d for d in bundle.diagnostics.sorted() if d.is_error] == []
    assert _params(path, "prog.add") == {"x": "<unknown>", "y": "<unknown>", "flag": "<unknown>"}


@requires_llvm
@requires_cc
def test_explain_says_where_a_type_came_from(tmp_path: Path):
    _write(tmp_path, CALLED)
    done = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.window_sum")
    assert "inferred (not annotated):" in done.stdout, done.stdout
    assert "xs: list[int], from 2 calls (prog.py:" in done.stdout
    assert "lo: int, from 1 call (prog.py:" in done.stdout
    assert "and the default value" in done.stdout
    assert "return: int, from the body's return statements" in done.stdout


@requires_llvm
@requires_cc
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_every_path_agrees_and_inferred_functions_go_native(tmp_path: Path, name: str):
    source, natives = PROGRAMS[name]
    expected = _expected(tmp_path, source)
    for args in (["-m", "ppy_compiler", "prog.py"], ["-m", "ppy_compiler", "run", "prog.py"]):
        done = _run(tmp_path, *args)
        assert done.returncode == 0, done.stderr
        assert _output(done).strip() == expected, (args, done.stderr)
    for function in natives:
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


@requires_standalone
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_a_standalone_binary_agrees(tmp_path: Path, name: str):
    expected = _expected(tmp_path, PROGRAMS[name][0])
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "--standalone", "prog.py", "-o", "dist")
    if built.returncode != 0:
        pytest.skip(f"not a standalone program: {built.stderr[-300:]}")
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


#: Only inferred functions and a `main` that calls them: what a standalone
#: build and emitted C can hold whole.
NATIVE_ONLY = """
def count_divisors(n):
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 2 if i * i != n else 1
        i += 1
    return count


def window_sum(xs, lo):
    total = 0
    for i in range(lo, len(xs)):
        total += xs[i]
    return total


def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def main():
    print(count_divisors(28), count_divisors(36))
    print(window_sum([1, 2, 3, 4], 0), window_sum([1, 2, 3, 4], 2))
    print(fib(15))


main()
"""


@requires_standalone
def test_a_standalone_binary_of_inferred_functions_agrees(tmp_path: Path):
    expected = _expected(tmp_path, NATIVE_ONLY)
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "--standalone", "prog.py", "-o", "dist")
    assert built.returncode == 0, built.stderr
    ran = subprocess.run(
        [str(tmp_path / "dist" / "prog")], capture_output=True, text=True, check=False
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == expected


@requires_llvm
@pytest.mark.parametrize("unsafe", [False, True])
@pytest.mark.parametrize("language", ["c", "cpp"])
def test_emitted_source_of_inferred_functions_frees_everything_once(
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
