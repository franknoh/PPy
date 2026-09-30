"""Nested functions, lambdas, and functions as values, native.

A function value is a closure: a handle to its entry's address and the cells
of the variables it shares with the function that made it. Late binding and
`nonlocal` go through the cells, as in CPython. Each program is held to
CPython under `ppy`, `ppy run`, a standalone binary, and emitted C and C++
(safe and `--unsafe`) under AddressSanitizer with leak detection, with
`ppy explain` confirming each function went native. A function value never
crosses to Python: a function that takes or returns one runs in Python when
Python calls it.
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

BASICS = """
from typing import Callable


def apply(f: Callable[[int], int], xs: list[int]) -> list[int]:
    out: list[int] = []
    for x in xs:
        out.append(f(x))
    return out


def double(x: int) -> int:
    return x * 2


def use_named(n: int) -> int:
    ys = apply(double, list(range(n)))
    return sum(ys)


def use_lambda(n: int) -> int:
    k = 3
    ys = apply(lambda x: x * k, list(range(n)))
    return sum(ys)


def counter(n: int) -> int:
    total = 0

    def add(v: int) -> None:
        nonlocal total
        total += v

    for i in range(n):
        add(i)
    return total


def make_adder(k: int) -> Callable[[int], int]:
    def add(x: int) -> int:
        return x + k

    return add


def adders(n: int) -> int:
    f = make_adder(10)
    g = make_adder(n)
    return f(n) * 100 + g(1)


def main() -> None:
    print(use_named(10), use_lambda(10), counter(10), adders(5))


main()
"""

KEYS = """
from typing import Callable


def keyed(n: int) -> int:
    xs = [(i * 7) % 11 for i in range(n)]
    xs.sort(key=lambda v: -v)
    ys = sorted(xs, key=lambda v: v % 3)
    zs = sorted(xs, key=lambda v: (v % 4, -v), reverse=True)
    return xs[0] * 1000 + ys[0] * 10 + zs[0] + max(xs, key=lambda v: v % 5) + min(xs, key=lambda v: (v - 5) * (v - 5))


def square(v: int) -> int:
    return v * v


def mapped(n: int) -> int:
    total = sum(map(lambda v: v + 1, range(n)))
    total += sum(map(square, range(n)))
    evens = 0
    for v in filter(lambda v: v % 2 == 0, range(n)):
        evens += v
    k = 3
    f: Callable[[int], int] = lambda v: v * k
    k = 4
    total += sum(map(f, range(n)))
    return total * 1000 + evens


def keyed_value(n: int) -> int:
    xs = list(range(n))
    f: Callable[[int], int] = lambda v: -v
    xs.sort(key=f)
    return xs[0] + max(xs, key=square)


def main() -> None:
    print(keyed(20), mapped(10), keyed_value(7))


main()
"""

VALUES = """
from dataclasses import dataclass
from typing import Callable


class Op:
    def __init__(self, name: str, f: Callable[[int, int], int]) -> None:
        self.name = name
        self.f = f

    def run(self, a: int, b: int) -> int:
        return self.f(a, b)


def add(a: int, b: int) -> int:
    return a + b


def table(n: int) -> int:
    ops: list[Callable[[int, int], int]] = [add, lambda a, b: a * b, lambda a, b: a - b]
    total = 0
    for i in range(n):
        total += ops[i % 3](i, 3)
    return total


def objects(n: int) -> int:
    op = Op("mul", lambda a, b: a * b + n)
    other = Op("add", add)
    return op.run(3, 4) + other.f(1, 2) + op.f(2, 2)


def compose(f: Callable[[int], int], g: Callable[[int], int]) -> Callable[[int], int]:
    return lambda x: f(g(x))


def composed(n: int) -> int:
    h = compose(lambda x: x + 1, lambda x: x * 2)
    hh = compose(h, h)
    return h(n) * 1000 + hh(n)


def fib_closure(n: int) -> int:
    def fib(k: int) -> int:
        if k < 2:
            return k
        return fib(k - 1) + fib(k - 2)

    return fib(n)


def counters(n: int) -> int:
    def make() -> Callable[[], int]:
        count = 0

        def step() -> int:
            nonlocal count
            count += 1
            return count

        return step

    a = make()
    b = make()
    for _ in range(n):
        a()
    return a() * 100 + b()


def strings(n: int) -> str:
    sep = "-"
    join: Callable[[str, str], str] = lambda a, b: a + sep + b
    out = "x"
    for i in range(n):
        out = join(out, str(i))
    return out


def main() -> None:
    print(table(10), objects(5), composed(3), fib_closure(15), counters(4), strings(4))


main()
"""

EDGES = """
from typing import Callable


def checked(n: int) -> int:
    def half(v: int) -> int:
        if v % 2:
            raise ValueError("odd")
        return v // 2

    total = 0
    for i in range(n):
        try:
            total += half(i)
        except ValueError:
            total += 100
    return total


def late(n: int) -> int:
    def get() -> int:
        return x

    x = n * 2
    first = get()
    x = n * 3
    return first * 1000 + get()


def words(n: int) -> str:
    names = [str(i * 37 % 11) for i in range(n)]
    longest = sorted(names, key=len, reverse=True)
    shout: Callable[[str], str] = lambda s: s + "!"
    out = ""
    for said in map(shout, longest[:3]):
        out += said
    return out


def main() -> None:
    print(checked(10), late(5), words(12))


main()
"""

PROGRAMS = {
    "basics": (BASICS, ["apply", "use_named", "use_lambda", "counter", "make_adder", "adders"]),
    "keys": (KEYS, ["keyed", "mapped", "keyed_value"]),
    "values": (
        VALUES,
        ["table", "objects", "composed", "fib_closure", "counters", "strings", "compose"],
    ),
    "edges": (EDGES, ["checked", "late", "words"]),
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


BOUNDARY = """
from typing import Callable


def counter(n: int) -> int:
    total = 0

    def add(v: int) -> None:
        nonlocal total
        total += v * v

    for i in range(n):
        add(i)
    return total


def apply(f: Callable[[int], int], v: int) -> int:
    return f(v)


def make_adder(k: int) -> Callable[[int], int]:
    return lambda v: v + k
"""


@requires_llvm
@requires_cc
def test_python_calls_a_function_using_closures_natively(tmp_path: Path):
    """A body that makes and calls closures is native from Python; a function
    that takes or returns a function value keeps the function in Python."""
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
    lowered = module.functions["prog.counter"]
    assert lowered.exposed, lowered.exposure_reason
    signature = lowered.boundary or lowered.signature
    bound = bind(signature, engine.address(signature.symbol), lambda *_: None)
    assert bound.wrapper(1000) == sum(i * i for i in range(1000))
    assert bound.calls == 1
    for name in ("apply", "make_adder"):
        found = module.functions.get(f"prog.{name}")
        assert found is None or not found.exposed, name


@requires_llvm
@requires_cc
def test_a_loop_variable_a_closure_shares_stays_in_python(tmp_path: Path):
    """Each lambda sees the loop variable's last value, as CPython's do."""
    source = """
    from typing import Callable


    def captured(n: int) -> int:
        fs: list[Callable[[], int]] = []
        for i in range(n):
            fs.append(lambda: i)
        return sum(f() for f in fs)


    def main() -> None:
        print(captured(4))


    main()
    """
    expected = _expected(tmp_path, source)
    assert expected == "12"
    done = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert _output(done).strip() == expected
    explained = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.captured")
    assert "shared with a closure" in explained.stdout, explained.stdout
