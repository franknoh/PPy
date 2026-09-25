"""Generic classes with bases, and classes with generic bases.

`class Counted[T](Stack[T])` passes its parameter on; `class IntStack(Stack[int])`
fixes it. The checker binds a base's parameters from what the class gives
it, through every level; `super()` is the base with those arguments; native
code lays out, instantiates, and dispatches by the same bindings. Each
program is held to CPython under `ppy`, `ppy run`, a standalone binary, and
emitted C and C++ under AddressSanitizer.
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

STACKS = """
from ppy import Vec


class Stack[T]:
    def __init__(self) -> None:
        self.items: Vec[T] = Vec[T]()

    def push(self, value: T) -> None:
        self.items.push(value)

    def pop(self) -> T:
        return self.items.pop()

    def __len__(self) -> int:
        return len(self.items)


class Counted[T](Stack[T]):
    def __init__(self) -> None:
        super().__init__()
        self.pushes: int = 0

    def push(self, value: T) -> None:
        self.pushes += 1
        super().push(value)


class Capped[T](Counted[T]):
    def __init__(self, cap: int) -> None:
        super().__init__()
        self.cap: int = cap

    def push(self, value: T) -> None:
        if len(self) < self.cap:
            super().push(value)


class IntStack(Stack[int]):
    def total(self) -> int:
        s = 0
        for v in self.items:
            s += v
        return s


def fill(stack: Stack[int], n: int) -> int:
    for i in range(n):
        stack.push(i)
    return len(stack)


def counted(n: int) -> int:
    c = Counted[int]()
    kept = fill(c, n)
    return kept * 1000 + c.pushes * 10 + c.pop()


def capped(n: int) -> int:
    c = Capped[int](5)
    kept = fill(c, n)
    return kept * 1000 + c.pushes * 10 + c.pop()


def fixed(n: int) -> int:
    s = IntStack()
    kept = fill(s, n)
    return kept * 100000 + s.total()


def floats(n: int) -> float:
    c = Counted[float]()
    for i in range(n):
        c.push(i * 0.5)
    return c.pop() + c.pushes


def main() -> None:
    print(counted(12), capped(12), fixed(12), floats(7))


main()
"""

PAIRS = """
class Box[T]:
    def __init__(self, value: T) -> None:
        self.value: T = value

    def get(self) -> T:
        return self.value


class Pair[A, B](Box[A]):
    def __init__(self, first: A, second: B) -> None:
        super().__init__(first)
        self.second: B = second

    def swapped(self) -> "Pair[B, A]":
        return Pair[B, A](self.second, self.get())


class Tagged(Pair[int, float]):
    def weight(self) -> float:
        return self.get() * self.second


def pairs(n: int) -> float:
    p = Pair[int, float](n, 0.25)
    q = p.swapped()
    t = Tagged(n, 1.5)
    return q.get() + q.second + t.weight() + t.get()


def main() -> None:
    print(pairs(8))


main()
"""

PROGRAMS = {
    "stacks": (STACKS, ["counted", "capped", "fixed", "floats"]),
    "pairs": (PAIRS, ["pairs"]),
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


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_the_checker_binds_a_bases_parameters_from_the_class(tmp_path: Path, name: str):
    _write(tmp_path, PROGRAMS[name][0])
    checked = _run(tmp_path, "-m", "ppy_compiler", "check", "prog.ppy")
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_a_mismatch_through_a_generic_base_is_still_one(write, codes):
    """`IntStack` fixes `Stack`'s `T` to `int`, so pushing a string is `E1301`,
    as it is for a `Stack[int]`."""
    path = write(
        "wrong.ppy",
        """
        from ppy import Vec


        class Stack[T]:
            def __init__(self) -> None:
                self.items: Vec[T] = Vec[T]()

            def push(self, value: T) -> None:
                self.items.push(value)


        class IntStack(Stack[int]):
            pass


        def wrong() -> None:
            IntStack().push("no")
        """,
    )
    assert "E1301" in codes(path)


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


@requires_llvm
@requires_cc
def test_a_subclass_whose_arguments_the_base_does_not_tell_stays_in_python(tmp_path: Path):
    """`Stack[int]` says nothing of `Tagged`'s `U`: a call through it cannot
    pick `Tagged[U].push`, so that function runs as Python."""
    source = """
    from ppy import Vec


    class Stack[T]:
        def __init__(self) -> None:
            self.items: Vec[T] = Vec[T]()

        def push(self, value: T) -> None:
            self.items.push(value)

        def __len__(self) -> int:
            return len(self.items)


    class Tagged[T, U](Stack[T]):
        def push(self, value: T) -> None:
            self.items.push(value)
            self.items.push(value)


    def fill(stack: Stack[int], n: int) -> int:
        for i in range(n):
            stack.push(i)
        return len(stack)


    def main() -> None:
        print(fill(Tagged[int, float](), 3), fill(Stack[int](), 3))


    main()
    """
    expected = _expected(tmp_path, source)
    done = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert done.returncode == 0, done.stderr
    assert _output(done).strip() == expected
    explained = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.fill")
    assert "do not follow" in explained.stdout, explained.stdout
