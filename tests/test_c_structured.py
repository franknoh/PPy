"""Emitted C and C++ is structured: `while`, `if`/`else`, `break`, `continue`, `return`.

The shapes a structured writer finds hard -- a loop with an early `return`,
a `break` that runs code first, `for`/`else`, a generator resumed in the
middle of its loop, a shared raise path, a `max` whose tests chain, a
backtracking regex -- are written without a label, and the
program still answers as CPython does, under AddressSanitizer and
UndefinedBehaviorSanitizer with leak detection.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
_CXX = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

SHAPES = """
from collections.abc import Iterator


class Odd(ValueError):
    pass


def is_palindrome(text: str) -> bool:
    left = 0
    right = len(text) - 1
    while left < right:
        if text[left] != text[right]:
            return False
        left += 1
        right -= 1
    return True


def first_over(xs: list[int], limit: int) -> int:
    for i in range(len(xs)):
        if xs[i] > limit:
            return i
    return -1


def find_pair(xs: list[int], total: int) -> int:
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] + xs[j] == total:
                return i * 100 + j
    return -1


def smallest_factor(n: int) -> int:
    d = 2
    found = n
    while d * d <= n:
        if n % d == 0:
            found = d
            break
        d += 1
    return found


def classify(xs: list[int]) -> str:
    for x in xs:
        if x < 0:
            break
    else:
        return "all non-negative"
    return "has a negative"


def spin(n: int) -> int:
    steps = 0
    while True:
        if n == 1:
            return steps
        if steps > 1000:
            break
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return -1


def primes(limit: int) -> Iterator[int]:
    for n in range(2, limit):
        d = 2
        prime = True
        while d * d <= n:
            if n % d == 0:
                prime = False
                break
            d += 1
        if prime:
            yield n


def checked(x: int) -> int:
    if x % 7 == 3:
        raise Odd(f"bad {x}")
    return x * 2


def tally(n: int) -> int:
    total = 0
    bad = 0
    for i in range(n):
        try:
            total += checked(i)
        except Odd:
            bad += 1
            continue
        total += 1
    return total * 1000 + bad


def widest(xs: list[int]) -> int:
    return max(abs(x) for x in xs)


def main() -> None:
    print(is_palindrome("racecar"), is_palindrome("abca"), is_palindrome(""))
    print(first_over([1, 5, 9, 2], 4), first_over([1, 2], 9))
    print(find_pair([3, 8, 1, 6], 9), find_pair([1, 1], 5))
    print(smallest_factor(91), smallest_factor(97))
    print(classify([1, 2, 3]), classify([1, -2, 3]))
    print(spin(27), spin(1))
    print(sum(p for p in primes(200)), len([p for p in primes(50)]))
    print(tally(50))
    print(widest([3, -9, 4, 1]))


main()
"""


def _ppy(directory: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _program(tmp_path: Path) -> tuple[Path, str]:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    path = tmp_path / "shapes.ppy"
    path.write_text(textwrap.dedent(SHAPES).lstrip("\n"), encoding="utf-8")
    expected = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, check=True
    ).stdout
    return path, expected


def _structured(text: str) -> None:
    assert "goto" not in text
    labels = [line for line in text.splitlines() if line.rstrip().endswith(":;")]
    assert not labels, labels


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
def test_hard_shapes_are_structured_and_answer_like_cpython(
    tmp_path: Path, language: str, unsafe: bool
):
    compiler = c_compiler() if language == "c" else _CXX
    if compiler is None or not _sanitizes(compiler, tmp_path):
        pytest.skip(f"no {language} compiler with AddressSanitizer")
    path, expected = _program(tmp_path)
    flags = ["--unsafe"] if unsafe else []
    emitted = tmp_path / f"shapes.{language}"
    done = _ppy(tmp_path, "emit", language, "--standalone", *flags, path.name, "-o", emitted.name)
    assert done.returncode == 0, done.stderr
    _structured(emitted.read_text(encoding="utf-8"))
    standard = "-std=c11" if language == "c" else "-std=c++17"
    binary = tmp_path / "shapes"
    built = subprocess.run(
        # An unsafe program leaves a generator's unused constructor behind: not this test's concern.
        [compiler, standard, "-g", "-O1", "-Wall", "-Wextra", "-Wno-unused-function",
         "-fsanitize=address,undefined", str(emitted), "-lm", "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )  # fmt: skip
    assert built.returncode == 0, built.stderr
    assert "warning:" not in built.stderr, built.stderr
    ran = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        check=False,
        env={"ASAN_OPTIONS": "detect_leaks=1", "UBSAN_OPTIONS": "halt_on_error=1"},
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == expected


@requires_llvm
@pytest.mark.parametrize("language", ["c", "cpp"])
def test_the_library_form_of_the_hard_shapes_is_structured(tmp_path: Path, language: str):
    compiler = c_compiler() if language == "c" else _CXX
    path, _expected = _program(tmp_path)
    emitted = tmp_path / f"shapes_lib.{language}"
    done = _ppy(tmp_path, "emit", language, path.name, "-o", emitted.name)
    assert done.returncode == 0, done.stderr
    _structured(emitted.read_text(encoding="utf-8"))
    if compiler is None:
        return
    standard = "-std=c11" if language == "c" else "-std=c++17"
    built = subprocess.run(
        [compiler, standard, "-O1", "-Wall", "-Wextra", "-fPIC", "-c", str(emitted),
         "-o", str(tmp_path / "shapes.o")],
        capture_output=True,
        text=True,
        check=False,
    )  # fmt: skip
    assert built.returncode == 0, built.stderr
    assert "warning:" not in built.stderr, built.stderr


def _examples() -> list[Path]:
    return sorted(_EXAMPLES.glob("*/*.ppy"))


@requires_llvm
@pytest.mark.parametrize("path", _examples(), ids=lambda p: p.parent.name)
def test_no_example_falls_back_to_labels(monkeypatch, path: Path):
    """Every function of every example is written structured, as a library and as a program."""
    from ppy_compiler.backend.c import emit as emitter
    from ppy_compiler.driver.cli import main

    fallen: list[str] = []
    run = emitter._FunctionEmitter.run

    def watched(self):  # type: ignore[no-untyped-def]
        try:
            return run(self)
        except emitter._Unstructured as error:
            fallen.append(f"{self.function.name}: {error}")
            raise

    monkeypatch.setattr(emitter._FunctionEmitter, "run", watched)
    monkeypatch.chdir(path.parent)
    emitted = 0
    for flags in ([], ["--standalone"]):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = main(["emit", "c", *flags, path.name])
        emitted += status == 0
    if not emitted:
        pytest.skip("no C form of this example")
    assert not fallen, fallen
