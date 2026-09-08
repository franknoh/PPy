"""`ppy build --python-extension`: an importable CPython module."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.wrapper_build import wrapper_toolchain

_ready, _detail = wrapper_toolchain()
requires_extension_toolchain = pytest.mark.skipif(
    not (llvm_available() and _ready), reason=f"no extension toolchain: {_detail}"
)

MODULE = """
    from dataclasses import dataclass


    GREETING = "hello"


    @dataclass
    class Point:
        x: float
        y: float


    def scale(xs: list[int], k: int) -> int:
        total = 0
        for x in xs:
            total += x * k
        return total


    def cube(n: int) -> int:
        return n * n * n


    def norm2(p: Point) -> float:
        return p.x * p.x + p.y * p.y


    def describe(n: int) -> str:
        return f"{GREETING} {cube(n)}"
    """


def _build(project: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", "geometry.ppy", *extra],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )


def _python(cwd: Path, code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


@requires_extension_toolchain
def test_the_extension_imports_and_runs_native_with_the_python_fallback(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "geometry.ppy").write_text(textwrap.dedent(MODULE).lstrip("\n"), encoding="utf-8")
    built = _build(tmp_path, "--python-extension", "-o", "dist")
    assert built.returncode == 0, built.stderr
    assert "extension: " in built.stderr
    library = tmp_path / "dist" / "geometry.so"
    assert library.is_file()
    ran = _python(
        tmp_path / "dist",
        """
        import geometry
        print(geometry.scale([1, 2, 3], 4), geometry.cube(3), geometry.norm2(geometry.Point(3.0, 4.0)))
        print(type(geometry.scale).__name__, type(geometry.describe).__name__)
        print(geometry.describe(2), geometry.GREETING)
        print(geometry.cube(3037000500))  # overflows a machine word: the Python definition answers
        print(geometry.scale([1, "x"], 1) if False else geometry.scale([], 7))
        """,
    )
    assert ran.returncode == 0, ran.stderr
    lines = ran.stdout.splitlines()
    assert lines[0] == "24 27 25.0"
    assert lines[1] == "builtin_function_or_method function"
    assert lines[2] == "hello 8 hello"
    assert lines[3] == str(3037000500**3)
    assert lines[4] == "0"


@requires_extension_toolchain
def test_an_extension_is_built_for_this_interpreter_only(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "geometry.ppy").write_text(textwrap.dedent(MODULE).lstrip("\n"), encoding="utf-8")
    from ppy_compiler.target import host_target

    other = "aarch64-linux-gnu" if host_target().architecture != "aarch64" else "x86_64-linux-gnu"
    built = _build(tmp_path, "--python-extension", "--target", other, "-o", "dist")
    assert built.returncode == 2
    assert "E1002" in built.stderr and "this interpreter" in built.stderr
