"""Artifact determinism: the same program and configuration make the same bytes, cold or warm.

Every artifact a build writes -- the program's object, the library, the
boundary wrapper, the manifest, the generated Python, the header -- and
every text `ppy emit` prints must come out byte for byte the same whether
the cache is empty, full, or emptied again. Nothing in an artifact's
identity may come from an object id or an unordered traversal.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

PACKAGE = {
    "shapes.ppy": """
        from dataclasses import dataclass


        @dataclass
        class Point:
            x: float
            y: float


        def norm(p: Point) -> float:
            return (p.x * p.x + p.y * p.y) ** 0.5


        def twice(n: int) -> int:
            if n > 10:
                return n + n
            return n * 3
        """,
    "app.ppy": """
        from ppy import native

        from shapes import Point, norm, twice


        @native.export(name="ppy_total")
        def total(xs: list[int]) -> int:
            acc = 0
            for x in xs:
                acc = acc + twice(x)
            return acc


        def spread(points: list[float]) -> float:
            best = 0.0
            for i in range(len(points)):
                d = norm(Point(points[i], 1.0))
                if d > best:
                    best = d
            return best


        def main() -> None:
            print(total([1, 2, 30]), round(spread([3.0, -4.0]), 6))


        main()
        """,
}


def _write(directory: Path) -> None:
    (directory / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    for name, source in PACKAGE.items():
        (directory / name).write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")


def _ppy(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _differences(left: dict[str, bytes], right: dict[str, bytes]) -> list[str]:
    names = sorted(set(left) | set(right))
    return [
        name for name in names if name not in left or name not in right or left[name] != right[name]
    ]


@requires_llvm
def test_a_build_makes_the_same_artifacts_cold_warm_and_cold_again(tmp_path: Path):
    _write(tmp_path)
    dist = tmp_path / "dist"
    cold = _ppy(tmp_path, "build", "app.ppy", "-o", "dist")
    assert cold.returncode == 0, cold.stderr
    first = _snapshot(dist)
    assert any(name.endswith("program.o") for name in first), sorted(first)
    assert "ppy-bindings.json" in first and any(name.endswith(".h") for name in first)

    shutil.rmtree(dist)
    warm = _ppy(tmp_path, "build", "app.ppy", "-o", "dist")
    assert warm.returncode == 0, warm.stderr
    second = _snapshot(dist)
    assert not _differences(first, second), f"the warm build differs: {_differences(first, second)}"

    shutil.rmtree(dist)
    assert _ppy(tmp_path, "clean").returncode == 0
    again = _ppy(tmp_path, "build", "app.ppy", "-o", "dist")
    assert again.returncode == 0, again.stderr
    third = _snapshot(dist)
    assert not _differences(first, third), f"a cold rebuild differs: {_differences(first, third)}"


@requires_llvm
@pytest.mark.parametrize("kind", ["ir", "linked-ir", "llvm-ir", "c", "cpp", "header"])
def test_emitted_text_is_the_same_from_an_empty_and_a_full_cache(tmp_path: Path, kind: str):
    _write(tmp_path)
    first = _ppy(tmp_path, "emit", kind, "app.ppy")
    assert first.returncode == 0, first.stderr
    second = _ppy(tmp_path, "emit", kind, "app.ppy")
    assert second.stdout == first.stdout, f"`ppy emit {kind}` changed with a warm cache"
    assert _ppy(tmp_path, "clean").returncode == 0
    third = _ppy(tmp_path, "emit", kind, "app.ppy")
    assert third.stdout == first.stdout, f"`ppy emit {kind}` changed after the cache was emptied"
    assert first.stdout.strip(), "the emission is not empty"
