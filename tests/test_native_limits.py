"""Where native lowering stops today, `ppy explain` says so, and the program still answers.

A value class built inside a native loop, and a tuple handed from one
native call straight into another, keep their function in Python: the
answer is Python's, the reason is on record, and a change in either
direction -- a lowering gained, or a native function lost -- shows here.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

PROGRAM = """
from dataclasses import dataclass

import ppy


@dataclass
class Vec3:
    x: float
    y: float
    z: float


@dataclass
class Ray:
    origin: float
    direction: float


@ppy.pure
@ppy.opt(3)
def norm2(v: Vec3) -> float:
    return v.x * v.x + v.y * v.y + v.z * v.z


@ppy.pure
@ppy.opt(3)
def orbit(count: int) -> float:
    total: float = 0.0
    for i in range(count):
        v: Vec3 = Vec3(float(i), 1.0, 0.5)
        total += norm2(v)
    return total


@ppy.pure
@ppy.opt(3)
def travel(ray: Ray, count: int) -> float:
    position: float = ray.origin
    total: float = 0.0
    for i in range(count):
        position = position * 0.999999 + ray.direction * (i % 3)
        total += position
    return total


@ppy.pure
@ppy.opt(3)
def midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


@ppy.pure
@ppy.opt(3)
def walk_forwarding(count: int) -> float:
    point: tuple[float, float] = (0.0, 0.0)
    for i in range(count):
        point = midpoint(point, (float(i), 1.0))
    return point[0] + point[1]


@ppy.pure
@ppy.opt(3)
def walk_scalars(count: int) -> float:
    x: float = 0.0
    y: float = 0.0
    for i in range(count):
        x, y = midpoint((x, y), (float(i), 1.0))
    return x + y


@ppy.pure
@ppy.opt(3)
def walk(start: tuple[float, float], count: int) -> float:
    x, y = start
    for i in range(count):
        x, y = (x + float(i)) / 2.0, (y + 1.0) / 2.0
    return x + y


print(f"{orbit(1000):.3f} {travel(Ray(1.0, 0.001), 1000):.3f}")
print(f"{walk_forwarding(1000):.3f} {walk_scalars(1000):.3f} {walk((0.0, 0.0), 1000):.3f}")
print(midpoint((0.0, 0.0), (2.0, 4.0)))
"""


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def _explain(workspace: Path, name: str) -> str:
    done = _run([sys.executable, "-m", "ppy_compiler", "explain", f"limits.{name}"], workspace)
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.fixture(name="workspace")
def _workspace(tmp_path: Path) -> Path:
    (tmp_path / "limits.ppy").write_text(textwrap.dedent(PROGRAM).lstrip("\n"), encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    return tmp_path


@requires_llvm
def test_the_documented_limits_are_the_ones_explain_reports(workspace: Path):
    """The two documented fallbacks, and the two shapes the examples use instead, by name."""
    assert "llvm backend: boxed: `Vec3` has no native lowering" in _explain(workspace, "orbit")
    assert "llvm backend: native" in _explain(workspace, "travel"), "a value class as a parameter"
    assert "llvm backend: boxed: `Tuple` has no native lowering" in _explain(
        workspace, "walk_forwarding"
    ), "a tuple local rebound from a native call"
    assert "llvm backend: boxed: a tuple result cannot be forwarded between native calls yet" in (
        _explain(workspace, "walk_scalars")
    ), "a tuple result unpacked straight into scalars"
    assert "llvm backend: native" in _explain(workspace, "walk"), "a tuple as a parameter"
    assert "llvm backend: native" in _explain(workspace, "midpoint"), "a tuple result, from Python"


@requires_llvm
def test_a_function_left_in_python_still_answers_as_python_does(workspace: Path):
    plain = _run([sys.executable, "limits.ppy"], workspace)
    native = _run([sys.executable, "-m", "ppy_compiler", "run", "limits.ppy"], workspace)
    assert plain.returncode == 0, plain.stderr
    assert native.returncode == 0, native.stderr
    assert native.stdout == plain.stdout
    assert native.stdout.splitlines()[2] == "(1.0, 2.0)"
