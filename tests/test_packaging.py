"""`ppy build --library` and `ppy build --target`: packaged, and for another machine."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import toolchain_status
from ppy_compiler.target import host_target

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
_usable, _detail = toolchain_status()
requires_toolchain = pytest.mark.skipif(not _usable, reason=f"no native toolchain: {_detail}")

LIBRARY = """\
from ppy import Buffer, native


@native.export(name="ppy_dot")
def dot(a: Buffer[float], b: Buffer[float], n: int) -> float:
    total = 0.0
    for i in range(n):
        total += a[i] * b[i]
    return total


@native.export()
def clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else (hi if x > hi else x)
"""

PLAIN = """\
def total(xs: list[int]) -> int:
    result = 0
    for x in xs:
        result += x
    return result
"""

#: ELF `e_machine` values, to read an object's architecture off its header.
_ELF_MACHINES = {"x86_64": 62, "aarch64": 183, "riscv64": 243}


def _build(project: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", *args],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
        env={"PPY_LOWERING": "ir", **__import__("os").environ},
    )


def _project(tmp_path: Path, source: str, name: str = "lib.ppy") -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / name).write_text(source, encoding="utf-8")
    return tmp_path / name


@requires_toolchain
def test_a_library_build_lays_out_lib_include_and_pkgconfig(tmp_path: Path):
    _project(tmp_path, LIBRARY)
    built = _build(tmp_path, "lib.ppy", "--library", "-o", "dist")
    assert built.returncode == 0, built.stderr
    name = tmp_path.name
    dist = tmp_path / "dist"
    library = dist / "lib" / f"libppy_{name}.so"
    header = dist / "include" / f"{name}.h"
    pc = dist / "lib" / "pkgconfig" / f"{name}.pc"
    assert library.is_file() and header.is_file() and pc.is_file()
    assert not (dist / f"libppy_{name}.so").exists(), "the library moved under lib/"
    assert not (dist / f"{name}.h").exists()
    assert "double ppy_dot(" in header.read_text(encoding="utf-8")
    spelled = pc.read_text(encoding="utf-8")
    assert f"Libs: -L${{libdir}} -lppy_{name}" in spelled and "Cflags: -I${includedir}" in spelled
    assert "prefix=${pcfiledir}/../.." in spelled
    manifest = json.loads((dist / "ppy-bindings.json").read_text(encoding="utf-8"))
    assert set(manifest["exports"]) == {"ppy_dot", "clamp"}
    assert "package:  dist" in built.stderr
    assert not list(dist.glob("ppy_wrappers_*")), "a C consumer needs no Python boundary"
    assert not (dist / "lib").is_file()


@requires_toolchain
def test_a_library_build_with_nothing_to_export_says_so(tmp_path: Path):
    _project(tmp_path, PLAIN)
    built = _build(tmp_path, "lib.ppy", "--library", "-o", "dist")
    assert built.returncode == 2
    assert "E1805" in built.stderr and "@ppy.native.export" in built.stderr


@requires_llvm
def test_a_cross_build_emits_objects_for_the_target_and_says_what_it_left_out(tmp_path: Path):
    _project(tmp_path, LIBRARY)
    host = host_target()
    other = "aarch64-linux-gnu" if host.architecture != "aarch64" else "x86_64-linux-gnu"
    built = _build(tmp_path, "lib.ppy", "--target", other, "-o", "dist")
    assert built.returncode == 0, built.stderr
    canonical = f"{other.split('-', maxsplit=1)[0]}-unknown-linux-gnu"
    assert f"target:   {canonical}" in built.stderr
    objects = list((tmp_path / "dist").glob("*.o"))
    assert objects, built.stderr
    header = objects[0].read_bytes()
    assert header[:4] == b"\x7fELF"
    assert int.from_bytes(header[18:20], "little") == _ELF_MACHINES[other.split("-", maxsplit=1)[0]]
    manifest = json.loads((tmp_path / "dist" / "ppy-bindings.json").read_text(encoding="utf-8"))
    assert manifest["target"] == canonical
    assert "built on that machine" in built.stderr, "the wrapper and launcher are not for here"
    assert not list((tmp_path / "dist").glob("ppy_wrappers_*"))
    assert not (tmp_path / "dist" / "lib").exists()
    assert (tmp_path / "dist" / f"{tmp_path.name}.h").is_file(), "the header is target-neutral"
    ran = subprocess.run(
        [
            sys.executable,
            "-m",
            "ppy_compiler",
            "run",
            "--prebuilt",
            "dist/ppy-bindings.json",
            "lib.ppy",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 2
    assert f"built for {canonical}" in ran.stderr


@requires_llvm
def test_a_target_this_compiler_cannot_describe_is_refused(tmp_path: Path):
    _project(tmp_path, LIBRARY)
    built = _build(tmp_path, "lib.ppy", "--target", "x86_64-plan9")
    assert built.returncode == 2
    assert "E1002" in built.stderr and "operating system" in built.stderr
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\ntarget = "foo-linux-gnu"\n', encoding="utf-8"
    )
    configured = _build(tmp_path, "lib.ppy")
    assert configured.returncode == 2 and "architecture" in configured.stderr
