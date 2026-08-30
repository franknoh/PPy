"""Native object emission, linking, launcher, and manifest (spec 4.2, 16.3, 26.2)."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import toolchain_status

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
_usable, _detail = toolchain_status()
requires_toolchain = pytest.mark.skipif(not _usable, reason=f"no native toolchain: {_detail}")

pytestmark = requires_llvm

SOURCE = """
import ppy
from ppy import f64


@ppy.pure
@ppy.opt(3)
def distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64:
    dx: f64 = x2 - x1
    dy: f64 = y2 - y1
    return (dx * dx + dy * dy) ** 0.5


@ppy.pure
def total(xs: list[int]) -> int:
    result: int = 0
    for x in xs:
        result += x
    return result


def main() -> None:
    print(round(distance(0.0, 0.0, 3.0, 4.0), 6), total([1, 2, 3]))


if __name__ == "__main__":
    main()
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "app.ppy").write_text(textwrap.dedent(SOURCE).lstrip("\n"), encoding="utf-8")
    return tmp_path


def _build(project: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", "app.ppy", "--backend", "llvm", *extra],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )


def test_build_writes_a_binding_manifest(project: Path):
    result = _build(project)
    assert result.returncode == 0, result.stderr

    manifest = json.loads((project / ".ppy-cache" / "native" / "ppy-bindings.json").read_text())
    assert manifest["abi_version"] == 1
    assert manifest["calling_convention"] == "c"

    entries = {entry["python_qualname"]: entry for entry in manifest["entries"]}
    assert "app.distance" in entries and "app.total" in entries

    distance = entries["app.distance"]
    assert distance["native_symbol"] == "ppy_app_distance"
    assert [a["native_type"] for a in distance["arguments"]] == ["double"] * 4
    assert distance["returns"]["passed_as"] == "out_parameter"
    assert distance["gil"] == "not_required"
    assert "re-run the Python implementation" in distance["status"]["meaning"]

    buffer = entries["app.total"]["arguments"][0]
    assert buffer["semantic_type"] == "list[int]"
    assert buffer["ownership"] == "borrowed"


@requires_toolchain
def test_build_emits_objects_and_links_a_library(project: Path):
    result = _build(project)
    assert result.returncode == 0, result.stderr

    native = project / ".ppy-cache" / "native"
    objects = list(native.glob("*.o"))
    libraries = list(native.glob("*.so"))
    assert objects and libraries
    assert libraries[0].stat().st_size > 0

    symbols = subprocess.run(
        ["nm", "-D", "--defined-only", str(libraries[0])],
        capture_output=True,
        text=True,
        check=False,
    )
    if symbols.returncode == 0:
        assert "ppy_app_distance" in symbols.stdout


@requires_toolchain
def test_build_produces_a_runnable_native_executable(project: Path):
    result = _build(project)
    assert result.returncode == 0, result.stderr

    launcher = project / ".ppy-cache" / "native" / "app"
    assert launcher.is_file()
    assert launcher.stat().st_mode & 0o111

    native = subprocess.run(
        [str(launcher)], cwd=project, capture_output=True, text=True, check=False
    )
    plain = subprocess.run(
        [sys.executable, "app.ppy"], cwd=project, capture_output=True, text=True, check=False
    )
    assert plain.returncode == 0, plain.stderr
    assert native.returncode == 0, native.stderr
    assert native.stdout == plain.stdout


@requires_toolchain
def test_the_launcher_forwards_program_arguments(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    (tmp_path / "args.ppy").write_text("import sys\n\nprint(sys.argv[1:])\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", "args.ppy", "--backend", "llvm"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    launcher = tmp_path / ".ppy-cache" / "native" / "args"
    if not launcher.is_file():
        pytest.skip("no launcher was produced")
    native = subprocess.run(
        [str(launcher), "one", "two"], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert native.stdout.strip() == "['one', 'two']"


@requires_toolchain
def test_the_launcher_runs_the_prebuilt_library_not_a_jit(project: Path):
    """The binary is `ppy run` in a compiled coat, fed by the built library.

    Deleting the library must break the launcher: a launcher that still
    worked would be quietly recompiling or, worse, quietly interpreting.
    """
    result = _build(project)
    assert result.returncode == 0, result.stderr
    native = project / ".ppy-cache" / "native"
    launcher = native / "app"
    if not launcher.is_file():
        pytest.skip("no launcher was produced")

    plain = subprocess.run(
        [sys.executable, "app.ppy"], cwd=project, capture_output=True, text=True, check=False
    )
    ran = subprocess.run([str(launcher)], cwd=project, capture_output=True, text=True, check=False)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == plain.stdout

    library = next(native.glob("libppy_*.so"))
    library.unlink()
    broken = subprocess.run(
        [str(launcher)], cwd=project, capture_output=True, text=True, check=False
    )
    assert broken.returncode != 0
    assert "E1801" in broken.stderr


@requires_toolchain
def test_run_binds_from_a_prebuilt_manifest(project: Path):
    result = _build(project)
    assert result.returncode == 0, result.stderr
    manifest = project / ".ppy-cache" / "native" / "ppy-bindings.json"

    plain = subprocess.run(
        [sys.executable, "app.ppy"], cwd=project, capture_output=True, text=True, check=False
    )
    ran = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "run", "--prebuilt", str(manifest), "app.ppy"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == plain.stdout


@requires_toolchain
def test_build_defaults_to_wrap_semantics_and_safe_restores_python_integers(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    (tmp_path / "spin.ppy").write_text(
        textwrap.dedent(
            """
            import ppy


            @ppy.pure
            def spin(n: int) -> int:
                value: int = 3
                for _i in range(n):
                    value = value * 2654435761 + 1
                return value


            print(spin(41))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    plain = subprocess.run(
        [sys.executable, "spin.ppy"], cwd=tmp_path, capture_output=True, text=True, check=False
    )

    def built(*extra: str) -> str:
        out = tmp_path / ("safe" if extra else "fast")
        result = subprocess.run(
            [sys.executable, "-m", "ppy_compiler", "build", *extra, "spin.ppy", "-o", str(out)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        launcher = out / "spin"
        if not launcher.is_file():
            pytest.skip("no launcher was produced")
        ran = subprocess.run(
            [str(launcher)], cwd=tmp_path, capture_output=True, text=True, check=False
        )
        assert ran.returncode == 0, ran.stderr
        return ran.stdout

    assert built().strip() == "8606135309836935036"
    assert built("--safe") == plain.stdout


@requires_toolchain
def test_the_launcher_never_imports_the_compiler(project: Path, tmp_path: Path):
    """A built artifact is compiled software: `ppy_compiler` may be gone.

    The compiler package is poisoned via PYTHONPATH; if any part of the
    runtime path imported it, the launcher would die on the spot.
    """
    result = _build(project)
    assert result.returncode == 0, result.stderr
    launcher = project / ".ppy-cache" / "native" / "app"
    if not launcher.is_file():
        pytest.skip("no launcher was produced")
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "ppy_compiler.py").write_text(
        'raise ImportError("a built artifact must not import the compiler")\n',
        encoding="utf-8",
    )
    plain = subprocess.run(
        [sys.executable, "app.ppy"], cwd=project, capture_output=True, text=True, check=False
    )
    import os

    environment = dict(os.environ, PYTHONPATH=str(poison))
    ran = subprocess.run(
        [str(launcher)],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == plain.stdout


@requires_toolchain
def test_the_manifest_records_the_program_and_the_abi(project: Path):
    result = _build(project)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((project / ".ppy-cache" / "native" / "ppy-bindings.json").read_text())
    program = manifest["program"]
    assert program["entry"] == "app"
    assert "app" in program["generated"]
    entry = next(e for e in manifest["entries"] if e["python_qualname"] == "app.distance")
    assert entry["module"] == "app"
    assert entry["binding"] == "distance"
    assert entry["abi"]["returns"] == ["double"]
    assert [p["kind"] for p in entry["abi"]["parameters"]] == ["float"] * 4


def test_build_honours_an_explicit_output_directory(project: Path, tmp_path: Path):
    destination = tmp_path / "out"
    result = _build(project, "-o", str(destination))
    assert result.returncode == 0, result.stderr
    assert (destination / "ppy-bindings.json").is_file()


def test_doctor_reports_the_native_toolchain(project: Path):
    result = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "doctor"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "native toolchain" in result.stdout
