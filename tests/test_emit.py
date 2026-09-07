"""`ppy emit` and building from `.ppyir`."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import toolchain_status
from ppy_compiler.ir import decode, verify

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
_usable, _detail = toolchain_status()
requires_toolchain = pytest.mark.skipif(not _usable, reason=f"no native toolchain: {_detail}")

KERNEL = """\
def scale(xs: list[int], k: int) -> int:
    total = 0
    for x in xs:
        total += x * k
    return total


def halve(n: int) -> int:
    return n // 2
"""


def _ppy(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    path = tmp_path / "kernel.ppy"
    path.write_text(KERNEL, encoding="utf-8")
    return path


@requires_llvm
def test_emit_ir_prints_a_ppyir_module_that_reads_back(tmp_path: Path):
    _project(tmp_path)
    done = _ppy(tmp_path, "emit", "ir", "kernel.ppy")
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("ppyir 1\nmodule @kernel\n")
    module = decode(done.stdout)
    assert not verify(module)
    assert sorted(module.functions) == ["kernel_halve", "kernel_scale"]
    scale = module.functions["kernel_scale"]
    assert scale.attributes["ppy.symbol"] == "ppy_kernel_scale"
    assert scale.param_attributes[0] == {"ppy.kind": "list", "ownership": "owned"}
    assert 'overflow = "python"' in done.stdout
    again = _ppy(tmp_path, "emit", "ir", "kernel.ppy")
    assert again.stdout == done.stdout, "one input, one text"


@requires_llvm
def test_emit_writes_where_told_and_refuses_to_guess_for_a_directory(tmp_path: Path):
    _project(tmp_path)
    (tmp_path / "other.ppy").write_text("def one() -> int:\n    return 1\n", encoding="utf-8")
    to_file = _ppy(tmp_path, "emit", "ir", "kernel.ppy", "-o", "out/kernel.ppyir")
    assert to_file.returncode == 0, to_file.stderr
    assert to_file.stdout == ""
    assert (tmp_path / "out" / "kernel.ppyir").read_text(encoding="utf-8").startswith("ppyir 1\n")
    no_output = _ppy(tmp_path, "emit", "ir", ".")
    assert no_output.returncode == 2
    assert "needs `-o DIR`" in no_output.stderr
    directory = _ppy(tmp_path, "emit", "ir", ".", "-o", "ir")
    assert directory.returncode == 0, directory.stderr
    assert sorted(p.name for p in (tmp_path / "ir").iterdir()) == ["kernel.ppyir", "other.ppyir"]
    llvm = _ppy(tmp_path, "emit", "llvm-ir", "kernel.ppy")
    assert llvm.returncode == 0, llvm.stderr
    assert "define" in llvm.stdout and "ppy_kernel_scale" in llvm.stdout


@requires_toolchain
def test_a_ppyir_file_builds_into_a_library_and_a_manifest(tmp_path: Path):
    _project(tmp_path)
    assert _ppy(tmp_path, "emit", "ir", "kernel.ppy", "-o", "kernel.ppyir").returncode == 0
    (tmp_path / "kernel.ppy").unlink()  # the IR is all the build has
    built = _ppy(tmp_path, "build", "kernel.ppyir", "-o", "dist")
    assert built.returncode == 0, built.stderr
    assert "library:" in built.stderr and "manifest:" in built.stderr
    manifest = json.loads((tmp_path / "dist" / "ppy-bindings.json").read_text(encoding="utf-8"))
    entries = {e["python_qualname"]: e for e in manifest["entries"]}
    assert set(entries) == {"kernel.scale", "kernel.halve"}
    scale = entries["kernel.scale"]
    assert scale["native_symbol"] == "ppy_kernel_scale"
    assert [p["kind"] for p in scale["abi"]["parameters"]] == ["list", "int"]
    assert scale["abi"]["parameters"][0]["element"] == "int"
    assert (tmp_path / "dist" / "libppy_kernel.so").is_file()


def test_a_foreign_or_stale_ppyir_is_refused_with_the_reason(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    stale = tmp_path / "stale.ppyir"
    stale.write_text("ppyir 42\nmodule @stale\n", encoding="utf-8")
    done = _ppy(tmp_path, "build", "stale.ppyir")
    assert done.returncode == 2
    assert "schema 42; this compiler reads schema 1" in done.stderr
    foreign = tmp_path / "foreign.ppyir"
    foreign.write_text("ppyir 1\nmodule @foreign\ndialect tpu 1\n", encoding="utf-8")
    done = _ppy(tmp_path, "build", "foreign.ppyir")
    assert done.returncode == 2
    assert "dialect 'tpu', which this compiler does not have" in done.stderr
