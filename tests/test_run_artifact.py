"""The artifact a warm `ppy run` launches is compiled for the machine that built it."""

from __future__ import annotations

from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.driver.reporting import Reporter

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

PROGRAM = """
    def total(n: int) -> int:
        acc = 0
        for i in range(n):
            acc = acc + i * i
        return acc


    print(total(1000))
    """


@requires_llvm
def test_the_run_artifact_is_compiled_for_the_host_cpu(write, analyze, tmp_path, monkeypatch):
    from ppy_compiler.backend import llvm

    path = write("hot.ppy", PROGRAM)
    (path.parent / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\npipeline = "ir"\n', encoding="utf-8"
    )
    bundle = analyze(path, backend="llvm")
    assert not bundle.diagnostics.has_errors(), [d.message for d in bundle.diagnostics.errors]
    seen: list[bool] = []
    real = llvm.emit_object

    def watched(engine, ir, destination, *, host_cpu=False, target=None):  # type: ignore[no-untyped-def]
        seen.append(host_cpu)
        return real(engine, ir, destination, host_cpu=host_cpu, target=target)

    monkeypatch.setattr(llvm, "emit_object", watched)
    manifest = llvm.compile_for_run(bundle, Reporter(quiet=True), tmp_path / "run", path.resolve())
    assert manifest is not None and manifest.is_file()
    assert seen and all(seen), "the run artifact is JIT code that never leaves this machine"

    seen.clear()
    built = llvm.compile_project(bundle, Reporter(quiet=True), output=tmp_path / "dist")
    assert built.objects
    assert seen and not any(seen), "a build is portable unless --host-cpu asks otherwise"


def test_the_run_key_carries_the_cpu(tmp_path: Path, monkeypatch):
    from ppy_compiler.driver import warm

    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    entry = tmp_path / "prog.ppy"
    entry.write_text("print(1)\n", encoding="utf-8")
    import argparse

    options = argparse.Namespace()
    before = warm.locate(entry, options).directory
    monkeypatch.setattr(warm, "_cpu_features", lambda: ("avx2", "fma", "some-other-machine"))
    after = warm.locate(entry, options).directory
    assert before != after, "another CPU is another run directory"
