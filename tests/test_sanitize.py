"""Sanitizers, the stage debugger, and the optimization report."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.driver.report import categorize
from ppy_compiler.ir.transforms import sanitizer_kinds

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

PROGRAM = """
    from ppy import native


    def pick(xs: list[int], i: int) -> int:
        return xs[i] * 2


    def grow(n: int) -> int:
        total = 1
        for _ in range(n):
            total = total * 3
        return total


    def deref(p: native.ptr[float]) -> float:
        return native.load(p) + 1.0
    """


def _write(directory: Path, extra: str = "") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\npipeline = "ir"\n' + extra, encoding="utf-8"
    )
    (directory / "san.ppy").write_text(textwrap.dedent(PROGRAM).lstrip("\n"), encoding="utf-8")


def _ppy(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_kinds_are_spelled_and_the_refused_ones_say_why():
    assert sanitizer_kinds("bounds, overflow") == {"bounds", "overflow"}
    assert sanitizer_kinds(["pointer", "alignment"]) == {"pointer", "alignment"}
    with pytest.raises(ValueError, match="not a sanitizer yet: stack lifetime"):
        sanitizer_kinds("lifetime")
    with pytest.raises(ValueError, match="`race` is not a sanitizer; the sanitizers are"):
        sanitizer_kinds("race")
    assert categorize("@f: 2 call(s) inlined") == "function inlined"
    assert categorize("sanitizer: 3 check(s) inserted in @f") == "sanitizer checks inserted"
    assert categorize("`k` compiled to PTX for sm_70") == "GPU kernel emitted"
    assert categorize("something else entirely") == "note"


@requires_llvm
def test_sanitized_ir_carries_the_checks_and_a_failed_check_raises(tmp_path: Path):
    _write(tmp_path)
    plain = _ppy(tmp_path, "emit", "ir", "san.ppy")
    assert plain.returncode == 0, plain.stderr
    assert "sanitize:" not in plain.stdout
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\npipeline = "ir"\nsafeguards = "off"\n'
        'sanitize = ["bounds", "overflow", "pointer", "alignment"]\n',
        encoding="utf-8",
    )
    sanitized = _ppy(tmp_path, "emit", "ir", "san.ppy")
    assert sanitized.returncode == 0, sanitized.stderr
    text = sanitized.stdout
    assert 'label = "sanitize:bounds"' in text and 'label = "sanitize:overflow"' in text
    assert 'label = "sanitize:pointer"' in text and 'label = "sanitize:alignment"' in text
    assert "ppy.checked_mul" in text and "core.const 0 : ptr<f64>" in text
    program = tmp_path / "main.ppy"
    program.write_text(
        textwrap.dedent(PROGRAM).lstrip("\n")
        + textwrap.dedent(
            """

            from ppy_runtime.binding import SanitizerFailure

            print(pick([1, 2, 3], 1), grow(3), deref(native.stack_alloc[float](1)))
            for attempt in (lambda: pick([1, 2, 3], 7), lambda: grow(60)):
                try:
                    print(attempt())
                except SanitizerFailure as failure:
                    print(str(failure).split(" (")[0])
                except IndexError:
                    print("IndexError: the frontend's own bounds check fell back to Python")
            """
        ),
        encoding="utf-8",
    )
    run = _ppy(tmp_path, "run", "--safeguards", "off", "--sanitize", "bounds,overflow", "main.ppy")
    assert run.returncode == 0, run.stderr
    assert run.stdout == (
        "4 27 1.0\n"
        "IndexError: the frontend's own bounds check fell back to Python\n"
        "sanitizer: a overflow check failed in `main.grow`\n"
    )
    wrapped = _ppy(tmp_path, "run", "--safeguards", "off", "main.ppy")
    assert wrapped.returncode == 0, wrapped.stderr
    assert wrapped.stdout.splitlines()[2] not in {"42391158275216203514294433201", "sanitizer"}, (
        "without the sanitizer a wrapping build answers with the wrapped number"
    )
    refused = _ppy(tmp_path, "run", "--sanitize", "lifetime", "main.ppy")
    assert refused.returncode == 2 and "not a sanitizer yet" in refused.stderr


@requires_llvm
def test_inspect_shows_every_stage_and_build_reports(tmp_path: Path):
    _write(tmp_path)
    for stage, marker in (
        ("analysis", "san.pick: (list[int], int) -> int"),
        ("ir", "core.guard %"),
        ("canonical", "func @san_grow"),
        ("optimized", "func @san_pick"),
        ("llvm", "define"),
    ):
        shown = _ppy(tmp_path, "inspect", "san.ppy", "--stage", stage)
        assert shown.returncode == 0, (stage, shown.stderr)
        assert f"[{stage}]" in shown.stdout and marker in shown.stdout, (stage, shown.stdout[:400])
    analysis = _ppy(tmp_path, "inspect", "san.ppy", "--stage", "analysis").stdout
    assert "native: eligible" in analysis and "boundary: bound" in analysis
    empty = _ppy(tmp_path, "inspect", "san.ppy", "--stage", "gpu")
    assert empty.returncode == 0 and empty.stdout == "", "no device code, nothing to show"
    built = _ppy(
        tmp_path, "build", "san.ppy", "--report-opt", "--report-opt-json", "out/report.json"
    )
    assert built.returncode == 0, built.stderr
    assert (
        "optimization report: " in built.stdout
        and "san.pick: native, bound to Python" in built.stdout
    )
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert (
        report["pipeline"] == "ir" and report["modules"]["san"]["functions"]["san.grow"]["native"]
    )
    assert isinstance(report["modules"]["san"]["remarks"], list)
    assert all("category" in remark for remark in report["modules"]["san"]["remarks"])
