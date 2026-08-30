"""The migration pass framework and `ppy migrate` reporting."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.diagnostics import Diagnostic, Severity
from ppy_compiler.migration import Classification, apply_passes, classify


def _ppy(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    return tmp_path


def test_literal_attribute_calls_become_plain_access():
    source = textwrap.dedent(
        """
        class Holder:
            pass


        holder = Holder()
        setattr(holder, "status", 3)
        value = getattr(holder, "status")
        delattr(holder, "status")
        """
    ).lstrip("\n")
    rewritten, rewrites = apply_passes(source)
    assert "holder.status = 3" in rewritten
    assert "value = holder.status" in rewritten
    assert "del holder.status" in rewritten
    assert "setattr" not in rewritten
    assert len(rewrites) == 3
    assert {r.pass_name for r in rewrites} == {"literal-attributes"}


def test_computed_names_and_defaults_are_left_alone():
    source = textwrap.dedent(
        """
        holder = object()
        name = "status"
        setattr(holder, name, 3)
        fallback = getattr(holder, "status", None)
        """
    ).lstrip("\n")
    rewritten, rewrites = apply_passes(source)
    assert rewritten == source
    assert not rewrites


def test_a_shadowed_builtin_stays_a_call():
    source = textwrap.dedent(
        """
        def setattr(target, name, value):
            print("custom", name)


        holder = object()
        setattr(holder, "status", 3)
        """
    ).lstrip("\n")
    rewritten, rewrites = apply_passes(source)
    assert rewritten == source
    assert not rewrites


def test_constant_import_module_becomes_an_import():
    source = 'import importlib\n\nmath = importlib.import_module("math")\n'
    rewritten, rewrites = apply_passes(source)
    assert "import math\n" in rewritten
    assert "import_module" not in rewritten
    # The rewrite consumed the only use, so the feeding import goes too.
    assert "import importlib" not in rewritten
    assert [r.pass_name for r in rewrites] == ["static-imports", "static-imports"]


def test_a_still_used_importlib_import_is_kept():
    source = (
        "import importlib\n\n"
        'math = importlib.import_module("math")\n'
        "other = importlib.import_module(pick())\n"
    )
    rewritten, rewrites = apply_passes(source)
    assert "import importlib\n" in rewritten
    assert [r.pass_name for r in rewrites] == ["static-imports"]


def test_module_namespace_writes_become_assignments():
    source = 'globals()["LIMIT"] = 128\nprint(LIMIT)\n'
    rewritten, rewrites = apply_passes(source)
    assert rewritten.startswith("LIMIT = 128\n")
    assert [r.pass_name for r in rewrites] == ["module-namespace-writes"]


def test_function_scope_namespace_writes_are_not_touched():
    source = textwrap.dedent(
        """
        def install() -> None:
            globals()["LIMIT"] = 128
        """
    ).lstrip("\n")
    rewritten, rewrites = apply_passes(source)
    assert rewritten == source
    assert not rewrites


@pytest.mark.parametrize(
    ("code", "severity", "expected"),
    [
        ("E1501", Severity.ERROR, Classification.UNSUPPORTED),
        ("E1504", Severity.WARNING, Classification.DYNAMIC_BOUNDARY),
        ("E1505", Severity.ERROR, Classification.DYNAMIC_BOUNDARY),
        ("R3003", Severity.REMARK, Classification.OPTIMIZATION_OPPORTUNITY),
        ("E1304", Severity.WARNING, Classification.REQUIRES_REWRITE),
        ("E1301", Severity.ERROR, Classification.REQUIRES_REWRITE),
        ("W2002", Severity.WARNING, None),
    ],
)
def test_leftover_findings_classify_by_meaning(code, severity, expected):
    assert classify(Diagnostic(code, severity, "x")) is expected


def test_migrate_applies_passes_and_writes_a_report(workspace: Path):
    (workspace / "legacy.py").write_text(
        textwrap.dedent(
            """
            class Holder:
                def __init__(self):
                    self.status = 0


            def run(snippet):
                return eval(snippet)


            holder = Holder()
            setattr(holder, "status", 3)
            globals()["LIMIT"] = 128
            print(holder.status, LIMIT, run("1 + 1"))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    result = _ppy(["migrate", "legacy.py", "--report", "migration.json"], workspace)
    assert result.returncode == 0, result.stderr
    converted = (workspace / "legacy.ppy").read_text(encoding="utf-8")
    assert "holder.status = 3" in converted
    assert "LIMIT" in converted and 'globals()["LIMIT"]' not in converted
    assert "migration summary" in result.stderr

    report = json.loads((workspace / "migration.json").read_text(encoding="utf-8"))
    assert report["files_changed"] == 1
    assert report["sites_rewritten"] == 2
    assert report["counts"]["autofixed"] == 2
    assert any(f["classification"] == "unsupported" for f in report["findings"])
    # The verdict is about the final output: the eval survives migration, so
    # the result is not yet strict PPY, and the report says so.
    assert report["strict_ready"] is False
    assert report["strict_errors"] >= 1
    listed = {(r["pass"], r["line"]) for r in report["rewrites"]}
    assert ("literal-attributes", 11) in listed
    assert ("module-namespace-writes", 12) in listed


def test_the_report_judges_the_final_output_not_the_input(workspace: Path):
    """A pinned unknown decorator leaves strict-invalid output; the report says so."""
    (workspace / "wrapped.py").write_text(
        textwrap.dedent(
            """
            def registry(fn):
                return fn


            @registry
            def opaque(x):
                return x * 2


            print(opaque(1))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    result = _ppy(["migrate", "wrapped.py", "--report", "migration.json"], workspace)
    assert result.returncode == 0, result.stderr
    report = json.loads((workspace / "migration.json").read_text(encoding="utf-8"))
    assert report["strict_ready"] is False
    codes = {f["code"] for f in report["findings"]}
    assert "E1204" in codes
    assert "E1201" in codes


def test_a_complete_migration_reports_strict_ready(workspace: Path):
    (workspace / "clean.py").write_text(
        "def double(x):\n    return x * 2\n\n\nprint(double(21))\n", encoding="utf-8"
    )
    result = _ppy(["migrate", "clean.py", "--report", "migration.json"], workspace)
    assert result.returncode == 0, result.stderr
    report = json.loads((workspace / "migration.json").read_text(encoding="utf-8"))
    assert report["strict_ready"] is True
    assert report["strict_errors"] == 0
    assert "strict ready:               yes" in result.stderr


def test_migrate_diff_shows_the_change_and_writes_nothing(workspace: Path):
    (workspace / "small.py").write_text(
        "def double(x):\n    return x * 2\n\n\nprint(double(21))\n", encoding="utf-8"
    )
    result = _ppy(["migrate", "small.py", "--diff"], workspace)
    assert result.returncode == 0, result.stderr
    assert not (workspace / "small.ppy").exists()
    assert "+def double(x: int) -> int:" in result.stdout
    assert "-def double(x):" in result.stdout


def test_migrated_output_still_runs(workspace: Path):
    (workspace / "steps.py").write_text(
        textwrap.dedent(
            """
            class Holder:
                def __init__(self):
                    self.count = 0


            holder = Holder()
            setattr(holder, "count", 2)
            print(getattr(holder, "count") * 3)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["migrate", "steps.py", "--in-place"], workspace).returncode == 0
    executed = subprocess.run(
        [sys.executable, "steps.ppy"], cwd=workspace, capture_output=True, text=True, check=False
    )
    assert executed.returncode == 0, executed.stderr
    assert executed.stdout == "6\n"
