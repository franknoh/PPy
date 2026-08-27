"""CLI surface: convert, check, fmt, explain, inspect, cache (spec 4.3)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.driver.cli import main


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
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    return tmp_path


def test_version_and_help(capsys):
    assert main(["--version"]) == 0
    assert "ppy" in capsys.readouterr().out
    assert main([]) == 0


def test_check_reports_success(workspace: Path):
    (workspace / "ok.ppy").write_text("def f(x: int) -> int:\n    return x\n", encoding="utf-8")
    result = _ppy(["check", "ok.ppy"], workspace)
    assert result.returncode == 0
    assert "no errors" in result.stderr


def test_check_reports_failure_with_a_code_and_span(workspace: Path):
    (workspace / "bad.ppy").write_text("def f(x):\n    return x\n", encoding="utf-8")
    result = _ppy(["check", "bad.ppy"], workspace)
    assert result.returncode == 1
    assert "error[E1201]" in result.stderr
    assert "bad.ppy:1:1" in result.stderr
    assert "= help:" in result.stderr


def test_check_scans_a_directory(workspace: Path):
    (workspace / "a.ppy").write_text("def a() -> int:\n    return 1\n", encoding="utf-8")
    (workspace / "b.ppy").write_text("def b() -> int:\n    return 2\n", encoding="utf-8")
    result = _ppy(["check", "."], workspace)
    assert result.returncode == 0
    assert "2 module(s)" in result.stderr


def test_convert_writes_a_ppy_file_and_preserves_the_source(workspace: Path):
    source = workspace / "mod.py"
    source.write_text(
        textwrap.dedent(
            '''
            """Docstring."""
            from __future__ import annotations


            def square(x):
                # keep me
                return x * x


            answer = square(7)
            '''
        ).lstrip("\n"),
        encoding="utf-8",
    )
    original = source.read_text(encoding="utf-8")
    result = _ppy(["convert", "mod.py"], workspace)
    assert result.returncode == 0

    converted = (workspace / "mod.ppy").read_text(encoding="utf-8")
    assert source.read_text(encoding="utf-8") == original
    assert "def square(x: int) -> int:" in converted
    assert "# keep me" in converted
    assert '"""Docstring."""' in converted
    assert "answer: int = square(7)" in converted


def test_convert_inserts_ppy_after_docstring_and_future(workspace: Path):
    source = workspace / "order.py"
    source.write_text(
        '"""Doc."""\nfrom __future__ import annotations\n\nX = 1\n', encoding="utf-8"
    )
    _ppy(["convert", "order.py"], workspace)
    lines = (workspace / "order.ppy").read_text(encoding="utf-8").splitlines()
    assert lines[0] == '"""Doc."""'
    assert lines[1] == "from __future__ import annotations"
    assert lines[2] == "import ppy"


def test_convert_refuses_to_clobber_without_force(workspace: Path):
    (workspace / "exists.py").write_text("X = 1\n", encoding="utf-8")
    (workspace / "exists.ppy").write_text("# handwritten\n", encoding="utf-8")
    result = _ppy(["convert", "exists.py"], workspace)
    assert result.returncode == 1
    assert "--force" in result.stderr
    assert (workspace / "exists.ppy").read_text(encoding="utf-8") == "# handwritten\n"


def test_convert_in_place_replaces_the_source(workspace: Path):
    source = workspace / "inplace.py"
    source.write_text("def f(x):\n    return x + 1\n\n\nf(1)\n", encoding="utf-8")
    result = _ppy(["convert", "inplace.py", "--in-place"], workspace)
    assert result.returncode == 0
    assert "def f(x: int) -> int:" in source.read_text(encoding="utf-8")
    assert not (workspace / "inplace.ppy").exists()


def test_convert_reports_uninferable_parameters(workspace: Path):
    (workspace / "vague.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    result = _ppy(["convert", "vague.py", "--dry-run"], workspace)
    assert "E1304" in result.stderr
    assert "Any" not in (workspace / "vague.py").read_text(encoding="utf-8")


def test_convert_flags_dynamic_features(workspace: Path):
    (workspace / "dyn.py").write_text("def f(src: str):\n    return eval(src)\n", encoding="utf-8")
    result = _ppy(["convert", "dyn.py", "--dry-run"], workspace)
    assert "E1504" in result.stderr
    assert "ppy.dynamic" in result.stderr


def test_converted_output_runs_on_all_paths(workspace: Path):
    (workspace / "roundtrip.py").write_text(
        textwrap.dedent(
            """
            def scale(values, factor):
                return [v * factor for v in values]


            def main():
                print(scale([1.0, 2.0], 2.0))


            main()
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "roundtrip.py"], workspace).returncode == 0
    converted = workspace / "roundtrip.ppy"

    plain = subprocess.run(
        [sys.executable, converted.name], cwd=workspace, capture_output=True, text=True, check=False
    )
    optimized = _ppy([converted.name], workspace)
    assert plain.returncode == 0, plain.stderr
    assert optimized.returncode == 0, optimized.stderr
    assert optimized.stdout == plain.stdout == "[2.0, 4.0]\n"


def test_fmt_normalizes_and_check_mode_reports(workspace: Path):
    target = workspace / "messy.ppy"
    target.write_text("def a() -> int:\n    return 1\ndef b() -> int:\n    return 2\n", encoding="utf-8")

    check = _ppy(["fmt", "messy.ppy", "--check"], workspace)
    assert check.returncode == 1
    assert "would reformat" in check.stderr

    assert _ppy(["fmt", "messy.ppy"], workspace).returncode == 0
    assert _ppy(["fmt", "messy.ppy", "--check"], workspace).returncode == 0


def test_fmt_reports_a_syntax_error(workspace: Path):
    (workspace / "broken.ppy").write_text("def f(:\n", encoding="utf-8")
    result = _ppy(["fmt", "broken.ppy"], workspace)
    assert result.returncode == 1
    assert "E1001" in result.stderr


def test_explain_reports_optimization_decisions(workspace: Path):
    (workspace / "ex.ppy").write_text(
        textwrap.dedent(
            """
            import ppy


            @ppy.pure
            @ppy.opt(3)
            def total(xs: list[float]) -> float:
                result: float = 0.0
                for x in xs:
                    result += x
                return result
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    result = _ppy(["explain", "ex.ppy:6"], workspace)
    assert result.returncode == 0
    assert "function: total" in result.stdout
    assert "optimization: O3" in result.stdout
    assert "effects:" in result.stdout
    assert "parallel:" in result.stdout
    assert "representation:" in result.stdout


def test_explain_reports_why_parallelization_was_rejected(workspace: Path):
    (workspace / "why.ppy").write_text(
        textwrap.dedent(
            """
            import ppy


            def log_and_total(xs: list[int]) -> int:
                result: int = 0
                for x in xs:
                    print(x)
                    result += x
                return result
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    result = _ppy(["explain", "log_and_total"], workspace)
    assert "parallel: rejected" in result.stdout
    assert "reason:" in result.stdout
    assert "I/O" in result.stdout


def test_explain_describes_a_diagnostic_code(workspace: Path):
    result = _ppy(["explain", "E1304"], workspace)
    assert result.returncode == 0
    assert "E1304" in result.stdout


def test_inspect_prints_generated_python(workspace: Path):
    (workspace / "gen.ppy").write_text(
        "import ppy\n\n\n@ppy.pure\ndef f() -> int:\n    return 2 + 3\n", encoding="utf-8"
    )
    result = _ppy(["inspect", "gen.ppy"], workspace)
    assert result.returncode == 0
    assert "return 5" in result.stdout
    assert "@ppy.pure" not in result.stdout


def test_cache_status_clean_and_gc(workspace: Path):
    (workspace / "c.ppy").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    assert _ppy(["c.ppy"], workspace).returncode == 0

    status = _ppy(["cache", "status"], workspace)
    assert status.returncode == 0
    assert "entries:" in status.stdout

    assert _ppy(["cache", "gc"], workspace).returncode == 0
    assert _ppy(["cache", "clean"], workspace).returncode == 0
    assert _ppy(["clean"], workspace).returncode == 0
    assert not (workspace / ".ppy-cache").exists()


def test_doctor_reports_the_toolchain(workspace: Path):
    result = _ppy(["doctor"], workspace)
    assert result.returncode == 0
    assert "llvm backend" in result.stdout
    assert "plugins:" in result.stdout
    assert "numpy" in result.stdout


def test_program_arguments_reach_the_script(workspace: Path):
    (workspace / "args.ppy").write_text(
        "import sys\n\nprint(sys.argv[1:])\n", encoding="utf-8"
    )
    result = _ppy(["args.ppy", "--", "one", "two"], workspace)
    assert result.returncode == 0
    assert result.stdout.strip() == "['one', 'two']"


def test_opt_level_flag_overrides_the_project_setting(workspace: Path):
    (workspace / "lvl.ppy").write_text(
        "import ppy\n\n\ndef f() -> int:\n    total: int = 0\n    for i in range(3):\n        total += i\n    return total\n",
        encoding="utf-8",
    )
    at_o3 = _ppy(["-O", "3", "inspect", "lvl.ppy"], workspace)
    at_o1 = _ppy(["-O", "1", "inspect", "lvl.ppy"], workspace)
    assert "for i in range" not in at_o3.stdout
    assert "for i in range" in at_o1.stdout


def test_differential_test_command(workspace: Path):
    (workspace / "same.ppy").write_text("print(1 + 1)\n", encoding="utf-8")
    result = _ppy(["test", "."], workspace)
    assert result.returncode == 0
    assert "matched plain CPython" in result.stderr


def test_missing_target_is_reported(workspace: Path):
    result = _ppy(["check", "nope.ppy"], workspace)
    assert result.returncode == 2
    assert "E1002" in result.stderr


def test_lsp_command_serves_a_session(workspace: Path):
    from ppy_compiler.lsp.protocol import encode

    (workspace / "a.ppy").write_text("def f(x: int) -> int:\n    return x\n", encoding="utf-8")
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"rootPath": str(workspace)}},
        {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}},
        {"jsonrpc": "2.0", "method": "exit"},
    ]
    result = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "lsp", "--root", "."],
        cwd=workspace,
        input=b"".join(encode(message) for message in messages),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert b'"serverInfo"' in result.stdout
    assert b'"hoverProvider":true' in result.stdout


def test_explain_reports_which_boundary_a_function_crosses(workspace: Path):
    (workspace / "boundary.ppy").write_text(
        textwrap.dedent(
            """
            import ppy


            @ppy.pure
            @ppy.opt(3)
            def square(x: int) -> int:
                return x * x


            @ppy.jit
            @ppy.pure
            def cube(x: int) -> int:
                return x * x * x
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    plain = _ppy(["explain", "square"], workspace)
    assert plain.returncode == 0
    assert "python boundary: a generated CPython-ABI wrapper" in plain.stdout

    jitted = _ppy(["explain", "cube"], workspace)
    assert "selects the specialization" in jitted.stdout
    assert "specializes on x" in jitted.stdout
