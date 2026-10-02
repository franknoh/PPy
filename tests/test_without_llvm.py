"""An install without the LLVM extra: `ppy run` runs on CPython, native-only
commands stop with one clear error.

Each test runs the CLI in a child interpreter in which `import llvmlite`
fails, as it does after `pip install ppy-lang` without `[llvm]`.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

#: Makes llvmlite unimportable in the child, then runs the CLI on its argv.
_WITHOUT_LLVMLITE = textwrap.dedent(
    """
    import sys

    class _NoLlvmlite:
        def find_spec(self, name, path=None, target=None):
            if name == "llvmlite" or name.startswith("llvmlite."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    sys.meta_path.insert(0, _NoLlvmlite())
    for loaded in [m for m in sys.modules if m == "llvmlite" or m.startswith("llvmlite.")]:
        del sys.modules[loaded]
    from ppy_compiler.driver.cli import main

    raise SystemExit(main(sys.argv[1:]))
    """
)

_PROGRAM = textwrap.dedent(
    """
    import sys


    def total(n: int) -> int:
        s = 0
        for i in range(n):
            s += i * i
        return s


    print(total(1000), sys.argv[1:])
    if len(sys.argv) > 2:
        print([1, 2][5])
    raise SystemExit(3)
    """
)


def _without_llvm(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _WITHOUT_LLVMLITE, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def _cpython(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args], cwd=cwd, capture_output=True, text=True, check=False
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    (tmp_path / "prog.py").write_text(_PROGRAM, encoding="utf-8")
    return tmp_path


def test_the_child_really_has_no_llvmlite(workspace: Path):
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            _WITHOUT_LLVMLITE.replace("from ppy_compiler", "import llvmlite\n#"),
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "No module named 'llvmlite'" in probe.stderr


@pytest.mark.parametrize("args", [[], ["a"], ["a", "b"]], ids=["exit", "argv", "raises"])
def test_run_warns_once_and_runs_on_cpython(workspace: Path, args: list[str]):
    expected = _cpython(["prog.py", *args], workspace)
    result = _without_llvm(["run", "prog.py", *args], workspace)
    assert result.stdout == expected.stdout
    assert result.returncode == expected.returncode
    assert result.stderr.count("warning[W2012]") == 1
    assert "ppy-lang[llvm]" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    if expected.stderr:
        assert result.stderr.splitlines()[-1] == expected.stderr.splitlines()[-1]


@pytest.mark.parametrize(
    "args",
    [
        ["build", "prog.py"],
        ["build", "--standalone", "prog.py"],
        ["build", "--warm", "prog.py"],
        ["run", "--profile", "prog.py"],
        ["inspect", "--ir", "prog.py"],
        ["emit", "llvm-ir", "prog.py"],
    ],
    ids=["build", "standalone", "warm", "profile", "inspect-ir", "emit-ir"],
)
def test_native_only_commands_stop_with_one_clear_error(workspace: Path, args: list[str]):
    result = _without_llvm(args, workspace)
    assert result.returncode == 2
    assert "error[E1801]" in result.stderr
    assert "Traceback" not in result.stderr
    assert "llvmlite is not installed" in result.stderr


def test_python_build_needs_no_llvm(workspace: Path):
    result = _without_llvm(["build", "--backend", "python", "prog.py"], workspace)
    assert result.returncode == 0, result.stderr
