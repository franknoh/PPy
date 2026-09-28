"""`int(x)` of a float: CPython's answer where native code has one, a guard where not.

The machine's conversion answers NaN, the infinities, and anything past 2**63
with whatever the instruction set says (-2**63 on x86; C calls it undefined).
CPython raises `ValueError` for NaN and `OverflowError` for an infinity, and
past 2**63 gives an integer no word holds. Under `ppy run` each of those
falls back to Python; a standalone binary, and emitted C and C++ with or
without `--unsafe`, print what CPython raises and exit with status 1.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler, standalone_toolchain_status

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")
_standalone, _standalone_detail = standalone_toolchain_status()
requires_standalone = pytest.mark.skipif(
    not (llvm_available() and _standalone), reason=f"no standalone toolchain: {_standalone_detail}"
)
_CXX = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)

CAUGHT = """
import ppy


@ppy.native
def to_int(x: float) -> int:
    return int(x)


def main() -> None:
    for text in ("3.7", "-3.7", "nan", "inf", "-inf", "1e300", "9.3e18", "-9.3e18"):
        try:
            print(text, to_int(float(text)))
        except (OverflowError, ValueError) as error:
            print(text, type(error).__name__, error)


main()
"""

READ = """
import ppy


def to_int(x: float) -> int:
    return int(x)


def main() -> None:
    print(to_int(ppy.input[float]()))


main()
"""

#: What a standalone binary answers for each input: its output, or CPython's
#: last line and status 1. Past 2**63 CPython answers a big integer; native
#: code has no word for it and says so, as every 64-bit result past a word does.
STANDALONE = {
    "3.7": ("3", 0),
    "-3.7": ("-3", 0),
    "-9223372036854775808.0": ("-9223372036854775808", 0),
    "nan": ("ValueError: cannot convert float NaN to integer", 1),
    "inf": ("OverflowError: cannot convert float infinity to integer", 1),
    "-inf": ("OverflowError: cannot convert float infinity to integer", 1),
    "9223372036854775808.0": ("OverflowError: the result does not fit in a 64-bit integer", 1),
    "1e300": ("OverflowError: the result does not fit in a 64-bit integer", 1),
}


def _write(tmp_path: Path, source: str) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    program = tmp_path / "prog.ppy"
    program.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return program


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    return subprocess.run(
        [sys.executable, *args], cwd=tmp_path, capture_output=True, text=True, check=False, env=env
    )


def _output(done: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(line for line in done.stdout.splitlines() if not line.startswith("compiling"))


@requires_llvm
@requires_cc
def test_ppy_run_answers_as_cpython_and_stays_native(tmp_path: Path):
    _write(tmp_path, CAUGHT)
    expected = _run(tmp_path, "prog.ppy")
    assert expected.returncode == 0, expected.stderr
    for args in (["-m", "ppy_compiler", "prog.ppy"], ["-m", "ppy_compiler", "run", "prog.ppy"]):
        done = _run(tmp_path, *args)
        assert done.returncode == 0, done.stderr
        assert _output(done) == expected.stdout.strip(), args
    explained = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.to_int")
    assert "llvm backend: native" in explained.stdout, explained.stdout


def _answers(binary: Path) -> dict[str, tuple[str, int]]:
    found = {}
    for text in STANDALONE:
        ran = subprocess.run(
            [str(binary)], input=text + "\n", capture_output=True, text=True, check=False
        )
        shown = (ran.stdout + ran.stderr).strip().splitlines()
        found[text] = (shown[-1] if shown else "", ran.returncode)
    return found


@requires_standalone
def test_a_standalone_binary_says_what_cpython_raises(tmp_path: Path):
    _write(tmp_path, READ)
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "--standalone", "prog.ppy", "-o", "dist")
    assert built.returncode == 0, built.stderr
    assert _answers(tmp_path / "dist" / "prog") == STANDALONE


@requires_llvm
@pytest.mark.parametrize("unsafe", [False, True])
@pytest.mark.parametrize("language", ["c", "cpp"])
def test_emitted_source_says_it_too(tmp_path: Path, language: str, unsafe: bool):
    """The guards are range checks, which `--unsafe` keeps; the emitted cast is
    never reached with a value C leaves undefined (UBSan would stop it)."""
    compiler = c_compiler() if language == "c" else _CXX
    if compiler is None:
        pytest.skip(f"no {language} compiler")
    _write(tmp_path, READ)
    emitted = tmp_path / f"prog.{language}"
    flags = ["--unsafe"] if unsafe else []
    done = _run(
        tmp_path, "-m", "ppy_compiler", "emit", language, "--standalone", *flags, "prog.ppy",
        "-o", emitted.name,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    standard = "-std=c11" if language == "c" else "-std=c++17"
    binary = tmp_path / "prog"
    sanitize = ["-fsanitize=float-cast-overflow", "-fno-sanitize-recover=all"]
    probe = subprocess.run(
        [compiler, *sanitize, "-x", "c", "-", "-o", str(tmp_path / "probe")],
        input="int main(void) { return 0; }\n",
        capture_output=True,
        text=True,
        check=False,
    )
    flags = sanitize if probe.returncode == 0 else []
    subprocess.run(
        [compiler, standard, "-O2", *flags, str(emitted), "-lm", "-o", str(binary)], check=True
    )
    assert _answers(binary) == STANDALONE
