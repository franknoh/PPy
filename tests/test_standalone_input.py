"""Reading numbers with no interpreter: floats, fixed widths, tuples, and buffers.

A standalone binary reads `ppy.input` and `ppy.scan` with the C runtime, and
prints a float the way Python's `repr` does. Both are held to CPython here:
the same program, the same input, the same output, and where CPython raises,
the binary stops and names the same exception.
"""

from __future__ import annotations

import math
import random
import shutil
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.c.runtime import support_source
from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler, standalone_toolchain_status

_standalone, _standalone_detail = standalone_toolchain_status()
requires_standalone = pytest.mark.skipif(
    not (llvm_available() and _standalone), reason=f"no standalone toolchain: {_standalone_detail}"
)
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")
_CXX = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
requires_cxx = pytest.mark.skipif(_CXX is None, reason="no C++ compiler on PATH")

READS = """
import ppy
from ppy import Buffer


def main() -> None:
    a, b = ppy.input[tuple[int, float]]()
    xs: Buffer[float] = ppy.input[Buffer[float]]()
    ys: list[int] = ppy.input[list[int]]()
    small: ppy.i16 = ppy.input[ppy.i16]()
    c: tuple[ppy.u8, float, int] = ppy.scan[tuple[ppy.u8, float, int]]()
    n: int = ppy.scan[int]()
    zs: Buffer[float] = ppy.scan[Buffer[float]](n)
    total: float = 0.0
    for i in range(len(xs)):
        total += xs[i]
    count: int = 0
    for y in ys:
        count += y
    rest: float = 0.0
    for i in range(n):
        rest += zs[i]
    print(a, b, small, c[0], c[1], c[2], len(xs), total, len(ys), count, rest)
    print(total / 3, 1e16, 0.1 + 0.2, -0.0, 1e-7, 123456789.125, ppy.scan[float]())


main()
"""

GOOD = "7 2.5\n1.5 2 -3e2 1_000.5\n4 5 6\n-32768\n200\n1.5 -2\n2 0.25 0.5\ninf\n"

#: Input CPython refuses, and the exception it raises for each.
BAD = [
    ("1 2 3\n", "ValueError"),
    ("x 2\n", "ValueError"),
    ("7 0x10\n", "ValueError"),
    ("7 2.5\n1 2\n3\n40000\n", "OverflowError"),
    ("7 2.5\n\n\n1\n300 1 2\n", "OverflowError"),
    ("7 2.5\n", "EOFError"),
    ("7 2.5\n1 2\n3\n1\n1 2 3\n2 0.5\n", "EOFError"),
]


def _python(program: Path, text: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(program)], input=text, capture_output=True, text=True, check=False
    )


def _native(binary: Path, text: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(binary)], input=text, capture_output=True, text=True, check=False)


def _agree(program: Path, binary: Path) -> None:
    expected = _python(program, GOOD)
    assert expected.returncode == 0, expected.stderr
    got = _native(binary, GOOD)
    assert got.returncode == 0, got.stderr
    assert got.stdout == expected.stdout
    for text, raised in BAD:
        python = _python(program, text)
        assert python.returncode != 0 and f"{raised}:" in python.stderr, (text, python.stderr)
        native = _native(binary, text)
        assert native.returncode == 1, (text, native.stdout)
        assert native.stderr.startswith(f"ppy: {raised}:"), (text, native.stderr)


@requires_standalone
def test_a_standalone_binary_reads_numbers_as_cpython_does(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    program = tmp_path / "reads.ppy"
    program.write_text(READS.lstrip("\n"), encoding="utf-8")
    built = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", "--standalone", "reads.ppy", "-o", "dist"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    _agree(program, tmp_path / "dist" / "reads")


@pytest.mark.parametrize("unsafe", [False, True])
@pytest.mark.parametrize("language", ["c", "cpp"])
def test_emitted_source_reads_numbers_as_cpython_does(tmp_path: Path, language: str, unsafe: bool):
    compiler = c_compiler() if language == "c" else _CXX
    if compiler is None:
        pytest.skip(f"no {language} compiler on PATH")
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    program = tmp_path / "reads.ppy"
    program.write_text(READS.lstrip("\n"), encoding="utf-8")
    source = tmp_path / f"reads.{language}"
    command = [sys.executable, "-m", "ppy_compiler", "emit", language, "--standalone"]
    emitted = subprocess.run(
        [*command, *(["--unsafe"] if unsafe else []), "reads.ppy", "-o", source.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert emitted.returncode == 0, emitted.stderr
    standard = "-std=c11" if language == "c" else "-std=c++17"
    binary = tmp_path / "reads"
    compiled = subprocess.run(
        [compiler, standard, "-O2", "-Wall", "-Werror", str(source), "-lm", "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    _agree(program, binary)


_HARNESS = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int ppy_rt_float_text(const int8_t *text, int64_t length, double *out);
void ppy_rt_print_f64(double value);
int main(void) {
    char line[1024];
    while (fgets(line, sizeof line, stdin)) {
        size_t n = strlen(line);
        if (n && line[n - 1] == '\n') {
            line[--n] = 0;
        }
        double value = 0.0;
        if (line[0] == 'P') {
            ppy_rt_print_f64(strtod(line + 1, NULL));
        } else if (ppy_rt_float_text((const int8_t *)line + 1, (int64_t)(n - 1), &value)) {
            ppy_rt_print_f64(value);
        } else {
            fputs("ValueError", stdout);
        }
        fputc('\n', stdout);
    }
    return 0;
}
"""

_TEXTS = [
    "1.5", "-3", "+4.", ".5", "5.", "1_000.5", "1e1_0", "1e+5", "1E-5", "inf", "-Infinity",
    "NaN", "1__0", "_1", "1_", "1._5", "1_.5", "1e", "e5", ".", "0x1p3", "1.5.5", "infinit",
    "--1", "1 2", "",
]  # fmt: skip


@requires_cc
def test_the_runtime_parses_and_prints_floats_as_python_does(tmp_path: Path):
    """`float()`'s grammar, and `repr`'s shortest round-trip digits, over random doubles."""
    compiler = c_compiler()
    assert compiler is not None
    (tmp_path / "support.c").write_text(support_source(), encoding="utf-8")
    (tmp_path / "harness.c").write_text(_HARNESS, encoding="utf-8")
    binary = tmp_path / "harness"
    subprocess.run(
        [compiler, "-O1", "support.c", "harness.c", "-lm", "-o", str(binary)],
        cwd=tmp_path,
        check=True,
    )
    rng = random.Random(7)
    values = [0.1, 1e16, 1e15, 9.999999999999999e15, 1.5e-5, 1e-4, 5e-324, 1.7976931348623157e308]
    for _ in range(2000):
        values.append(struct.unpack("d", struct.pack("Q", rng.getrandbits(64)))[0])
        values.append(rng.random() * 10 ** rng.randint(-20, 20))
    lines = [f"P{value!r}" for value in values if not math.isnan(value)]
    lines += [f"T{text}" for text in _TEXTS]
    ran = subprocess.run(
        [str(binary)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True
    )
    for line, got in zip(lines, ran.stdout.splitlines(), strict=True):
        try:
            expected = repr(float(line[1:]))
        except ValueError:
            expected = "ValueError"
        assert got == expected, line


def test_the_runtime_rejects_what_the_width_cannot_hold():
    """`ppy.i32` and friends read as `int` and then must fit, on CPython too."""
    program = textwrap.dedent(
        """
        import ppy
        try:
            print(ppy.input[ppy.i32]())
        except OverflowError as error:
            print("OverflowError", error)
        """
    )
    ran = subprocess.run(
        [sys.executable, "-c", program],
        input="2147483648\n",
        capture_output=True,
        text=True,
        check=True,
    )
    assert ran.stdout.strip() == "OverflowError 2147483648 does not fit in ppy.i32"
