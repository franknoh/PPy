"""The C and C++ backends: the same answers as the LLVM road, from C."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_lowering import CASES, KERNELS, _call, _lower

from ppy_compiler.backend.c import HeaderOnlyError, Language, emit_module
from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.ir_pipeline import optimize
from ppy_compiler.backend.llvm.jit import JitEngine
from ppy_compiler.backend.llvm.link import c_compiler
from ppy_compiler.lowering import lower_module_to_ir
from ppy_runtime.abi import STATUS_FALLBACK, STATUS_OK

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")
_CXX = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
requires_cxx = pytest.mark.skipif(_CXX is None, reason="no C++ compiler on PATH")

BUFFERS = """
    from ppy import Buffer, native


    def scale(xs: list[int], k: int) -> int:
        total = 0
        for x in xs:
            total += x * k
        return total


    @native.export(name="ppy_dot")
    def dot(a: Buffer[float], b: Buffer[float], n: int) -> float:
        total = 0.0
        for i in range(n):
            total += a[i] * b[i]
        return total


    def pair(n: int) -> tuple[int, bool]:
        return n * 2, n > 0
    """

READER = """
    import ppy


    def main() -> int:
        n = ppy.input[int]()
        print(n * 2)
        return 0
    """


def _ir_module(analyze, path: Path, **options):  # type: ignore[no-untyped-def]
    """The canonical IR of one file after the shared passes."""
    from ppy_compiler.backend.llvm import _definitions, _value_class_layouts

    bundle = analyze(path, backend="llvm")
    name = path.stem
    symbols = bundle.symbols.modules[name]
    analysis = bundle.analysis.modules[name]
    candidates = {}
    for owner, node in _definitions(symbols.module.tree):
        info = symbols.functions.get(node.name) if not owner else None
        if info is not None and info.qualname in analysis.functions:
            candidates[info.qualname] = (info, analysis.functions[info.qualname], node)
    lowered = lower_module_to_ir(
        analysis, candidates, _value_class_layouts(bundle), safeguards="inline", **options
    )
    optimize(lowered.module, 2)
    return lowered


def _compile(text: str, directory: Path, language: Language) -> ctypes.CDLL:
    """The emitted unit as a shared library, loaded."""
    suffix = ".cpp" if language is Language.CPP else ".c"
    source = directory / f"unit{suffix}"
    source.write_text(text, encoding="utf-8")
    library = directory / f"libunit_{language}.so"
    compiler = _CXX if language is Language.CPP else c_compiler()
    standard = "-std=c++17" if language is Language.CPP else "-std=c11"
    done = subprocess.run(
        [
            compiler,
            standard,
            "-Wall",
            "-O1",
            "-shared",
            "-fPIC",
            "-o",
            str(library),
            str(source),
            "-lm",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, f"{done.stderr}\n{text}"
    assert "warning:" not in done.stderr, done.stderr
    return ctypes.CDLL(str(library))


class _Library:
    """A loaded C unit answering `address(symbol)` like a JIT engine."""

    def __init__(self, handle: ctypes.CDLL) -> None:
        self.handle = handle

    def address(self, symbol: str) -> int:
        return ctypes.cast(getattr(self.handle, symbol), ctypes.c_void_p).value or 0


@requires_llvm
@requires_cc
@pytest.mark.parametrize("language", [Language.C, pytest.param(Language.CPP, marks=requires_cxx)])
def test_the_c_unit_answers_like_the_llvm_road_on_every_input(write, analyze, tmp_path, language):
    """Same value, same status, from a C compiler instead of LLVM."""
    path = write("kernels.ppy", KERNELS)
    new = _lower(analyze, path)
    engine = JitEngine(opt_level=2).open()
    engine.add(new.ir)
    engine.finalize()
    lowered = _ir_module(analyze, path)
    text = emit_module(lowered.module, language)
    assert emit_module(lowered.module, language) == text, "one module, one text"
    unit = _Library(_compile(text, tmp_path, language))
    checked = 0
    for name, cases in CASES.items():
        signature = new.functions[f"kernels.{name}"].signature
        for arguments in cases:
            expected = _call(engine, signature, arguments)
            got = _call(unit, signature, arguments)
            assert got == expected, f"{name}{arguments}: C {got}, LLVM {expected}"
            checked += 1
    assert checked == sum(len(cases) for cases in CASES.values())
    minimum = -(2**63)
    floor_mod = new.functions["kernels.floor_mod"].signature
    assert _call(unit, floor_mod, (minimum, -1))[0] == STATUS_FALLBACK
    assert _call(unit, floor_mod, (-7, 2)) == (STATUS_OK, -3)


@requires_llvm
@requires_cc
def test_buffers_tuples_and_exports_cross_the_c_boundary(write, analyze, tmp_path):
    path = write("kernels.ppy", BUFFERS)
    lowered = _ir_module(analyze, path)
    text = emit_module(lowered.module, Language.C)
    assert "typedef struct ppy_agg0" in text, "a tuple is a struct"
    assert "double ppy_dot(double *a, int64_t a_len, double *b, int64_t b_len, int64_t n)" in text
    handle = _compile(text, tmp_path, Language.C)
    xs = (ctypes.c_int64 * 4)(1, 2, 3, 4)
    out = ctypes.c_int64(0)
    scale = handle.ppy_kernels_scale
    scale.restype = ctypes.c_int32
    assert scale(xs, 4, 3, ctypes.byref(out)) == STATUS_OK
    assert out.value == 30
    a = (ctypes.c_double * 3)(1.0, 2.0, 3.0)
    b = (ctypes.c_double * 3)(4.0, 5.0, 6.0)
    handle.ppy_dot.restype = ctypes.c_double
    handle.ppy_dot.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    assert handle.ppy_dot(a, 3, b, 3, 3) == 32.0
    pair = handle.ppy_kernels_pair
    pair.restype = ctypes.c_int32
    flag = ctypes.c_int8(0)
    assert pair(5, ctypes.byref(out), ctypes.byref(flag)) == STATUS_OK
    assert (out.value, flag.value) == (10, 1)
    assert pair(-5, ctypes.byref(out), ctypes.byref(flag)) == STATUS_OK
    assert (out.value, flag.value) == (-10, 0)


@requires_llvm
@requires_cc
def test_a_program_carries_the_runtime_shims_it_calls(write, analyze, tmp_path):
    path = write("reader.ppy", READER)
    lowered = _ir_module(analyze, path, standalone=True)
    text = emit_module(lowered.module, Language.C)
    assert "static int64_t ppy_rt_read_int(void)" in text
    assert "static void ppy_rt_print_i64(int64_t value)" in text
    assert "ppy_rt_print_str" not in text, "only the shims the unit calls"
    source = tmp_path / "reader.c"
    source.write_text(
        text + "\nint main(void) { int64_t out; return ppy_reader_main(&out); }\n", encoding="utf-8"
    )
    binary = tmp_path / "reader"
    built = subprocess.run(
        [c_compiler(), "-std=c11", "-Wall", "-O1", "-o", str(binary), str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(binary)], input="21\n", capture_output=True, text=True, check=False)
    assert ran.stdout == "42\n"
    with pytest.raises(HeaderOnlyError, match="ppy_rt_read_int"):
        emit_module(lowered.module, Language.C, header_only=True)


@requires_llvm
@requires_cc
@requires_cxx
def test_a_header_only_unit_is_included_from_two_translation_units(write, analyze, tmp_path):
    path = write("kernels.ppy", BUFFERS)
    lowered = _ir_module(analyze, path)
    for language, compiler, standard in (
        (Language.C, c_compiler(), "-std=c11"),
        (Language.CPP, _CXX, "-std=c++17"),
    ):
        header = emit_module(lowered.module, language, header_only=True)
        assert header.startswith(
            f"/* kernels: generated by ppy, {'C++17' if language is Language.CPP else 'C11'} */"
        )
        assert "#ifndef PPY_KERNELS_H" in header
        assert "static inline double ppy_dot(" in header
        suffix = ".hpp" if language is Language.CPP else ".h"
        (tmp_path / f"kernels{suffix}").write_text(header, encoding="utf-8")
        units = []
        for index in range(2):
            unit = tmp_path / f"user{index}{suffix.replace('h', 'c', 1)}"
            body = f'#include "kernels{suffix}"\n#include "kernels{suffix}"\n'
            body += (
                "int main(void) { double a[1] = {2.0}; return (int)ppy_dot(a, 1, a, 1, 1) - 4; }\n"
                if index == 0
                else "int other(void) { return 0; }\n"
            )
            unit.write_text(body, encoding="utf-8")
            units.append(str(unit))
        binary = tmp_path / f"user_{language}"
        built = subprocess.run(
            [compiler, standard, "-Wall", "-o", str(binary), *units, "-lm"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert built.returncode == 0, built.stderr
        assert subprocess.run([str(binary)], check=False).returncode == 0


def _ppy(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_cli_offers_every_emit_kind():
    from ppy_compiler.driver import cli, emit

    assert cli._EMIT_KINDS == emit.KINDS


@requires_llvm
def test_emit_c_cpp_and_header_follow_the_emit_rules(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "kernels.ppy").write_text(
        "from ppy import native\n\n\n@native.export()\ndef twice(n: int) -> int:\n    return n * 2\n",
        encoding="utf-8",
    )
    (tmp_path / "reader.ppy").write_text(
        "import ppy\n\n\ndef main() -> int:\n    print(ppy.input[int]())\n    return 0\n",
        encoding="utf-8",
    )
    c = _ppy(tmp_path, "emit", "c", "kernels.ppy")
    assert c.returncode == 0, c.stderr
    assert c.stdout.startswith("/* kernels: generated by ppy, C11 */\n")
    assert "int64_t twice(int64_t n)" in c.stdout and 'extern "C"' not in c.stdout
    cpp = _ppy(tmp_path, "emit", "cpp", "kernels.ppy")
    assert cpp.returncode == 0, cpp.stderr
    assert "#include <cstdint>" in cpp.stdout and 'extern "C" {' in cpp.stdout
    assert "static_cast<" in cpp.stdout or "std::abort" in cpp.stdout
    header = _ppy(tmp_path, "emit", "header", "kernels.ppy")
    assert header.returncode == 0, header.stderr
    assert "int64_t twice(int64_t n);" in header.stdout and "#ifdef __cplusplus" in header.stdout
    inline = _ppy(tmp_path, "emit", "c", "--header-only", "kernels.ppy", "-o", "out/kernels.h")
    assert inline.returncode == 0, inline.stderr
    assert "static inline int64_t twice(int64_t n)" in (tmp_path / "out" / "kernels.h").read_text()
    directory = _ppy(tmp_path, "emit", "cpp", ".", "-o", "gen")
    assert directory.returncode == 0, directory.stderr
    assert [p.name for p in (tmp_path / "gen").iterdir()] == ["kernels.cpp"], (
        "reader has no native part"
    )
    program = _ppy(tmp_path, "emit", "c", "--standalone", "reader.ppy")
    assert program.returncode == 0, program.stderr
    assert "static int64_t ppy_rt_read_int(void)" in program.stdout
    assert program.stdout.rstrip().endswith(
        'int main(void) {\n    int64_t out = 0;\n    int32_t status = ppy_reader_main(&out);\n    if (status != 0) {\n        fputs("ppy: a native guard failed and there is no Python to fall back to\\n", stderr);\n        return 70;\n    }\n    return 0;\n}'
    )
    if c_compiler() is not None:
        (tmp_path / "reader.c").write_text(program.stdout, encoding="utf-8")
        built = subprocess.run(
            [
                c_compiler(),
                "-std=c11",
                "-Wall",
                "-o",
                str(tmp_path / "reader"),
                str(tmp_path / "reader.c"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert built.returncode == 0 and built.stderr == "", built.stderr
        ran = subprocess.run(
            [str(tmp_path / "reader")], input="7", capture_output=True, text=True, check=False
        )
        assert ran.stdout == "7\n"
    refused = _ppy(tmp_path, "emit", "c", "--standalone", "--header-only", "reader.ppy")
    assert refused.returncode == 2
    assert "E1804" in refused.stderr and "ppy_rt_read_int" in refused.stderr
    not_a_program = _ppy(tmp_path, "emit", "c", "--standalone", "kernels.ppy")
    assert not_a_program.returncode == 1 and "E1803" in not_a_program.stderr
    misplaced = _ppy(tmp_path, "emit", "ir", "--header-only", "kernels.ppy")
    assert misplaced.returncode == 2 and "applies to" in misplaced.stderr
