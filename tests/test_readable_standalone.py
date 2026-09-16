"""Readable standalone source keeps the program's observable behavior."""

from __future__ import annotations

import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm.link import c_compiler


def _ppy(directory: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(params=["c", "cpp"])
def language(request):
    return request.param


def _emit(path: Path, language: str, *, formatted: bool = False) -> str:
    flags = ["--format"] if formatted else []
    done = _ppy(path.parent, "emit", language, "--standalone", "--unsafe", *flags, path.name)
    assert done.returncode == 0, done.stderr
    return done.stdout


def _compile(path: Path, language: str, text: str) -> Path:
    compiler = (
        c_compiler()
        if language == "c"
        else next((shutil.which(c) for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
    )
    if compiler is None:
        pytest.skip("no compiler for the requested language")
    source = path.with_suffix(f".{language}")
    source.write_text(text, encoding="utf-8")
    executable = path.with_suffix(".exe")
    standard = "c11" if language == "c" else "c++17"
    done = subprocess.run(
        [
            compiler,
            f"-std={standard}",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, f"{done.stderr}\n{text}"
    return executable


PROGRAM = """
import ppy


def parking_fee(minutes: int) -> int:
    if minutes <= 30:
        return 0
    chargeable_minutes: int = minutes - 30
    units: int = (chargeable_minutes + 9) // 10
    fee: int = units * 500
    if fee > 5000:
        fee = 5000
    return fee


def main() -> None:
    print("===Parking Fee Calculator===")
    print("Number of cars: ", end="", flush=True)
    number_of_cars: int = ppy.input[int]()
    total_fee: int = 0
    free_cars: int = 0
    car: int = 1
    while car <= number_of_cars:
        print(f"Car {car} parking time (minutes): ", end="", flush=True)
        parking_time: int = ppy.input[int]()
        fee: int = parking_fee(parking_time)
        print(f"Car {car} fee: {fee} won")
        total_fee += fee
        if fee == 0:
            free_cars += 1
        car += 1
    print(f"Total fee: {total_fee} won, Free cars: {free_cars}")


main()
"""


@pytest.mark.parametrize("formatted", [False, True])
def test_parking_program(write, language, formatted):
    if formatted and shutil.which("clang-format") is None:
        pytest.skip("clang-format is not installed")
    path = write("problem1.ppy", PROGRAM)
    checked = _ppy(path.parent, "check", path.name)
    assert checked.returncode == 0, checked.stderr
    text = _emit(path, language, formatted=formatted)
    assert "parking_fee(" in text
    assert "int main(void)" in text if language == "c" else "int main()" in text
    for unwanted in (
        "ppy_problem1_",
        "ppy_ovf_",
        "ppy_rt_print",
        "ppy_rt_scan",
        "ppy_str_",
        "*out",
    ):
        assert unwanted not in text
    assert "scanf(" in text and "fflush(stdout)" in text
    assert "total_fee += fee;" in text and "car += 1;" in text
    assert "minutes - 30" in text and "units * 500" in text
    assert '"Car %" PRId64 " fee: %" PRId64 " won\\n"' in text
    if language == "cpp":
        assert "namespace problem1" in text and "problem1::parking_fee(" in text
        assert "std::printf(" in text and "std::scanf(" in text
        assert "std::int64_t" in text and "#include <cinttypes>" in text
    executable = _compile(path, language, text)
    ran = subprocess.run(
        [str(executable)], input=b"3\n20\n45\n300\n", capture_output=True, check=False
    )
    expected = subprocess.run(
        [sys.executable, str(path)], input=b"3\n20\n45\n300\n", capture_output=True, check=False
    )
    assert ran.returncode == expected.returncode == 0, (ran.stderr, expected.stderr)
    assert ran.stdout == expected.stdout
    failed = subprocess.run([str(executable)], input=b"bad\n", capture_output=True, check=False)
    assert failed.returncode == 1


@pytest.mark.parametrize("kind", ["ir", "llvm-ir", "header", "stablehlo", "cuda", "hip", "ptx"])
def test_unsafe_rejects_unrelated_emit_formats(write, kind):
    path = write("app.ppy", "def main() -> None:\n    print(1)\n\nmain()\n")
    done = _ppy(path.parent, "emit", kind, "--unsafe", path.name)
    assert done.returncode == 2
    assert "`--unsafe` applies to `emit c` and `emit cpp`" in done.stderr


def test_multi_module_names(write, language):
    write("billing.ppy", "def calculate(x: int) -> int:\n    return x * 2\n")
    write("utils/__init__.ppy", "")
    write("utils/math.ppy", "def calculate(x: int) -> int:\n    return x + 3\n")
    path = write(
        "app.ppy",
        """
        import billing
        from utils import math

        def main() -> None:
            print(billing.calculate(10), math.calculate(10))

        main()
    """,
    )
    text = _emit(path, language)
    if language == "cpp":
        assert "namespace billing" in text and "namespace utils::math" in text
        assert "billing::calculate(" in text and "utils::math::calculate(" in text
    else:
        assert "billing_calculate(" in text and "utils_math_calculate(" in text
    executable = _compile(path, language, text)
    ran = subprocess.run([str(executable)], capture_output=True, check=False)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == b"20 13\n"


def test_print_bytes_and_effect_order(write, language):
    path = write(
        "app.ppy",
        r"""
        def observed(n: int) -> int:
            print(n, end="!")
            return n

        def main() -> None:
            print("100%\"\\\t\r\n끝\0f", True, False, end="!")
            print("start", f"{observed(1)}:{observed(2)}", observed(3), sep="|")

        main()
    """,
    )
    text = _emit(path, language)
    assert "ppy_str_" not in text and "ppy_rt_print" not in text
    executable = _compile(path, language, text)
    ran = subprocess.run([str(executable)], capture_output=True, check=False)
    expected = subprocess.run([sys.executable, str(path)], capture_output=True, check=False)
    assert ran.returncode == expected.returncode == 0
    assert ran.stdout == expected.stdout


@pytest.mark.parametrize(
    "source, expected",
    [
        ('def main() -> None:\n    print(1, -2, True, False, "??/n")\n', b"1 -2 True False ??/n\n"),
        (
            'def observed() -> int:\n    print("called")\n    return 7\n\ndef main() -> int:\n    return observed()\n',
            b"called\n",
        ),
        (
            "def new(x: int) -> int:\n    return x + 1\n\ndef new_(x: int) -> int:\n    return x + 2\n\ndef main() -> None:\n    print(new(1), new_(1))\n",
            b"2 3\n",
        ),
        (
            "import ppy\n\ndef main() -> None:\n    scanf: int = 1\n    fflush: int = 2\n    stdout: int = 3\n    x: int = ppy.input[int]()\n    print(scanf, fflush, stdout, x, flush=True)\n",
            b"1 2 3 42\n",
        ),
        ('def foo() -> None:\n    print("foo")\n\ndef main() -> None:\n    foo()\n', b"foo\n"),
        ("def main() -> None:\n    pass\n", b""),
    ],
)
def test_source_edge_cases(write, language, source, expected):
    path = write("app.ppy", source + "\nmain()\n")
    executable = _compile(path, language, _emit(path, language))
    ran = subprocess.run([str(executable)], input=b"42\n", capture_output=True, check=False)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == expected


def test_failed_input_in_helper_stops_program(write, language):
    path = write(
        "app.ppy",
        """
        import ppy

        def read() -> int:
            return ppy.input[int]()

        def main() -> None:
            print(read())
            print("unreachable")

        main()
    """,
    )
    executable = _compile(path, language, _emit(path, language))
    for data in (b"", b"bad\n"):
        ran = subprocess.run([str(executable)], input=data, capture_output=True, check=False)
        assert ran.returncode == 1
        assert ran.stdout == b""


def test_imported_side_effects_are_rejected(write, language):
    write("helper.ppy", 'print("side effect")\n\ndef value() -> int:\n    return 1\n')
    path = write(
        "app.ppy", "import helper\n\ndef main() -> None:\n    print(helper.value())\n\nmain()\n"
    )
    done = _ppy(path.parent, "emit", language, "--standalone", "--unsafe", path.name)
    assert done.returncode == 1
    assert "cannot run without CPython" in done.stderr


def test_unsigned_print(write, language):
    path = write(
        "app.ppy",
        """
        import ppy

        def show(value: ppy.u64) -> None:
            print(value)

        def main() -> None:
            show(42)

        main()
    """,
    )
    executable = _compile(path, language, _emit(path, language))
    ran = subprocess.run([str(executable)], capture_output=True, check=False)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == b"42\n"


def test_canonical_unsigned_print(tmp_path, language):
    from ppy_compiler.backend.c import Language, emit_module
    from ppy_compiler.ir import U64, Builder, IRModule
    from ppy_compiler.ir.dialects import core

    module = IRModule("app")
    function = module.add_function("entry", [], [], attributes={"ppy.symbol": "entry"})
    builder = Builder(function.add_entry_block())
    core.call_extern(builder, "ppy_rt_print_u64", (core.const(builder, 2**64 - 1, U64),), ())
    core.call_extern(builder, "ppy_rt_print_nl", (), ())
    core.ret(builder)
    text = emit_module(module, Language(language), entry="entry", readable=True)
    assert "PRIu64" in text
    executable = _compile(tmp_path / "app.ppy", language, text)
    ran = subprocess.run([str(executable)], capture_output=True, check=False)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == b"18446744073709551615\n"


def test_flush_before_input(write, language):
    path = write(
        "app.ppy",
        """
        import ppy

        def main() -> None:
            print("x: ", end="", flush=True)
            x: int = ppy.input[int]()
            print("x =", x, "ok =", x > 0)

        main()
    """,
    )
    text = _emit(path, language)
    assert text.count("printf(") == 2
    executable = _compile(path, language, text)
    with (
        subprocess.Popen(
            [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process,
        ThreadPoolExecutor(max_workers=1) as reader,
    ):
        assert process.stdout is not None
        pending = reader.submit(process.stdout.read, 3)
        try:
            assert pending.result(timeout=10) == b"x: "
            stdout, stderr = process.communicate(b"42\n", timeout=10)
            assert process.returncode == 0, stderr
            assert stdout == b"x = 42 ok = True\n"
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_unsafe_semantics_precede_source_emission(write, analyze):
    from ppy_compiler.backend.llvm.standalone import standalone_ir
    from ppy_compiler.driver.reporting import Reporter
    from ppy_compiler.ir import verify

    path = write(
        "app.ppy",
        """
        def add(x: int, y: int) -> int:
            return x + y

        def main() -> None:
            print(add(1, 2))

        main()
    """,
    )
    bundle = analyze(path, backend="llvm")
    bundle.project.config.llvm.safeguards = "off"
    module = standalone_ir(bundle, Reporter(color=False), path)
    assert not isinstance(module, int)
    assert not verify(module)
    additions = [
        op for f in module.functions.values() for op in f.operations() if op.name == "core.add"
    ]
    assert additions and all(op.attributes["overflow"] == "native" for op in additions)
    bundle.project.config.llvm.safeguards = "inline"
    safe = standalone_ir(bundle, Reporter(color=False), path)
    assert not isinstance(safe, int)
    additions = [
        op for f in safe.functions.values() for op in f.operations() if op.name == "core.add"
    ]
    assert additions and all(op.attributes["overflow"] == "python" for op in additions)


def test_conditional_print_types(write, language):
    path = write(
        "app.ppy",
        """
        import ppy

        def main() -> None:
            x: int = ppy.input[int]()
            print(1 if x > 0 else 2)
            print(x > 0 if x % 2 else x < 0)

        main()
    """,
    )
    executable = _compile(path, language, _emit(path, language))
    for data, expected in [(b"1\n", b"1\nTrue\n"), (b"2\n", b"1\nFalse\n")]:
        ran = subprocess.run([str(executable)], input=data, capture_output=True, check=False)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == expected


def test_native_expressions_keep_machine_width(write, language):
    path = write(
        "app.ppy",
        """
        def int64_t(x: int) -> int:
            return x + 1

        def main() -> None:
            x: int = 1
            y: int = (2000000000 if x else 1) + 1000000000
            z: int = (1000000000 if x else 1) * 4
            print(y, z, int64_t(2))

        main()
    """,
    )
    executable = _compile(path, language, _emit(path, language))
    ran = subprocess.run([str(executable)], capture_output=True, check=False)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == b"3000000000 4000000000 3\n"
