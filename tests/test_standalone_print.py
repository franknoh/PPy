"""Standalone printing has the same observable output through C and LLVM."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler


def _ppy(directory: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(params=["llvm", "c"])
def backend(request):
    if c_compiler() is None:
        pytest.skip("no C compiler on PATH")
    if request.param == "llvm" and not llvm_available():
        pytest.skip("llvmlite is not installed")
    return request.param


def _build(directory: Path, backend: str) -> Path:
    if backend == "llvm":
        built = _ppy(directory, "build", "--standalone", "app.ppy", "-o", "out")
        assert built.returncode == 0, built.stderr
        return directory / "out" / "app"
    emitted = _ppy(directory, "emit", "c", "--standalone", "app.ppy")
    assert emitted.returncode == 0, emitted.stderr
    source = directory / "app.c"
    source.write_text(emitted.stdout, encoding="utf-8")
    executable = directory / "app"
    compiler = c_compiler()
    assert compiler is not None
    built = subprocess.run(
        [compiler, "-std=c11", "-Wall", "-Werror", str(source), "-o", str(executable)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    return executable


PROMPT = """
import ppy


def identity(x: int) -> int:
    return x


def main() -> None:
    print("value: ", end="", flush=True)
    value: int = ppy.input[int]()
    print(f"got {identity(value)}")


main()
"""


@pytest.mark.parametrize(("end", "prompt"), [("", b"value: "), ("!", b"value: !")])
def test_standalone_print_flushes_before_reading(write, backend, end: str, prompt: bytes):
    path = write("app.ppy", PROMPT.replace('end=""', f"end={end!r}"))
    executable = _build(path.parent, backend)
    # Keep stdin open and empty: output must arrive before input is supplied
    # or the process exits, so accepting flush=True without flushing fails.
    with (
        subprocess.Popen(
            [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process,
        ThreadPoolExecutor(max_workers=1) as reader,
    ):
        assert process.stdout is not None
        pending = reader.submit(process.stdout.read, len(prompt))
        try:
            assert pending.result(timeout=10) == prompt
            stdout, stderr = process.communicate(b"42\n", timeout=10)
            assert process.returncode == 0, stderr
            assert stdout == b"got 42\n"
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('print(123)\nprint("hello")\nprint(True, False)', "123\nhello\nTrue False\n"),
        ('print(1, end="")\nprint(2, end="\\n", flush=False)', "12\n"),
        ('print(1, True, "hi", sep="::", end="!")', "1::True::hi!"),
        ('print(1, 2, sep="", end="끝\\0")', "12끝\0"),
        ('print()\nprint(end="tail", flush=True)\nprint(end="")', "\ntail"),
        ('print(f"value = {123}, flag = {True}", "ok", sep="|")', "value = 123, flag = True|ok\n"),
    ],
)
def test_standalone_print_output(write, backend, body: str, expected: str):
    path = write(
        "app.ppy", "def main() -> None:\n" + textwrap.indent(body, "    ") + "\n\nmain()\n"
    )
    executable = _build(path.parent, backend)
    ran = subprocess.run([str(executable)], capture_output=True, check=False)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == expected.encode("utf-8")


def test_standalone_print_evaluates_arguments_before_writing(write, backend):
    path = write(
        "app.ppy",
        """
        def observed(n: int) -> int:
            print(n, end="!")
            return n


        def main() -> None:
            print("start", f"{observed(1)}:{observed(2)}", observed(3), sep="|")


        main()
        """,
    )
    executable = _build(path.parent, backend)
    ran = subprocess.run([str(executable)], capture_output=True, check=False)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == b"1!2!3!start|1:2|3\n"


@pytest.mark.parametrize("command", [("build",), ("emit", "c")])
@pytest.mark.parametrize(
    ("call", "message"),
    [
        ("print(1, file=None)", "standalone print does not support `file`"),
        ('print(1, end="A", end="B")', "standalone print repeats keyword `end`"),
        ('print(1, **{"end": ""})', "standalone print does not support `**kwargs`"),
        ("print(1, end=ppy.input[str]())", "standalone print `end` must be a string literal"),
        ("print(1, end=1)", "standalone print `end` must be a string literal"),
        ("print(1, sep=ppy.input[str]())", "standalone print `sep` must be a string literal"),
        ("print(1, flush=bool(1))", "standalone print `flush` must be a boolean literal"),
        ("print(1, flush=1)", "standalone print `flush` must be a boolean literal"),
        # A spec with a field of its own is Python's; a literal spec is native.
        ('print(f"{1:{4}}")', "a format spec with fields has no native lowering"),
        ('print(f"{1:q}")', "the format spec `q` has no native lowering"),
        ("identity(x=42)", "keyword arguments have no native ABI"),
    ],
)
def test_standalone_print_rejects_unsupported_calls(write, command, call: str, message: str):
    path = write(
        "app.ppy",
        f"import ppy\n\ndef identity(x: int) -> int:\n    return x\n\n"
        f"def main() -> None:\n    {call}\n\nmain()\n",
    )
    result = _ppy(path.parent, *command, "--standalone", "app.ppy")
    assert result.returncode == 1, result.stderr
    assert "E1803" in result.stderr and message in result.stderr
    assert "Traceback" not in result.stderr


def test_converted_prompt_builds_standalone(write, backend):
    path = write(
        "app.py",
        PROMPT.replace(
            'print("value: ", end="", flush=True)\n    value: int = ppy.input[int]()',
            'value: int = int(input("value: "))',
        ),
    )
    converted = _ppy(path.parent, "convert", "app.py")
    assert converted.returncode == 0, converted.stderr
    executable = _build(path.parent, backend)
    ran = subprocess.run([str(executable)], input=b"42\n", capture_output=True, check=False)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == b"value: got 42\n"


def test_standalone_print_flush_false_needs_no_flush_shim(write):
    path = write("app.ppy", "def main() -> None:\n    print(1, flush=False)\n\nmain()\n")
    emitted = _ppy(path.parent, "emit", "c", "--standalone", "app.ppy")
    assert emitted.returncode == 0, emitted.stderr
    assert "fflush" not in emitted.stdout


@pytest.mark.parametrize("command", [("build",), ("emit", "c")])
def test_shadowed_print_keeps_generic_keyword_rejection(write, command):
    path = write(
        "app.ppy",
        "def print(n: int, flush: bool = False) -> int:\n    return n + int(flush)\n\n"
        "def main() -> None:\n    print(7, flush=True)\n\nmain()\n",
    )
    result = _ppy(path.parent, *command, "--standalone", "app.ppy")
    assert result.returncode == 1, result.stderr
    assert "E1803" in result.stderr and "keyword arguments have no native ABI" in result.stderr
