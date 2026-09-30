"""Effects in native code under `ppy run`: `print`, `input`, calls into Python, text files.

A native function that prints holds its output until it answers and drops it
when it falls back, so nothing is printed twice. One that reaches a barrier
(`input()`, `print(flush=True)`, a call into Python, a file) must not fall
back after it: a check CPython raises for raises natively there, and a
function where anything else could fall back after a barrier stays in Python,
which `ppy explain` says. Each program is held to CPython under `ppy run`,
with its input on stdin, and `pyio.c` runs clean under AddressSanitizer.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")

PRINTS = """
import sys


def show(n: int) -> int:
    total = 0
    for i in range(n):
        total += i
        print("step", i, total)
    print("done", total, sep=": ", end="!\\n")
    print("to stderr", n, file=sys.stderr)
    print("a", "b", sep=None, end=None)
    print()
    print(1.5, True, None, [1, 2], "x", sep="|")
    return total


def main() -> None:
    print("before")
    print(show(3))
    print("between", flush=True)
    print(show(4))
    print("after")


main()
"""

FALLS_BACK = """
def grow(n: int) -> int:
    x = 1
    for i in range(n):
        print("i", i)
        x = x * 1000
    return x


def main() -> None:
    print(grow(3))
    print(grow(10))
    print(grow(2))


main()
"""

FLUSHES = """
def ticks(n: int) -> int:
    for i in range(n):
        print("tick", i, flush=True)
    return n


def main() -> None:
    print(ticks(3))


main()
"""

INPUT = """
def ask(n: int) -> str:
    print("asking", n)
    name = input("name? ")
    print("hello", name)
    return name


def divide(n: int) -> float:
    got = input()
    print("got", got)
    return 10 / n


def lines(n: int) -> str:
    out = ""
    for _ in range(n):
        out = out + input() + ","
    return out


def safe(n: int) -> str:
    try:
        return lines(n)
    except EOFError:
        print("eof")
        return "none"


def main() -> None:
    print(ask(1))
    try:
        print(divide(0))
    except ZeroDivisionError as e:
        print("caught", e)
    print(safe(2))
    print(safe(9))


main()
"""

KEPT_IN_PYTHON = """
def after(n: int) -> int:
    s = input()
    return n * n + len(s)


def parse() -> int:
    return int(input())


def key(d: dict[str, int]) -> int:
    s = input()
    return d[s]


def copied(xs: list[int], k: int) -> int:
    for i in range(k):
        xs.append(i)
    xs.append(len(input()))
    return len(xs)


def main() -> None:
    print(after(3))
    print(parse())
    data = [1, 2]
    print(copied(data, 2), data)
    try:
        print(key({"a": 1}))
    except KeyError as e:
        print("missing", e)


main()
"""

UNCAUGHT = """
def lines(n: int) -> str:
    out = ""
    for _ in range(n):
        out = out + input() + ","
    return out


def main() -> None:
    print(lines(1))
    print(lines(5))


main()
"""

PYTHON_CALLS = """
import math
import sys


def log(msg: str) -> None:
    sys.stdout.write("log: " + msg + "\\n")


def hyp(a: float, b: float) -> float:
    return math.hypot(a, b)


def work(n: int) -> str:
    out = ""
    for i in range(n):
        print("step", i)
        log(str(i))
        out = out + str(i) + ";"
    print(hyp(3.0, 4.0))
    return out


def bad(n: int) -> None:
    log("bad")
    if n > 0:
        raise ValueError("nope")


def main() -> None:
    print(work(3))
    try:
        bad(1)
    except ValueError as e:
        print("caught", e)


main()
"""

FILES = """
def write_file(path: str, n: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write("line " + str(i) + " \\u00e9\\r\\n")
        f.write("tail")


def collect(path: str) -> list[str]:
    found: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("line"):
                found.append(line.strip())
    return found


def first_and_rest(path: str) -> str:
    with open(path, newline="") as f:
        first = f.readline()
        rest = f.read()
    return first + "|" + rest


def every(path: str) -> list[str]:
    with open(path, "r") as f:
        return f.readlines()


def missing(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return "missing"


def main() -> None:
    p = "a.txt"
    write_file(p, 4)
    print(collect(p))
    print(repr(first_and_rest(p)))
    print(every(p))
    print(missing("nope.txt"))


main()
"""

REENTRANT = """
import ppy


def churn(k: int) -> int:
    xs: list[int] = []
    for i in range(k):
        xs.append(i * i)
    total = 0
    for x in xs:
        total += x
    return total * 1000000000000


def outer(n: int) -> int:
    kept = [i for i in range(n)]
    total = sum(kept)
    got = input("prompt> ")
    print(len(kept), total, got)
    return len(kept)


with ppy.dynamic:
    import contextlib
    import io

    class Tee(io.StringIO):
        def __init__(self):
            super().__init__()
            self.seen = []

        def write(self, text):
            self.seen.append(churn(len(text) + 3000))
            return super().write(text)

    tee = Tee()
    with contextlib.redirect_stdout(tee):
        print(outer(50))
        print(outer(7))
    print(tee.getvalue(), end="")
    print(tee.seen[0])
"""

#: name -> (program, its stdin, functions that go native, functions kept in Python).
PROGRAMS: dict[str, tuple[str, str, list[str], list[str]]] = {
    "prints": (PRINTS, "", ["show"], []),
    "falls_back": (FALLS_BACK, "", ["grow"], []),
    "flushes": (FLUSHES, "", ["ticks"], []),
    "input": (INPUT, "bob\nline2\na\nb\nc\n", ["ask", "divide", "lines", "safe"], []),
    "kept_in_python": (
        KEPT_IN_PYTHON,
        "x\n42\nzz\nb\n",
        [],
        ["after", "parse", "key", "copied"],
    ),
    "uncaught": (UNCAUGHT, "a\nb\n", ["lines"], []),
    "python_calls": (PYTHON_CALLS, "", ["log", "hyp", "work", "bad"], []),
    "files": (FILES, "", ["write_file", "collect", "first_and_rest", "every"], []),
    "reentrant": (REENTRANT, "x\ny\n", ["outer"], []),
}


def _write(tmp_path: Path, source: str, strict: bool = True) -> None:
    (tmp_path / "pyproject.toml").write_text(
        f"[tool.ppy]\nstrict = {'true' if strict else 'false'}\n", encoding="utf-8"
    )
    (tmp_path / "prog.ppy").write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")


def _run(tmp_path: Path, stdin: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    return subprocess.run(
        [sys.executable, *args],
        cwd=tmp_path,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _output(done: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(line for line in done.stdout.splitlines() if not line.startswith("compiling"))


def _last_line(text: str) -> str:
    lines = [
        line
        for line in text.strip().splitlines()
        if line and not line.startswith((" ", "compiling "))
    ]
    return lines[-1] if lines else ""


@requires_llvm
@requires_cc
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_ppy_run_agrees_with_cpython(tmp_path: Path, name: str):
    source, stdin, natives, kept = PROGRAMS[name]
    _write(tmp_path, source)
    expected = _run(tmp_path, stdin, "prog.ppy")
    done = _run(tmp_path, stdin, "-m", "ppy_compiler", "run", "prog.ppy")
    assert done.returncode == expected.returncode, done.stderr
    assert _output(done) == expected.stdout.rstrip("\n")
    assert _last_line(done.stderr) == _last_line(expected.stderr)
    for function in natives:
        explained = _run(tmp_path, "", "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)
    for function in kept:
        explained = _run(tmp_path, "", "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: boxed" in explained.stdout, (function, explained.stdout)


@requires_llvm
@requires_cc
def test_explain_names_the_rule(tmp_path: Path):
    _write(tmp_path, INPUT)
    held = _run(tmp_path, "", "-m", "ppy_compiler", "explain", "prog.divide")
    assert "nothing falls back after the first barrier (`input()`)" in held.stdout
    _write(tmp_path, FALLS_BACK)
    held = _run(tmp_path, "", "-m", "ppy_compiler", "explain", "prog.grow")
    assert "output is held until the call returns" in held.stdout
    _write(tmp_path, KEPT_IN_PYTHON)
    kept = _run(tmp_path, "", "-m", "ppy_compiler", "explain", "prog.after")
    assert "integer arithmetic that may not fit 64 bits can follow `input()`" in kept.stdout
    kept = _run(tmp_path, "", "-m", "ppy_compiler", "explain", "prog.parse")
    assert "can follow `input()`" in kept.stdout
    kept = _run(tmp_path, "", "-m", "ppy_compiler", "explain", "prog.copied")
    assert "takes `xs` by copy, which Python could read or change at `input()`" in kept.stdout


@requires_llvm
@requires_cc
def test_output_that_falls_back_is_printed_once(tmp_path: Path):
    _write(tmp_path, FALLS_BACK)
    done = _run(tmp_path, "", "-m", "ppy_compiler", "run", "prog.ppy")
    lines = _output(done).splitlines()
    assert lines.count("i 9") == 1
    assert lines.count("i 0") == 3


@requires_llvm
@requires_cc
def test_a_python_result_of_the_wrong_type_raises_type_error(tmp_path: Path):
    source = """
    import ppy


    @ppy.dynamic
    def shout(text: str) -> str:
        return ppy.assume[str](eval("len(text)"))


    def call(n: int) -> str:
        out = ""
        for _ in range(n):
            out = out + shout("abc")
        return out


    def main() -> None:
        print(call(1))


    main()
    """
    _write(tmp_path, source, strict=False)
    done = _run(tmp_path, "", "-m", "ppy_compiler", "run", "prog.ppy")
    explained = _run(tmp_path, "", "-m", "ppy_compiler", "explain", "prog.call")
    assert "llvm backend: native" in explained.stdout, explained.stdout
    assert _last_line(done.stderr).startswith("TypeError: `shout` returned int"), done.stderr


_HARNESS = r"""
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int64_t *ppy_io_state(void);
void ppy_io_set_hook(int64_t hook);
void ppy_io_put(int64_t stream, const int8_t *data, int64_t bytes);
int64_t ppy_io_commit(void);
void ppy_io_discard(void);
int64_t ppy_io_enter(void);
int64_t ppy_io_leave(int64_t outer);
int64_t ppy_io_call(const int8_t *name, int64_t bytes, int64_t kind, int64_t method);
void ppy_io_push_text(int8_t *text);
int8_t *ppy_io_result_text(void);
void ppy_io_answer_text(const int8_t *data, int64_t bytes);
void ppy_io_answer(int64_t kind, int64_t word);
void ppy_io_pending(int64_t tag, int64_t number, const int8_t *name, int64_t name_bytes,
                    const int8_t *text, int64_t text_bytes);
int8_t *ppy_str_new(const int8_t *data, int64_t bytes);
int64_t ppy_str_bytes(int8_t *handle);
int8_t *ppy_exc_take(void);
void ppy_coll_release(int8_t *handle);
void ppy_coll_sweep(void);

static char written[4096];
static int64_t at = 0;
static int calls = 0;

static int64_t hook(int64_t op, int64_t a, int64_t b, int64_t c) {
    if (op == 1) {
        written[at++] = (char)('0' + c);
        memcpy(written + at, (const char *)(intptr_t)a, (size_t)b);
        at += b;
        return 0;
    }
    if (op == 3) {
        calls++;
        /* A nested native call that fails: what it made is swept, and the
           outer call's strings survive the park. */
        int8_t *made = ppy_str_new((const int8_t *)"nested garbage text", 19);
        (void)made;
        ppy_coll_sweep();
        if (calls == 2) {
            ppy_io_pending(7, 0, (const int8_t *)"EOFError", 8, (const int8_t *)"eof", 3);
            return -1;
        }
        ppy_io_answer_text((const int8_t *)"answer", 6);
        ppy_io_answer(4, 0);
        return 0;
    }
    return 0;
}

int main(void) {
    ppy_io_set_hook((int64_t)(intptr_t)hook);
    int64_t outer = ppy_io_enter();
    int8_t *kept = ppy_str_new((const int8_t *)"kept across the call", 20);
    ppy_io_put(1, (const int8_t *)"one ", 4);
    ppy_io_put(1, (const int8_t *)"two\n", 4);
    ppy_io_put(2, (const int8_t *)"err\n", 4);
    ppy_io_discard();
    ppy_io_put(1, (const int8_t *)"held\n", 5);
    ppy_io_push_text(kept);
    if (ppy_io_call((const int8_t *)"m:f", 3, 4, 0) != 0) return 1;
    int8_t *answer = ppy_io_result_text();
    if (ppy_str_bytes(answer) != 6 || ppy_str_bytes(kept) != 20) return 2;
    ppy_io_put(2, (const int8_t *)"after\n", 6);
    if (ppy_io_call((const int8_t *)"m:g", 3, 4, 0) != -1) return 3;
    int8_t *raised = ppy_exc_take();
    if (raised == NULL) return 4;
    ppy_coll_release(raised);
    if (ppy_io_commit() != 0) return 5;
    if (!ppy_io_leave(outer)) return 6;
    ppy_coll_release(answer);
    ppy_coll_release(kept);
    for (int i = 0; i < 1000; i++) {
        ppy_io_put(1, (const int8_t *)"grow the buffer past its first block ", 37);
    }
    ppy_io_discard();
    written[at] = 0;
    fputs(written, stdout);
    return 0;
}
"""


def _sanitizes(compiler: str, directory: Path) -> bool:
    probe = directory / "probe.c"
    probe.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    done = subprocess.run(
        [compiler, "-fsanitize=address", "-x", "c", str(probe), "-o", str(directory / "probe")],
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


@requires_cc
def test_pyio_runs_clean_under_address_sanitizer(tmp_path: Path):
    """The buffer, the park around the hook, and an exception from Python, in C
    with the hook a C function: no leak, no use after free."""
    from ppy_runtime.collections import library_source

    compiler = c_compiler()
    assert compiler is not None
    if not _sanitizes(compiler, tmp_path):
        pytest.skip("no C compiler with AddressSanitizer")
    runtime = tmp_path / "runtime.c"
    runtime.write_text(library_source(), encoding="utf-8")
    harness = tmp_path / "harness.c"
    harness.write_text(_HARNESS, encoding="utf-8")
    binary = tmp_path / "harness"
    subprocess.run(
        [compiler, "-std=c11", "-g", "-O1", "-fsanitize=address,undefined", str(runtime),
         str(harness), "-lm", "-o", str(binary)],
        check=True,
    )  # fmt: skip
    ran = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        check=False,
        env={"ASAN_OPTIONS": "detect_leaks=1"},
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == "1held\n2after\n"
