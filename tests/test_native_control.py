"""`raise`, `try`, `assert`, and generators, lowered natively.

A module that raises or catches has its checks as exceptions native code can
catch: an index out of range inside a `try` is an `IndexError` the handler
takes, with no fall back to Python. An exception crosses native calls by
the raised status and lets go of what each frame held. One nothing native
catches ends the call: Python runs it again and raises it, and a standalone
binary prints CPython's last line and exits with status 1.

A generator is lowered into the loop that consumes it, and a generator
expression too. Each program is held to CPython under `ppy`, `ppy run`, a
standalone binary, and emitted C and C++ under AddressSanitizer, with
`ppy explain` confirming every function went native.
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

EXCEPTIONS = """
from ppy import HashMap, Vec


class BadInput(ValueError):
    pass


class Missing(LookupError):
    pass


def check(n: int) -> int:
    if n < 0:
        raise BadInput(f"negative: {n}")
    if n == 13:
        raise KeyError("unlucky")
    return n * 2


def safe_get(v: Vec[int], i: int) -> int:
    try:
        return v[i]
    except IndexError as error:
        return -len(str(error))


def loops(n: int) -> int:
    got = 0
    for i in range(-2, n):
        try:
            got += check(i)
        except BadInput as error:
            got += 1000 + len(str(error))
        except LookupError as error:
            got += 7 + len(f"{error}")
        else:
            got += 1
        finally:
            got += 100
    return got


def arithmetic(a: int, b: int) -> int:
    try:
        assert b != 3, "three"
        return a // b
    except ZeroDivisionError:
        return -1
    except AssertionError as error:
        return -len(str(error))


def lookup(table: HashMap[str, int], key: str) -> int:
    scratch = Vec[int]()
    scratch.push(len(key))
    if key not in table:
        raise Missing(f"no {key}")
    return table[key] + scratch[0]


def middle(table: HashMap[str, int], keys: Vec[str]) -> int:
    held = Vec[str]()
    total = 0
    for key in keys:
        held.push(key)
        total += lookup(table, key)
    return total + len(held)


def frames(n: int) -> int:
    table = HashMap[str, int]()
    for i in range(n):
        table[f"k{i}"] = i
    keys = Vec[str]()
    for i in range(n + 2):
        keys.push(f"k{i}")
    try:
        return middle(table, keys)
    except Missing as error:
        return -len(str(error))
    finally:
        table["z"] = 1


def leaving(n: int) -> int:
    total = 0
    for i in range(n):
        try:
            if i == 2:
                continue
            if i == 5:
                break
            total += i
        finally:
            total += 100
    try:
        return total
    finally:
        total += 1_000_000


def nested(n: int) -> int:
    total = 0
    try:
        try:
            if n > 3:
                raise KeyError(n)
            total += 1
        except KeyError:
            total += 10
            raise
        finally:
            total += 100
    except LookupError as error:
        total += 1000 + len(str(error))
        if isinstance(error, KeyError):
            total += 5
        if isinstance(error, IndexError):
            total += 50
    return total


def main() -> None:
    v = Vec[int]()
    v.push(5)
    print(safe_get(v, 0), safe_get(v, 4), loops(15), arithmetic(7, 0), arithmetic(7, 2))
    print(arithmetic(7, 3), frames(5), frames(0), leaving(8), nested(1), nested(9))


main()
"""

GENERATORS = """
from collections.abc import Iterator

from ppy import HashMap, Vec


def countdown(n: int) -> Iterator[int]:
    while n > 0:
        yield n
        n -= 1


def evens(limit: int) -> Iterator[int]:
    for i in range(limit):
        if i % 2 == 0:
            yield i


def chained(limit: int) -> Iterator[int]:
    yield from evens(limit)
    yield from countdown(3)
    yield -1


def squares(v: Vec[int]) -> Iterator[int]:
    for x in v:
        yield x * x


def first_big(limit: int) -> Iterator[int]:
    for i in range(limit):
        if i * i > 50:
            yield i
            return


def words(text: str) -> Iterator[str]:
    for part in text.split():
        yield part.lower()


def checked(limit: int) -> Iterator[int]:
    for i in range(limit):
        if i == 7:
            raise ValueError(f"seven at {i}")
        yield i * 10


def consume(n: int) -> int:
    total = 0
    for x in countdown(n):
        if x == 3:
            continue
        total = total * 7 + x
        total %= 1000003
    for y in chained(n):
        if y > 100:
            break
        total += y
    else:
        total += 5
    return total


def reduce_all(n: int) -> int:
    v = Vec[int](evens(n))
    s = sum(x * 3 for x in countdown(n) if x % 3 != 0)
    m = max(countdown(n))
    lo = min(i - 50 for i in evens(n))
    flags = int(any(x > n for x in countdown(n))) * 10 + int(all(x > 0 for x in countdown(n)))
    return s + m * 1000 + len(v) * 100000 + lo + flags * 7 + sum(squares(v)) + next(first_big(n), -9)


def tally(text: str) -> int:
    seen = HashMap[str, int]()
    for w in words(text):
        seen[w] = seen.get(w, 0) + 1
    joined = Vec[str](words(text))
    return len(seen) * 1000 + seen["the"] * 10 + len(joined)


def guarded(limit: int) -> int:
    total = 0
    try:
        for x in checked(limit):
            total += x
    except ValueError as error:
        total = -total - len(str(error))
    return total


def main() -> None:
    text = "The cat and THE dog saw the Fox"
    print(consume(12), reduce_all(20), tally(text), guarded(5), guarded(20))


main()
"""

PROGRAMS = {
    "exceptions": (
        EXCEPTIONS,
        ["check", "safe_get", "loops", "arithmetic", "frames", "leaving", "nested"],
    ),
    "generators": (GENERATORS, ["consume", "reduce_all", "tally", "guarded"]),
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


def _expected(tmp_path: Path, source: str) -> str:
    _write(tmp_path, source)
    done = _run(tmp_path, "prog.ppy")
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@requires_llvm
@requires_cc
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_every_path_agrees_and_each_function_goes_native(tmp_path: Path, name: str):
    source, natives = PROGRAMS[name]
    expected = _expected(tmp_path, source)
    for args in (["-m", "ppy_compiler", "prog.ppy"], ["-m", "ppy_compiler", "run", "prog.ppy"]):
        done = _run(tmp_path, *args)
        assert done.returncode == 0, done.stderr
        assert _output(done).strip() == expected, args
    for function in natives:
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" in explained.stdout, (function, explained.stdout)


@requires_standalone
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_a_standalone_binary_agrees(tmp_path: Path, name: str):
    """A standalone binary has no Python to fall back to: what it prints is what
    native code caught and computed itself."""
    expected = _expected(tmp_path, PROGRAMS[name][0])
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "--standalone", "prog.ppy", "-o", "dist")
    assert built.returncode == 0, built.stderr
    ran = subprocess.run(
        [str(tmp_path / "dist" / "prog")], capture_output=True, text=True, check=False
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == expected


def _sanitizes(compiler: str, directory: Path) -> bool:
    probe = directory / "probe.c"
    probe.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    done = subprocess.run(
        [compiler, "-fsanitize=address", "-x", "c", str(probe), "-o", str(directory / "probe")],
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


def _emitted(tmp_path: Path, language: str, unsafe: bool) -> Path:
    compiler = c_compiler() if language == "c" else _CXX
    if compiler is None or not _sanitizes(compiler, tmp_path):
        pytest.skip(f"no {language} compiler with AddressSanitizer")
    emitted = tmp_path / f"prog.{language}"
    flags = ["--unsafe"] if unsafe else []
    done = _run(
        tmp_path, "-m", "ppy_compiler", "emit", language, "--standalone", *flags, "prog.ppy",
        "-o", emitted.name,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    standard = "-std=c11" if language == "c" else "-std=c++17"
    binary = tmp_path / "prog"
    subprocess.run(
        [compiler, standard, "-g", "-O1", "-fsanitize=address,undefined", str(emitted), "-lm",
         "-o", str(binary)],
        check=True,
    )  # fmt: skip
    return binary


@requires_llvm
@pytest.mark.parametrize("unsafe", [False, True])
@pytest.mark.parametrize("language", ["c", "cpp"])
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_emitted_source_frees_everything_once(
    tmp_path: Path, name: str, language: str, unsafe: bool
):
    expected = _expected(tmp_path, PROGRAMS[name][0])
    binary = _emitted(tmp_path, language, unsafe)
    ran = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        check=False,
        env={"ASAN_OPTIONS": "detect_leaks=1"},
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == expected


UNCAUGHT = """
from ppy import Vec


class Broken(RuntimeError):
    pass


def deep(v: Vec[int], n: int) -> int:
    if n == 0:
        raise Broken(f"at the bottom of {len(v)}")
    v.push(n)
    return deep(v, n - 1) + 1


def top(n: int) -> int:
    v = Vec[int]()
    try:
        return deep(v, n)
    except KeyError:
        return -1


def main() -> None:
    print(top(3))


main()
"""


@requires_llvm
@requires_cc
def test_an_exception_nothing_catches_is_cpythons(tmp_path: Path):
    """Under `ppy run` Python runs the call again and raises it; a standalone
    binary prints the line CPython's traceback ends with and exits 1."""
    _write(tmp_path, UNCAUGHT)
    expected = _run(tmp_path, "prog.ppy")
    assert expected.returncode == 1
    last = expected.stderr.strip().splitlines()[-1]
    assert last == "Broken: at the bottom of 3"
    done = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert done.returncode == 1
    # `ppy run` imports the program as a module, whose name CPython spells.
    assert done.stderr.strip().splitlines()[-1].endswith(last)
    explained = _run(tmp_path, "-m", "ppy_compiler", "explain", "prog.top")
    assert "llvm backend: native" in explained.stdout, explained.stdout


@requires_standalone
def test_a_standalone_binary_says_what_nothing_caught(tmp_path: Path):
    _write(tmp_path, UNCAUGHT)
    built = _run(tmp_path, "-m", "ppy_compiler", "build", "--standalone", "prog.ppy", "-o", "dist")
    assert built.returncode == 0, built.stderr
    ran = subprocess.run(
        [str(tmp_path / "dist" / "prog")], capture_output=True, text=True, check=False
    )
    assert ran.returncode == 1
    assert ran.stderr.strip().splitlines()[-1] == "Broken: at the bottom of 3"


@requires_llvm
@pytest.mark.parametrize("language", ["c", "cpp"])
def test_emitted_source_says_it_and_frees_every_frame(tmp_path: Path, language: str):
    _write(tmp_path, UNCAUGHT)
    binary = _emitted(tmp_path, language, False)
    ran = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        check=False,
        env={"ASAN_OPTIONS": "detect_leaks=1"},
    )
    assert ran.returncode == 1
    assert ran.stderr.strip().splitlines()[-1] == "Broken: at the bottom of 3"


PYTHON_ONLY = """
from collections.abc import Iterator


def numbers(n: int) -> Iterator[int]:
    for i in range(n):
        yield i


def arguments(n: int) -> int:
    try:
        raise ValueError(n)
    except ValueError as error:
        return len(error.args)


def chained(n: int) -> int:
    try:
        raise ValueError("inner") from KeyError(n)
    except ValueError:
        return 1


def stepped(n: int) -> int:
    made = numbers(n)
    return next(made) + next(made)


def main() -> None:
    print(arguments(1), chained(1), stepped(3))


main()
"""


@requires_llvm
@requires_cc
def test_what_native_code_does_not_take_stays_in_python(tmp_path: Path):
    """`e.args`, `raise ... from ...`, and a generator stepped twice keep the
    function on its Python body, which answers."""
    expected = _expected(tmp_path, PYTHON_ONLY)
    done = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.ppy")
    assert done.returncode == 0, done.stderr
    assert _output(done).strip() == expected
    for function in ("arguments", "chained", "stepped"):
        explained = _run(tmp_path, "-m", "ppy_compiler", "explain", f"prog.{function}")
        assert "llvm backend: native" not in explained.stdout, (function, explained.stdout)


def test_a_function_returning_str_binds_without_the_collection_crossing():
    """A `str` result comes back as text: the binding calls the native code,
    as a `ppy build` library's does, rather than running the Python body."""
    import ctypes  # pylint: disable=import-outside-toplevel

    from ppy_runtime.abi import TEXT, NativeParam, NativeSignature
    from ppy_runtime.binding import bind

    libc = ctypes.CDLL(None)
    libc.strdup.restype = ctypes.c_void_p
    libc.strdup.argtypes = (ctypes.c_char_p,)

    @ctypes.CFUNCTYPE(
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int64),
    )
    def native(n, address, length):  # type: ignore[no-untyped-def]
        text = f"native {n}".encode()
        address[0] = libc.strdup(text)
        length[0] = len(text)
        return 0

    signature = NativeSignature(
        "m.f", "ppy_m_f", (NativeParam("n", "int"),), (TEXT,), returned="str"
    )
    binding = bind(signature, ctypes.cast(native, ctypes.c_void_p).value, lambda n: f"python {n}")
    assert binding.wrapper(3) == "native 3"
