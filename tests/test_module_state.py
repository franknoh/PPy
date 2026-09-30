"""Module state under `ppy run`: settled globals passed to native code.

A module global bound once by the module's body, and never rebound, is read
natively as the object it names when the function is called: the boundary
reads it from the module and passes it as one more parameter, and a native
caller passes on the one it was given.
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


def _write(tmp_path: Path, source: str, name: str = "prog.ppy") -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    program = tmp_path / name
    program.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return program


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    return subprocess.run(
        [sys.executable, *args], cwd=tmp_path, capture_output=True, text=True, check=False, env=env
    )


def _analyze(path: Path):  # type: ignore[no-untyped-def]
    from ppy_compiler.driver.pipeline import analyze_paths, open_project

    return analyze_paths(open_project(path), [path], backend="llvm")


def test_settled_globals_are_bound_once_by_the_module(tmp_path: Path):
    """Bound once at the top level and never again: settled. A second binding
    anywhere in the module's scope (a loop, an augmented assignment, `del`) or
    a `global` statement that assigns it is not; a local of the same name in
    a function is its own name."""
    path = _write(
        tmp_path,
        """
        TABLE = [i * i for i in range(10)]
        NAMES = {"a": 1}
        COUNT = 0
        LOOPED = 0
        for LOOPED in range(3):
            pass
        GROWN = [1]
        GROWN += [2]
        GONE = [3]
        del GONE
        LIMIT = 10


        def bump() -> None:
            global COUNT
            COUNT += 1


        def shadow() -> int:
            NAMES = {"b": 2}
            return len(NAMES)
        """,
    )
    module = _analyze(path).analysis.modules["prog"]
    # A literal is a constant, folded where it is read, and not in this set.
    assert module.symbols.settled_globals == {"TABLE", "NAMES"}


def test_implicit_globals_close_over_calls(tmp_path: Path):
    """A function takes the globals it reads and every global its callees
    read, marked written when it or a callee writes one; a global it reads
    that is not settled keeps it, and its callers, in Python."""
    path = _write(
        tmp_path,
        """
        TABLE: list[int] = [i * i for i in range(10)]
        CACHE: dict[int, int] = {}
        MOVING = [1]


        def lookup(i: int) -> int:
            return TABLE[i % 10]


        def remember(i: int) -> int:
            CACHE[i] = lookup(i)
            return len(CACHE)


        def outer(i: int) -> int:
            return remember(i) + lookup(i)


        def moving() -> int:
            return len(MOVING)


        def calls_moving() -> int:
            return moving() + lookup(1)


        def rebind() -> None:
            global MOVING
            MOVING = [2]
        """,
    )
    functions = _analyze(path).analysis.modules["prog"].functions

    def held(name: str) -> list[tuple[str, bool]]:
        return [(g.key, g.written) for g in functions[f"prog.{name}"].implicit_globals]

    assert held("lookup") == [("prog:TABLE", False)]
    assert held("remember") == [("prog:CACHE", True), ("prog:TABLE", False)]
    assert held("outer") == [("prog:CACHE", True), ("prog:TABLE", False)]
    assert functions["prog.outer"].globals_native
    assert not functions["prog.moving"].globals_native
    assert not functions["prog.calls_moving"].globals_native


BOUNDARY = """
TABLE: list[int] = [i * i for i in range(10)]
SCALE = int("3")
CACHE: dict[int, int] = {}
WORD = "ab" + "c"


def lookup(n: int) -> int:
    total = 0
    for k in range(n):
        total += TABLE[k % 10] * SCALE
    for v in TABLE:
        total += v
    return total


def remember(n: int) -> int:
    for i in range(n):
        if i not in CACHE:
            CACHE[i] = lookup(i)
    return len(CACHE)


def spelled(n: int) -> int:
    total = 0
    for _ in range(n):
        for c in WORD:
            total += ord(c)
    return total
"""


@requires_llvm
@requires_cc
def test_python_calls_read_the_globals_at_the_call(tmp_path: Path):
    """The boundary reads each global from the module when the function is
    called: a rebinding the analysis could not see is the value the call
    uses, a write comes back to the module's object, and a global that is
    gone or holds another type runs the Python body."""
    from ppy_compiler.backend.llvm import _collect
    from ppy_compiler.backend.llvm.jit import JitEngine
    from ppy_compiler.backend.llvm.runtime import bind
    from ppy_runtime.collections import library_path

    path = _write(tmp_path, BOUNDARY)
    module = _collect(_analyze(path))["prog"]
    runtime = library_path()
    assert runtime is not None
    engine = JitEngine(opt_level=2).open()
    engine.load_library(str(runtime))
    engine.add(module.ir)
    engine.finalize()
    namespace: dict[str, object] = {}
    exec(compile(BOUNDARY, "prog.ppy", "exec"), namespace)  # noqa: S102 - the program itself
    fell: list[str] = []

    # A plain function's globals are its module's; one made by exec here is.
    source = "def fallback(*arguments):\n    FELL.append(NAME)\n    return PYTHON(*arguments)\n"

    def bound(name: str):  # type: ignore[no-untyped-def]
        lowered = module.functions[f"prog.{name}"]
        assert lowered.exposed, (name, lowered.exposure_reason)
        signature = lowered.boundary or lowered.signature
        assert signature.reads_globals
        own = dict(namespace)
        exec(source, own)  # noqa: S102
        own.update(FELL=fell, NAME=name, PYTHON=namespace[name])
        return bind(signature, engine.address(signature.symbol), own["fallback"]), own

    lookup, lookup_globals = bound("lookup")
    expected = namespace["lookup"](25)  # type: ignore[operator]
    assert lookup.wrapper(25) == expected and lookup.calls == 1 and fell == []
    lookup_globals["TABLE"] = [1] * 10
    assert lookup.wrapper(25) == 25 * 3 + 10 and lookup.calls == 2
    lookup_globals["SCALE"] = 2
    assert lookup.wrapper(25) == 25 * 2 + 10
    # The Python body the fallback runs reads the program's own module.
    del lookup_globals["SCALE"]
    assert lookup.wrapper(25) == expected
    assert fell == ["lookup"] and lookup.fallbacks == 1
    lookup_globals["SCALE"] = 3
    lookup_globals["TABLE"] = ("not", "a", "list")
    assert lookup.wrapper(25) == expected
    assert fell == ["lookup", "lookup"] and lookup.fallbacks == 2

    remember, remember_globals = bound("remember")
    cache = remember_globals["CACHE"]
    assert remember.wrapper(5) == 5 and remember.calls == 1
    assert cache == {i: namespace["lookup"](i) for i in range(5)}  # type: ignore[operator]
    assert remember_globals["CACHE"] is cache
    assert remember.wrapper(7) == 7 and len(cache) == 7

    spelled, _ = bound("spelled")
    assert spelled.wrapper(3) == namespace["spelled"](3)  # type: ignore[operator]
    assert spelled.calls == 1


@requires_llvm
@requires_cc
def test_ppy_run_passes_globals_across_modules(tmp_path: Path):
    """A native function that calls another module's function passes the
    globals that module's function reads; the program's answer is CPython's
    and both functions answer natively."""
    _write(
        tmp_path,
        """
        WEIGHTS: list[int] = [w * 3 for w in range(8)]
        SEEN: dict[int, int] = {}


        def weigh(n: int) -> int:
            total = 0
            for w in WEIGHTS:
                total += w * n
            SEEN[n] = total
            return total
        """,
        name="tables.py",
    )
    _write(
        tmp_path,
        """
        import tables

        OFFSETS: list[int] = [o + 1 for o in range(4)]


        def score(n: int) -> int:
            total = 0
            for i in range(n):
                total += tables.weigh(i)
                for o in OFFSETS:
                    total += o * i
            return total


        def main() -> None:
            print(score(50), len(tables.SEEN), tables.SEEN[49])
            tables.WEIGHTS[0] = 100
            print(score(50))
            print(hasattr(score, "__ppy_native__"), hasattr(tables.weigh, "__ppy_native__"))


        main()
        """,
        name="prog.py",
    )
    python = _run(tmp_path, "prog.py")
    assert python.returncode == 0, python.stderr
    ran = _run(tmp_path, "-m", "ppy_compiler", "run", "prog.py")
    assert ran.returncode == 0, ran.stderr
    lines = [line for line in ran.stdout.splitlines() if not line.startswith("compiling")]
    assert lines[:2] == python.stdout.splitlines()[:2]
    assert lines[2] == "True True"
