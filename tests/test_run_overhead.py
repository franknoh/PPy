"""What keeps `ppy run` from being slower than `python`, checked without a clock.

Timing belongs to `scripts/run_overhead.py`; these hold the decisions that
produce the times: which functions cross the Python boundary, what the warm
path accepts and remembers, and the layouts the inlined string code relies on.
"""

from __future__ import annotations

import ctypes
import os
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.driver import fastrun
from ppy_compiler.driver.config import _MARKERS

PROGRAM = """
from ppy import Vec


def add(a: int, b: int) -> int:
    return a + b


def ends(v: Vec[int]) -> int:
    total = 0
    for i in range(3):
        total += v[i] + v[len(v) - 1 - i]
    return total


def total(v: Vec[int]) -> int:
    s = 0
    for x in v:
        s += x
    return s


def fill(rows: Vec[Vec[int]], n: int) -> None:
    for i in range(n):
        rows[i % len(rows)].push(i)


def vowels(text: str) -> int:
    n = 0
    for c in text:
        if c in "aeiou":
            n += 1
    return n
"""


@pytest.fixture
def exposure(write, analyze):  # type: ignore[no-untyped-def]
    from ppy_compiler.backend.llvm.lowering import should_lower_native
    from ppy_compiler.driver.ir_pipeline import value_class_layouts

    path = write("prog.ppy", PROGRAM)
    bundle = analyze(path, backend="llvm")
    layouts = value_class_layouts(bundle)

    def decided(name: str) -> tuple[bool, str]:
        info = bundle.symbols.functions[f"prog.{name}"]
        return should_lower_native(info, bundle.analysis.function(f"prog.{name}"), layouts)

    return decided


def test_a_tiny_body_stays_off_the_boundary(exposure):  # type: ignore[no-untyped-def]
    exposed, reason = exposure("add")
    assert not exposed and "crossing costs more" in reason


def test_a_collection_the_body_barely_touches_is_not_copied_across(exposure):  # type: ignore[no-untyped-def]
    """`ends` reads three elements from each end of a `Vec` the boundary would
    copy whole, in and back, on every call."""
    exposed, reason = exposure("ends")
    assert not exposed and "copying the collections" in reason


@pytest.mark.parametrize("name", ["total", "fill", "vowels"])
def test_work_that_grows_with_the_argument_crosses(exposure, name: str):  # type: ignore[no-untyped-def]
    exposed, reason = exposure(name)
    assert exposed, reason


def test_the_warm_path_finds_projects_as_the_configuration_does():
    assert fastrun.MARKERS == _MARKERS


@pytest.mark.parametrize(
    "argv",
    [
        ["run"],
        ["check", "prog.ppy"],
        ["run", "--unsafe", "prog.ppy"],
        ["run", "prog.ppy", "--unsafe"],
        ["run", "notes.txt"],
    ],
)
def test_the_warm_path_takes_only_a_plain_run(argv: list[str]):
    assert fastrun.try_warm(argv) is None


def test_the_warm_path_runs_only_what_nothing_since_has_changed(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    program = project / "prog.ppy"
    program.write_text("print(1)\n", encoding="utf-8")
    other = project / "helper.ppy"
    other.write_text("X = 1\n", encoding="utf-8")
    manifest = tmp_path / "ppy-bindings.json"
    manifest.write_text("{}", encoding="utf-8")

    taken = fastrun.signature(str(program))
    fastrun.remember(str(program), taken, str(manifest))
    assert fastrun.signature(str(program)) == taken
    # Any source under the root, a configuration file, or a new file changes it.
    other.write_text("X = 22\n", encoding="utf-8")
    assert fastrun.signature(str(program)) != taken
    other.write_text("X = 1\n", encoding="utf-8")
    os.utime(other, ns=(0, 0))
    assert fastrun.signature(str(program)) != taken
    (project / "new.ppy").write_text("", encoding="utf-8")
    assert fastrun.signature(str(program)) != taken
    # A remembered manifest that is gone is a miss, not an error.
    manifest.unlink()
    monkeypatch.chdir(project)
    assert fastrun.try_warm(["run", "prog.ppy"]) is None


def test_the_static_characters_are_where_the_inlined_code_looks():
    """`for c in s` and `c in "..."` compute a static character's address as
    the table's start plus `byte * 26` words; the runtime must lay them out so."""
    from ppy_compiler.lowering.strings import _CHAR_WORDS
    from ppy_runtime.collections import library_path

    built = library_path()
    if built is None:
        pytest.skip("no C compiler to build the runtime")
    runtime = ctypes.CDLL(str(built))
    runtime.ppy_str_char.restype = ctypes.c_void_p
    runtime.ppy_str_char.argtypes = [ctypes.c_int64]
    runtime.ppy_str_ascii_table.restype = ctypes.c_void_p
    start = runtime.ppy_str_ascii_table()
    for byte in (0, 1, ord("a"), 127):
        assert runtime.ppy_str_char(byte) == start + byte * _CHAR_WORDS * 8


def test_a_literal_is_one_handle_for_good():
    """A literal is looked up once per function and never let go of; the runtime
    keeps it off every heap list, with a count no use brings to zero."""
    from ppy_runtime.collections import library_path

    built = library_path()
    if built is None:
        pytest.skip("no C compiler to build the runtime")
    runtime = ctypes.CDLL(str(built))
    runtime.ppy_str_interned.restype = ctypes.c_void_p
    runtime.ppy_str_interned.argtypes = [ctypes.c_char_p, ctypes.c_int64]
    runtime.ppy_coll_release.argtypes = [ctypes.c_void_p]
    runtime.ppy_coll_live_handles.restype = ctypes.c_int64
    text = ctypes.create_string_buffer(textwrap.dedent("hello, world").encode())
    live = runtime.ppy_coll_live_handles()
    first = runtime.ppy_str_interned(text, 12)
    for _ in range(3):
        runtime.ppy_coll_release(first)
    assert runtime.ppy_str_interned(text, 12) == first
    assert runtime.ppy_coll_live_handles() == live
