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


def triangle(base: float, height: float) -> float:
    return 0.5 * base * height


def power(base: int, exponent: int) -> int:
    return base * power(base, exponent - 1) if exponent else 1


def greet(name: str) -> int:
    return len(name) + 1


def shout(n: int) -> int:
    print(n)
    return n + 1


def check(n: int) -> None:
    if n < 0:
        raise ValueError("negative")


def squares(xs: list[int], n: int) -> None:
    for i in range(n):
        xs.append(i * i)


def lookups(d: dict[int, int]) -> int:
    s = 0
    for k in d:
        s += d[k]
    return s


def weighed(d: dict[int, int]) -> int:
    s = 0
    for k, v in d.items():
        s += k * v + (v >> 1)
    return s


class Link:
    def __init__(self, value: int) -> None:
        self.value = value
        self.next: Link | None = None


def chain(head: Link) -> int:
    s = 0
    node: Link | None = head
    while node is not None:
        s += node.value
        node = node.next
    return s
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


@pytest.mark.parametrize(
    ("name", "exposed", "why"),
    [
        # The generated wrapper costs what a Python call does: two operations gain.
        ("triangle", True, "straight-line work"),
        # Its depth is the argument's: CPython raises `RecursionError` where
        # native code has no limit to stop at.
        ("power", False, "recursion limit"),
        # A string is made natively on the way in, which a short body does not repay.
        ("greet", False, "crossing costs more"),
        # One line printed through Python costs more than CPython's `print`.
        ("shout", False, "crossing costs more"),
        ("check", False, "returns nothing"),
        # A loop that fills the caller's list returns nothing, and pays.
        ("squares", True, ""),
        # A dict's entry costs a hash and a put to copy: a lookup per entry
        # does not repay it, arithmetic on each does.
        ("lookups", False, "copying the collections"),
        ("weighed", True, ""),
        # Copying a chain of objects costs more than a short walk over it.
        ("chain", False, "object"),
    ],
)
def test_the_crossing_is_taken_where_it_is_cheaper(exposure, name: str, exposed: bool, why: str):  # type: ignore[no-untyped-def]
    found, reason = exposure(name)
    assert found == exposed, reason
    assert why in reason


def test_the_gil_is_dropped_only_around_a_call_that_may_run_long(write, analyze):  # type: ignore[no-untyped-def]
    """Dropping the GIL and taking it back costs what a two-operation body does."""
    from ppy_compiler.backend.llvm.lowering import _signature

    path = write("prog.ppy", PROGRAM)
    bundle = analyze(path, backend="llvm")

    def releases(name: str) -> bool:
        info = bundle.symbols.functions[f"prog.{name}"]
        return _signature(info, None, bundle.analysis.function(f"prog.{name}")).releases_gil

    assert not releases("add") and not releases("triangle")
    assert releases("vowels")


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
    """The signature covers the artifact's own sources, the directories its
    imports resolve through, and the project's configuration; not the rest of
    the project, and not a source edited while the artifact was built."""
    import hashlib
    import json

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    program = project / "prog.ppy"
    program.write_text("print(1)\n", encoding="utf-8")
    unrelated = project / "notes" / "other.ppy"
    unrelated.parent.mkdir()
    unrelated.write_text("X = 1\n", encoding="utf-8")
    manifest = tmp_path / "run" / "ppy-bindings.json"
    manifest.parent.mkdir()

    def built_from(text: str) -> None:
        digest = hashlib.blake2b(text.encode(), digest_size=16).hexdigest()
        program_section = {"sources": {str(program): digest}, "search_paths": [str(project)]}
        manifest.write_text(json.dumps({"program": program_section}), encoding="utf-8")

    def index():  # type: ignore[no-untyped-def]
        import marshal

        with open(fastrun._index_path(str(program)), "rb") as held:
            return marshal.load(held)

    built_from("print(1)\n")
    fastrun.remember(str(program), str(manifest))
    _version, sources, directories, taken, _manifest, _light, racy = index()
    assert sources == (str(program),)
    assert fastrun.signature(str(program), sources, directories) == taken
    # A file the program does not import changes nothing.
    unrelated.write_text("X = 22\n", encoding="utf-8")
    assert fastrun.signature(str(program), sources, directories) == taken
    # Its own source, a new file where imports resolve, or the configuration do.
    # A same-size edit this soon after `remember` may keep the stats; the file
    # is racy, and its content decides.
    assert racy == {str(program): hashlib.blake2b(b"print(1)\n", digest_size=16).hexdigest()}
    assert fastrun.current(str(program), sources, directories, taken, racy)
    program.write_text("print(2)\n", encoding="utf-8")
    assert not fastrun.current(str(program), sources, directories, taken, racy)
    program.write_text("print(1)\n", encoding="utf-8")
    os.utime(program, ns=(0, 0))
    assert not fastrun.current(str(program), sources, directories, taken, racy)
    # The directory's clock is coarse: set its time back, so that a file made
    # within the same tick as `remember` still changes it.
    os.utime(project, ns=(0, 0))
    fastrun.remember(str(program), str(manifest))
    _version, sources, directories, taken, _manifest, _light, racy = index()
    (project / "shadow.ppy").write_text("", encoding="utf-8")
    assert fastrun.signature(str(program), sources, directories) != taken
    fastrun.remember(str(program), str(manifest))
    _version, sources, directories, taken, _manifest, _light, racy = index()
    (project / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    assert fastrun.signature(str(program), sources, directories) != taken

    # A same-size edit that keeps the very stats `remember` took (a coarse clock)
    # is still seen: the racy source's content is checked.
    program.write_text("print(1)\n", encoding="utf-8")
    fastrun.remember(str(program), str(manifest))
    _version, sources, directories, taken, _manifest, _light, racy = index()
    kept = os.stat(program)
    program.write_text("print(7)\n", encoding="utf-8")
    os.utime(program, ns=(kept.st_atime_ns, kept.st_mtime_ns))
    assert fastrun.signature(str(program), sources, directories) == taken
    assert not fastrun.current(str(program), sources, directories, taken, racy)
    program.write_text("print(1)\n", encoding="utf-8")

    # A source that no longer matches what the build read is not remembered.
    os.unlink(fastrun._index_path(str(program)))
    built_from("print(0)\n")
    fastrun.remember(str(program), str(manifest))
    assert not os.path.exists(fastrun._index_path(str(program)))
    # A remembered manifest that is gone is a miss, not an error.
    built_from("print(1)\n")
    fastrun.remember(str(program), str(manifest))
    manifest.unlink()
    monkeypatch.chdir(project)
    assert fastrun.try_warm(["run", "prog.ppy"]) is None


LIGHT_PROGRAM = """
def shout(text: str) -> str:
    return text.upper() + "!"


def main() -> None:
    import sys

    print(shout("light"), sys.argv[1:])
    raise SystemExit(3)


main()
"""


def test_a_program_python_never_calls_natively_runs_without_the_launcher(tmp_path: Path):
    """No native entry crosses to Python here, so the build writes a light plan,
    and a warm run imports none of the launcher, `json`, or `ctypes`."""
    import subprocess
    import sys

    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "prog.ppy").write_text(textwrap.dedent(LIGHT_PROGRAM).lstrip(), encoding="utf-8")
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")}
    env.pop("PPY_LOWERING", None)
    run = [sys.executable, "-m", "ppy_compiler", "run", "prog.ppy", "--", "a", "b"]
    first = subprocess.run(run, cwd=tmp_path, capture_output=True, text=True, env=env, check=False)
    assert first.returncode == 3, first.stderr
    assert list((tmp_path / ".ppy-cache" / "run").glob("*/light.marshal"))
    traced = subprocess.run(
        [sys.executable, "-X", "importtime", *run[1:]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert traced.returncode == 3
    assert traced.stdout.strip().splitlines()[-1] == "LIGHT! ['a', 'b']"
    imported = {line.rsplit("|", 1)[-1].strip() for line in traced.stderr.splitlines()}
    assert not imported & {"ppy_runtime.launch", "json", "ctypes", "dataclasses"}


def test_the_launcher_caches_what_it_parsed_and_compiled(tmp_path: Path):
    """A warm run of a program with native entries reads the manifest and its
    compiled modules from the launcher's cache: no `json`, no `ast`."""
    import subprocess
    import sys

    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    # No `import ppy` here: `ppy` imports `dataclasses` itself, under `python` too.
    source = (
        "def total_squares(n: int) -> int:\n    s = 0\n    for i in range(n):\n"
        "        s += i * i\n    return s\n\n\nprint(total_squares(300))\n"
    )
    (tmp_path / "prog.ppy").write_text(source, encoding="utf-8")
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")}
    env.pop("PPY_LOWERING", None)
    run = [sys.executable, "-m", "ppy_compiler", "run", "prog.ppy"]
    for _ in range(2):
        done = subprocess.run(
            run, cwd=tmp_path, capture_output=True, text=True, env=env, check=False
        )
        assert done.returncode == 0, done.stderr
    assert list((tmp_path / ".ppy-cache" / "run").glob("*/launch.marshal"))
    traced = subprocess.run(
        [sys.executable, "-X", "importtime", *run[1:]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert traced.stdout.strip().splitlines()[-1] == str(sum(i * i for i in range(300)))
    imported = {line.rsplit("|", 1)[-1].strip() for line in traced.stderr.splitlines()}
    assert "ppy_runtime.launch" in imported
    assert not imported & {"json", "ast", "dataclasses", "inspect"}


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
