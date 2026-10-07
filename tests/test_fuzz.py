"""The differential fuzzer, CI-sized: a few seeds on every path, and every
regression it ever found.

`scripts/fuzz.py` runs thousands of generated programs; this runs a fixed
handful through CPython, the Python backend, `ppy run`, a standalone binary,
and emitted C and C++ under AddressSanitizer and UBSan, and replays each
program saved under `tests/fuzz_regressions/` on the path it failed on.
Paths whose toolchain this machine lacks are left out, and say so.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler, standalone_toolchain_status
from ppy_compiler.testing.fuzz import (
    ALL_PATHS,
    OVERFLOW_64,
    STATE_PATHS,
    TIMED_OUT,
    UNANNOTATED_MARK,
    Result,
    compare,
    generate_program,
    run_program,
)

REGRESSIONS = Path(__file__).parent / "fuzz_regressions"

#: Seeds run on every path in CI; each takes a few seconds.
SEEDS = (0, 1, 2, 3)


def _available() -> tuple[str, ...]:
    paths = ["python", "ppy"]
    if llvm_available():
        paths.append("run")
        if standalone_toolchain_status()[0]:
            paths.append("standalone")
        if c_compiler() is not None:
            paths.extend(("c", "cpp"))
    return tuple(paths)


def test_the_generator_is_deterministic_and_valid_python():
    assert generate_program(11) == generate_program(11)
    assert generate_program(11) != generate_program(12)
    compile(generate_program(11), "prog.ppy", "exec")


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_programs_agree_on_every_path(seed: int):
    paths = _available()
    results = run_program(generate_program(seed), paths)
    assert results["python"].status == 0 or results["python"].last_error, results["python"]
    mismatches = compare(results)
    assert not mismatches, [
        (m.path, m.reason, m.expected.last_error, m.found.last_error, m.found.stderr[-2000:])
        for m in mismatches
    ]


@pytest.mark.parametrize("seed", SEEDS[:2])
def test_programs_with_module_state_agree(seed: int):
    """Module globals read and written, and objects Python made walked and
    written, on the paths with a Python boundary."""
    paths = tuple(p for p in _available() if p in STATE_PATHS)
    source = generate_program(seed, state=True)
    assert "G_TABLE" in source and "class Link" in source
    mismatches = compare(run_program(source, paths))
    assert not mismatches, [
        (m.path, m.reason, m.expected.last_error, m.found.last_error, m.found.stderr[-2000:])
        for m in mismatches
    ]


@pytest.mark.parametrize("seed", SEEDS[:2])
def test_unannotated_programs_agree(seed: int):
    """Functions without annotations, typed from `main`'s calls under
    `--no-strict`, then called from Python with other types: the native
    entries refuse those and the Python bodies run."""
    paths = tuple(p for p in _available() if p in STATE_PATHS)
    source = generate_program(seed, unannotated=True)
    assert source.startswith(UNANNOTATED_MARK) and "getattr(here," in source
    mismatches = compare(run_program(source, paths))
    assert not mismatches, [
        (m.path, m.reason, m.expected.last_error, m.found.last_error, m.found.stderr[-2000:])
        for m in mismatches
    ]


@pytest.mark.parametrize("seed", SEEDS[:1])
def test_unannotated_programs_with_inference_evidence_agree(seed: int):
    """A `functools.wraps` decorator, operators on a value class, a `list`
    parameter, mixed `int` and `float` calls, `argparse`, and parameters
    typed by their use, each called from Python with other types too."""
    paths = tuple(p for p in _available() if p in STATE_PATHS)
    source = generate_program(seed, unannotated=True, inference=True)
    assert "@functools.wraps(fn)" in source and "def __lt__(self, other):" in source
    mismatches = compare(run_program(source, paths))
    assert not mismatches, [
        (m.path, m.reason, m.expected.last_error, m.found.last_error, m.found.stderr[-2000:])
        for m in mismatches
    ]


@pytest.mark.parametrize("seed", SEEDS[:1])
def test_unannotated_programs_with_acting_decorators_agree(seed: int):
    """Project decorators that scale a result, print, count, cache, swap
    arguments, or hand back another function: every call by the name, from
    Python and from functions that go native, reaches what they returned."""
    paths = tuple(p for p in _available() if p in STATE_PATHS)
    source = generate_program(seed, unannotated=True, decorators=True)
    assert "seen[n] = fn(n)" in source and "wrapper.calls += 1" in source
    mismatches = compare(run_program(source, paths))
    assert not mismatches, [
        (m.path, m.reason, m.expected.last_error, m.found.last_error, m.found.stderr[-2000:])
        for m in mismatches
    ]


@pytest.mark.parametrize("seed", SEEDS[:2])
def test_programs_that_edit_structures_agree(seed: int):
    """Classes with unannotated fields, a tree with parent links and a doubly
    linked list, relinked in place by methods Python calls natively: every
    shape, link, and identity is CPython's."""
    paths = tuple(p for p in _available() if p in STATE_PATHS)
    source = generate_program(seed, structures=True)
    assert source.startswith(UNANNOTATED_MARK) and "class STree" in source
    mismatches = compare(run_program(source, paths))
    assert not mismatches, [
        (m.path, m.reason, m.expected.last_error, m.found.last_error, m.found.stderr[-2000:])
        for m in mismatches
    ]


def _regressions() -> list[Path]:
    return sorted(REGRESSIONS.glob("*.ppy"))


@pytest.mark.parametrize("file", _regressions(), ids=lambda path: path.stem)
def test_every_saved_regression_stays_fixed(file: Path):
    header, _, source = file.read_text(encoding="utf-8").partition("\n")
    path = header.split("path=", 1)[1].split()[0]
    if path not in _available():
        pytest.skip(f"no toolchain for the {path} path here")
    mismatches = compare(run_program(source, ("python", path)))
    assert not mismatches, [
        (m.reason, m.expected.last_error, m.found.last_error) for m in mismatches
    ]


def test_the_one_allowed_difference_is_the_64_bit_overflow():
    """Native code with no Python to fall back to stops where CPython computes
    an integer past a word, having printed the same lines before; any other
    difference is a mismatch."""
    reference = Result("python", "1\n2\n18446744073709551616\n", 0, "")
    stopped = Result("standalone", "1\n2\n", 1, OVERFLOW_64)
    assert not compare({"python": reference, "standalone": stopped})
    assert compare({"python": reference, "run": stopped})
    wrong = Result("c", "1\n3\n", 1, OVERFLOW_64)
    assert compare({"python": reference, "c": wrong})
    assert set(ALL_PATHS) >= set(_available())


def _left_running(marker: str) -> list[int]:
    """The processes whose command line holds `marker`."""
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                line = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            if marker.encode() in line:
                found.append(int(entry.name))
    return found


def _gone(marker: str) -> bool:
    import time  # pylint: disable=import-outside-toplevel

    deadline = time.monotonic() + 10
    while _left_running(marker) and time.monotonic() < deadline:
        time.sleep(0.1)
    return not _left_running(marker)


def _spawner(tmp_path: Path, marker: str, then: str, holding: bool = True) -> Path:
    """A script that starts grandchildren (sleeping, `marker` in their command
    lines; one holding the output pipes open when `holding`, one not) and
    then does `then`."""
    script = tmp_path / "spawner.py"
    held = f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)', {marker!r}])"
    script.write_text(
        "import subprocess, sys, time\n"
        + (held + "\n" if holding else "")
        + f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)', {marker!r}],"
        " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"{then}\n",
        encoding="utf-8",
    )
    return script


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_a_timeout_ends_every_process_the_path_started(tmp_path: Path):
    """A child that keeps the output pipe open does not hold the runner past
    its timeout: the whole process group is killed, grandchildren too, and
    the timeout is a finding, not a pass."""
    import time  # pylint: disable=import-outside-toplevel

    from ppy_compiler.testing.fuzz import _execute  # pylint: disable=import-outside-toplevel

    marker = f"ppy-fuzz-timeout-{tmp_path.name}"
    script = _spawner(tmp_path, marker, "time.sleep(600)")
    started = time.monotonic()
    status, _out, err = _execute([sys.executable, str(script), marker], tmp_path, timeout=2.0)
    assert status == TIMED_OUT and err == "timed out"
    assert time.monotonic() - started < 30
    assert _gone(marker), _left_running(marker)
    reference = Result("python", "1\n", 0, "")
    hung = Result("run", "", TIMED_OUT, "timed out")
    assert [m.reason for m in compare({"python": reference, "run": hung})] == ["timed out"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_a_path_that_exits_leaves_no_process_behind(tmp_path: Path):
    from ppy_compiler.testing.fuzz import _execute  # pylint: disable=import-outside-toplevel

    marker = f"ppy-fuzz-exited-{tmp_path.name}"
    script = _spawner(tmp_path, marker, "print('done')", holding=False)
    status, out, _err = _execute([sys.executable, str(script), marker], tmp_path, timeout=30.0)
    assert (status, out) == (0, "done\n")
    assert _gone(marker), _left_running(marker)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
@pytest.mark.parametrize("how", ["SIGTERM", "SIGKILL"])
def test_a_fuzz_run_killed_from_outside_takes_its_path_along(tmp_path: Path, how: str):
    """The runner itself killed mid-path: with SIGTERM (`scripts/fuzz.py`
    turns it into an exit) the group goes on the way out; with SIGKILL the
    kernel kills the path it started (`PR_SET_PDEATHSIG`)."""
    import signal  # pylint: disable=import-outside-toplevel
    import subprocess  # pylint: disable=import-outside-toplevel
    import time  # pylint: disable=import-outside-toplevel

    marker = f"ppy-fuzz-{how}-{tmp_path.name}"
    looping = tmp_path / "looping.py"
    looping.write_text("while True:\n    pass\n", encoding="utf-8")
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import signal, sys\n"
        "from pathlib import Path\n"
        "from ppy_compiler.testing.fuzz import _execute\n"
        "signal.signal(signal.SIGTERM, lambda n, f: sys.exit(1))\n"
        f"_execute([sys.executable, {str(looping)!r}, {marker!r}], Path({str(tmp_path)!r}), 600)\n",
        encoding="utf-8",
    )
    process = subprocess.Popen([sys.executable, str(runner)])  # pylint: disable=consider-using-with
    try:
        deadline = time.monotonic() + 60
        while not _left_running(marker) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert _left_running(marker)
        process.send_signal(getattr(signal, how))
        process.wait(timeout=30)
        assert _gone(marker), _left_running(marker)
    finally:
        process.kill()
        for pid in _left_running(marker):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)


def test_the_minimizer_never_leaves_a_loop_that_cannot_end():
    """A deletion inside a `while` (the step that ends it) or inside a
    structure method (a link that keeps the structure acyclic) is not
    offered: a reduced candidate once looped forever under CPython."""
    from ppy_compiler.testing.fuzz import _statements  # pylint: disable=import-outside-toplevel

    source = (
        "class STree:\n"
        "    def insert(self, key):\n"
        "        node = self.root\n"
        "        node.left = key\n"
        "def countdown(n):\n"
        "    total = 0\n"
        "    while n > 0:\n"
        "        total += n\n"
        "        n -= 2\n"
        "    return total\n"
    )
    offered = {source.splitlines()[start - 1].strip() for start, _end in _statements(source)}
    assert offered == {"total = 0", "while n > 0:", "return total"}
