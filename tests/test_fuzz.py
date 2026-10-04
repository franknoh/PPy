"""The differential fuzzer, CI-sized: a few seeds on every path, and every
regression it ever found.

`scripts/fuzz.py` runs thousands of generated programs; this runs a fixed
handful through CPython, the Python backend, `ppy run`, a standalone binary,
and emitted C and C++ under AddressSanitizer and UBSan, and replays each
program saved under `tests/fuzz_regressions/` on the path it failed on.
Paths whose toolchain this machine lacks are left out, and say so.
"""

from __future__ import annotations

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


def test_a_timeout_ends_every_process_the_path_started(tmp_path: Path):
    """A child that keeps the output pipe open does not hold the runner past
    its timeout: the whole process group is killed, and the timeout is a
    finding, not a pass."""
    import time  # pylint: disable=import-outside-toplevel

    from ppy_compiler.testing.fuzz import _execute  # pylint: disable=import-outside-toplevel

    script = tmp_path / "spawner.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        "time.sleep(600)\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    status, _out, err = _execute([sys.executable, str(script)], tmp_path, timeout=2.0)
    assert status == TIMED_OUT and err == "timed out"
    assert time.monotonic() - started < 30
    reference = Result("python", "1\n", 0, "")
    hung = Result("run", "", TIMED_OUT, "timed out")
    assert [m.reason for m in compare({"python": reference, "run": hung})] == ["timed out"]
