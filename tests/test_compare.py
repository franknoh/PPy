"""`examples/compare.py` holds every counterpart to one answer before it prints a timing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent.parent / "examples" / "compare.py"


def _tool(*lines: str) -> str:
    """A command that prints `lines`, as one tool of the comparison."""
    program = "\\n".join(lines)
    return f'{sys.executable} -c "print(\\"{program}\\")"'


def _compare(
    *tools: str, runs: int = 2, reference: str | None = None
) -> subprocess.CompletedProcess:
    arguments = [sys.executable, str(HARNESS), str(runs), *tools]
    if reference is not None:
        arguments.extend(["--reference", reference])
    return subprocess.run(arguments, capture_output=True, text=True, check=False)


def test_agreeing_tools_get_a_table():
    done = _compare(
        "a=" + _tool("42", "# k: 1.5 ms"),
        "b=" + _tool("42", "# k: 2.5 ms"),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "every tool agrees with a on 2 run(s) each" in done.stdout
    assert "| k | 1.50 ± 0.00 | 2.50 ± 0.00 |" in done.stdout


def test_a_tool_with_its_own_consistent_wrong_answer_is_refused():
    done = _compare(
        "ppy=" + _tool("123", "# k: 1 ms"),
        "numba=" + _tool("124", "# k: 1 ms"),
        "mojo=" + _tool("777", "# k: 1 ms"),
    )
    assert done.returncode == 1
    assert "numba: answers differ from ppy: ['124'] vs ['123']" in done.stdout
    assert "mojo: answers differ from ppy: ['777'] vs ['123']" in done.stdout
    assert "| k |" not in done.stdout, "no timings until every tool agrees"


def test_the_reference_may_be_named():
    done = _compare(
        "a=" + _tool("1", "# k: 1 ms"),
        "b=" + _tool("2", "# k: 1 ms"),
        reference="b",
    )
    assert done.returncode == 1
    assert "a: answers differ from b" in done.stdout


def test_a_nondeterministic_tool_is_refused(tmp_path: Path):
    counter = tmp_path / "count"
    counter.write_text("0", encoding="utf-8")
    flaky = (
        f'{sys.executable} -c "import pathlib; p = pathlib.Path({str(counter)!r}); '
        "n = int(p.read_text()) + 1; p.write_text(str(n)); print(n); print('# k: 1 ms')\""
    )
    done = _compare("flaky=" + flaky, runs=3)
    assert done.returncode == 1
    assert "flaky: not deterministic" in done.stdout


def test_a_missing_timing_and_a_failing_tool_are_refused():
    done = _compare(
        "a=" + _tool("7", "# k: 1 ms"),
        "b=" + _tool("7"),
        f'c={sys.executable} -c "import sys; sys.exit(3)"',
    )
    assert done.returncode == 1
    assert "b: no timing for k" in done.stdout
    assert "c: run 1 failed: exit status 3" in done.stdout


def test_the_kernel_line_format_carries_its_answer():
    done = _compare(
        "a=" + _tool("sieve 2e6      9.1 ms   -> 148933"),
        "b=" + _tool("sieve 2e6      8.2 ms   -> 148933"),
    )
    assert done.returncode == 0, done.stdout
    assert "sieve 2e6=148933" in done.stdout
    bad = _compare(
        "a=" + _tool("sieve 2e6      9.1 ms   -> 148933"),
        "b=" + _tool("sieve 2e6      8.2 ms   -> 148934"),
    )
    assert bad.returncode == 1 and "answers differ" in bad.stdout
