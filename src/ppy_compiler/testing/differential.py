"""Differential execution across the three PPY paths (spec 33.1)."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..diagnostics import Diagnostic, Severity

__all__ = ["Outcome", "Comparison", "run_paths", "compare", "run_suite"]

_TIMEOUT = 120


@dataclass(frozen=True, slots=True)
class Outcome:
    path: str
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass(frozen=True, slots=True)
class Comparison:
    file: Path
    outcomes: dict[str, Outcome]
    mismatches: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.mismatches


def _run(command: list[str], cwd: Path) -> Outcome:
    try:
        completed = subprocess.run(  # noqa: S603 - commands are constructed here
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Outcome(" ".join(command), 124, "", "timed out")
    return Outcome(" ".join(command), completed.returncode, completed.stdout, completed.stderr)


def run_paths(file: Path, *, include_llvm: bool = True) -> dict[str, Outcome]:
    """Run the same program through plain CPython, `ppy`, and `ppy run`."""
    cwd = file.parent
    outcomes = {
        "python": _run([sys.executable, file.name], cwd),
        "ppy": _run([sys.executable, "-m", "ppy_compiler", file.name], cwd),
    }
    if include_llvm:
        outcomes["ppy run"] = _run([sys.executable, "-m", "ppy_compiler", "run", file.name], cwd)
    return outcomes


def compare(file: Path, outcomes: dict[str, Outcome]) -> Comparison:
    baseline = outcomes.get("python")
    mismatches: list[str] = []
    if baseline is None:
        return Comparison(file, outcomes, ("no plain CPython baseline",))
    for name, outcome in outcomes.items():
        if name == "python":
            continue
        if outcome.stdout != baseline.stdout:
            mismatches.append(f"{name}: stdout differs from plain CPython")
        if outcome.exit_code != baseline.exit_code:
            mismatches.append(
                f"{name}: exit code {outcome.exit_code} != {baseline.exit_code}"
            )
    return Comparison(file, outcomes, tuple(mismatches))


def run_suite(target: Path, reporter) -> int:  # type: ignore[no-untyped-def]
    from ..driver.pipeline import collect_sources

    files = [p for p in collect_sources(target, ppy_only=True)]
    if not files:
        reporter.note(f"no .ppy sources found under {target}")
        return 0

    failures = 0
    for file in files:
        outcomes = run_paths(file)
        comparison = compare(file, outcomes)
        if comparison.ok:
            reporter.note(f"ok       {file}")
            continue
        failures += 1
        reporter.emit(
            Diagnostic(
                "E1802",
                Severity.ERROR,
                f"differential mismatch for {file}: " + "; ".join(comparison.mismatches),
            )
        )
        for name, outcome in outcomes.items():
            if outcome.stderr.strip():
                reporter.note(f"  [{name}] {outcome.stderr.strip().splitlines()[-1]}")
    reporter.note(f"{len(files) - failures}/{len(files)} program(s) matched plain CPython")
    return 1 if failures else 0
