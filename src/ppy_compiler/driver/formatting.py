"""`ppy fmt`: source-preserving formatting for PPY sources (spec 4.3, 7.2)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import libcst as cst

from ..diagnostics import Diagnostic, Severity, Span
from .pipeline import collect_sources
from .reporting import Reporter

__all__ = ["run_fmt", "format_source"]

_EXTERNAL = ("ruff", "black")


def run_fmt(options: argparse.Namespace, reporter: Reporter) -> int:
    target: Path = options.path
    if not target.exists():
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{target} does not exist"))
        return 2

    sources = collect_sources(target, ppy_only=target.is_dir())
    if not sources:
        reporter.note(f"no PPY sources found under {target}")
        return 0

    changed: list[Path] = []
    failed = 0
    for path in sources:
        try:
            original = path.read_text(encoding="utf-8")
        except OSError as exc:
            reporter.emit(Diagnostic("E1002", Severity.ERROR, f"cannot read {path}: {exc}"))
            failed += 1
            continue
        try:
            formatted = format_source(original, path)
        except cst.ParserSyntaxError as exc:
            reporter.emit(
                Diagnostic("E1001", Severity.ERROR, str(exc), Span(path, getattr(exc, "raw_line", 1), 0))
            )
            failed += 1
            continue
        if formatted == original:
            continue
        changed.append(path)
        if not options.check:
            path.write_text(formatted, encoding="utf-8")

    if options.check:
        for path in changed:
            reporter.note(f"would reformat {path}")
        if changed:
            reporter.note(f"{len(changed)} file(s) would be reformatted")
            return 1
        reporter.note(f"{len(sources)} file(s) already formatted")
        return 0 if not failed else 1

    reporter.note(f"formatted {len(changed)} of {len(sources)} file(s)")
    return 1 if failed else 0


def format_source(source: str, path: Path | None = None) -> str:
    """Format PPY source, delegating to an installed formatter when present."""
    external = _external_format(source, path)
    if external is not None:
        return external
    return _normalize(source)


def _external_format(source: str, path: Path | None) -> str | None:
    """`.ppy` is ordinary Python, so an existing formatter can be reused."""
    for tool in _EXTERNAL:
        executable = shutil.which(tool)
        if executable is None:
            continue
        command = (
            [executable, "format", "--stdin-filename", str(path or "source.py"), "-"]
            if tool == "ruff"
            else [executable, "-q", "-"]
        )
        try:
            completed = subprocess.run(  # noqa: S603 - explicit executable path
                command,
                input=source,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode == 0 and completed.stdout:
            return completed.stdout
    return None


def _normalize(source: str) -> str:
    """Conservative normalization through the concrete syntax tree."""
    module = cst.parse_module(source)
    module = module.with_changes(body=_space_top_level(list(module.body)))
    code = module.code
    code = "\n".join(line.rstrip() for line in code.splitlines())
    return code + "\n" if code and not code.endswith("\n") else code


def _space_top_level(body: list[cst.BaseStatement]) -> list[cst.BaseStatement]:
    """Two blank lines before each top-level `def` or `class`."""
    spaced: list[cst.BaseStatement] = []
    for index, statement in enumerate(body):
        if (
            index > 0
            and isinstance(statement, (cst.FunctionDef, cst.ClassDef))
            and not _decorated_continuation(body[index - 1])
        ):
            statement = statement.with_changes(
                leading_lines=_blank_lines(statement.leading_lines, 2)
            )
        spaced.append(statement)
    return spaced


def _decorated_continuation(previous: cst.BaseStatement) -> bool:
    return isinstance(previous, cst.Decorator)


def _blank_lines(existing: tuple[cst.EmptyLine, ...], count: int) -> list[cst.EmptyLine]:
    comments = [line for line in existing if line.comment is not None]
    blanks = [cst.EmptyLine(indent=False) for _ in range(count)]
    return [*blanks, *comments]
