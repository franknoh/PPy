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

__all__ = [
    "normalize_source",
    "run_fmt",
    "format_source",
]

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
    return normalize_source(source)


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


def normalize_source(source: str) -> str:
    """Conservative normalization through the concrete syntax tree.

    Deterministic and self-contained, so a converted file is byte-identical
    wherever it is produced.
    """
    module = cst.parse_module(source)
    module = module.with_changes(body=_space_top_level(list(module.body)))
    module = module.visit(_WrapSignatures())
    code = module.code
    code = "\n".join(line.rstrip() for line in code.splitlines())
    return code + "\n" if code and not code.endswith("\n") else code


#: Annotating a signature can push it past a line limit that the source kept.
_LINE_LIMIT = 100


class _WrapSignatures(cst.CSTTransformer):
    """Put each parameter on its own line when a signature grows too long."""

    def leave_FunctionDef(
        self, original: cst.FunctionDef, updated: cst.FunctionDef
    ) -> cst.FunctionDef:
        params = updated.params
        # `star_arg` is a sentinel rather than None when absent, so it has to be
        # tested by type.
        variadic = isinstance(params.star_arg, cst.Param) or params.star_kwarg is not None
        if not params.params or variadic or params.kwonly_params:
            return updated
        header = _header_width(updated)
        if header <= _LINE_LIMIT:
            return updated
        wrapped = [
            param.with_changes(comma=_break_after(last=index == len(params.params) - 1))
            for index, param in enumerate(params.params)
        ]
        return updated.with_changes(
            params=params.with_changes(params=wrapped),
            whitespace_before_params=cst.ParenthesizedWhitespace(
                first_line=cst.TrailingWhitespace(),
                indent=True,
                last_line=cst.SimpleWhitespace("    "),
            ),
        )


def _break_after(*, last: bool) -> cst.Comma:
    """A trailing comma that starts the next line, closing at column 0 on the last."""
    return cst.Comma(
        whitespace_after=cst.ParenthesizedWhitespace(
            first_line=cst.TrailingWhitespace(),
            indent=True,
            last_line=cst.SimpleWhitespace("" if last else "    "),
        )
    )


def _header_width(node: cst.FunctionDef) -> int:
    """How wide the `def` line is once its body is set aside."""
    stripped = node.with_changes(
        body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
        leading_lines=(),
    )
    rendered = cst.Module(body=[stripped]).code.splitlines()
    return max((len(line) for line in rendered[:-1]), default=0)


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
