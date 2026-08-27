"""CPython-parser integration (spec 7.1).

PPY introduces no grammar of its own: the configured CPython parser is the
single source of truth for what a `.ppy` file means syntactically.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ..diagnostics import Diagnostic, Severity, Span
from .source import SourceFile, load

__all__ = ["parse_source", "parse_file", "parse_text"]


def parse_source(source: SourceFile) -> Diagnostic | None:
    try:
        source.tree = ast.parse(
            source.text,
            filename=str(source.path),
            type_comments=True,
        )
    except SyntaxError as exc:
        return _syntax_diagnostic(source.path, exc)
    except ValueError as exc:
        return Diagnostic("E1001", Severity.ERROR, str(exc), Span(source.path, 1, 0))
    return None


def _syntax_diagnostic(path: Path, exc: SyntaxError) -> Diagnostic:
    span = Span(
        path=path,
        line=exc.lineno or 1,
        column=max(0, (exc.offset or 1) - 1),
        end_line=exc.end_lineno,
        end_column=max(0, (exc.end_offset or 1) - 1) if exc.end_offset else None,
    )
    return Diagnostic(
        "E1001",
        Severity.ERROR,
        exc.msg or "invalid syntax",
        span,
        help="a .ppy file must parse as ordinary Python for the configured version",
    )


def parse_file(
    path: Path, overlays: dict[Path, str] | None = None
) -> tuple[SourceFile | None, Diagnostic | None]:
    source, diagnostic = load(path, overlays)
    if source is None:
        return None, diagnostic
    diagnostic = parse_source(source)
    return source, diagnostic


def parse_text(text: str, path: Path | str = "<string>") -> tuple[SourceFile | None, Diagnostic | None]:
    source = SourceFile(path=Path(path), text=text)
    return source, parse_source(source)
