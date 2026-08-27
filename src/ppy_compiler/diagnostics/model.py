"""Diagnostic model and renderer (spec 29)."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

__all__ = ["Severity", "Span", "Diagnostic", "DiagnosticBag", "render", "PPyError"]


class Severity(enum.StrEnum):
    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"
    REMARK = "remark"


@dataclass(frozen=True, slots=True)
class Span:
    path: Path
    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.column + 1}"

    @property
    def width(self) -> int:
        if self.end_line == self.line and self.end_column is not None:
            return max(1, self.end_column - self.column)
        return 1


@dataclass(slots=True)
class Diagnostic:
    code: str
    severity: Severity
    message: str
    span: Span | None = None
    notes: list[str] = field(default_factory=list)
    help: str | None = None

    def with_note(self, note: str) -> "Diagnostic":
        self.notes.append(note)
        return self

    @property
    def is_error(self) -> bool:
        return self.severity is Severity.ERROR


class PPyError(Exception):
    """Raised when compilation cannot continue."""

    def __init__(self, diagnostics: Sequence[Diagnostic]) -> None:
        self.diagnostics = list(diagnostics)
        super().__init__(f"{len(self.diagnostics)} PPY diagnostic(s)")


class DiagnosticBag:
    def __init__(self) -> None:
        self._items: list[Diagnostic] = []
        self._seen: set[tuple] = set()

    def add(self, diagnostic: Diagnostic) -> Diagnostic:
        """Record a diagnostic, ignoring one already reported at that place.

        A loop body is analyzed repeatedly to reach a fixpoint, so the same
        finding can be produced several times for one source location.
        """
        span = diagnostic.span
        key = (
            diagnostic.code,
            diagnostic.message,
            None if span is None else (str(span.path), span.line, span.column),
        )
        if key in self._seen:
            return diagnostic
        self._seen.add(key)
        self._items.append(diagnostic)
        return diagnostic

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        for diagnostic in diagnostics:
            self.add(diagnostic)

    def error(self, code: str, message: str, span: Span | None = None, help: str | None = None) -> Diagnostic:
        return self.add(Diagnostic(code, Severity.ERROR, message, span, help=help))

    def warning(self, code: str, message: str, span: Span | None = None, help: str | None = None) -> Diagnostic:
        return self.add(Diagnostic(code, Severity.WARNING, message, span, help=help))

    def remark(self, code: str, message: str, span: Span | None = None) -> Diagnostic:
        return self.add(Diagnostic(code, Severity.REMARK, message, span))

    @property
    def items(self) -> list[Diagnostic]:
        return list(self._items)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self._items if d.is_error]

    def has_errors(self) -> bool:
        return any(d.is_error for d in self._items)

    def raise_if_errors(self) -> None:
        if self.has_errors():
            raise PPyError(self.errors)

    def sorted(self) -> list[Diagnostic]:
        def key(d: Diagnostic) -> tuple:
            if d.span is None:
                return ("", 0, 0, d.code)
            return (str(d.span.path), d.span.line, d.span.column, d.code)

        return sorted(self._items, key=key)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)


_COLORS = {
    Severity.ERROR: "\033[1;31m",
    Severity.WARNING: "\033[1;33m",
    Severity.NOTE: "\033[1;36m",
    Severity.REMARK: "\033[1;34m",
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def render(diagnostic: Diagnostic, *, source: str | None = None, color: bool = False) -> str:
    def paint(text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if color else text

    head = paint(f"{diagnostic.severity}[{diagnostic.code}]", _COLORS[diagnostic.severity])
    lines = [f"{head}: {paint(diagnostic.message, _BOLD)}"]
    span = diagnostic.span
    if span is not None:
        lines.append(f"  --> {span}")
        snippet = _snippet(span, source)
        if snippet:
            lines.append("")
            lines.extend(snippet)
    for note in diagnostic.notes:
        lines.append(f"  = note: {note}")
    if diagnostic.help:
        lines.append(f"  = help: {diagnostic.help}")
    return "\n".join(lines)


def _snippet(span: Span, source: str | None) -> list[str]:
    if source is None:
        try:
            source = span.path.read_text(encoding="utf-8")
        except OSError:
            return []
    lines = source.splitlines()
    if not 1 <= span.line <= len(lines):
        return []
    text = lines[span.line - 1]
    gutter = str(span.line)
    pad = " " * len(gutter)
    caret = " " * span.column + "^" * span.width
    return [f"{gutter} | {text}", f"{pad} | {caret}"]
