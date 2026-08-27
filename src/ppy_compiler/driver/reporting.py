"""Diagnostic and remark reporting for the CLI."""

from __future__ import annotations

import sys
from pathlib import Path

from ..diagnostics import Diagnostic, DiagnosticBag, Severity, render

__all__ = ["Reporter"]


class Reporter:
    def __init__(self, *, color: bool | None = None, quiet: bool = False, stream=None) -> None:
        self.stream = stream or sys.stderr
        self.quiet = quiet
        self.color = self.stream.isatty() if color is None else color
        self._sources: dict[Path, str] = {}

    def source(self, path: Path) -> str | None:
        if path not in self._sources:
            try:
                self._sources[path] = path.read_text(encoding="utf-8")
            except OSError:
                return None
        return self._sources[path]

    def emit(self, diagnostic: Diagnostic) -> None:
        if self.quiet and diagnostic.severity is not Severity.ERROR:
            return
        source = self.source(diagnostic.span.path) if diagnostic.span else None
        print(render(diagnostic, source=source, color=self.color), file=self.stream)
        print(file=self.stream)

    def report(self, bag: DiagnosticBag, *, show_remarks: bool = False) -> int:
        errors = warnings = 0
        for diagnostic in bag.sorted():
            if diagnostic.severity is Severity.REMARK and not show_remarks:
                continue
            if diagnostic.severity is Severity.ERROR:
                errors += 1
            elif diagnostic.severity is Severity.WARNING:
                warnings += 1
            self.emit(diagnostic)
        return errors

    def summary(self, errors: int, warnings: int, *, subject: str = "") -> None:
        if self.quiet:
            return
        if errors:
            print(f"{errors} error(s), {warnings} warning(s){subject}", file=self.stream)
        elif warnings:
            print(f"no errors, {warnings} warning(s){subject}", file=self.stream)

    def note(self, message: str) -> None:
        if not self.quiet:
            print(message, file=self.stream)
