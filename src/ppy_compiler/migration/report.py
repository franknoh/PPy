"""Structured migration findings and the report `ppy migrate` renders.

A migration is not judged by how little the source changed; it is judged by
how much of the project became strict PPY and how precisely the rest is
described. Every finding carries one of five classifications, so the report
can say "418 sites rewritten, 12 need a boundary, 2 are unsupported" instead
of a wall of diagnostics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..diagnostics import Diagnostic, Severity

__all__ = ["Classification", "MigrationReport", "Rewrite", "classify"]


class Classification(Enum):
    """What one migration finding means for the person migrating."""

    AUTOFIXED = "autofixed"
    REQUIRES_REWRITE = "requires-rewrite"
    UNSUPPORTED = "unsupported"
    DYNAMIC_BOUNDARY = "dynamic-boundary"
    OPTIMIZATION_OPPORTUNITY = "optimization-opportunity"


@dataclass(frozen=True, slots=True)
class Rewrite:
    """One site a migration pass changed automatically."""

    line: int
    pass_name: str
    message: str


#: `eval`/`exec` build code at runtime; no rewrite recovers static source.
_UNSUPPORTED = frozenset({"E1501"})
#: Constructs whose escape hatch is an explicit `ppy.dynamic` boundary.
_BOUNDARY = frozenset({"E1504", "E1505"})


def classify(diagnostic: Diagnostic) -> Classification | None:
    """What a leftover checker finding means for the migration.

    Errors classify; advisory warnings do too when they are migration-shaped
    (`E1504` dynamic-feature advisories, `E1304` inference gaps). A plain
    warning is not a migration finding.
    """
    code = diagnostic.code
    if code in _UNSUPPORTED:
        return Classification.UNSUPPORTED
    if code in _BOUNDARY:
        return Classification.DYNAMIC_BOUNDARY
    if code == "R3003":
        return Classification.OPTIMIZATION_OPPORTUNITY
    if code == "E1304":
        return Classification.REQUIRES_REWRITE
    if diagnostic.severity is Severity.ERROR:
        return Classification.REQUIRES_REWRITE
    return None


@dataclass(slots=True)
class MigrationReport:
    """Everything one `ppy migrate` run decided, counted honestly."""

    root: Path
    files_scanned: int = 0
    files_changed: int = 0
    annotations: dict[str, int] = field(default_factory=dict)
    rewrites: dict[Path, list[Rewrite]] = field(default_factory=dict)
    findings: list[tuple[Classification, Diagnostic]] = field(default_factory=list)

    def add_rewrites(self, path: Path, rewrites: list[Rewrite]) -> None:
        if rewrites:
            self.rewrites.setdefault(path, []).extend(rewrites)

    def add_finding(self, diagnostic: Diagnostic) -> None:
        found = classify(diagnostic)
        if found is not None:
            self.findings.append((found, diagnostic))

    def count(self, classification: Classification) -> int:
        return sum(1 for c, _ in self.findings if c is classification)

    @property
    def sites_rewritten(self) -> int:
        return sum(len(entries) for entries in self.rewrites.values())

    @property
    def annotations_written(self) -> int:
        return sum(self.annotations.values())

    def summary_lines(self) -> list[str]:
        lines = [
            "migration summary",
            f"  files scanned:              {self.files_scanned}",
            f"  files changed:              {self.files_changed}",
            f"  annotations written:        {self.annotations_written}",
            f"  sites rewritten:            {self.sites_rewritten}",
        ]
        labeled = (
            ("dynamic boundaries needed", Classification.DYNAMIC_BOUNDARY),
            ("manual rewrites required", Classification.REQUIRES_REWRITE),
            ("unsupported constructs", Classification.UNSUPPORTED),
            ("optimization opportunities", Classification.OPTIMIZATION_OPPORTUNITY),
        )
        for label, classification in labeled:
            count = self.count(classification)
            if count:
                lines.append(f"  {label}:{' ' * max(1, 28 - len(label) - 1)}{count}")
        return lines

    def to_json(self) -> str:
        def relative(path: Path) -> str:
            try:
                return str(path.resolve().relative_to(self.root.resolve()))
            except ValueError:
                return str(path)

        payload = {
            "files_scanned": self.files_scanned,
            "files_changed": self.files_changed,
            "annotations": dict(sorted(self.annotations.items())),
            "sites_rewritten": self.sites_rewritten,
            "rewrites": [
                {
                    "file": relative(path),
                    "line": rewrite.line,
                    "pass": rewrite.pass_name,
                    "message": rewrite.message,
                }
                for path, entries in sorted(self.rewrites.items())
                for rewrite in entries
            ],
            "findings": [
                {
                    "classification": classification.value,
                    "code": diagnostic.code,
                    "file": relative(diagnostic.span.path) if diagnostic.span else None,
                    "line": diagnostic.span.line if diagnostic.span else None,
                    "message": diagnostic.message,
                }
                for classification, diagnostic in self.findings
            ],
            "counts": {
                classification.value: self.count(classification)
                for classification in Classification
                if classification is not Classification.AUTOFIXED
            }
            | {Classification.AUTOFIXED.value: self.sites_rewritten},
        }
        return json.dumps(payload, indent=2, sort_keys=False) + "\n"
