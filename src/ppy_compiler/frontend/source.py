"""Source files and span construction (spec 7.1)."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from ..diagnostics import Diagnostic, Severity, Span

__all__ = ["SOURCE_SUFFIXES", "SourceFile", "load", "span_of"]

SOURCE_SUFFIXES = (".ppy", ".py")


@dataclass(slots=True)
class SourceFile:
    path: Path
    text: str
    tree: ast.Module | None = None
    _lines: list[str] | None = field(default=None, repr=False)
    _nodes: tuple[ast.AST, ...] | None = field(default=None, repr=False)

    @property
    def lines(self) -> list[str]:
        if self._lines is None:
            self._lines = self.text.splitlines()
        return self._lines

    @property
    def nodes(self) -> tuple[ast.AST, ...]:
        """Every node of the tree, in `ast.walk` order, walked once.

        A whole-project pass asks for every node of every module, and a
        conversion makes a dozen such passes over the same trees; the
        generator walk was a tenth of its time. The tree is not changed
        after parsing, so the tuple stands.
        """
        if self._nodes is None:
            assert self.tree is not None
            self._nodes = tuple(ast.walk(self.tree))
        return self._nodes

    @property
    def is_ppy(self) -> bool:
        return self.path.suffix == ".ppy"

    def digest(self) -> str:
        return hashlib.blake2b(self.text.encode("utf-8"), digest_size=16).hexdigest()

    def span(self, node: ast.AST) -> Span:
        return span_of(self.path, node)


def span_of(path: Path, node: ast.AST) -> Span:
    line = getattr(node, "lineno", 1)
    column = getattr(node, "col_offset", 0)
    return Span(
        path=path,
        line=line,
        column=column,
        end_line=getattr(node, "end_lineno", None),
        end_column=getattr(node, "end_col_offset", None),
    )


def load(
    path: Path, overlays: dict[Path, str] | None = None
) -> tuple[SourceFile | None, Diagnostic | None]:
    """Read a source file, preferring an editor's unsaved buffer (spec 28.1)."""
    if overlays is not None and path in overlays:
        return SourceFile(path=path, text=overlays[path]), None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, Diagnostic(
            "E1002",
            Severity.ERROR,
            f"cannot read {path}: {exc.strerror or exc}",
            Span(path, 1, 0),
        )
    return SourceFile(path=path, text=text), None
