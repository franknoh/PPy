"""One pass over every source in a project, for the indexes that need them all.

`Final` needs to know who writes a name anywhere; materializing an annotation
needs to know who reads annotations anywhere. Each of those questions used to
answer itself by walking the project, parsing every file, and running the
lexical scan over each tree -- twice, in the reflection index's case. The
files are the same files. This walks them once and hands the trees and their
bindings to whoever asks.
"""

from __future__ import annotations

import ast
import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..frontend.modules import resolve_module_name
from .lexical import LexicalBindings, scan_module

__all__ = ["ProjectScan", "ScannedModule", "scan_project"]

#: Directories whose sources are not the project's own.
_SKIP = frozenset(
    {".venv", "venv", ".git", "__pycache__", "build", "dist", ".ppy-cache", ".tox", "node_modules"}
)
_SUFFIXES = (".py", ".ppy")


@dataclass(slots=True)
class ScannedModule:
    path: Path
    module: str
    tree: ast.Module
    is_package: bool
    bindings: LexicalBindings
    _nodes: tuple[ast.AST, ...] | None = field(default=None, repr=False)

    @property
    def nodes(self) -> tuple[ast.AST, ...]:
        """Every node of the tree, in `ast.walk` order, walked once for all
        the indexes built over this scan."""
        if self._nodes is None:
            self._nodes = tuple(ast.walk(self.tree))
        return self._nodes


@dataclass(slots=True)
class ProjectScan:
    """Every parseable source under the root, with its lexical bindings."""

    modules: list[ScannedModule] = field(default_factory=list)
    #: A file could not be read or parsed, so any index built from this scan
    #: is incomplete and must vouch for nothing.
    tainted: bool = False


def scan_project(root: Path, source_roots: tuple[str, ...] = ("src", ".")) -> ProjectScan:
    search_paths = [root / entry for entry in source_roots if (root / entry).is_dir()]
    if not search_paths:
        search_paths = [root]
    scan = ProjectScan()
    for path in sorted(_sources(root)):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            scan.tainted = True
            continue
        module = resolve_module_name(path, search_paths)
        is_package = path.name.startswith("__init__.")
        bindings = scan_module(tree, module, is_package=is_package)
        scan.modules.append(ScannedModule(path, module, tree, is_package, bindings))
    return scan


def _sources(root: Path):  # type: ignore[no-untyped-def]
    pending = [root]
    while pending:
        directory = pending.pop()
        with contextlib.suppress(OSError), os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in _SKIP:
                        pending.append(Path(entry.path))
                elif entry.name.endswith(_SUFFIXES):
                    yield Path(entry.path)
