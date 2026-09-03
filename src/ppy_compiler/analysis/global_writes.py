"""Which module-level names the rest of the project assigns to.

`Final` is a promise about the whole program, so proving it takes the whole
program: `config.LIMIT = 5` in a file that was not being converted rebinds the
name just as surely as a second assignment at home would. This index is built
once over every source under the project root -- including files outside the
current conversion -- and answers the one question the converter asks.

Name resolution comes from the shared lexical bindings (`analysis/lexical.py`):
scope-aware, point-sensitive, alias-propagating, `global`-aware. When a file
cannot be parsed at all, the index fails closed and vouches for nothing.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from .lexical import LexicalBindings
from .project_scan import ProjectScan, scan_project

__all__ = ["GlobalWriteIndex", "build_write_index"]


@dataclass(slots=True)
class GlobalWriteIndex:
    """`module -> names` for every cross-module attribute write."""

    writes: dict[str, set[str]] = field(default_factory=dict)
    #: Modules hit by a `setattr` whose name is not a literal. No name in
    #: them is provably unbound.
    dynamic: set[str] = field(default_factory=set)
    #: A project file could not be read or parsed, so the scan is incomplete
    #: and the index must vouch for nothing.
    tainted: bool = False

    def can_emit_final(self, module: str, name: str) -> bool:
        """Is `module.name` free of assignments anywhere else in the project?

        `module` is the converter's qualified name for the file; imports may
        reach it under a shorter spelling, so both directions of suffix match
        count as the same module. Missing evidence is a no.
        """
        if self.tainted:
            return False
        for target, names in self.writes.items():
            if _same_module(module, target) and name in names:
                return False
        return not any(_same_module(module, target) for target in self.dynamic)


def _same_module(module: str, target: str) -> bool:
    return module == target or module.endswith("." + target) or target.endswith("." + module)


def build_write_index(
    root: Path, source_roots: tuple[str, ...] = ("src", "."), *, scan: ProjectScan | None = None
) -> GlobalWriteIndex:
    """Index every cross-module write in the project.

    `scan` is the project already walked and parsed, shared with the other
    whole-project indexes; without one this walks the project itself.
    """
    if scan is None:
        scan = scan_project(root, source_roots)
    index = GlobalWriteIndex()
    # An unreadable file could hold any write; claiming `Final` anyway would
    # be promising something the scan never saw.
    index.tainted = scan.tainted
    for scanned in scan.modules:
        _scan(scanned.tree, scanned.module, scanned.bindings, index)
    return index


def _scan(
    tree: ast.Module, module: str, bindings: LexicalBindings, index: GlobalWriteIndex
) -> None:

    def record(base: ast.expr, attr: str) -> None:
        for target in bindings.targets_at(base):
            index.writes.setdefault(target, set()).add(attr)

    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = list(node.targets)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"setattr", "delattr"}
            and node.args
        ):
            found = bindings.targets_at(node.args[0])
            written = node.args[1] if len(node.args) > 1 else None
            for target in found:
                if isinstance(written, ast.Constant) and isinstance(written.value, str):
                    index.writes.setdefault(target, set()).add(written.value)
                else:
                    index.dynamic.add(target)
            continue
        for target in targets:
            if isinstance(target, ast.Attribute):
                record(target.value, target.attr)
