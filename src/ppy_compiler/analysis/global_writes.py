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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .project_scan import ProjectScan, scan_project

if TYPE_CHECKING:
    from ..cache import CacheStore

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
    root: Path,
    source_roots: tuple[str, ...] = ("src", "."),
    *,
    scan: ProjectScan | None = None,
    store: CacheStore | None = None,
) -> GlobalWriteIndex:
    """Index every cross-module write in the project.

    `scan` is the project already walked and reduced, shared with the other
    whole-project indexes; without one this walks the project itself, off
    the `store` where it can.
    """
    if scan is None:
        scan = scan_project(root, source_roots, store=store)
    index = GlobalWriteIndex()
    # An unreadable file could hold any write; claiming `Final` anyway would
    # be promising something the scan never saw.
    index.tainted = scan.tainted
    for scanned in scan.modules:
        for target, names in scanned.facts.writes:
            index.writes.setdefault(target, set()).update(names)
        index.dynamic.update(scanned.facts.dynamic_writes)
    return index
