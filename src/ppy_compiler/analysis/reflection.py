"""Who in the project looks at annotations at runtime.

Materializing an inferred annotation is invisible -- until someone reads
`f.__annotations__`, calls `inspect.signature(f)`, or prints the module's
`__annotations__`, at which point the conversion has changed the program's
output. This scan finds those readers across the whole project so the
converter can leave the observed objects exactly as their author wrote them.

Resolution is best-effort and failure is conservative: a reflective read
whose target cannot be named blocks materialization everywhere, because the
target could be anything. One shape gets more precision, because it is
everywhere: a function reading annotations off its *own parameter* -- the
classic introspecting decorator -- observes exactly what is passed to it, so
only its call sites' arguments and whatever it decorates are blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .file_facts import READERS
from .project_scan import ProjectScan, scan_project

if TYPE_CHECKING:
    from ..cache import CacheStore

__all__ = ["ReflectionIndex", "build_reflection_index"]


#: Callables whose argument's annotations become observable, by canonical
#: name -- the lexical bindings resolve `sig`, `i.signature`, and the rest
#: down to these before the check.
@dataclass(slots=True)
class ReflectionIndex:
    """Names whose annotations the project observes at runtime."""

    #: Dotted spellings whose `__annotations__`/signature someone reads.
    observed: set[str] = field(default_factory=set)
    #: Modules whose own `__annotations__` mapping is read.
    module_annotations: set[str] = field(default_factory=set)
    #: A reflective read whose target could not be named: everything may be
    #: observed, so nothing may be materialized.
    dynamic: bool = False

    def blocks_function(self, name: str, qualname: str) -> bool:
        del name
        if self.dynamic:
            return True
        # Spellings are canonical (module-qualified), so `homemade.f` blocks
        # only that `f`; matching bare tails would block every `f` there is.
        return any(
            qualname == spelling
            or qualname.endswith("." + spelling)
            or spelling.endswith("." + qualname)
            for spelling in self.observed
        )

    def blocks_module_globals(self, module: str) -> bool:
        if self.dynamic:
            return True
        return any(
            module == seen or module.endswith("." + seen) or seen.endswith("." + module)
            for seen in self.module_annotations
        )


def build_reflection_index(
    root: Path,
    source_roots: tuple[str, ...] = ("src", "."),
    *,
    scan: ProjectScan | None = None,
    store: CacheStore | None = None,
) -> ReflectionIndex:
    """Index every runtime reader of annotations in the project.

    `scan` is the project already walked and reduced, shared with the other
    whole-project indexes; without one this walks the project itself, off
    the `store` where it can.
    """
    if scan is None:
        scan = scan_project(root, source_roots, store=store)
    index = ReflectionIndex()
    index.dynamic = scan.tainted
    observers = frozenset(name for scanned in scan.modules for name in scanned.facts.observers)

    def record(found: tuple[str, ...], is_parameter: bool) -> None:
        if found:
            index.observed.update(found)
            index.module_annotations.update(found)
            return
        if is_parameter:
            return
        # A value nobody can name had its annotations read; anything could
        # be observed, so nothing may be materialized.
        index.dynamic = True

    for scanned in scan.modules:
        facts = scanned.facts
        if facts.names_own_annotations:
            index.module_annotations.add(scanned.module)
        for reader, found, is_parameter in facts.reads:
            if reader is None or not observers.isdisjoint(reader) or not READERS.isdisjoint(reader):
                # Whatever reaches a library reader or an observer has its
                # annotations read.
                record(found, is_parameter)
        for decorators, qualname in facts.decorated:
            if not observers.isdisjoint(decorators):
                index.observed.add(qualname)
    return index
