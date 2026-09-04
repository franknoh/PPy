"""One pass over every source in a project, for the indexes that need them all.

`Final` needs to know who writes a name anywhere; materializing an annotation
needs to know who reads annotations anywhere. Each of those questions used to
answer itself by walking the project, parsing every file, and running the
lexical scan over each tree -- twice, in the reflection index's case. The
files are the same files. This walks them once, reduces each to the facts the
indexes combine, and keeps those facts in the cache store as one record per
project: a file that has not changed since the last run is not parsed again,
so converting one file of a large project costs a stat per file, not a parse.
"""

from __future__ import annotations

import ast
import contextlib
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..frontend.modules import resolve_module_name
from ..version import compiler_fingerprint
from .file_facts import FileFacts, facts_of
from .lexical import scan_module

if TYPE_CHECKING:
    from ..cache import CacheStore

__all__ = ["ProjectScan", "ScannedModule", "scan_project"]

#: Directories whose sources are not the project's own.
_SKIP = frozenset(
    {".venv", "venv", ".git", "__pycache__", "build", "dist", ".ppy-cache", ".tox", "node_modules"}
)
_SUFFIXES = (".py", ".ppy")
#: Bumped when what a record holds, or how it is read off a file, changes.
_FORMAT = 1


@dataclass(slots=True)
class ScannedModule:
    path: Path
    module: str
    is_package: bool
    facts: FileFacts


@dataclass(slots=True)
class ProjectScan:
    """Every parseable source under the root, reduced to its facts."""

    modules: list[ScannedModule] = field(default_factory=list)
    #: A file could not be read or parsed, so any index built from this scan
    #: is incomplete and must vouch for nothing.
    tainted: bool = False
    #: How many files took their record from the store rather than a parse.
    reused: int = 0


def scan_project(
    root: Path, source_roots: tuple[str, ...] = ("src", "."), *, store: CacheStore | None = None
) -> ProjectScan:
    """Reduce every source under `root` to its facts.

    With a `store`, the records of the last scan of this root by this
    compiler are read back as one entry; a file whose size and modification
    time still match takes its record from there, the rest are parsed, and
    the entry is written once more if anything moved.
    """
    search_paths = [root / entry for entry in source_roots if (root / entry).is_dir()]
    if not search_paths:
        search_paths = [root]
    scan = ProjectScan()
    key = ""
    kept: dict[str, tuple[int, int, FileFacts]] = {}
    if store is not None:
        key = f"scan:{_FORMAT}:{compiler_fingerprint()}:{root}"
        kept = _load(store, key)
    records: dict[str, tuple[int, int, FileFacts]] = {}
    for path in sorted(_sources(root)):
        module = resolve_module_name(path, search_paths)
        is_package = path.name.startswith("__init__.")
        stamp: tuple[int, int] | None = None
        if store is not None:
            try:
                stat = path.stat()
            except OSError:
                scan.tainted = True
                continue
            stamp = (stat.st_size, stat.st_mtime_ns)
            record = kept.get(str(path))
            if record is not None and record[:2] == stamp:
                records[str(path)] = record
                scan.modules.append(ScannedModule(path, module, is_package, record[2]))
                scan.reused += 1
                continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            scan.tainted = True
            continue
        bindings = scan_module(tree, module, is_package=is_package)
        facts = facts_of(tree, module, bindings)
        if stamp is not None:
            records[str(path)] = (*stamp, facts)
        scan.modules.append(ScannedModule(path, module, is_package, facts))
    if store is not None and records != kept:
        _save(store, key, records)
    return scan


def _load(store: CacheStore, key: str) -> dict[str, tuple[int, int, FileFacts]]:
    try:
        data = store.read(key)
    except OSError:
        return {}
    if data is None:
        return {}
    try:
        records = pickle.loads(data)
    except (pickle.UnpicklingError, EOFError, AttributeError, TypeError, ValueError):
        return {}
    found: dict[str, tuple[int, int, FileFacts]] = {}
    if not isinstance(records, dict):
        return found
    for path, value in records.items():
        record = _record(value)
        if isinstance(path, str) and record is not None:
            found[path] = record
    return found


def _record(value: object) -> tuple[int, int, FileFacts] | None:
    """A stored record, or None for anything the store handed back that is
    not one: a stale format, a truncated object."""
    if not isinstance(value, tuple) or len(value) != 3:
        return None
    size, mtime, facts = value
    if isinstance(size, int) and isinstance(mtime, int) and isinstance(facts, FileFacts):
        return (size, mtime, facts)
    return None


def _save(store: CacheStore, key: str, records: dict[str, tuple[int, int, FileFacts]]) -> None:
    with contextlib.suppress(OSError):
        store.put(key, pickle.dumps(records, protocol=pickle.HIGHEST_PROTOCOL), kind="scan")


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
