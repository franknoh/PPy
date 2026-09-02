"""Content-addressed artifact store with a SQLite index (spec 27)."""

from __future__ import annotations

import atexit
import contextlib
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .keys import CacheKey, digest

__all__ = ["CacheEntry", "CacheStats", "CacheStore"]

_LAYOUT = ("objects", "python", "llvm", "native", "jit", "plugins", "metadata", "locks", "gc")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    key         TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    object      TEXT NOT NULL,
    size        INTEGER NOT NULL,
    created     REAL NOT NULL,
    accessed    REAL NOT NULL,
    source      TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS dependencies (
    key        TEXT NOT NULL,
    depends_on TEXT NOT NULL,
    PRIMARY KEY (key, depends_on)
);
CREATE TABLE IF NOT EXISTS roots (
    key   TEXT PRIMARY KEY,
    label TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS artifacts_kind ON artifacts(kind);
CREATE INDEX IF NOT EXISTS artifacts_source ON artifacts(source);
"""


@dataclass(frozen=True, slots=True)
class CacheEntry:
    key: str
    kind: str
    path: Path
    size: int
    created: float
    accessed: float
    source: str


@dataclass(frozen=True, slots=True)
class CacheStats:
    entries: int
    total_bytes: int
    by_kind: dict[str, tuple[int, int]]
    root: Path

    def human_total(self) -> str:
        size = float(self.total_bytes)
        for unit in ("B", "KiB", "MiB", "GiB"):
            if size < 1024 or unit == "GiB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GiB"


#: What a damaged index says. Anything else -- busy, locked, read-only,
#: out of disk -- is a working index this process cannot have right now, and
#: destroying it would be the wrong repair.
_DAMAGE = ("malformed", "not a database", "corrupt", "encrypted")


def _is_damage(error: BaseException) -> bool:
    return any(mark in str(error).lower() for mark in _DAMAGE)


def _in_memory() -> sqlite3.Connection:
    """A cache that answers every lookup with a miss and keeps nothing."""
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.executescript(_SCHEMA)
    return connection


class CacheStore:
    """Atomic, race-safe, content-addressed artifact storage."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._connection: sqlite3.Connection | None = None
        #: Where a damaged index was moved, when one was found.
        self.quarantined: Path | None = None
        #: True once the index could not be rebuilt on disk, so every lookup
        #: is a miss and every store is dropped.
        self.disabled = False
        # A process that opened the index should not leave it to the garbage
        # collector, which reports the open handle as a ResourceWarning.
        atexit.register(self.close)

    def ensure(self) -> None:
        for directory in _LAYOUT:
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        return self.root / "index.sqlite"

    def connect(self) -> sqlite3.Connection:
        if self._connection is None:
            try:
                self._connection = self._open()
            except sqlite3.DatabaseError as error:
                if not _is_damage(error):
                    # Busy, locked, or unwritable: another process is using a
                    # perfectly good index. Work without one rather than
                    # destroying theirs.
                    self.disabled = True
                    self._connection = _in_memory()
                else:
                    # The index is optional acceleration state. A damaged one
                    # costs a rebuild, never the answer (spec 27.5).
                    self._connection = self._recover()
        return self._connection

    def _open(self) -> sqlite3.Connection:
        try:
            self.ensure()
        except OSError as error:
            # Nowhere to put a cache is a slow build, not a broken one.
            raise sqlite3.OperationalError(f"cannot prepare the cache: {error}") from error
        connection = sqlite3.connect(self.index_path, timeout=30.0, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript(_SCHEMA)
        return connection

    def _quarantine(self) -> Path | None:
        """Move a damaged index aside, keeping it for anyone who wants to look."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        kept: Path | None = None
        for suffix in ("", "-wal", "-shm"):
            broken = self.index_path.with_name(self.index_path.name + suffix)
            if not broken.exists():
                continue
            aside = broken.with_name(f"{broken.name}.corrupt-{stamp}")
            try:
                broken.replace(aside)
            except OSError:
                with contextlib.suppress(OSError):
                    broken.unlink()
                continue
            if not suffix:
                kept = aside
        return kept

    def _recover(self) -> sqlite3.Connection:
        """Rebuild the index after damage, or run without one.

        Every artifact is content-addressed on disk, so losing the index
        costs recompilation and nothing else. If even a fresh index cannot be
        opened -- a read-only cache directory, a full disk -- the store keeps
        working in memory and every lookup is simply a miss.
        """
        with contextlib.suppress(sqlite3.DatabaseError):
            self.close()
        aside = self._quarantine()
        self.quarantined = aside
        where = f" (kept at {aside})" if aside is not None else ""
        print(
            f"warning[W2101]: the build cache index was damaged and has been rebuilt{where}",
            file=sys.stderr,
        )
        try:
            return self._open()
        except (sqlite3.DatabaseError, OSError):
            self.disabled = True
            return _in_memory()

    def close(self) -> None:
        if self._connection is not None:
            with contextlib.suppress(sqlite3.DatabaseError):
                self._connection.close()
            self._connection = None

    def exists(self) -> bool:
        return self.index_path.exists()

    def object_path(self, content_hash: str) -> Path:
        return self.root / "objects" / content_hash[:2] / content_hash[2:]

    def put(
        self,
        key: CacheKey | str,
        data: bytes | str,
        *,
        kind: str | None = None,
        source: str = "",
        dependencies: Iterable[str] = (),
        suffix: str = "",
    ) -> Path:
        """Store an artifact atomically and index it."""
        try:
            return self._put(
                key, data, kind=kind, source=source, dependencies=dependencies, suffix=suffix
            )
        except sqlite3.DatabaseError as error:
            self._after(error)
            return self.object_path(
                digest(data if isinstance(data, str) else data.decode("utf-8", "surrogateescape"))
                + suffix
            )

    @contextlib.contextmanager
    def _transaction(self):  # type: ignore[no-untyped-def]
        """One unit of work across the tables an artifact spans."""
        connection = self.connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            with contextlib.suppress(sqlite3.DatabaseError):
                connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")

    def _put(
        self,
        key: CacheKey | str,
        data: bytes | str,
        *,
        kind: str | None = None,
        source: str = "",
        dependencies: Iterable[str] = (),
        suffix: str = "",
    ) -> Path:
        payload = data.encode("utf-8") if isinstance(data, str) else data
        key_hex = key.hex() if isinstance(key, CacheKey) else key
        kind = kind or (key.kind if isinstance(key, CacheKey) else "object")
        content_hash = (
            digest(payload.decode("utf-8", "surrogateescape"))
            if isinstance(data, str)
            else digest(payload)
        )
        target = self.object_path(content_hash + suffix)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                self._write_atomic(target, payload)
        except OSError:
            # Nowhere to keep it: the caller still gets the path it asked
            # about, the lookup later misses, and the build recomputes.
            self.disabled = True
            return target
        now = time.time()
        # Recording an artifact touches two tables. A reader arriving between
        # them would see a row whose dependencies are not there yet.
        with self._transaction() as connection:
            self._record(
                connection, key_hex, kind, content_hash + suffix, len(payload), source, now
            )
            for dependency in dependencies:
                connection.execute(
                    "INSERT OR IGNORE INTO dependencies(key, depends_on) VALUES(?,?)",
                    (key_hex, dependency),
                )
        return target

    @staticmethod
    def _record(
        connection: sqlite3.Connection,
        key_hex: str,
        kind: str,
        obj: str,
        size: int,
        source: str,
        now: float,
    ) -> None:
        connection.execute(
            "INSERT INTO artifacts(key, kind, object, size, created, accessed, source) "
            "VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET object=excluded.object, "
            "size=excluded.size, accessed=excluded.accessed",
            (key_hex, kind, obj, size, now, now, source),
        )

    def get(self, key: CacheKey | str) -> Path | None:
        if self.disabled:
            return None
        try:
            return self._get(key)
        except sqlite3.DatabaseError as error:
            # Damage can show up mid-query as easily as at open; a miss is
            # always a safe answer either way, but only damage earns a rebuild.
            self._after(error)
            return None

    def _get(self, key: CacheKey | str) -> Path | None:
        key_hex = key.hex() if isinstance(key, CacheKey) else key
        connection = self.connect()
        row = connection.execute("SELECT object FROM artifacts WHERE key=?", (key_hex,)).fetchone()
        if row is None:
            return None
        path = self.object_path(row[0])
        if not path.exists():
            connection.execute("DELETE FROM artifacts WHERE key=?", (key_hex,))
            return None
        connection.execute("UPDATE artifacts SET accessed=? WHERE key=?", (time.time(), key_hex))
        return path

    def read(self, key: CacheKey | str) -> bytes | None:
        path = self.get(key)
        if path is None:
            return None
        try:
            return path.read_bytes()
        except OSError:
            # The index knew about an object the filesystem no longer has.
            return None

    def read_text(self, key: CacheKey | str) -> str | None:
        data = self.read(key)
        return data.decode("utf-8") if data is not None else None

    def mark_root(self, key: CacheKey | str, label: str = "") -> None:
        key_hex = key.hex() if isinstance(key, CacheKey) else key
        try:
            self.connect().execute(
                "INSERT OR REPLACE INTO roots(key, label) VALUES(?,?)", (key_hex, label)
            )
        except sqlite3.DatabaseError as error:
            self._after(error)

    def _after(self, error: sqlite3.DatabaseError) -> None:
        """Rebuild after damage; step aside for anything else."""
        if _is_damage(error):
            self._connection = self._recover()
            return
        with contextlib.suppress(sqlite3.DatabaseError):
            self.close()
        self.disabled = True
        self._connection = _in_memory()

    def invalidate_source(self, source: str) -> int:
        connection = self.connect()
        cursor = connection.execute("DELETE FROM artifacts WHERE source=?", (source,))
        return cursor.rowcount or 0

    def _write_atomic(self, target: Path, payload: bytes) -> None:
        handle, temporary = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp-")
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def entries(self) -> Iterator[CacheEntry]:
        connection = self.connect()
        for key, kind, obj, size, created, accessed, source in connection.execute(
            "SELECT key, kind, object, size, created, accessed, source FROM artifacts"
        ):
            yield CacheEntry(key, kind, self.object_path(obj), size, created, accessed, source)

    def stats(self) -> CacheStats:
        if not self.exists():
            return CacheStats(0, 0, {}, self.root)
        connection = self.connect()
        by_kind: dict[str, tuple[int, int]] = {}
        total = count = 0
        for kind, entries, size in connection.execute(
            "SELECT kind, COUNT(*), COALESCE(SUM(size), 0) FROM artifacts GROUP BY kind"
        ):
            by_kind[kind] = (entries, size)
            count += entries
            total += size
        return CacheStats(count, total, by_kind, self.root)

    def clean(self) -> None:
        """Remove every artifact but keep the cache layout."""
        self.close()
        for directory in _LAYOUT:
            path = self.root / directory
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        self.index_path.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(str(self.index_path) + suffix).unlink(missing_ok=True)
        self.ensure()

    def gc(
        self, *, max_age_days: float | None = 30.0, max_bytes: int | None = None
    ) -> tuple[int, int]:
        """Drop unreachable and expired artifacts. Returns (removed, bytes freed)."""
        if not self.exists():
            return 0, 0
        connection = self.connect()
        reachable = self._reachable()
        removed = freed = 0
        cutoff = time.time() - (max_age_days * 86400) if max_age_days else None

        candidates = list(
            connection.execute(
                "SELECT key, object, size, accessed FROM artifacts ORDER BY accessed ASC"
            )
        )
        live_bytes = sum(row[2] for row in candidates)
        for key, _obj, size, accessed in candidates:
            expired = cutoff is not None and accessed < cutoff
            over_budget = max_bytes is not None and live_bytes > max_bytes
            if key in reachable and not expired and not over_budget:
                continue
            connection.execute("DELETE FROM artifacts WHERE key=?", (key,))
            connection.execute("DELETE FROM dependencies WHERE key=?", (key,))
            removed += 1
            freed += size
            live_bytes -= size
        self._collect_orphans()
        connection.execute("VACUUM")
        return removed, freed

    def _reachable(self) -> set[str]:
        connection = self.connect()
        roots = {row[0] for row in connection.execute("SELECT key FROM roots")}
        reachable = set(roots)
        frontier = list(roots)
        while frontier:
            key = frontier.pop()
            for (dependency,) in connection.execute(
                "SELECT depends_on FROM dependencies WHERE key=?", (key,)
            ):
                if dependency not in reachable:
                    reachable.add(dependency)
                    frontier.append(dependency)
        return reachable

    def _collect_orphans(self) -> None:
        connection = self.connect()
        referenced = {row[0] for row in connection.execute("SELECT object FROM artifacts")}
        objects = self.root / "objects"
        if not objects.is_dir():
            return
        for shard in objects.iterdir():
            if not shard.is_dir():
                continue
            for blob in shard.iterdir():
                if f"{shard.name}{blob.name}" not in referenced:
                    blob.unlink(missing_ok=True)
            if not any(shard.iterdir()):
                shard.rmdir()
