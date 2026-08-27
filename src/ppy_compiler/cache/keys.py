"""Cache key construction (spec 27.2)."""

from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import dataclass
from typing import Iterable

__all__ = ["digest", "CacheKey", "environment_fingerprint", "FRONTEND_SCHEMA_VERSION"]

#: Bump when the semantic AST or analysis artifact layout changes.
FRONTEND_SCHEMA_VERSION = 1


def digest(*parts: object) -> str:
    """A fast cryptographic content digest over the given parts."""
    hasher = hashlib.blake2b(digest_size=32)
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\x1e")
    return hasher.hexdigest()


def environment_fingerprint() -> str:
    """Everything that invalidates every artifact when it changes."""
    return digest(
        "ppy",
        FRONTEND_SCHEMA_VERSION,
        f"{sys.version_info.major}.{sys.version_info.minor}",
        sys.implementation.name,
        getattr(sys, "abiflags", ""),
        platform.machine(),
        platform.system(),
    )


@dataclass(frozen=True, slots=True)
class CacheKey:
    """A content-addressed key for one compilation artifact."""

    kind: str
    source: str
    compiler_version: str
    target: str
    options: str
    dependencies: str = ""
    plugins: str = ""

    @staticmethod
    def build(
        kind: str,
        *,
        source_digest: str,
        compiler_version: str,
        opt_level: int,
        directives: Iterable[str] = (),
        target: str = "python",
        dependency_hashes: Iterable[str] = (),
        plugin_fingerprints: Iterable[str] = (),
        extra: Iterable[object] = (),
    ) -> "CacheKey":
        return CacheKey(
            kind=kind,
            source=source_digest,
            compiler_version=compiler_version,
            target=digest(target, environment_fingerprint()),
            options=digest(opt_level, tuple(sorted(directives)), tuple(extra)),
            dependencies=digest(tuple(sorted(dependency_hashes))),
            plugins=digest(tuple(sorted(plugin_fingerprints))),
        )

    def hex(self) -> str:
        return digest(
            self.kind, self.source, self.compiler_version,
            self.target, self.options, self.dependencies, self.plugins,
        )

    def __str__(self) -> str:
        return f"{self.kind}:{self.hex()[:16]}"
