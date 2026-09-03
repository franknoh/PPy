"""What identifies this compiler, for `--version` and for cache keys.

A leaf module on purpose: the backend keys its own caches on the same
identity the driver does, and importing the driver to learn it would make
the dependency run backwards.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import os
from pathlib import Path

__all__ = ["COMPILER_VERSION", "compiler_fingerprint"]

#: The released version. `src/ppy/__init__.py` and the packaging metadata
#: carry the same string, and a test holds all three together.
COMPILER_VERSION = "0.1.1a1"


@functools.cache
def compiler_fingerprint() -> str:
    """What identifies this compiler build, for cache keys.

    A version string only changes at releases; the compiler changes at every
    edit. A cache keyed on the string alone will happily serve IR and
    generated code from before a codegen fix -- silently wrong, or silently
    slow. So a development tree (anything outside `site-packages`) hashes its
    own sources once per process, while a regular install -- immutable until
    the version moves -- keeps the free constant. `PPY_COMPILER_BUILD`
    overrides both for build systems that already know their identity.
    """
    override = os.environ.get("PPY_COMPILER_BUILD")
    if override:
        return override
    package = Path(__file__).resolve().parent
    if "site-packages" in package.parts or "dist-packages" in package.parts:
        return COMPILER_VERSION
    # Sizes and modification times, not contents: reading every source cost a
    # third of a second on every command in a development tree, and an edit
    # moves the stamp just as surely as it moves the bytes. Build systems
    # settle for this, and a warm `ppy run` has to fit in a few dozen ms.
    digest = hashlib.sha256()
    for relative, size, mtime in sorted(_sources(package)):
        digest.update(f"{relative}:{size}:{mtime}".encode())
    return digest.hexdigest()[:16]


def _sources(package: Path):  # type: ignore[no-untyped-def]
    """Every `.py` under the package with its size and mtime, in one walk.

    `scandir` hands back the stat with the entry on most filesystems, which
    is what makes this cheap; `rglob` followed by `stat` asks twice.
    """
    pending = [package]
    while pending:
        directory = pending.pop()
        with contextlib.suppress(OSError), os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name != "__pycache__":
                        pending.append(Path(entry.path))
                elif entry.name.endswith(".py"):
                    with contextlib.suppress(OSError):
                        stat = entry.stat()
                        relative = str(Path(entry.path).relative_to(package))
                        yield relative, stat.st_size, stat.st_mtime_ns
