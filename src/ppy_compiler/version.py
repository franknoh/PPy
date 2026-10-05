"""What identifies this compiler, for `--version` and for cache keys.

A leaf module on purpose: the backend keys its own caches on the same
identity the driver does, and importing the driver to learn it would make
the dependency run backwards.
"""

from __future__ import annotations

import os

__all__ = ["COMPILER_VERSION", "compiler_fingerprint"]

#: The released version. The packaging build reads this literal;
#: `ppy_runtime.version` carries the runtime's copy, `ppy.__version__` reads
#: that, and a test holds them together.
COMPILER_VERSION = "0.7.0"


#: The fingerprint once worked out; one per process.
_FINGERPRINT: list[str] = []


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
    if _FINGERPRINT:
        return _FINGERPRINT[0]
    _FINGERPRINT.append(_fingerprint())
    return _FINGERPRINT[0]


def _clear() -> None:
    _FINGERPRINT.clear()


#: As `functools.cache` offered: the next call works it out again.
compiler_fingerprint.cache_clear = _clear  # type: ignore[attr-defined]


def _fingerprint() -> str:
    # Plain `os.path` and a lazy `hashlib`: this runs on every warm `ppy run`,
    # where `pathlib` and `functools` cost more than the answer.
    override = os.environ.get("PPY_COMPILER_BUILD")
    if override:
        return override
    package = os.path.dirname(os.path.realpath(__file__))
    parts = package.split(os.sep)
    if "site-packages" in parts or "dist-packages" in parts:
        return COMPILER_VERSION
    import hashlib  # pylint: disable=import-outside-toplevel

    # Sizes and modification times, not contents: reading every source cost a
    # third of a second on every command in a development tree, and an edit
    # moves the stamp just as surely as it moves the bytes. Build systems
    # settle for this, and a warm `ppy run` has to fit in a few dozen ms.
    digest = hashlib.sha256()
    for relative, size, mtime in sorted(_sources(package)):
        digest.update(f"{relative}:{size}:{mtime}".encode())
    # The C runtimes a built artifact compiles in (`ppy_runtime/*.c`) are
    # part of what it was built from, as much as the compiler is.
    try:
        with os.scandir(os.path.join(os.path.dirname(package), "ppy_runtime")) as entries:
            for entry in sorted(entries, key=lambda e: e.name):
                if entry.name.endswith(".c"):
                    stat = entry.stat()
                    digest.update(
                        f"ppy_runtime/{entry.name}:{stat.st_size}:{stat.st_mtime_ns}".encode()
                    )
    except OSError:
        pass
    return digest.hexdigest()[:16]


def _sources(package: str):  # type: ignore[no-untyped-def]
    """Every `.py` and `.c` under the package with its size and mtime, in one walk.

    `scandir` hands back the stat with the entry on most filesystems, which
    is what makes this cheap; `rglob` followed by `stat` asks twice.
    """
    prefix = len(package) + 1
    pending = [package]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                if entry.name != "__pycache__":
                    pending.append(entry.path)
            elif entry.name.endswith((".py", ".c")):
                # The C a wrapper is generated with (`backend/llvm/crossing.c`)
                # is the compiler too.
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                # Relative by slicing: `Path.relative_to` per file cost more
                # than the walk itself.
                relative = entry.path[prefix:].replace(os.sep, "/")
                yield relative, stat.st_size, stat.st_mtime_ns
