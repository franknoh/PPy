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
COMPILER_VERSION = "0.1.0a1"


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
    digest = hashlib.sha256()
    for source in sorted(package.rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        digest.update(source.name.encode())
        with contextlib.suppress(OSError):
            digest.update(source.read_bytes())
    return digest.hexdigest()[:16]
