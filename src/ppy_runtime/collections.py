"""The collections runtime: `collections.c`, and the ways a program reaches it.

The C text is written once, as an ordinary C file, and read here into one
entry per function: its result, its parameters, its body, and the other
runtime functions it calls. A standalone binary and emitted C carry the
functions they call as shims; `ppy run` loads them from a shared library
compiled from the same text on first use; an ahead-of-time library build
compiles the file in.

See `collections.c` for the layout of a collection and its families.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import threading
from pathlib import Path

__all__ = ["FUNCTIONS", "HEADERS", "library_path", "library_source", "source_path"]

#: The file, beside this module.
SOURCE = Path(__file__).with_name("collections.c")

#: What every function needs from the C library.
HEADERS = ("stdint.h", "stdio.h", "stdlib.h", "string.h")

#: name -> (result, parameters, body, needs).
Definitions = dict[str, tuple[str, tuple[str, ...], str, tuple[str, ...]]]

_DEFINITION = re.compile(
    r"^(?P<result>[a-z][a-z0-9_]*(?: \*)?)\s*\**?(?P<name>ppy_[a-z0-9_]+)"
    r"\((?P<parameters>[^)]*)\)\s*(?P<body>\{.*?^\})",
    re.MULTILINE | re.DOTALL,
)


def _parse(text: str) -> Definitions:
    """Each top-level function of the file, in the order it defines them."""
    found: Definitions = {}
    for match in _DEFINITION.finditer(text):
        parameters = " ".join(match.group("parameters").split())
        spelled = tuple(p.strip() for p in parameters.split(",")) if parameters != "void" else ()
        result = match.group("result").strip()
        if text[match.start("name") - 1] == "*" and not result.endswith("*"):
            result += " *"
        found[match.group("name")] = (result, spelled, match.group("body"), ())
    names = set(found)
    for name, (result, parameters, body, _needs) in found.items():
        called = {other for other in re.findall(r"\b(ppy_[a-z0-9_]+)\(", body) if other in names}
        found[name] = (result, parameters, body, tuple(sorted(called - {name})))
    return found


FUNCTIONS: Definitions = _parse(SOURCE.read_text(encoding="utf-8"))


def library_source() -> str:
    """The whole runtime as one C file."""
    lines = [f"#include <{header}>" for header in HEADERS] + [""]
    lines.append(SOURCE.read_text(encoding="utf-8"))
    return "\n".join(lines)


def _cache_directory() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "ppy" / "collections"


_lock = threading.Lock()
#: The compiled runtime once built, and whether building it was tried.
_built: Path | None = None
_tried = False


def source_path() -> Path:
    """The runtime's text on disk with its headers, for a build that compiles it in."""
    text = library_source()
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    target = _cache_directory() / f"ppy_collections-{digest}.c"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        draft = target.with_name(f"{target.name}.{os.getpid()}.part")
        draft.write_text(text, encoding="utf-8")
        draft.replace(target)
    return target


def library_path() -> Path | None:
    """The runtime compiled for `ppy run`, built into the cache on first use."""
    global _built, _tried  # noqa: PLW0603 - one runtime per process
    with _lock:
        if _tried:
            return _built
        _tried = True
        from .aio import compiler  # pylint: disable=import-outside-toplevel

        cc = compiler()
        if cc is None:
            return None
        source = source_path()
        target = source.with_suffix(".so")
        if not target.is_file():
            draft = target.with_name(f"{target.name}.{os.getpid()}.part")
            command = [cc, "-std=c11", "-O2", "-shared", "-fPIC", "-o", str(draft), str(source)]
            done = subprocess.run(command, capture_output=True, text=True, check=False)
            if done.returncode != 0:
                draft.unlink(missing_ok=True)
                return None
            draft.replace(target)
        _built = target
        return target
