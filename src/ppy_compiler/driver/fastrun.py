"""The warm `ppy run FILE` before anything else is imported.

A warm run used to import the command line (argparse builds the whole parser),
read `pyproject.toml`, and read and hash every source under the project root to
recompute the key of an artifact it had built before: on a program that runs
in 30 ms, that alone made `ppy run` twice as slow as `python`. The key is only
a function of what is on disk, so this module remembers, per program, the key's
inputs as file stats and the manifest the key led to. When every stat still
matches, the manifest is the one `warm.locate` would find, and it runs with no
parser, no configuration, and no hashing.

A stat signature is what build systems trust, and what `compiler_fingerprint`
already uses: an edit moves a file's size or its modification time, a new file
or a removed one changes the walk, and anything that does not match falls
through to the full path, which computes the key and writes the index again.
The signature is taken before that key is, so a file edited while a build ran
makes the next run miss, never hit a stale artifact.

Only `ppy run FILE` and `ppy run FILE -- ARGS...` take this path; a flag of
any kind means the full command line.
"""

from __future__ import annotations

import contextlib
import marshal
import os
import sys

__all__ = ["MARKERS", "remember", "signature", "try_warm"]

#: What marks a project root; `config.find_project_root` looks for the same,
#: and a test holds the two together.
MARKERS = ("pyproject.toml", "ppy.toml", ".git")

_SKIPPED = frozenset({"__pycache__", ".ppy-cache", ".git", ".venv"})
_SUFFIXES = (".ppy", ".py")
_INDEX_VERSION = 1


def _root(file: str) -> str:
    start = os.path.dirname(file)
    directory = start
    while True:
        for marker in MARKERS:
            if os.path.exists(os.path.join(directory, marker)):
                return directory
        parent = os.path.dirname(directory)
        if parent == directory:
            return start
        directory = parent


def _index_path(file: str) -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    identity = f"{file}\0{sys.executable}\0{sys.version}"
    # `hash()` is salted per process; this name has to be the same next time.
    name = format(_stable_hash(identity), "016x")
    return os.path.join(base, "ppy", "warm-index", f"{name}.marshal")


def _stable_hash(text: str) -> int:
    value = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def signature(file: str) -> tuple:  # type: ignore[type-arg]
    """Everything `warm.locate`'s key reads from disk, as stats: the sources under
    the project root, its configuration files, the installed distributions, and
    the compiler's own identity."""
    from ..version import compiler_fingerprint

    root = _root(file)
    found: list[tuple[str, int, int]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        with contextlib.suppress(OSError), os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in _SKIPPED:
                        pending.append(entry.path)
                elif entry.name.endswith(_SUFFIXES) or entry.name in ("pyproject.toml", "ppy.toml"):
                    with contextlib.suppress(OSError):
                        stat = entry.stat()
                        found.append((entry.path, stat.st_size, stat.st_mtime_ns))
    found.sort()
    installed: set[str] = set()
    for place in sys.path:
        with contextlib.suppress(OSError), os.scandir(place or ".") as entries:
            installed.update(e.name for e in entries if e.name.endswith(".dist-info"))
    return (
        _INDEX_VERSION,
        root,
        compiler_fingerprint(),
        tuple(found),
        tuple(sorted(installed)),
        tuple(sorted((k, v) for k, v in os.environ.items() if k.startswith("PPY_"))),
    )


def remember(file: str, taken: tuple, manifest: str) -> None:  # type: ignore[type-arg]
    """Record that `taken` (a `signature` from before the key was computed) led
    to `manifest`. Best effort: a cache that cannot be written is a slower run."""
    path = _index_path(file)
    with contextlib.suppress(OSError, ValueError):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        draft = f"{path}.{os.getpid()}.part"
        with open(draft, "wb") as out:
            marshal.dump((taken, manifest), out)
        os.replace(draft, path)


def try_warm(argv: list[str]) -> int | None:
    """Run the remembered artifact of `ppy run FILE [-- ARGS...]`, or None to
    take the full path."""
    if len(argv) < 2 or argv[0] != "run":
        return None
    named, rest = argv[1], argv[2:]
    if named.startswith("-") or not named.endswith(_SUFFIXES) or (rest and rest[0] != "--"):
        return None
    file = os.path.abspath(named)
    if not os.path.isfile(file):
        return None
    try:
        with open(_index_path(file), "rb") as held:
            taken, manifest = marshal.load(held)
    except (OSError, EOFError, ValueError, TypeError):
        return None
    if not os.path.isfile(manifest) or taken != signature(file):
        return None
    from ppy_runtime.launch import main as launch

    return launch(manifest, rest[1:])
