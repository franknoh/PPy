"""The warm path of `ppy run`: find the last build of this program, cheaply.

A warm `ppy run` used to cost what a cold one did minus the code generation:
importing the compiler, initializing LLVM, and analyzing the program again to
learn that nothing had changed. All of that is spent before the first line of
the program runs, and none of it is needed when the answer is already on
disk. So this module imports nothing heavy -- no analysis, no LLVM, no cache
store -- and asks one question: is there an artifact for exactly this
program, on exactly this compiler, under exactly this configuration? When
there is, `ppy_runtime.launch` runs it the way a built launcher would.

The key over-approximates on purpose. It covers every source under the
project root rather than the ones the program imports, because knowing the
imports means analyzing, and analyzing is the cost being avoided. A change
to an unrelated file rebuilds the artifact; the per-module caches underneath
make that rebuild cheap, and a stale artifact would be a wrong answer.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from ..version import COMPILER_VERSION, compiler_fingerprint
from .config import Config, find_project_root, load_config

__all__ = ["JIT_MARKER", "MANIFEST", "Warm", "locate", "resolved_safeguards", "run_directory"]

#: What a run directory holds: a manifest the launcher can run, or a note
#: that this program needs the in-process JIT and must not be cached.
MANIFEST = "ppy-bindings.json"
JIT_MARKER = "needs-jit"

#: Directories `collect_sources` never looks in, so the key never does either.
_SKIPPED = frozenset({"__pycache__", ".ppy-cache", ".git", ".venv"})
_SUFFIXES = (".ppy", ".py")


@dataclass(frozen=True, slots=True)
class Warm:
    """Where this program's run artifact lives, and what is there now."""

    root: Path
    config: Config
    directory: Path
    manifest: Path | None
    needs_jit: bool


def resolved_safeguards(options: argparse.Namespace, configured: str | None, command: str) -> str:
    """The guard mode a command runs with, from flags, then config, then default.

    One implementation, used both to build the artifact and to name it: the
    key and the compiler must agree on what `--unsafe` meant.
    """
    explicit = getattr(options, "safeguards", None)
    if explicit:
        return str(explicit)
    if getattr(options, "unsafe", False):
        return "off"
    if getattr(options, "safe", False):
        return "hoisted"
    if configured:
        return configured
    return "off" if command == "build" else "hoisted"


def run_directory(config: Config, key: str) -> Path:
    return config.cache_path / "run" / key[:24]


def locate(file: Path, options: argparse.Namespace) -> Warm:
    """Find this program's run directory and say what it currently holds."""
    root = find_project_root(file)
    config = load_config(root)
    key = _key(file.resolve(), root, config, options)
    directory = run_directory(config, key)
    manifest = directory / MANIFEST
    return Warm(
        root=root,
        config=config,
        directory=directory,
        manifest=manifest if manifest.is_file() else None,
        needs_jit=(directory / JIT_MARKER).is_file(),
    )


def _key(file: Path, root: Path, config: Config, options: argparse.Namespace) -> str:
    """Everything that could change what `ppy run FILE` builds."""
    hasher = hashlib.blake2b(digest_size=32)

    def feed(*parts: object) -> None:
        for part in parts:
            hasher.update(str(part).encode("utf-8"))
            hasher.update(b"\x1e")

    feed("ppy-run", COMPILER_VERSION, compiler_fingerprint())
    feed(sys.version_info[:3], sys.implementation.cache_tag, sys.platform, _machine())
    # Where it is matters: the manifest records absolute search paths, so a
    # project that moved needs its artifact built again where it now lives.
    feed(
        str(root.resolve()),
        str(file.relative_to(root.resolve()) if file.is_relative_to(root) else file),
    )
    feed(
        config.strict and not getattr(options, "no_strict", False),
        getattr(options, "opt_level", None) or config.opt_level,
        config.dynamic_boundaries,
        config.inference.implicit_any,
        config.llvm.host_cpu,
        config.llvm.jit,
        config.llvm.target,
        config.parallel.enabled,
        config.parallel.threads,
        resolved_safeguards(options, config.llvm.safeguards, "run"),
        getattr(options, "prover", None) or config.llvm.prover or "off",
        sorted((name, sorted(asdict(plugin).items())) for name, plugin in config.plugins.items()),
    )
    for path, contents in _sources(root):
        feed(path, hashlib.blake2b(contents, digest_size=16).hexdigest())
    # Installed packages stand in for plugin fingerprints, which would import
    # the libraries to ask their versions: listing `*.dist-info` costs a
    # directory read, and any install or upgrade changes it.
    feed(*_installed())
    return hasher.hexdigest()


def _machine() -> str:
    try:
        return os.uname().machine
    except AttributeError:  # Windows has no uname
        return os.environ.get("PROCESSOR_ARCHITECTURE", "")


def _sources(root: Path):  # type: ignore[no-untyped-def]
    """Every source under the root with its bytes, in path order."""
    found: list[tuple[str, bytes]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        with contextlib.suppress(OSError), os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in _SKIPPED:
                        pending.append(Path(entry.path))
                elif entry.name.endswith(_SUFFIXES):
                    with contextlib.suppress(OSError):
                        found.append(
                            (os.path.relpath(entry.path, root), Path(entry.path).read_bytes())
                        )
    # The configuration file is a source too: a changed `[tool.ppy]` is a
    # changed program, whatever the `.ppy` files say.
    with contextlib.suppress(OSError):
        found.append(("pyproject.toml", (root / "pyproject.toml").read_bytes()))
    found.sort()
    return found


def _installed() -> list[str]:
    names: set[str] = set()
    for place in sys.path:
        with contextlib.suppress(OSError), os.scandir(place or ".") as entries:
            names.update(e.name for e in entries if e.name.endswith(".dist-info"))
    return sorted(names)
