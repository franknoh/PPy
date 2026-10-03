"""Compiling and loading generated CPython-ABI wrappers (spec 16.5).

The wrapper is built against the exact interpreter it will run in, so the
artifact is keyed by that interpreter's version and ABI and is never reused
across a mismatch.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ...cache import digest
from .lowering import NativeSignature
from .wrapper import WrapperModule, generate, wrapped_in_c

__all__ = ["BuiltWrappers", "build_wrappers", "wrapper_toolchain"]


@dataclass(slots=True)
class BuiltWrappers:
    module: object | None = None
    entries: dict[str, int] = field(default_factory=dict)
    path: Path | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.module is not None

    def bind(  # type: ignore[no-untyped-def]
        self,
        qualname: str,
        address: int,
        types: tuple,
        fallback=None,
        resolve=None,
        arity: int = 0,
    ) -> object | None:
        """Point one wrapper at its native code, and hand back the fast entry.

        With `fallback`, the wrapper holds the Python implementation itself and
        invokes it from C when a guard refuses the call; without one it returns
        `NotImplemented` and the caller must watch for it. `resolve` finds the
        Python classes of the objects that cross (`collection_boundary.resolver`).
        With `arity`, the entry's parameter count, a call with keywords or
        defaults left out is bound in Python and made through the entry.
        """
        index = self.entries.get(qualname)
        if self.module is None or index is None:
            return None
        try:
            given = (
                (address, types, fallback)
                if resolve is None
                else (address, types, fallback, resolve)
            )
            named = getattr(self.module, f"bind_{index}")(*given)
        except Exception:  # noqa: BLE001 - a refusal keeps the slower path
            return None
        found = named
        if found is None:
            # A wrapper module built before the entry points carried their names
            # still answers to the index.
            found = getattr(self.module, qualname, None) or getattr(
                self.module, f"call_{index}", None
            )
        keyed_set = getattr(self.module, f"keyed_{index}", None)
        if found is not None and fallback is not None and arity and keyed_set is not None:
            from ppy_runtime.binding import keyed  # pylint: disable=import-outside-toplevel

            # Keywords and defaults bound as Python binds them, then the entry.
            keyed_set(keyed(fallback, found, arity))
        return found

    def attach_runtime(self, library=None) -> bool:  # type: ignore[no-untyped-def]
        """Point the wrappers that copy containers at the collections runtime."""
        if self.module is None:
            return False
        from ppy_runtime.collection_boundary import (
            attach,  # pylint: disable=import-outside-toplevel
        )

        return attach(self.module, library)

    def attach_effects(self, library=None) -> bool:  # type: ignore[no-untyped-def]
        """Point the wrappers of functions with effects at the held output."""
        if self.module is None:
            return False
        from ppy_runtime.effects import attach  # pylint: disable=import-outside-toplevel

        return attach(self.module, library)

    def registrar(self, qualname: str):  # type: ignore[no-untyped-def]
        """A callable that hands one specialization to the generated wrapper."""
        index = self.entries.get(qualname)
        if self.module is None or index is None:
            return None
        register = getattr(self.module, f"specialize_{index}", None)
        if register is None:
            return None

        def add(address: int, pins: tuple) -> bool:
            try:
                return bool(register(address, pins))
            except Exception:  # noqa: BLE001 - a refusal keeps the generic code
                return False

        return add


def wrapper_toolchain() -> tuple[bool, str]:
    compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        return False, "no C compiler (cc or gcc) is on PATH"
    include = Path(sysconfig.get_paths()["include"])
    if not (include / "Python.h").is_file():
        return False, f"CPython headers are missing from {include}"
    return True, f"{compiler}, headers in {include}"


def _fingerprint(source: str) -> str:
    return digest(
        source,
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        getattr(sys, "abiflags", ""),
        sysconfig.get_config_var("EXT_SUFFIX") or "",
    )[:16]


def _publish(draft: Path, final: Path) -> None:
    """Move a finished artifact onto its shared name, atomically.

    The bytes are the same whoever wrote them -- the name is their digest --
    so losing the race is not a failure. Windows refuses to replace a library
    another process has already loaded, which is exactly that case.
    """
    try:
        os.replace(draft, final)
    except OSError:
        draft.unlink(missing_ok=True)
        if not final.is_file():
            raise


def _shared_directory() -> Path | None:
    """Where every project keeps its compiled wrappers: the user's cache.

    A wrapper is named by the digest of its source, and its file name carries
    the interpreter's ABI (`EXT_SUFFIX`), so one built for any project serves
    every other with the same signatures on the same Python. Kept per project,
    each new program, and each cleared project cache, paid a C compile again.
    """
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    directory = Path(base) / "ppy" / "wrappers"
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return directory if os.access(directory, os.W_OK) else None


def build_wrappers(
    module_name: str,
    signatures: dict[str, NativeSignature],
    cache_directory: Path,
    *,
    notify=None,
    shared: bool = False,
) -> BuiltWrappers:
    """Generate, compile, and import the Python-ABI wrappers for a module.

    With `shared`, the compiled wrapper goes to the user's cache, where every
    project finds it, rather than to `cache_directory`."""
    # A coroutine hands back a future the Python side wraps; no C wrapper for it.
    signatures = {name: s for name, s in signatures.items() if not s.future}
    # A collection the generated wrapper cannot copy crosses through the
    # Python-level binding, and so does a global the binding reads.
    crossing = {name for name, s in signatures.items() if not wrapped_in_c(s)}
    signatures = {name: s for name, s in signatures.items() if name not in crossing}
    if not signatures:
        # Every function here crosses through the Python-level binding: there
        # is nothing slower about that to warn of.
        return BuiltWrappers(reason="" if crossing else "no native function to wrap")
    ready, detail = wrapper_toolchain()
    if not ready:
        if notify is not None:
            # The ctypes boundary is correct and several times slower per
            # call; a node whose Python lacks its headers should hear that
            # once, from the run, not only from `ppy doctor`.
            notify(
                f"python boundary is ctypes ({detail}); the generated CPython ABI is "
                "several times faster per call -- install the CPython headers, or use a "
                "uv-managed interpreter, and see `ppy doctor`"
            )
        return BuiltWrappers(reason=detail)

    draft: WrapperModule = generate("ppy_wrappers_placeholder", signatures)
    name = f"ppy_wrappers_{_fingerprint(draft.source)}"
    built: WrapperModule = generate(name, signatures)

    directory = (_shared_directory() if shared else None) or cache_directory / "wrappers"
    directory.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    library = directory / f"{name}{suffix}"
    source_path = directory / f"{name}.c"

    if not library.is_file():
        if notify is not None:
            notify(f"compiling {len(signatures)} Python-ABI wrapper(s)")
        # Two compilations of the same wrapper are the normal case -- one
        # cache, several processes -- and they agree on the name, because it
        # is the source's digest. So build somewhere only this thread knows
        # and publish with a rename: a reader sees the whole library or the
        # one that was there before, never a file still being written.
        # The source keeps its final name inside a directory only this thread
        # knows, and the compiler runs from there: it records the input's name
        # in the object, so a name that varied would make bytes that varied.
        stamp = f".{os.getpid()}.{threading.get_ident():x}.part"
        draft_directory = directory / f"{name}{stamp}"
        draft_directory.mkdir(exist_ok=True)
        draft_source = draft_directory / f"{name}.c"
        draft_library = library.with_name(library.name + stamp)
        draft_source.write_text(built.source, encoding="utf-8")
        compiler = os.environ.get("CC") or "cc"
        command = [
            compiler,
            "-O3",
            "-shared",
            "-fPIC",
            "-I",
            sysconfig.get_paths()["include"],
            draft_source.name,
            "-o",
            str(draft_library),
        ]
        completed = subprocess.run(
            command, cwd=draft_directory, capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            shutil.rmtree(draft_directory, ignore_errors=True)
            draft_library.unlink(missing_ok=True)
            return BuiltWrappers(
                reason=f"the wrapper did not compile: {detail[-1] if detail else 'unknown'}"
            )
        _publish(draft_library, library)
        _publish(draft_source, source_path)
        shutil.rmtree(draft_directory, ignore_errors=True)

    spec = importlib.util.spec_from_file_location(name, library)
    if spec is None or spec.loader is None:
        return BuiltWrappers(reason="the compiled wrapper could not be loaded")
    extension = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(extension)
    except Exception as exc:  # noqa: BLE001
        return BuiltWrappers(reason=f"the compiled wrapper could not be imported: {exc}")

    return BuiltWrappers(module=extension, entries=built.entries, path=library)
