"""Compiling and loading generated CPython-ABI wrappers (spec 16.5).

The wrapper is built against the exact interpreter it will run in, so the
artifact is keyed by that interpreter's version and ABI and is never reused
across a mismatch.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import sysconfig
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ...cache import digest
from .lowering import NativeSignature
from .wrapper import WrapperModule, generate

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

    def bind(self, qualname: str, address: int, types: tuple, fallback=None) -> object | None:  # type: ignore[no-untyped-def]
        """Point one wrapper at its native code, and hand back the fast entry.

        With `fallback`, the wrapper holds the Python implementation itself and
        invokes it from C when a guard refuses the call; without one it returns
        `NotImplemented` and the caller must watch for it.
        """
        index = self.entries.get(qualname)
        if self.module is None or index is None:
            return None
        try:
            getattr(self.module, f"bind_{index}")(address, types, fallback)
        except Exception:  # noqa: BLE001 - a refusal keeps the slower path
            return None
        return getattr(self.module, f"call_{index}", None)

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
    import shutil

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


def build_wrappers(
    module_name: str,
    signatures: dict[str, NativeSignature],
    cache_directory: Path,
    *,
    notify=None,
) -> BuiltWrappers:
    """Generate, compile, and import the Python-ABI wrappers for a module."""
    if not signatures:
        return BuiltWrappers(reason="no native function to wrap")
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

    directory = cache_directory / "wrappers"
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
        # The stamp goes before the suffix, not after it: a compiler reads
        # the input language from the extension, and `.c.1234.part` is not C.
        stamp = f".{os.getpid()}.{threading.get_ident():x}.part"
        draft_source = directory / f"{name}{stamp}.c"
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
            str(draft_source),
            "-o",
            str(draft_library),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            draft_source.unlink(missing_ok=True)
            draft_library.unlink(missing_ok=True)
            return BuiltWrappers(
                reason=f"the wrapper did not compile: {detail[-1] if detail else 'unknown'}"
            )
        _publish(draft_library, library)
        _publish(draft_source, source_path)

    spec = importlib.util.spec_from_file_location(name, library)
    if spec is None or spec.loader is None:
        return BuiltWrappers(reason="the compiled wrapper could not be loaded")
    extension = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(extension)
    except Exception as exc:  # noqa: BLE001
        return BuiltWrappers(reason=f"the compiled wrapper could not be imported: {exc}")

    return BuiltWrappers(module=extension, entries=built.entries, path=library)
