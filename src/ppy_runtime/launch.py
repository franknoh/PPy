"""Run a built artifact: load, bind, execute -- and never compile.

This is the whole runtime path of an AOT build. It reads the manifest,
opens the native library, rebuilds the guarded bindings from the recorded
ABI, loads the generated Python the build wrote, and runs the entry module.
No parsing, no analysis, no LLVM: those all happened at build time.
"""

from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

from .binding import bind
from .dispatch import LibraryBinder
from .execute import execute, format_traceback
from .generated import GeneratedModule
from .manifest import Manifest, ManifestError, load

__all__ = ["main"]


class PrebuiltBinder(LibraryBinder):
    """Serves the entries a manifest recorded, out of one shared library."""

    def __init__(self, manifest: Manifest, library) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self._library = library
        self._entries: dict[str, dict[str, object]] = {}
        for entry in manifest.entries:
            self._entries.setdefault(entry.module, {})[entry.binding] = entry.signature

    def names(self, module: str) -> frozenset[str]:
        return frozenset(self._entries.get(module, {}))

    def bind(self, module: str, function: str, fallback):  # type: ignore[no-untyped-def]
        signature = self._entries.get(module, {}).get(function)
        if signature is None or not callable(fallback):
            return fallback
        try:
            symbol = getattr(self._library, signature.symbol)  # type: ignore[union-attr]
        except AttributeError:
            return fallback
        address = ctypes.cast(symbol, ctypes.c_void_p).value or 0
        if not address:
            return fallback
        binding = bind(signature, address, fallback, owner=self._library)
        return binding.wrapper


def _generated_modules(manifest: Manifest) -> dict[str, GeneratedModule]:
    modules: dict[str, GeneratedModule] = {}
    for name, file in manifest.generated.items():
        payload = json.loads(file.read_text(encoding="utf-8"))
        modules[name] = GeneratedModule(
            name=payload["name"],
            source_path=Path(payload["source"]),
            code=payload["code"],
            artifact=Path(payload["artifact"]),
            key=payload["key"],
            line_map={int(k): v for k, v in payload["line_map"].items()},
            fused_symbols=tuple(payload.get("fused_symbols", ())),
        )
    return modules


def main(manifest_path: Path, argv: list[str]) -> int:
    try:
        manifest = load(Path(manifest_path))
    except ManifestError as error:
        print(f"error[E1801]: {error}", file=sys.stderr)
        return 2

    library = None
    if manifest.library is not None:
        try:
            library = ctypes.CDLL(str(manifest.library))
        except OSError as error:
            print(f"error[E1801]: cannot load {manifest.library}: {error}", file=sys.stderr)
            return 2

    modules = _generated_modules(manifest)
    entry = modules.get(manifest.entry_module)
    if entry is None:
        print(
            f"error[E1801]: the manifest names no runnable entry module "
            f"({manifest.entry_module!r}) -- rebuild the artifact with `ppy build`",
            file=sys.stderr,
        )
        return 2

    binder = PrebuiltBinder(manifest, library)
    result = execute(
        entry,
        modules,
        argv,
        search_paths=manifest.search_paths,
        natives=binder,
        entry_name=manifest.entry_module,
    )
    if result.exception is not None:
        sys.stderr.write(format_traceback(result.exception))
    return result.exit_code
