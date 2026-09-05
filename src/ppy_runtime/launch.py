"""Run a built artifact: load, bind, execute -- and never compile.

This is the whole runtime path of an AOT build. It reads the manifest,
opens the native library, rebuilds the guarded bindings from the recorded
ABI, loads the generated Python the build wrote, and runs the entry module.
No parsing, no analysis, no LLVM: those all happened at build time.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import sys
from pathlib import Path

from .binding import bind, value_class_types
from .dispatch import LibraryBinder
from .execute import execute, format_traceback
from .generated import GeneratedModule
from .manifest import Manifest, ManifestError, load

__all__ = ["PrebuiltBinder", "generated_modules", "main"]


def _wrapper_module(manifest: Manifest):  # type: ignore[no-untyped-def]
    """Import the wrapper extension the build shipped, or None without one."""
    if manifest.wrapper_library is None:
        return None
    # The module name is the filename before the interpreter's ABI tag.
    name = manifest.wrapper_library.name.partition(".")[0]
    spec = importlib.util.spec_from_file_location(name, manifest.wrapper_library)
    if spec is None or spec.loader is None:
        return None
    extension = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(extension)
    except Exception:  # noqa: BLE001 - a stale wrapper means the slower path
        return None
    return extension


class PrebuiltBinder(LibraryBinder):
    """Serves the entries a manifest recorded, out of one shared library."""

    def __init__(self, manifest: Manifest, library) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self._library = library
        self._entries: dict[str, dict[str, object]] = {}
        self._wrappers = _wrapper_module(manifest)
        self._wrapper_entries = manifest.wrapper_entries or {}
        self._region_libraries = manifest.regions or {}
        self._extensions: dict[Path, object | None] = {}
        for entry in manifest.entries:
            self._entries.setdefault(entry.module, {})[entry.binding] = entry.signature

    def names(self, module: str) -> frozenset[str]:
        return frozenset(self._entries.get(module, {}))

    def region_names(self, module: str) -> frozenset[str]:
        shipped = self._region_libraries.get(module)
        return frozenset(shipped.entries) if shipped is not None else frozenset()

    def region(self, module: str, function: str, fallback):  # type: ignore[no-untyped-def]
        """Serve a compiled ATen region out of the extension the build shipped."""
        from .regions import bind_region

        shipped = self._region_libraries.get(module)
        symbol = shipped.entries.get(function) if shipped is not None else None
        compiled = None
        if shipped is not None and symbol is not None:
            compiled = getattr(self._extension(shipped.library), symbol, None)
        binding = bind_region(function, compiled, fallback)
        self.region_bindings.append(binding)
        return binding.wrapper

    def _extension(self, library: Path):  # type: ignore[no-untyped-def]
        """Load a region extension once; None when it will not load here."""
        if library in self._extensions:
            return self._extensions[library]
        extension = None
        try:
            # The extension links against libtorch, which importing torch
            # loads; without torch there is nothing for the region to call.
            import torch  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import

            spec = importlib.util.spec_from_file_location(library.stem, library)
            if spec is not None and spec.loader is not None:
                extension = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(extension)
        except Exception:  # noqa: BLE001 - a region that will not load is the Python body
            extension = None
        self._extensions[library] = extension
        return extension

    def _fast_entry(self, signature, address: int, fallback):  # type: ignore[no-untyped-def]
        """Bind the shipped C wrapper, which holds the fallback itself."""
        index = self._wrapper_entries.get(signature.qualname)
        if self._wrappers is None or index is None:
            return None
        types = value_class_types(signature, fallback)
        if types is None:
            return None
        try:
            getattr(self._wrappers, f"bind_{index}")(address, types, fallback)
        except Exception:  # noqa: BLE001 - a refusal keeps the slower path
            return None
        return getattr(self._wrappers, f"call_{index}", None)

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
        entry = self._fast_entry(signature, address, fallback)
        if entry is not None:
            return entry
        binding = bind(signature, address, fallback, owner=self._library)
        return binding.wrapper


def generated_modules(manifest: Manifest) -> dict[str, GeneratedModule]:
    """The generated Python modules a manifest names, read off disk."""
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

    modules = generated_modules(manifest)
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
