"""Run a built artifact: load, bind, execute -- and never compile.

This is the whole runtime path of an AOT build. It reads the manifest,
opens the native library, rebuilds the guarded bindings from the recorded
ABI, loads the generated Python the build wrote, and runs the entry module.
No parsing, no analysis, no LLVM: those all happened at build time.
"""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path

from .abi import NativeSignature
from .binding import as_method, bind, bind_globals, keyed, remember, value_class_types
from .dispatch import LibraryBinder
from .execute import execute, format_traceback
from .generated import GeneratedModule
from .manifest import LAUNCH_CACHE, Manifest, ManifestError, host_runs, launch_cache, load

__all__ = ["LIGHT", "PrebuiltBinder", "generated_modules", "main", "write_light"]


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
        self._entries: dict[str, dict[str, NativeSignature]] = {}
        self._wrappers = _wrapper_module(manifest)
        self._wrapper_entries = manifest.wrapper_entries or {}
        self._region_libraries = manifest.regions or {}
        for module, entries in (manifest.staged or {}).items():
            for function, file in entries.items():
                self.add_exported(module, function, file.read_bytes())
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

    def _fast_entry(self, signature, address: int, fallback, spelled=None):  # type: ignore[no-untyped-def]
        """Bind the shipped C wrapper, which holds the fallback itself: `spelled`
        where the function is passed module globals (`bind_globals`)."""
        index = self._wrapper_entries.get(signature.qualname)
        if self._wrappers is None or index is None:
            return None
        types = value_class_types(signature, fallback, globals_read=spelled is not None)
        if types is None:
            return None
        if signature.crosses_collections:
            from .collection_boundary import attach  # pylint: disable=import-outside-toplevel

            if not attach(self._wrappers, self._library):
                return None
        if signature.draws:
            from .binding import attach_random  # pylint: disable=import-outside-toplevel

            if not attach_random(self._wrappers, self._library):
                return None
        if signature.effects:
            from .effects import attach as attach_effects  # pylint: disable=import-outside-toplevel
            from .effects import register_function  # pylint: disable=import-outside-toplevel

            if not attach_effects(self._wrappers, self._library):
                return None
            register_function(signature.qualname, fallback)
        resolve = None
        if signature.classes:
            from .collection_boundary import resolver  # pylint: disable=import-outside-toplevel

            resolve = resolver(signature, fallback)
        held = spelled if spelled is not None else fallback
        given = (address, types, held) if resolve is None else (address, types, held, resolve)
        try:
            named = getattr(self._wrappers, f"bind_{index}")(*given)
        except Exception:  # noqa: BLE001 - a refusal keeps the slower path
            return None
        if named is None:
            # The entry point bears the function's qualified name in the library
            # `ppy build` writes beside the manifest.
            named = getattr(self._wrappers, signature.qualname, None)
        keyed_set = getattr(self._wrappers, f"keyed_{index}", None)
        if named is not None and spelled is None and keyed_set is not None:
            # Keywords and defaults bound as Python binds them, then the entry.
            keyed_set(keyed(fallback, named, len(signature.parameters)))
        return named

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
        # A coroutine's future needs the Python-side wrapping; the C wrapper
        # would hand back the bare handle. A function that draws has
        # `random`'s state saved around it, which the C wrapper does too; one
        # that reads globals and draws, only the Python side.
        if signature.reads_globals and not (signature.future or signature.draws):
            # Python reads the globals and passes them after its arguments.
            read = bind_globals(
                signature,
                address,
                fallback,
                self._library,
                lambda spelled: self._fast_entry(signature, address, fallback, spelled),
            )
            if read is not None:
                return read.wrapper
        entry = (
            None
            if signature.future or signature.reads_globals
            else self._fast_entry(signature, address, fallback)
        )
        if entry is not None:
            remember(entry, signature, fallback)
            return as_method(entry, fallback, function)
        binding = bind(signature, address, fallback, owner=self._library)
        return binding.wrapper


def generated_modules(manifest: Manifest) -> dict[str, GeneratedModule]:
    """The generated Python modules a manifest names, read off disk, or off the
    launcher's cache with their compiled code."""
    cached = launch_cache(manifest.path) or {}
    payloads = cached.get("generated", {})
    compiled = cached.get("compiled", {})
    modules: dict[str, GeneratedModule] = {}
    for name, file in manifest.generated.items():
        payload = payloads.get(name)
        if payload is None:
            import json  # pylint: disable=import-outside-toplevel

            payload = json.loads(file.read_text(encoding="utf-8"))
        modules[name] = GeneratedModule(
            name=payload["name"],
            source_path=Path(payload["source"]),
            code=payload["code"],
            artifact=Path(payload["artifact"]),
            key=payload["key"],
            line_map={int(k): v for k, v in payload["line_map"].items()},
            fused_symbols=tuple(payload.get("fused_symbols", ())),
            compiled=dict(compiled.get(name, {})),
        )
    return modules


def _write_launch_cache(manifest: Manifest, modules: dict[str, GeneratedModule]) -> None:
    """After a run, what it parsed and compiled, marshaled beside the manifest for
    the next: no `json`, no `ast`, no `compile`. Best effort, and written only
    when something new was compiled."""
    cached = launch_cache(manifest.path) or {}
    old = cached.get("compiled", {})
    compiled = {name: dict(module.compiled) for name, module in modules.items()}
    if compiled == old and cached:
        return
    import json  # pylint: disable=import-outside-toplevel
    import marshal  # pylint: disable=import-outside-toplevel
    import os  # pylint: disable=import-outside-toplevel

    try:
        stat = os.stat(manifest.path)
        payload = json.loads(manifest.path.read_text(encoding="utf-8"))
        generated = {
            name: json.loads(file.read_text(encoding="utf-8"))
            for name, file in manifest.generated.items()
        }
        blob = marshal.dumps(
            {
                "stamp": (stat.st_size, stat.st_mtime_ns, sys.version),
                "payload": payload,
                "generated": generated,
                "compiled": compiled,
            }
        )
        target = manifest.path.with_name(LAUNCH_CACHE)
        draft = target.with_name(f"{target.name}.{os.getpid()}.part")
        draft.write_bytes(blob)
        draft.replace(target)
    except (OSError, ValueError):
        pass


#: Beside a manifest: what a program whose native code Python never calls needs
#: to run, precompiled (`write_light`, read by the compiler's `fastrun`).
LIGHT = "light.marshal"


def write_light(manifest_path: Path) -> Path | None:
    """Write the light plan of a manifest whose program runs as Python alone.

    A manifest with no native entry Python calls, and no wrappers, regions,
    staged exports, or fused kernels, needs none of the native library: its
    generated modules run as they are. Their compiled code, the entry, the
    search paths, and whether the program names `ppy` are marshaled beside
    the manifest, so a warm run imports neither this module nor `json`,
    `ctypes`, or `dataclasses`, and compiles nothing. Marshal is tied to this
    interpreter, which the plan records and the reader checks. Returns the
    plan, or None when the program needs the launcher.
    """
    import marshal  # pylint: disable=import-outside-toplevel

    try:
        manifest = load(Path(manifest_path))
    except ManifestError:
        return None
    import json  # pylint: disable=import-outside-toplevel

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (
        manifest.entries
        or manifest.wrapper_library is not None
        or manifest.regions
        or manifest.staged
        or payload.get("fused_kernels")
        or payload.get("exports")
    ):
        return None
    modules = generated_modules(manifest)
    if manifest.entry_module not in modules or any(m.fused_symbols for m in modules.values()):
        return None
    compiled = {
        name: (str(module.source_path), module.compile()) for name, module in modules.items()
    }
    plan = {
        "python": sys.version,
        "entry": manifest.entry_module,
        "search_paths": [str(p) for p in manifest.search_paths],
        "uses_ppy": manifest.uses_ppy,
        "modules": compiled,
    }
    target = Path(manifest_path).with_name(LIGHT)
    draft = target.with_name(f"{target.name}.part")
    draft.write_bytes(marshal.dumps(plan))
    draft.replace(target)
    return target


def main(manifest_path: Path, argv: list[str]) -> int:
    try:
        manifest = load(Path(manifest_path))
    except ManifestError as error:
        print(f"error[E1801]: {error}", file=sys.stderr)
        return 2

    if not host_runs(manifest.target):
        print(
            f"error[E1801]: the artifact was built for {manifest.target} and this machine "
            f"is another -- build it here, or for here, with `ppy build`",
            file=sys.stderr,
        )
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
        uses_ppy=manifest.uses_ppy,
    )
    _write_launch_cache(manifest, modules)
    if result.exception is not None:
        sys.stderr.write(format_traceback(result.exception))
    return result.exit_code
