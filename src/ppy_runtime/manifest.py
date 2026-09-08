"""Loading the binding manifest: the artifact's contract with this runtime.

The manifest is data a build wrote once; loading it must stay cheap and
must never reconstruct compiler state. Validation is schema, ABI version,
interpreter compatibility, and file presence -- nothing that reads source.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from .abi import NativeParam, NativeSignature

__all__ = ["Manifest", "ManifestError", "NativeEntry", "RegionLibrary", "load"]

#: 2: the 0.2.0 artifact -- it carries its torch regions, and a 0.1.x
#: manifest is refused with the rebuild message rather than half-served.
SUPPORTED_ABI = 2


class ManifestError(RuntimeError):
    """The artifact cannot be run, with the reason and the remedy."""


@dataclass(frozen=True, slots=True)
class NativeEntry:
    module: str
    binding: str
    signature: NativeSignature


@dataclass(frozen=True, slots=True)
class RegionLibrary:
    """One generated module's compiled ATen regions: the extension holding
    them, and the C++ symbol that serves each function."""

    library: Path
    entries: dict[str, str]


@dataclass(slots=True)
class Manifest:
    path: Path
    library: Path | None
    entries: list[NativeEntry]
    entry_module: str
    search_paths: list[Path]
    generated: dict[str, Path]
    safeguards: str
    #: The prebuilt CPython-ABI wrapper extension, when the build shipped one:
    #: its path next to the manifest, and the wrapper index per qualname.
    #: The triple the objects were compiled for ("" in an older artifact).
    target: str = ""
    wrapper_library: Path | None = None
    wrapper_entries: dict[str, int] | None = None
    #: Compiled torch regions per generated module, when the build shipped
    #: any and their libraries are still beside the manifest.
    regions: dict[str, RegionLibrary] | None = None
    #: Staged exports per generated module -- a kernel's PTX, an XLA module --
    #: as files beside the manifest, when the build shipped any.
    staged: dict[str, dict[str, Path]] | None = None


def _signature(payload: dict) -> NativeSignature:
    abi = payload["abi"]
    parameters = tuple(
        NativeParam(
            name=parameter["name"],
            kind=parameter["kind"],
            element=parameter.get("element", ""),
            elements=tuple(parameter.get("elements", ())),
            fields=tuple((f, s) for f, s in parameter.get("fields", ())),
            class_name=parameter.get("class_name", ""),
        )
        for parameter in abi["parameters"]
    )
    return NativeSignature(
        qualname=payload["python_qualname"],
        symbol=payload["native_symbol"],
        parameters=parameters,
        returns=tuple(abi["returns"]),
        releases_gil=bool(abi.get("releases_gil", False)),
        cpu_features=tuple(str(f) for f in abi.get("cpu_features", ())),
        future=str(abi.get("future", "")),
    )


def load(path: Path) -> Manifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"cannot read the binding manifest {path}: {exc}") from exc
    except ValueError as exc:
        raise ManifestError(f"{path} is not a binding manifest: {exc}") from exc

    if payload.get("abi_version") != SUPPORTED_ABI:
        raise ManifestError(
            f"{path} speaks ABI {payload.get('abi_version')!r}; this runtime "
            f"speaks {SUPPORTED_ABI} -- rebuild the artifact with `ppy build`"
        )
    program = payload.get("program")
    if not isinstance(program, dict):
        raise ManifestError(
            f"{path} has no program section; it predates the AOT runtime -- "
            "rebuild the artifact with `ppy build`"
        )
    built_for = payload.get("python", "")
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    if built_for and built_for != running:
        raise ManifestError(
            f"the artifact was built for Python {built_for} and this is "
            f"{running} -- rebuild the artifact with `ppy build`"
        )

    library = None
    spelled = payload.get("native_library")
    if spelled:
        library = path.parent / Path(spelled).name
        if not library.is_file():
            raise ManifestError(
                f"the native library named by {path} is missing -- "
                "rebuild the artifacts with `ppy build`"
            )

    entries = [
        NativeEntry(
            module=entry["module"],
            binding=entry["binding"],
            signature=_signature(entry),
        )
        for entry in payload.get("entries", ())
        if "abi" in entry and "module" in entry
    ]
    generated: dict[str, Path] = {}
    for module, filename in program.get("generated", {}).items():
        target = path.parent / Path(filename).name
        candidate = path.parent / "generated" / Path(filename).name
        generated[module] = candidate if candidate.is_file() else target
    wrapper_library = None
    wrapper_entries = None
    wrappers = payload.get("wrappers")
    if isinstance(wrappers, dict) and wrappers.get("library"):
        candidate = path.parent / Path(wrappers["library"]).name
        # A missing wrapper is a slower boundary, not a broken artifact.
        if candidate.is_file():
            wrapper_library = candidate
            wrapper_entries = {
                str(name): int(index) for name, index in wrappers.get("entries", {}).items()
            }
    regions: dict[str, RegionLibrary] = {}
    for module, section in (payload.get("regions") or {}).items():
        if not isinstance(section, dict) or not section.get("library"):
            continue
        candidate = path.parent / Path(str(section["library"])).name
        # A missing region library is the Python body, not a broken artifact:
        # the region only ever removed Python round trips.
        if candidate.is_file():
            regions[str(module)] = RegionLibrary(
                candidate,
                {str(name): str(symbol) for name, symbol in section.get("entries", {}).items()},
            )
    staged: dict[str, dict[str, Path]] = {}
    for module, section in (payload.get("staged") or {}).items():
        if not isinstance(section, dict):
            continue
        for function, filename in section.items():
            candidate = path.parent / Path(str(filename)).name
            # A missing payload is the Python definition, not a broken artifact.
            if candidate.is_file():
                staged.setdefault(str(module), {})[str(function)] = candidate
    return Manifest(
        path=path,
        library=library,
        target=str(payload.get("target") or ""),
        entries=entries,
        entry_module=program.get("entry", ""),
        search_paths=[Path(p) for p in program.get("search_paths", ())],
        generated=generated,
        safeguards=program.get("safeguards", "hoisted"),
        wrapper_library=wrapper_library,
        wrapper_entries=wrapper_entries,
        regions=regions or None,
        staged=staged or None,
    )


def host_runs(target: str) -> bool:
    """Whether this machine is the one `target` names, by architecture and OS.

    The runtime has no LLVM to ask, so it compares the words a triple is
    made of with what the platform says of itself.
    """
    if not target:
        return True
    parts = target.lower().split("-")
    architecture = {"amd64": "x86_64", "arm64": "aarch64"}.get(parts[0], parts[0])
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    system = {"linux": "linux", "darwin": "darwin", "win32": "windows"}.get(
        sys.platform, sys.platform
    )
    names_os = any(
        part.startswith(system) or (system == "darwin" and part.startswith("macos"))
        for part in parts[1:]
    )
    return architecture == machine and names_os
