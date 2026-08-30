"""Loading the binding manifest: the artifact's contract with this runtime.

The manifest is data a build wrote once; loading it must stay cheap and
must never reconstruct compiler state. Validation is schema, ABI version,
interpreter compatibility, and file presence -- nothing that reads source.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .abi import NativeParam, NativeSignature

__all__ = ["Manifest", "ManifestError", "NativeEntry", "load"]

SUPPORTED_ABI = 1


class ManifestError(RuntimeError):
    """The artifact cannot be run, with the reason and the remedy."""


@dataclass(frozen=True, slots=True)
class NativeEntry:
    module: str
    binding: str
    signature: NativeSignature


@dataclass(slots=True)
class Manifest:
    path: Path
    library: Path | None
    entries: list[NativeEntry]
    entry_module: str
    search_paths: list[Path]
    generated: dict[str, Path]
    safeguards: str


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
    return Manifest(
        path=path,
        library=library,
        entries=entries,
        entry_module=program.get("entry", ""),
        search_paths=[Path(p) for p in program.get("search_paths", ())],
        generated=generated,
        safeguards=program.get("safeguards", "hoisted"),
    )
