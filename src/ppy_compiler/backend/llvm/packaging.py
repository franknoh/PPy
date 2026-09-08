"""What a build becomes beyond a manifest: an extension, or a library package.

`--python-extension` takes the entry module -- its native objects, its
exposed signatures, its optimized Python -- and makes one importable
CPython module of them. `--library` takes the exports and lays out what a
C consumer expects: `lib/` with the shared library, `include/` with the
header, `lib/pkgconfig/` with the `.pc` that names both, and the manifest
that describes the ABI.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ...driver.pipeline import COMPILER_VERSION
from ...target import TargetInfo
from .extension import bound_python, build_python_extension
from .link import BuildArtifacts, write_pkg_config

__all__ = ["PackagingError", "build_extension", "package_library"]


class PackagingError(RuntimeError):
    """A build extra that cannot be made, with the diagnostic it is reported as."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _entry_module(bundle, entry: Path | None) -> str:  # type: ignore[no-untyped-def]
    if entry is not None:
        for name, symbols in bundle.symbols.modules.items():
            if symbols.path == entry.resolve():
                return name
    raise PackagingError("E1002", "the build takes the module's file as its target")


def build_extension(
    bundle, artifacts: BuildArtifacts, entry: Path | None, output: Path | None
) -> Path:  # type: ignore[no-untyped-def]
    """The importable module for the entry file; recorded on `artifacts`."""
    module = _entry_module(bundle, entry)
    generated = artifacts.generated.get(module)
    if generated is None:
        raise PackagingError("E1002", f"`{module}` produced no Python to carry")
    prefix = f"{module}."
    signatures = {
        qualname: signature
        for qualname, signature in artifacts.signatures.items()
        if qualname.startswith(prefix)
    }
    keys = {qualname: qualname[len(prefix) :] for qualname in signatures}
    name = module.rpartition(".")[2]
    directory = output or (artifacts.manifest.parent if artifacts.manifest else Path.cwd())
    artifacts.extension = build_python_extension(
        name,
        signatures,
        keys,
        bound_python(generated.code, generated.line_map, frozenset(keys.values())),  # type: ignore[attr-defined]
        Path(generated.source_path),  # type: ignore[attr-defined]
        artifacts.objects,
        directory,
        libraries=artifacts.libraries,
    )
    return artifacts.extension


def package_library(
    bundle, artifacts: BuildArtifacts, output: Path | None, target: TargetInfo
) -> Path:  # type: ignore[no-untyped-def]
    """`lib/`, `include/`, `lib/pkgconfig/`, and the manifest, in one directory."""
    if not artifacts.exports:
        raise PackagingError(
            "E1805",
            "a library build has nothing to export: mark the functions to publish with "
            "`@ppy.native.export`",
        )
    if artifacts.library is None or artifacts.header is None:
        detail = "; ".join(artifacts.notes) or "the library was not linked"
        raise PackagingError("E1801", f"a library build needs the linked library: {detail}")
    name = bundle.project.root.name
    root = output or artifacts.library.parent / "package"
    lib_dir = root / "lib"
    include_dir = root / "include"
    lib_dir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)
    library = lib_dir / artifacts.library.name
    header = include_dir / artifacts.header.name
    for source, destination in ((artifacts.library, library), (artifacts.header, header)):
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
            if output is not None and source.parent.resolve() == root.resolve():
                source.unlink()
    write_pkg_config(lib_dir / "pkgconfig" / f"{name}.pc", name, COMPILER_VERSION)
    if artifacts.manifest is not None and artifacts.manifest.parent.resolve() != root.resolve():
        shutil.copy2(artifacts.manifest, root / artifacts.manifest.name)
    artifacts.library = library
    artifacts.header = header
    artifacts.package = root
    del target
    return root
