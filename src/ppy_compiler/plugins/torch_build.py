"""Compiling generated ATen regions against the installed PyTorch (spec 20.3, 20.7).

The generated C++ is built against the exact installed build, so the artifact
is keyed by the PyTorch version, its C++ ABI, and the accelerator runtime, and
is never reused across a mismatch.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .torch_region import TorchRegion, emit_source

__all__ = ["CompiledRegions", "compile_regions", "toolchain_ready"]


@dataclass(slots=True)
class CompiledRegions:
    module: object | None = None
    entry_points: dict[str, object] = field(default_factory=dict)
    source: str = ""
    directory: Path | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.module is not None and bool(self.entry_points)


def _ensure_scripts_on_path() -> None:
    """Torch shells out to `ninja` by name, so the interpreter's own script
    directory has to be reachable even when the environment is not activated."""
    import sysconfig

    scripts = sysconfig.get_path("scripts")
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if scripts and scripts not in entries:
        os.environ["PATH"] = os.pathsep.join([scripts, *entries])


def toolchain_ready() -> tuple[bool, str]:
    """Is everything present to compile a region against libtorch?"""
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"torch is not importable: {exc}"
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        return False, "no C++ compiler (c++ or g++) is on PATH"
    _ensure_scripts_on_path()
    ninja = shutil.which("ninja")
    if ninja is None:
        return False, "ninja is not on PATH; install the `torch` dependency group"
    return True, f"{compiler}, {ninja}"


def _fingerprint(regions: list[TorchRegion], source: str) -> str:
    from ..cache import digest

    import torch

    return digest(
        source,
        torch.__version__,
        torch.compiled_with_cxx11_abi(),
        getattr(torch.version, "cuda", None) or "none",
        tuple(region.symbol for region in regions),
    )[:16]


def compile_regions(
    regions: list[TorchRegion],
    cache_directory: Path,
    *,
    verbose: bool = False,
    notify=None,
) -> CompiledRegions:
    """Build one extension module holding every region of a PPY module."""
    usable = [region for region in regions if region.body]
    if not usable:
        return CompiledRegions(reason="no function translated to an ATen region")

    ready, detail = toolchain_ready()
    if not ready:
        return CompiledRegions(reason=detail)

    source = emit_source(usable)
    name = f"ppy_torch_{_fingerprint(usable, source)}"
    directory = cache_directory / "torch" / name
    warm = directory.is_dir() and any(directory.glob("*.so"))
    directory.mkdir(parents=True, exist_ok=True)
    if not warm and notify is not None:
        # Building against libtorch takes tens of seconds the first time; every
        # later build with the same fingerprint reuses this directory.
        notify(f"compiling {len(usable)} ATen region(s) against the installed PyTorch")

    from torch.utils.cpp_extension import load_inline

    try:
        extension = load_inline(
            name=name,
            cpp_sources=[source],
            functions=[region.symbol for region in usable],
            build_directory=str(directory),
            extra_cflags=["-O3"],
            verbose=verbose,
        )
    except Exception as exc:  # noqa: BLE001 - a build failure keeps the Python path
        detail = str(exc).strip().splitlines()
        return CompiledRegions(
            source=source,
            directory=directory,
            reason=f"the region did not compile: {detail[-1] if detail else exc}",
        )

    if notify is not None and not warm:
        notify(f"built {directory}")
    entry_points = {
        region.info.name: getattr(extension, region.symbol)
        for region in usable
        if hasattr(extension, region.symbol)
    }
    return CompiledRegions(
        module=extension, entry_points=entry_points, source=source, directory=directory
    )
