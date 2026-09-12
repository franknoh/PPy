"""The compiler's own backends, described through the backend interface.

Each says what it is called, what `ppy emit` may ask it for, whether its
toolchain is here, and what identifies its code generation for the cache.
Their emission and builds run on the driver's own roads -- several of
their formats are whole-program or flag-shaped (`--standalone`,
`--header-only`, `linked-ir`) and the LLVM road keeps a lowering cache of
its own -- so `emit` and `build` here are not called by the driver; the
registry, `ppy doctor`, format ownership, and the cache keys see them
through the same interface an external backend implements.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Mapping

from ..cache.keys import digest
from .base import Backend, EmitFormat, ToolchainStatus

__all__ = ["BUILTIN_BACKENDS", "builtin_backend", "distribution_version"]


def distribution_version(name: str) -> str:
    """An installed distribution's version, or `absent`, without importing it."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


class IrBackend(Backend):
    """The canonical IR itself: what every other backend receives."""

    name = "ir"

    def emit_formats(self) -> tuple[EmitFormat, ...]:
        return (
            EmitFormat("ir", ".ppyir", description="the canonical IR after the shared passes"),
            EmitFormat(
                "linked-ir", ".ppyir", description="the whole program as one optimized module"
            ),
        )

    def fingerprint(self) -> str:
        from ..ir import IR_SCHEMA_VERSION

        return digest("ir", IR_SCHEMA_VERSION)


class LlvmBackend(Backend):
    """Canonical IR to LLVM IR, objects, libraries, and the JIT."""

    name = "llvm"

    def emit_formats(self) -> tuple[EmitFormat, ...]:
        return (EmitFormat("llvm-ir", ".ll", description="what the LLVM backend makes of it"),)

    def fingerprint(self) -> str:
        # The LLVM that optimizes and emits the object code: another version
        # of it is other code, and a cached object must not outlive it.
        return digest("llvm", distribution_version("llvmlite"))

    def toolchain_status(self) -> ToolchainStatus:
        from .llvm import llvm_status, toolchain_status

        status, detail = llvm_status()
        if status != "available":
            return ToolchainStatus(False, detail or status)
        usable, toolchain = toolchain_status()
        version = distribution_version("llvmlite")
        if not usable:
            return ToolchainStatus(False, f"llvmlite {version}; no native toolchain: {toolchain}")
        return ToolchainStatus(True, f"llvmlite {version}, {toolchain}")


class PythonBackend(Backend):
    """Optimized Python under CPython: the backend `ppy FILE.ppy` runs on."""

    name = "python"

    def fingerprint(self) -> str:
        from ..version import COMPILER_VERSION

        return digest("python", COMPILER_VERSION)


class CBackend(Backend):
    """Canonical IR as C11, C++17, CUDA, or HIP source, or a C header."""

    name = "c"

    def emit_formats(self) -> tuple[EmitFormat, ...]:
        return (
            EmitFormat("c", ".c", description="one C11 translation unit per module"),
            EmitFormat("cpp", ".cpp", description="one C++17 translation unit per module"),
            EmitFormat("header", ".h", description="the C declarations of the exports"),
            EmitFormat("cuda", ".cu", description="the kernels and their launches as CUDA C++"),
            EmitFormat("hip", ".hip", description="the kernels and their launches as HIP C++"),
        )

    def toolchain_status(self) -> ToolchainStatus:
        return ToolchainStatus(True, "source only; any C11 or C++17 compiler builds it")


class NvvmBackend(Backend):
    """Device code as NVVM IR and PTX through LLVM's NVPTX target."""

    name = "nvvm"

    def emit_formats(self) -> tuple[EmitFormat, ...]:
        return (
            EmitFormat("nvvm-ir", ".nvvm.ll", description="the kernels as LLVM IR for NVPTX"),
            EmitFormat("ptx", ".ptx", description="the kernels as PTX"),
        )

    def fingerprint(self) -> str:
        return digest("nvvm", distribution_version("llvmlite"))

    def toolchain_status(self) -> ToolchainStatus:
        from .nvvm import available

        if available():
            return ToolchainStatus(True, f"llvmlite {distribution_version('llvmlite')} with NVPTX")
        return ToolchainStatus(False, "this LLVM has no NVPTX backend")


class StableHloBackend(Backend):
    """`@ppy.xla.jit` functions as StableHLO for XLA."""

    name = "stablehlo"

    def emit_formats(self) -> tuple[EmitFormat, ...]:
        return (
            EmitFormat(
                "stablehlo", ".mlir", description="the @ppy.xla.jit functions as an MLIR module"
            ),
        )

    def toolchain_status(self) -> ToolchainStatus:
        version = distribution_version("jax")
        if version == "absent":
            return ToolchainStatus(True, "emits without JAX; running it needs jax")
        return ToolchainStatus(True, f"jax {version} runs it")


#: The builtin backends by name, each constructed like an external one.
BUILTIN_BACKENDS: dict[str, Callable[[Mapping[str, object]], Backend]] = {
    "ir": IrBackend,
    "llvm": LlvmBackend,
    "python": PythonBackend,
    "c": CBackend,
    "nvvm": NvvmBackend,
    "stablehlo": StableHloBackend,
}


def builtin_backend(name: str, options: Mapping[str, object] | None = None) -> Backend:
    return BUILTIN_BACKENDS[name](options or {})
