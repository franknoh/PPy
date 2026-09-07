"""`TargetInfo`: everything the compiler knows about the machine it targets.

One record holds the triple, the CPU and its features, the pointer width,
the endianness, the ABI, the operating system, the object format, and the
LLVM data layout -- so a backend, a linker, a cache key, or a doctor
report asks the target rather than `sys.platform`. The host is one target
among others; `--target TRIPLE` names a different one, and a build then
emits objects for it and links them if a toolchain for it is on the path.
"""

from __future__ import annotations

import contextlib
import platform
import sys
from dataclasses import dataclass
from functools import cache, lru_cache

__all__ = ["TargetError", "TargetInfo", "configured_target", "host_target", "parse_target"]


class TargetError(ValueError):
    """A triple this compiler cannot make a target of."""


#: Architectures by their triple spelling: (canonical name, pointer width,
#: endianness). What LLVM would say, kept here so no LLVM is needed to
#: describe a target.
_ARCHITECTURES: dict[str, tuple[str, int, str]] = {
    "x86_64": ("x86_64", 64, "little"),
    "amd64": ("x86_64", 64, "little"),
    "i686": ("i686", 32, "little"),
    "i386": ("i386", 32, "little"),
    "aarch64": ("aarch64", 64, "little"),
    "arm64": ("aarch64", 64, "little"),
    "aarch64_be": ("aarch64_be", 64, "big"),
    "armv7": ("armv7", 32, "little"),
    "riscv64": ("riscv64", 64, "little"),
    "riscv32": ("riscv32", 32, "little"),
    "powerpc64le": ("powerpc64le", 64, "little"),
    "ppc64le": ("powerpc64le", 64, "little"),
    "powerpc64": ("powerpc64", 64, "big"),
    "s390x": ("s390x", 64, "big"),
    "wasm32": ("wasm32", 32, "little"),
    "wasm64": ("wasm64", 64, "little"),
}

#: Operating systems by their triple spelling, with the object format and
#: the ABI a bare triple implies.
_SYSTEMS: dict[str, tuple[str, str, str]] = {
    "linux": ("linux", "elf", "gnu"),
    "darwin": ("darwin", "macho", "darwin"),
    "macos": ("darwin", "macho", "darwin"),
    "macosx": ("darwin", "macho", "darwin"),
    "ios": ("ios", "macho", "darwin"),
    "windows": ("windows", "coff", "msvc"),
    "win32": ("windows", "coff", "msvc"),
    "freebsd": ("freebsd", "elf", "gnu"),
    "openbsd": ("openbsd", "elf", "gnu"),
    "netbsd": ("netbsd", "elf", "gnu"),
    "wasi": ("wasi", "wasm", "wasi"),
    "none": ("none", "elf", "eabi"),
}

_ABIS = frozenset(
    {"gnu", "gnueabihf", "gnueabi", "musl", "musleabihf", "msvc", "eabi", "android", "elf"}
)


@dataclass(frozen=True, slots=True)
class TargetInfo:
    """One machine a build targets."""

    triple: str
    architecture: str
    cpu: str
    features: str
    pointer_width: int
    endianness: str
    abi: str
    os: str
    object_format: str

    @property
    def data_layout(self) -> str:
        """LLVM's data layout for the triple, or empty where LLVM is absent.

        Asked for only when object code is emitted, so describing a target
        -- the host's, for a cache key -- never loads LLVM.
        """
        return _data_layout(self.triple)

    @property
    def is_host(self) -> bool:
        return self.triple == host_target().triple

    @property
    def shared_library_suffix(self) -> str:
        if self.os == "windows":
            return ".dll"
        if self.object_format == "macho":
            return ".dylib"
        return ".so"

    @property
    def executable_suffix(self) -> str:
        return ".exe" if self.os == "windows" else ""

    @property
    def extension_suffix(self) -> str:
        """The suffix an importable CPython extension carries on this target."""
        return ".pyd" if self.os == "windows" else ".so"

    @property
    def long_width(self) -> int:
        """The width of C `long`: 32 on Windows and every 32-bit target."""
        return 32 if self.os == "windows" else self.pointer_width

    @property
    def link_flags(self) -> tuple[str, ...]:
        """What a clang driver needs to link for this target."""
        return () if self.is_host else (f"--target={self.triple}",)

    def describe(self) -> str:
        cpu = f", cpu {self.cpu}" if self.cpu else ""
        return (
            f"{self.triple} ({self.pointer_width}-bit {self.endianness}-endian, "
            f"{self.object_format}, {self.abi or 'no'} abi{cpu})"
        )


def parse_target(triple: str, *, cpu: str = "", features: str = "") -> TargetInfo:
    """A target from a triple, in the shapes people write them.

    `aarch64-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-apple-darwin`,
    and `x86_64-pc-windows-msvc` all parse; the canonical four-part triple
    is what LLVM and the cache key see.
    """
    parts = triple.strip().lower().split("-")
    if len(parts) < 2 or not all(parts):
        raise TargetError(f"`{triple}` is not a target triple (arch-vendor-os[-abi])")
    arch_spelling = parts[0]
    described = _ARCHITECTURES.get(arch_spelling)
    if described is None:
        for known, value in _ARCHITECTURES.items():
            if arch_spelling.startswith(known):
                described = value
                break
    if described is None:
        raise TargetError(f"`{triple}` names an architecture this compiler does not know")
    architecture, width, endianness = described
    rest = parts[1:]
    vendor = "unknown"
    if len(rest) >= 2 and not rest[0].startswith(tuple(_SYSTEMS)):
        vendor = rest[0]
        rest = rest[1:]
    os_spelling = rest[0]
    system = next((s for s in _SYSTEMS if os_spelling.startswith(s)), None)
    if system is None:
        raise TargetError(f"`{triple}` names an operating system this compiler does not know")
    os_name, object_format, default_abi = _SYSTEMS[system]
    abi = default_abi
    explicit = len(rest) > 1
    if explicit:
        abi = rest[1]
        if abi not in _ABIS:
            raise TargetError(f"`{triple}` names an ABI this compiler does not know: {abi}")
    if os_name == "darwin" and vendor == "unknown":
        vendor = "apple"
    elif os_name == "windows" and vendor == "unknown":
        vendor = "pc"
    spelled_abi = f"-{abi}" if explicit or os_name == "linux" else ""
    canonical = f"{architecture}-{vendor}-{os_spelling}{spelled_abi}"
    return TargetInfo(
        triple=canonical,
        architecture=architecture,
        cpu=cpu,
        features=features,
        pointer_width=width,
        endianness=endianness,
        abi=abi,
        os=os_name,
        object_format=object_format,
    )


@cache
def _data_layout(triple: str) -> str:
    try:
        from llvmlite import binding
    except ImportError:
        return ""
    try:
        for step in ("initialize_all_targets", "initialize_all_asmprinters"):
            initializer = getattr(binding, step, None)
            if initializer is not None:
                with contextlib.suppress(RuntimeError):
                    initializer()
        machine = binding.Target.from_triple(triple).create_target_machine(opt=0, reloc="pic")
        return str(machine.target_data)
    except RuntimeError:
        return ""


def configured_target(spelled: str | None) -> TargetInfo:
    """The target `[tool.ppy.llvm] target` or `--target` names: the host by default."""
    if not spelled or spelled in {"native", "host"}:
        return host_target()
    return parse_target(spelled)


@lru_cache(maxsize=1)
def host_target() -> TargetInfo:
    """The machine this process runs on, described without loading LLVM.

    The triple is what LLVM would say of the process (`x86_64-unknown-
    linux-gnu`), made from what the platform says of itself, so a warm run
    or a fully cached build can key its artifacts without the code
    generator. The host's own CPU and features stay with the JIT, which is
    the only code that targets them.
    """
    machine = platform.machine().lower() or "unknown"
    system = {"linux": "linux", "darwin": "darwin", "win32": "windows"}.get(
        sys.platform, sys.platform
    )
    abi = ""
    if system == "linux":
        libc, _version = platform.libc_ver()
        abi = "-musl" if libc == "musl" else "-gnu"
    elif system == "windows":
        abi = "-msvc"
    vendor = {"darwin": "apple", "windows": "pc"}.get(system, "unknown")
    return parse_target(f"{machine}-{vendor}-{system}{abi}")
