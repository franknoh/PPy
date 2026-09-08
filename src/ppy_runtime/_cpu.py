"""What this machine's CPU can do, asked once and answered the same way
by the compiler folding `cpu.features()` and by the program running it.

The names are LLVM's lower-case feature names (`avx2`, `fma`, `neon`,
`sse4.2`), so `@cpu.target("avx2")` and `"avx2" in cpu.features()` speak
one vocabulary. The answer comes from the operating system -- `/proc/
cpuinfo`, `sysctl` -- so no compiler is needed to give it.
"""

from __future__ import annotations

import sys
from functools import cache

__all__ = ["features", "vector_bits", "vector_width"]

#: Linux spells a few features differently from LLVM.
_LINUX_NAMES = {
    "sse4_1": "sse4.1",
    "sse4_2": "sse4.2",
    "pni": "sse3",
    "avx512f": "avx512f",
    "asimd": "neon",
    "fp": "fp-armv8",
    "atomics": "lse",
}
_DARWIN_OPTIONAL = {
    "hw.optional.avx2_0": "avx2",
    "hw.optional.avx1_0": "avx",
    "hw.optional.fma": "fma",
    "hw.optional.sse4_2": "sse4.2",
    "hw.optional.sse4_1": "sse4.1",
    "hw.optional.neon": "neon",
    "hw.optional.arm.FEAT_LSE": "lse",
    "hw.optional.armv8_2_sha512": "sha3",
}


@cache
def features() -> tuple[str, ...]:
    """The CPU features of this machine, sorted, in LLVM's spelling."""
    import platform

    found: set[str] = set()
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    key, _, value = line.partition(":")
                    if key.strip() in {"flags", "Features"}:
                        for flag in value.split():
                            found.add(_LINUX_NAMES.get(flag.lower(), flag.lower()))
                        break
        except OSError:
            pass
    elif sys.platform == "darwin":
        import subprocess

        for key, name in _DARWIN_OPTIONAL.items():
            try:
                answer = subprocess.run(
                    ["sysctl", "-n", key], capture_output=True, text=True, check=False
                )
            except OSError:
                break
            if answer.returncode == 0 and answer.stdout.strip() == "1":
                found.add(name)
        if platform.machine().lower() in {"arm64", "aarch64"}:
            found.add("neon")
    if not found and platform.machine().lower() in {"arm64", "aarch64"}:
        found.add("neon")
    return tuple(sorted(found))


def vector_bits() -> int:
    """The widest vector register this machine computes with, in bits."""
    import platform

    have = set(features())
    if "avx512f" in have:
        return 512
    if "avx2" in have or "avx" in have:
        return 256
    if have & {"sse2", "neon", "sve"} or platform.machine().lower() in {"x86_64", "amd64"}:
        return 128
    return 64


def vector_width(element_bits: int) -> int:
    """How many elements of `element_bits` fit one vector register."""
    return max(vector_bits() // element_bits, 1)
