"""The PJRT bridge: StableHLO text compiled and run on an XLA device.

Everything XLA sits behind this module (spec 65). Nothing here is imported
until a program reaches for a device: `available()` looks without loading,
`Client` loads the bindings on first use. The bindings are XLA's own
(`jaxlib.xla_client`) for compiling and running; device buffers are made
through JAX's `device_put` while that is the only public way to place a
NumPy array on the client XLA compiles for, so the bridge needs JAX
installed even though the compiler that produced the StableHLO did not.

A compiled executable is cached by the StableHLO module's digest together
with the bindings' version, the platform, and the device (spec 67), in
memory and -- serialized by PJRT -- on disk under the user's cache
directory, so a second program with the same module does not compile it
again and a binary for one platform is never handed to another.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["Client", "available", "cache_directory", "default_platform"]

_client: Client | None = None


def available() -> bool:
    """Are the XLA bindings and JAX's buffer placement importable? (No import happens.)"""
    return (
        importlib.util.find_spec("jaxlib") is not None
        and importlib.util.find_spec("jax") is not None
    )


def default_platform() -> str:
    return os.environ.get("PPY_XLA_PLATFORM", "cpu")


def cache_directory() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "ppy" / "xla"


class Client:
    """One XLA client: the devices of one platform, and what was compiled for them."""

    def __init__(self, platform: str | None = None) -> None:
        import jax
        import jax.extend as jex
        from jaxlib import xla_client

        # A `float64` argument stays `float64` on the device; XLA computes what
        # the program wrote, not a narrower approximation of it.
        jax.config.update("jax_enable_x64", True)
        self.platform = platform or default_platform()
        self._jax = jax
        self._xla = xla_client
        self._backend = jex.backend.get_backend(self.platform)
        self._executables: dict[str, Any] = {}

    @property
    def version(self) -> str:
        return f"{self._xla._version}:{self._backend.platform_version}"

    def devices(self) -> list[str]:
        return [str(device) for device in self._backend.devices()]

    def key(self, stablehlo: str) -> str:
        """What identifies a compiled module: its text, the bindings, the platform, the device."""
        digest = hashlib.sha256()
        for part in (stablehlo, self.version, self.platform, ",".join(self.devices())):
            digest.update(part.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def compile(self, stablehlo: str) -> Any:
        """The loaded executable for `stablehlo`, from memory, disk, or the compiler."""
        key = self.key(stablehlo)
        found = self._executables.get(key)
        if found is not None:
            return found
        devices = self._xla.DeviceList(tuple(self._backend.devices()))
        cached = cache_directory() / f"{key}.pjrt"
        executable = None
        if cached.exists():
            try:
                executable = self._backend.deserialize_executable(cached.read_bytes(), devices)
            except Exception:  # noqa: BLE001 - a stale binary is recompiled, not served
                executable = None
        if executable is None:
            executable = self._backend.compile_and_load(
                stablehlo, devices, self._xla.CompileOptions()
            )
            try:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(executable.serialize())
            except OSError:
                pass
        self._executables[key] = executable
        return executable

    def execute(self, executable: Any, arguments: Sequence[Any]) -> list[Any]:
        """Run `executable` over host arrays; the results come back as NumPy arrays."""
        import numpy

        buffers = [self._jax.device_put(numpy.asarray(a)) for a in arguments]
        return [numpy.asarray(result) for result in executable.execute(buffers)]


def client(platform: str | None = None) -> Client:
    """The process's client for `platform`, made on first use."""
    global _client  # noqa: PLW0603
    if _client is None or (platform is not None and _client.platform != platform):
        _client = Client(platform)
    return _client
