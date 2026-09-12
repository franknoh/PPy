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
    """The platform a client is made for: `PPY_XLA_PLATFORM`, else the one JAX picked.

    JAX's default backend is the accelerator its plugin found -- `gpu` on a
    machine with a CUDA or ROCm plugin installed, `tpu` on one with a TPU --
    and `cpu` only where there is nothing else; the bridge follows it, so a
    program on a GPU machine runs its StableHLO on the GPU rather than on a
    `cpu:0` it never asked for. `JAX_PLATFORMS` still steers JAX, and
    `PPY_XLA_PLATFORM` steers the bridge alone. What is not done is to read
    a failure as the CPU: a JAX that will not import or initialize raises,
    and a JAX that came up on the CPU while an accelerator plugin is
    installed (and `JAX_PLATFORMS` did not ask for the CPU) raises too,
    naming the plugin, since that is the plugin failing, not a machine
    without a GPU. Without JAX at all there is no platform, and `cpu` is the
    answer `available()` already qualifies.
    """
    spelled = os.environ.get("PPY_XLA_PLATFORM")
    if spelled:
        return spelled
    if not available():
        return "cpu"
    import jax

    backend = str(jax.default_backend())
    asked = os.environ.get("JAX_PLATFORMS", "")
    if backend == "cpu" and not asked:
        plugins = accelerator_plugins()
        if plugins:
            raise RuntimeError(
                "JAX initialized on the CPU although an accelerator plugin is installed "
                f"({', '.join(plugins)}): the plugin failed to initialize, which is not a "
                "machine without an accelerator; JAX_PLATFORMS=cpu asks for the CPU on purpose"
            )
    return backend


def accelerator_plugins() -> list[str]:
    """The JAX accelerator plugin packages installed: `jax-cuda12-plugin`, `jax-rocm7-plugin`."""
    import importlib.metadata

    found = []
    for distribution in importlib.metadata.distributions():
        name = (distribution.metadata["Name"] or "").lower()
        if name.startswith(("jax-cuda", "jax_cuda", "jax-rocm", "jax_rocm")) and "plugin" in name:
            found.append(f"{name} {distribution.version}")
    return sorted(found)


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
        # One device: a module compiled for every device of the platform expects
        # an argument shard per device, and a machine with two GPUs then refuses
        # the single buffers the bridge places. The first device is the one JAX
        # calls the default; `PPY_XLA_DEVICE` names another by its index.
        devices = self._backend.devices()
        index = int(os.environ.get("PPY_XLA_DEVICE", "0"))
        if not 0 <= index < len(devices):
            raise ValueError(f"PPY_XLA_DEVICE={index}: the platform has {len(devices)} device(s)")
        self._device = devices[index]
        self._executables: dict[str, Any] = {}

    @property
    def version(self) -> str:
        return f"{self._xla._version}:{self._backend.platform_version}"

    def devices(self) -> list[str]:
        """Every device of the platform; the bridge runs on `device()`."""
        return [str(device) for device in self._backend.devices()]

    def device(self) -> str:
        return str(self._device)

    def key(self, stablehlo: str) -> str:
        """What identifies a compiled module: its text, the bindings, the platform, the device."""
        digest = hashlib.sha256()
        for part in (stablehlo, self.version, self.platform, self.device()):
            digest.update(part.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def compile(self, stablehlo: str) -> Any:
        """The loaded executable for `stablehlo`, from memory, disk, or the compiler."""
        key = self.key(stablehlo)
        found = self._executables.get(key)
        if found is not None:
            return found
        devices = self._xla.DeviceList((self._device,))
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

        buffers = [self._jax.device_put(numpy.asarray(a), self._device) for a in arguments]
        return [numpy.asarray(result) for result in executable.execute(buffers)]


def client(platform: str | None = None) -> Client:
    """The process's client for `platform`, made on first use."""
    global _client  # noqa: PLW0603
    if _client is None or (platform is not None and _client.platform != platform):
        _client = Client(platform)
    return _client
