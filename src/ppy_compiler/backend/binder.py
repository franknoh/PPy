"""Binding hooks a generated module calls as it loads (spec 15.4, 16.4)."""

from __future__ import annotations

__all__ = ["LibraryBinder"]


class LibraryBinder:
    """Supplies guarded entry points to generated modules.

    The Python backend uses this directly; the LLVM backend extends it with
    native functions and fused kernels.
    """

    def __init__(self) -> None:
        self.exported_bindings: list = []
        self.region_bindings: list = []
        self._exported: dict[str, dict[str, bytes]] = {}
        self._regions: dict[str, dict[str, object]] = {}

    def add_exported(self, module: str, function: str, payload: bytes) -> None:
        self._exported.setdefault(module, {})[function] = payload

    def exported_names(self, module: str) -> frozenset[str]:
        return frozenset(self._exported.get(module, {}))

    def exported(self, module: str, function: str, fallback):  # type: ignore[no-untyped-def]
        """Serve a build-time exported graph region, guarded by its signature."""
        from .exported_runtime import bind_exported

        payload = self._exported.get(module, {}).get(function)
        if payload is None:
            return fallback
        binding = bind_exported(function, payload, fallback)
        self.exported_bindings.append(binding)
        return binding.wrapper

    def add_region(self, module: str, function: str, compiled: object) -> None:
        self._regions.setdefault(module, {})[function] = compiled

    def region_names(self, module: str) -> frozenset[str]:
        return frozenset(self._regions.get(module, {}))

    def region(self, module: str, function: str, fallback):  # type: ignore[no-untyped-def]
        """Serve a compiled ATen region, guarded by its curated domain."""
        from .region_runtime import bind_region

        compiled = self._regions.get(module, {}).get(function)
        binding = bind_region(function, compiled, fallback)
        self.region_bindings.append(binding)
        return binding.wrapper

    def names(self, module: str) -> frozenset[str]:
        return frozenset()

    def bind(self, module: str, function: str, fallback):  # type: ignore[no-untyped-def]
        return fallback

    def fused(self, module: str, symbol: str, fallback):  # type: ignore[no-untyped-def]
        return fallback
