"""Plugin loading for a project (spec 18, 30)."""

from __future__ import annotations

from ..driver.config import Config
from .base import PluginRegistry

__all__ = ["AVAILABLE", "load_plugins"]

AVAILABLE = ("numpy", "torch", "jax", "uvicorn", "pydantic")


def load_plugins(config: Config) -> PluginRegistry:
    registry = PluginRegistry()
    for name in AVAILABLE:
        settings = config.plugin(name)
        if not settings.enabled:
            continue
        plugin = _construct(name, dict(settings.options))
        if plugin is not None:
            registry.register(plugin)
    return registry


def _construct(name: str, options: dict[str, object]):  # type: ignore[no-untyped-def]
    match name:
        case "numpy":
            from .numpy_plugin import NumPyPlugin

            return NumPyPlugin(options)
        case "torch":
            from .torch_plugin import TorchPlugin

            return TorchPlugin(options)
        case "jax":
            from .jax_plugin import JaxPlugin

            return JaxPlugin(options)
        case "uvicorn":
            from .uvicorn_plugin import UvicornPlugin

            return UvicornPlugin(options)
        case "pydantic":
            from .pydantic_plugin import PydanticPlugin

            return PydanticPlugin(options)
    return None
