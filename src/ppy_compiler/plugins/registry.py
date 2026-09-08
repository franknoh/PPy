"""Plugin loading for a project (spec 18, 30).

Builtin plugins load when their section does not disable them. External
plugins are discovered through the `ppy.plugins` entry-point group without
importing anything, and one is imported and loaded only when the project
names it under `[tool.ppy.plugins.<name>]` and does not disable it: an
installed plugin is not a running one. Two plugins claiming one module is
a problem the driver reports, never a question of who registered last.
"""

from __future__ import annotations

import importlib.metadata

from ..driver.config import Config
from .base import Plugin, PluginError, PluginRegistry

__all__ = ["AVAILABLE", "ENTRY_POINT_GROUP", "discover_external", "load_plugins"]

AVAILABLE = ("numpy", "torch", "jax", "uvicorn", "pydantic", "scipy", "pandas", "pyarrow")
ENTRY_POINT_GROUP = "ppy.plugins"


def discover_external() -> dict[str, importlib.metadata.EntryPoint]:
    """Every installed plugin by name, without importing any of them."""
    found: dict[str, importlib.metadata.EntryPoint] = {}
    try:
        entries = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 - a broken distribution must not break discovery
        return found
    for entry in entries:
        found.setdefault(entry.name, entry)
    return found


def load_plugins(config: Config) -> PluginRegistry:
    registry = PluginRegistry()
    for name in AVAILABLE:
        settings = config.plugin(name)
        if not settings.enabled:
            continue
        plugin = _construct(name, dict(settings.options))
        if plugin is not None:
            _register(registry, plugin)
    external = discover_external()
    for name, entry in sorted(external.items()):
        if name in AVAILABLE or name not in config.plugins:
            # A builtin name is the builtin; a plugin the project does not
            # mention stays unloaded, however many are installed.
            continue
        settings = config.plugin(name)
        if not settings.enabled:
            continue
        try:
            factory = entry.load()
            plugin = factory(dict(settings.options))
        except Exception as error:  # noqa: BLE001 - reported, not raised through analysis
            registry.problems.append(f"plugin {name!r} could not be loaded: {error}")
            continue
        if getattr(plugin, "name", "") == "":
            plugin.name = name
        _register(registry, plugin)
    return registry


def _register(registry: PluginRegistry, plugin: Plugin) -> None:
    try:
        registry.register(plugin)
    except PluginError as error:
        registry.problems.append(str(error))


def _construct(name: str, options: dict[str, object]) -> Plugin | None:
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
        case "scipy":
            from .scipy_plugin import SciPyPlugin

            return SciPyPlugin(options)
        case "pandas":
            from .pandas_plugin import PandasPlugin

            return PandasPlugin(options)
        case "pyarrow":
            from .pyarrow_plugin import PyArrowPlugin

            return PyArrowPlugin(options)
    return None
