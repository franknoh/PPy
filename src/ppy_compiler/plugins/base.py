"""Library plugin interface (spec 18)."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ..analysis import types as T
from ..analysis.effects import EffectSet
from ..analysis.refinements import Facts

__all__ = [
    "Lowering",
    "CallResult",
    "CallAdjustment",
    "Plugin",
    "PluginContext",
    "PluginRegistry",
]


class Lowering(enum.StrEnum):
    """The lowering a plugin selects for a recognized operation (spec 18.2)."""

    INTRINSIC = "Intrinsic"
    DIRECT_NATIVE_CALL = "DirectNativeCall"
    GRAPH_REGION = "GraphRegion"
    PYTHON_FALLBACK = "PythonFallback"
    REJECT = "Reject"


@dataclass(frozen=True, slots=True)
class CallResult:
    """A plugin's verdict for one recognized call."""

    type: T.Type
    facts: Facts = field(default_factory=Facts)
    effects: EffectSet = field(default_factory=EffectSet)
    lowering: Lowering = Lowering.PYTHON_FALLBACK
    reason: str = ""
    guards: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CallAdjustment:
    """A framework-level rewrite of one call's arguments (spec 15.4, 22.2)."""

    qualname: str
    replace_first_argument: str | None = None
    add_keywords: tuple[tuple[str, str], ...] = ()
    reason: str = ""


@dataclass(slots=True)
class PluginContext:
    """Per-project state a plugin may consult."""

    options: dict[str, object] = field(default_factory=dict)
    enabled: bool = True


class Plugin(Protocol):
    """A library plugin.

    A plugin never guesses a native signature: every direct path must come from
    a documented public API, a binding manifest, or a version-pinned adapter
    (spec 18.3).
    """

    name: str
    modules: tuple[str, ...]

    def fingerprint(self) -> str:
        """ABI/version identity that must enter every cache key (spec 18.3)."""

    def external_types(self) -> dict[str, str]:
        """Qualified names this plugin can type, mapped to display names."""

    def attribute_type(self, qualname: str) -> tuple[T.Type, Facts] | None:
        """Type of a module-level attribute such as `numpy.pi`."""

    def call(
        self,
        qualname: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult | None:
        """Type, effects, and lowering for a recognized call."""


class PluginRegistry:
    """Holds the plugins enabled for one project."""

    def __init__(self) -> None:
        self._plugins: list[Plugin] = []
        self._by_module: dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> None:
        self._plugins.append(plugin)
        for module in plugin.modules:
            self._by_module[module] = plugin

    def for_module(self, module: str) -> Plugin | None:
        root = module.partition(".")[0]
        return self._by_module.get(module) or self._by_module.get(root)

    def for_qualname(self, qualname: str) -> Plugin | None:
        parts = qualname.split(".")
        for index in range(len(parts) - 1, 0, -1):
            found = self._by_module.get(".".join(parts[:index]))
            if found is not None:
                return found
        return None

    @property
    def plugins(self) -> list[Plugin]:
        return list(self._plugins)

    def external_types(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for plugin in self._plugins:
            merged.update(plugin.external_types())
        return merged

    def fingerprints(self) -> tuple[str, ...]:
        return tuple(sorted(f"{p.name}:{p.fingerprint()}" for p in self._plugins))

    def __len__(self) -> int:
        return len(self._plugins)

    def __iter__(self):
        return iter(self._plugins)
