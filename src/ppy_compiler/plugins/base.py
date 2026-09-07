"""Library plugin interface, version 2 (spec 18).

A plugin extends the compiler's semantics through explicit hooks and
nothing else: it types the calls and attributes of the modules it claims,
declares their effects, says how each recognized operation lowers -- as a
backend-neutral spec, never as backend code -- and may register dialects,
passes, patterns, and lowerings for the IR. Every hook has a no-op default
here, so the compiler calls them directly; there is no probing for
optional capabilities.

A plugin is loaded for a project, never for the process: discovery reads
entry points without importing anything, and a plugin's module is imported
only when a project enables it.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..analysis import types as T
from ..analysis.effects import EffectSet
from ..analysis.refinements import Facts

if TYPE_CHECKING:
    import ast

    from ..analysis.decorators import DecoratorSemantics
    from ..ir import DialectRegistry, PassManager

__all__ = [
    "PLUGIN_API_VERSION",
    "CallAdjustment",
    "CallResult",
    "DialectOperationSpec",
    "DirectCallSpec",
    "FallbackSpec",
    "GraphRegionSpec",
    "IntrinsicSpec",
    "Lowering",
    "LoweringSpec",
    "Plugin",
    "PluginContext",
    "PluginError",
    "PluginRegistry",
    "RejectSpec",
]

#: The plugin interface's version, part of every plugin fingerprint: a
#: plugin written against another one is a different plugin to the cache,
#: and one written against an older one is refused with the reason.
PLUGIN_API_VERSION = 2


class PluginError(Exception):
    """A plugin the project cannot use: the wrong interface, or a collision."""


class Lowering(enum.StrEnum):
    """The kind of lowering a plugin selects for a recognized operation (spec 18.2)."""

    INTRINSIC = "Intrinsic"
    DIALECT_OPERATION = "DialectOperation"
    DIRECT_NATIVE_CALL = "DirectNativeCall"
    GRAPH_REGION = "GraphRegion"
    PYTHON_FALLBACK = "PythonFallback"
    REJECT = "Reject"


# -- typed lowering specs (spec 19) ------------------------------------------
# What a plugin answers about an operation is an object a backend reads,
# never LLVM IR or CUDA text: the same answer serves every backend.


@dataclass(frozen=True, slots=True)
class IntrinsicSpec:
    """A named intrinsic the backends know: `math.sqrt`, a fused kernel."""

    name: str

    @property
    def kind(self) -> Lowering:
        return Lowering.INTRINSIC


@dataclass(frozen=True, slots=True)
class DialectOperationSpec:
    """An operation of a dialect: `special.erf`, `columnar.filter`."""

    dialect: str
    operation: str
    attributes: tuple[tuple[str, object], ...] = ()

    @property
    def kind(self) -> Lowering:
        return Lowering.DIALECT_OPERATION


@dataclass(frozen=True, slots=True)
class DirectCallSpec:
    """A call into a library through its documented C symbol."""

    symbol: str
    library: str = ""
    abi: str = "c"

    @property
    def kind(self) -> Lowering:
        return Lowering.DIRECT_NATIVE_CALL


@dataclass(frozen=True, slots=True)
class GraphRegionSpec:
    """A region a framework compiles as one graph: an ATen region, a staged export."""

    framework: str
    region: str = ""

    @property
    def kind(self) -> Lowering:
        return Lowering.GRAPH_REGION


@dataclass(frozen=True, slots=True)
class FallbackSpec:
    """The Python implementation runs; `reason` says why that is right."""

    reason: str = ""

    @property
    def kind(self) -> Lowering:
        return Lowering.PYTHON_FALLBACK


@dataclass(frozen=True, slots=True)
class RejectSpec:
    """The construct is refused outright, with the reason."""

    reason: str = ""

    @property
    def kind(self) -> Lowering:
        return Lowering.REJECT


LoweringSpec = (
    IntrinsicSpec
    | DialectOperationSpec
    | DirectCallSpec
    | GraphRegionSpec
    | FallbackSpec
    | RejectSpec
)


@dataclass(frozen=True, slots=True)
class CallResult:
    """A plugin's verdict for one recognized call."""

    type: T.Type
    facts: Facts = field(default_factory=Facts)
    effects: EffectSet = field(default_factory=EffectSet)
    lowering: Lowering | LoweringSpec = Lowering.PYTHON_FALLBACK
    reason: str = ""
    guards: tuple[str, ...] = ()

    @property
    def kind(self) -> Lowering:
        """The kind of lowering, whether the plugin answered a kind or a spec."""
        lowering = self.lowering
        return lowering if isinstance(lowering, Lowering) else lowering.kind

    @property
    def spec(self) -> LoweringSpec:
        """The lowering as a spec; a bare kind becomes the spec with no detail."""
        lowering = self.lowering
        if not isinstance(lowering, Lowering):
            return lowering
        if lowering is Lowering.INTRINSIC:
            return IntrinsicSpec("")
        if lowering is Lowering.DIRECT_NATIVE_CALL:
            return DirectCallSpec("")
        if lowering is Lowering.GRAPH_REGION:
            return GraphRegionSpec("")
        if lowering is Lowering.REJECT:
            return RejectSpec(self.reason)
        if lowering is Lowering.DIALECT_OPERATION:
            return DialectOperationSpec("", "")
        return FallbackSpec(self.reason)


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


class Plugin:
    """A library plugin: the base every plugin extends.

    Every hook has a default that claims nothing, so a plugin implements
    what it knows and the compiler calls the rest without asking whether it
    exists. A plugin never guesses a native signature: every direct path
    comes from a documented public API, a binding manifest, or a
    version-pinned adapter (spec 18.3).
    """

    #: The plugin's name: its section under `[tool.ppy.plugins]`.
    name: str = ""
    #: The modules it claims, by root: every qualified name under one of
    #: them is asked of this plugin.
    modules: tuple[str, ...] = ()
    #: The interface this plugin was written against.
    api_version: int = PLUGIN_API_VERSION

    def __init__(self, options: dict[str, object] | None = None) -> None:
        self.options: dict[str, object] = dict(options or {})

    # -- identity -----------------------------------------------------------

    def fingerprint(self) -> str:
        """ABI/version identity that enters every cache key (spec 18.3).

        The default names the interface and the plugin; a plugin over a
        library adds the library's version and build.
        """
        return f"api{self.api_version}:{self.name}"

    # -- typing ---------------------------------------------------------------

    def external_types(self) -> dict[str, str]:
        """Qualified names this plugin can type, mapped to display names."""
        return {}

    def attribute_type(self, qualname: str) -> tuple[T.Type, Facts] | None:
        """Type of a module-level attribute such as `numpy.pi`."""
        return None

    def instance_attribute(
        self, type_name: str, attribute: str, facts: Facts | None = None
    ) -> tuple[T.Type, Facts] | None:
        """Type of an attribute or method of an instance of an external class."""
        return None

    def subscript(
        self, type_name: str, *, is_slice: bool, tupled: bool
    ) -> tuple[T.Type, Facts] | None:
        """Type of indexing an instance of an external class."""
        return None

    def call(
        self,
        qualname: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult | None:
        """Type, effects, and lowering for a recognized call."""
        return None

    def operator(self, symbol: str) -> str | None:
        """The library function a Python operator on this plugin's types runs."""
        return None

    def call_alias(self, type_name: str) -> str | None:
        """The method that calling an instance of an external class runs, if
        the plugin knows one: `forward` for `torch.nn.Module`."""
        return None

    def decorator_semantics(self, name: str) -> DecoratorSemantics | None:
        """What a decorator this plugin vouches for does to a function."""
        return None

    def adjust_call(self, qualname: str, node: ast.Call, symbols: object) -> CallAdjustment | None:
        """A framework-level rewrite of one call's arguments (spec 22.2)."""
        return None

    def stage(self, decorators: Sequence[str]) -> CallResult | None:
        """The build-time stage a decorated function enters, if this plugin
        stages it: a JAX export, an ATen region."""
        return None

    def tensor_operation(self, qualname: str) -> DialectOperationSpec | None:
        """The shared tensor operation this call converges onto, if any (spec 42).

        `numpy.add`, `torch.add`, and `jax.numpy.add` are all `tensor.add`:
        the compiler reads the answer to fuse expressions across libraries
        and to build their kernels from the tensor dialect, whatever the
        call's own lowering is.
        """
        return None

    # -- the IR ---------------------------------------------------------------

    def register_dialects(self, registry: DialectRegistry) -> None:
        """Dialects this plugin defines: `registry.register(MyDialect())`."""

    def register_passes(self, manager: PassManager) -> None:
        """Passes this plugin adds: `manager.register_stage_pass(stage, factory)`."""

    def register_patterns(self, registry: DialectRegistry) -> None:
        """Rewrite patterns this plugin adds: `registry.add_pattern(pattern)`."""

    def register_lowerings(self, registry: DialectRegistry) -> None:
        """Lowerings of this plugin's dialects to others or to a backend."""

    def __repr__(self) -> str:
        return f"<plugin {self.name}>"


class PluginRegistry:
    """Holds the plugins enabled for one project."""

    def __init__(self) -> None:
        self._plugins: list[Plugin] = []
        self._by_module: dict[str, Plugin] = {}
        #: What could not be loaded, for the driver to report.
        self.problems: list[str] = []

    def register(self, plugin: Plugin) -> None:
        if not isinstance(plugin, Plugin):
            raise PluginError(
                f"{type(plugin).__module__}.{type(plugin).__qualname__} is not a Plugin: "
                f"a plugin extends `ppy_compiler.plugins.Plugin` (interface {PLUGIN_API_VERSION})"
            )
        if plugin.api_version != PLUGIN_API_VERSION:
            raise PluginError(
                f"plugin {plugin.name!r} was written against plugin interface "
                f"{plugin.api_version}; this compiler speaks {PLUGIN_API_VERSION}"
            )
        for module in plugin.modules:
            other = self._by_module.get(module)
            if other is not None and other is not plugin:
                raise PluginError(
                    f"plugin {other.name!r} and plugin {plugin.name!r} both claim module {module!r}"
                )
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

    def fingerprints(self, modules: Iterable[str] | None = None) -> tuple[str, ...]:
        """Version fingerprints of the plugins that can affect a compilation.

        Computing a fingerprint imports the library, which for an accelerator
        runtime costs seconds. A module that never imports torch cannot be
        affected by torch's version, so restricting this to the modules
        actually imported keeps that cost off every other compilation. Pass
        `None` to fingerprint everything.
        """
        selected = self._plugins
        if modules is not None:
            roots = {name.partition(".")[0] for name in modules}
            selected = [
                plugin
                for plugin in self._plugins
                if roots & {module.partition(".")[0] for module in plugin.modules}
            ]
        return tuple(sorted(f"{p.name}:{p.fingerprint()}" for p in selected))

    def dialect_registry(self) -> DialectRegistry:
        """The IR registry for this project: the builtin dialects plus every
        dialect, pattern, and lowering the enabled plugins register."""
        from ..ir.dialect import DialectRegistry
        from ..ir.dialects import builtin_dialects

        registry = DialectRegistry()
        for dialect in builtin_dialects():
            registry.register(dialect)
        for plugin in self._plugins:
            plugin.register_dialects(registry)
            plugin.register_patterns(registry)
            plugin.register_lowerings(registry)
        return registry

    def register_passes(self, manager: PassManager) -> None:
        for plugin in self._plugins:
            plugin.register_passes(manager)

    def __len__(self) -> int:
        return len(self._plugins)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._plugins)
