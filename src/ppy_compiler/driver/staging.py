"""Build-time staging of library graph regions (spec 21.3, 31).

Some integrations can only be prepared by running project code. That is done
here, once per build, behind an explicit permission and with the result cached
under a fingerprint of everything that could change it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..cache import CacheKey, digest
from ..diagnostics import Diagnostic, DiagnosticBag, Severity, Span
from ..plugins.jax_export import ExportRequest, accelerator_fingerprint, export_function
from ..plugins.jax_plugin import JaxPlugin, staged_functions

__all__ = [
    "StagedArtifact",
    "StagingResult",
    "RegionResult",
    "stage_project",
    "compile_torch_regions",
]


@dataclass(frozen=True, slots=True)
class StagedArtifact:
    module: str
    function: str
    key: str
    payload: bytes
    signature: str = ""
    platforms: tuple[str, ...] = ()


@dataclass(slots=True)
class StagingResult:
    artifacts: dict[str, dict[str, StagedArtifact]] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def names(self, module: str) -> frozenset[str]:
        return frozenset(self.artifacts.get(module, {}))

    @property
    def count(self) -> int:
        return sum(len(entries) for entries in self.artifacts.values())


def stage_project(bundle, diagnostics: DiagnosticBag | None = None) -> StagingResult:  # type: ignore[no-untyped-def]
    """Export every staged region the project both declares and permits."""
    result = StagingResult()
    plugin = next((p for p in bundle.project.plugins if isinstance(p, JaxPlugin)), None)
    if plugin is None:
        return result

    permitted, reason = plugin.export_permitted(bundle.project.config.build_execution)
    store = bundle.project.store

    for module_name, symbols in bundle.symbols.modules.items():
        for staged in staged_functions(symbols):
            span = Span(symbols.path, staged.info.node.lineno, 0)
            if not staged.exportable:
                result.skipped.append((staged.info.qualname, staged.reason))
                result.diagnostics.append(
                    Diagnostic(
                        "W2004",
                        Severity.WARNING,
                        f"`{staged.info.name}` is staged but cannot be exported: {staged.reason}",
                        span,
                        help="annotate each parameter with `ppy.Shape(...)` and `ppy.DType(...)`",
                    )
                )
                continue
            if not permitted:
                result.skipped.append((staged.info.qualname, reason))
                result.diagnostics.append(
                    Diagnostic("R3001", Severity.REMARK, f"`{staged.info.name}` not exported: {reason}", span)
                )
                continue

            request = ExportRequest(
                module_path=symbols.path,
                module_name=module_name,
                function=staged.info.name,
                shapes=staged.shapes,
                dtypes=staged.dtypes,
                search_paths=tuple(bundle.project.search_paths),
            )
            key = CacheKey.build(
                "jax-export",
                source_digest=symbols.module.source.digest(),
                compiler_version="0.1.0",
                opt_level=bundle.project.config.opt_level,
                target="pjrt",
                plugin_fingerprints=bundle.project.plugins.fingerprints(),
                extra=(digest(request.fingerprint_input()), accelerator_fingerprint()),
            )
            cached = store.read(key)
            if cached is not None:
                result.artifacts.setdefault(module_name, {})[staged.info.name] = StagedArtifact(
                    module_name, staged.info.name, key.hex(), cached
                )
                continue

            exported = export_function(request)
            if not exported.ok:
                result.skipped.append((staged.info.qualname, exported.reason))
                result.diagnostics.append(
                    Diagnostic(
                        "W2004",
                        Severity.WARNING,
                        f"`{staged.info.name}` could not be exported: {exported.reason}",
                        span,
                    )
                )
                continue

            store.put(key, exported.payload, kind="metadata", source=str(symbols.path), suffix=".mlir")
            store.mark_root(key, f"jax:{staged.info.qualname}")
            result.artifacts.setdefault(module_name, {})[staged.info.name] = StagedArtifact(
                module_name,
                staged.info.name,
                key.hex(),
                exported.payload,
                exported.signature,
                exported.platforms,
            )
            result.diagnostics.append(
                Diagnostic(
                    "R3001",
                    Severity.REMARK,
                    f"`{staged.info.name}` exported to StableHLO for {', '.join(exported.platforms) or 'the default platform'}",
                    span,
                )
            )
    if diagnostics is not None:
        diagnostics.extend(result.diagnostics)
    return result


@dataclass(slots=True)
class RegionResult:
    """Compiled ATen regions, per module."""

    compiled: dict[str, dict[str, object]] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return sum(len(entries) for entries in self.compiled.values())


def compile_torch_regions(  # type: ignore[no-untyped-def]
    bundle, diagnostics: DiagnosticBag | None = None, notify=None
) -> RegionResult:
    """Compile every PPY function that translates wholly into ATen C++ calls.

    Each `at::` call inside a region still goes through the dispatcher, so
    autograd and device selection are unchanged; what the region removes is one
    Python round trip per operation (spec 20.3, 20.5).
    """
    from ..plugins.torch_build import compile_regions
    from ..plugins.torch_plugin import TorchPlugin
    from ..plugins.torch_region import find_regions

    result = RegionResult()
    plugin = next((p for p in bundle.project.plugins if isinstance(p, TorchPlugin)), None)
    if plugin is None or not bool(plugin.options.get("cpp-regions", True)):
        return result

    for module_name, symbols in bundle.symbols.modules.items():
        analysis = bundle.analysis.modules.get(module_name)
        if analysis is None:
            continue
        regions = find_regions(symbols, analysis)
        if not regions:
            continue

        for region in regions:
            if not region.body:
                result.rejected.append((region.info.qualname, region.reason))
                result.diagnostics.append(
                    Diagnostic(
                        "R3001",
                        Severity.REMARK,
                        f"`{region.name}` stays on the Python path: {region.reason}",
                        Span(symbols.path, region.info.node.lineno, 0),
                    )
                )

        built = compile_regions(regions, bundle.project.config.cache_path, notify=notify)
        if not built.ok:
            if any(region.body for region in regions):
                result.diagnostics.append(
                    Diagnostic(
                        "W2004",
                        Severity.WARNING,
                        f"no ATen region was compiled for `{module_name}`: {built.reason}",
                        Span(symbols.path, 1, 0),
                    )
                )
            continue

        for region in regions:
            entry_point = built.entry_points.get(region.name)
            if entry_point is None:
                continue
            result.compiled.setdefault(module_name, {})[region.name] = entry_point
            result.diagnostics.append(
                Diagnostic(
                    "R3001",
                    Severity.REMARK,
                    f"`{region.name}` compiled into one ATen region: "
                    f"{', '.join(region.operations)}",
                    Span(symbols.path, region.info.node.lineno, 0),
                )
            )
    if diagnostics is not None:
        diagnostics.extend(result.diagnostics)
    return result
