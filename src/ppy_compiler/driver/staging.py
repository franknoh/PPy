"""Build-time staging of library graph regions (spec 21.3, 31).

Some integrations can only be prepared by running project code. That is done
here, once per build, behind an explicit permission and with the result cached
under a fingerprint of everything that could change it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..cache import CacheKey, digest
from ..diagnostics import Diagnostic, DiagnosticBag, Severity, Span
from ..plugins.jax_export import ExportRequest, accelerator_fingerprint, export_function
from ..plugins.jax_plugin import JaxPlugin, staged_functions
from ..version import COMPILER_VERSION

__all__ = [
    "RegionResult",
    "StagedArtifact",
    "StagingResult",
    "compile_torch_regions",
    "stage_project",
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


def _export_help(reason: str) -> str:
    """Advice that matches why the export was declined, not a fixed suggestion."""
    if "VJP" in reason:
        return (
            "the exported artifact would run the forward pass only; keep the "
            "differentiated function jitted, or export a function that is not "
            "differentiated"
        )
    return "annotate each parameter with `ppy.Shape(...)` and `ppy.DType(...)`"


def stage_project(bundle, diagnostics: DiagnosticBag | None = None) -> StagingResult:  # type: ignore[no-untyped-def]
    """Export every staged region the project both declares and permits."""
    result = StagingResult()
    _stage_xla(bundle, result)
    plugin = next((p for p in bundle.project.plugins if isinstance(p, JaxPlugin)), None)
    if plugin is None:
        if diagnostics is not None:
            diagnostics.extend(result.diagnostics)
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
                        help=_export_help(staged.reason),
                    )
                )
                continue
            if not permitted:
                result.skipped.append((staged.info.qualname, reason))
                result.diagnostics.append(
                    Diagnostic(
                        "R3001",
                        Severity.REMARK,
                        f"`{staged.info.name}` not exported: {reason}",
                        span,
                    )
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

            store.put(
                key, exported.payload, kind="metadata", source=str(symbols.path), suffix=".mlir"
            )
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
                    f"`{staged.info.name}` exported to StableHLO for "
                    + (", ".join(exported.platforms) or "the default platform"),
                    span,
                )
            )
    if diagnostics is not None:
        diagnostics.extend(result.diagnostics)
    return result


# -- ppy.xla: functions XLA compiles ----------------------------------------------------------


def _stage_xla(bundle, result: StagingResult) -> None:  # type: ignore[no-untyped-def]
    """Every `@ppy.xla.jit` function as a StableHLO payload the runtime compiles once.

    The function is lowered to the IR like any other, its stack slots
    promoted to values, and emitted as StableHLO; a function XLA cannot
    take -- a branch, a buffer, a guard -- is reported and stays where it
    is. No device is touched here: staging writes text, running compiles it.
    """
    from ..backend.llvm import _value_class_layouts, prover_for
    from ..backend.stablehlo import StableHloError, emit_module, prepare
    from ..lowering import lower_module_to_ir
    from ..lowering.ast_to_ir import Unsupported

    config = bundle.project.config
    layouts = None
    for module_name, symbols in bundle.symbols.modules.items():
        analysis = bundle.analysis.modules.get(module_name)
        if analysis is None:
            continue
        for info in symbols.functions.values():
            if info.directive("xla.jit") is None:
                continue
            function_analysis = analysis.functions.get(info.qualname)
            if function_analysis is None:
                continue
            span = Span(symbols.path, info.node.lineno, 0)
            key = CacheKey.build(
                "xla-stage",
                source_digest=symbols.module.source.digest(),
                compiler_version=COMPILER_VERSION,
                opt_level=bundle.project.config.opt_level,
                target="stablehlo",
                plugin_fingerprints=(),
                extra=(info.qualname,),
            )
            cached = bundle.project.store.read(key)
            if cached is not None:
                result.artifacts.setdefault(module_name, {})[info.name] = StagedArtifact(
                    module_name, info.name, key.hex(), cached, "", ("xla",)
                )
                continue
            if layouts is None:
                layouts = _value_class_layouts(bundle)
            try:
                lowered = lower_module_to_ir(
                    analysis,
                    {info.qualname: (info, function_analysis, info.node)},
                    layouts,
                    safeguards=config.llvm.safeguards or "hoisted",
                    prover=prover_for(config),
                    root=bundle.project.root,
                )
                if info.qualname in lowered.rejected:
                    raise Unsupported(lowered.rejected[info.qualname])
                function = next(
                    f
                    for f in lowered.module.functions.values()
                    if f.attributes.get("ppy.qualname") == info.qualname
                )
                prepare(lowered.module)
                text = emit_module(lowered.module, (function.name,), entry=function.name)
            except (Unsupported, StableHloError) as error:
                result.skipped.append((info.qualname, str(error)))
                result.diagnostics.append(
                    Diagnostic(
                        "W2007",
                        Severity.WARNING,
                        f"`{info.name}` is marked `@ppy.xla.jit` but XLA cannot take it: {error}",
                        span,
                    )
                )
                continue
            payload = json.dumps(
                {
                    "kind": "ppy.xla",
                    "function": info.qualname,
                    "stablehlo": text,
                    "params": [_kind_of(p.type) for p in info.params],
                    "results": [_kind_of(info.ret)],
                }
            ).encode("utf-8")
            bundle.project.store.put(
                key, payload, kind="metadata", source=str(symbols.path), suffix=".mlir"
            )
            bundle.project.store.mark_root(key, f"xla:{info.qualname}")
            result.artifacts.setdefault(module_name, {})[info.name] = StagedArtifact(
                module_name, info.name, key.hex(), payload, "", ("xla",)
            )
            result.diagnostics.append(
                Diagnostic(
                    "R3001",
                    Severity.REMARK,
                    f"`{info.name}` emitted as StableHLO for XLA",
                    span,
                )
            )


def _kind_of(t) -> str:  # type: ignore[no-untyped-def]
    from ..analysis import types as T

    base = T.strip_literal(t)
    if base == T.FLOAT:
        return "float"
    if base == T.INT:
        return "int"
    if base == T.BOOL:
        return "bool"
    return "array"


@dataclass(slots=True)
class RegionResult:
    """Compiled ATen regions, per module."""

    compiled: dict[str, dict[str, object]] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    #: Per module: the extension library holding its regions, and the C++
    #: symbol of each -- what a build records so the launched artifact can
    #: load them without the compiler.
    libraries: dict[str, tuple[Path, dict[str, str]]] = field(default_factory=dict)

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

        if built.library is not None and built.symbols:
            result.libraries[module_name] = (built.library, dict(built.symbols))
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
