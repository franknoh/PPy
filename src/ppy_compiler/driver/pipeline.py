"""Compiler driver: project discovery, analysis, and backend selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from ..analysis import stdlib
from ..analysis.checker import ProjectAnalysis, analyze
from ..analysis.contracts import ContractReport, verify
from ..analysis.symbols import ProjectSymbols
from ..cache import CacheKey, CacheStore
from ..cache.keys import digest, environment_fingerprint
from ..diagnostics import Diagnostic, DiagnosticBag, Severity
from ..frontend.modules import ModuleGraph, build_graph
from ..opt.manager import OptimizationResult, Optimizer
from ..plugins.base import PluginRegistry
from ..plugins.registry import load_plugins
from ..version import COMPILER_VERSION, compiler_fingerprint
from .config import Config, find_project_root, load_config

__all__ = [
    "COMPILER_VERSION",
    "AnalysisBundle",
    "BuildOutput",
    "Project",
    "analyze_paths",
    "build_python",
    "collect_sources",
    "compiler_fingerprint",
    "open_project",
]

_SOURCE_SUFFIXES = (".ppy", ".py")


@dataclass(slots=True)
class Project:
    config: Config
    root: Path
    search_paths: list[Path]
    store: CacheStore
    plugins: PluginRegistry

    @property
    def cache(self) -> CacheStore:
        return self.store


@dataclass(slots=True)
class AnalysisBundle:
    project: Project
    graph: ModuleGraph
    symbols: ProjectSymbols
    analysis: ProjectAnalysis
    reports: dict[str, ContractReport]
    diagnostics: DiagnosticBag
    entry: str | None = None
    #: Project-wide cross-module write index, filled in by conversion.
    global_writes: object | None = None
    #: Project-wide annotation-reflection index, filled in by conversion.
    reflection: object | None = None

    @property
    def ok(self) -> bool:
        return not self.diagnostics.has_errors()


@dataclass(slots=True)
class BuildOutput:
    bundle: AnalysisBundle
    generated: dict[str, object] = field(default_factory=dict)
    remarks: list[Diagnostic] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


def open_project(target: Path, *, config_overrides: dict[str, object] | None = None) -> Project:
    """Discover the project root, configuration, cache, and plugins."""
    root = find_project_root(target)
    config = load_config(root)
    if config_overrides:
        config = config.with_overrides(**config_overrides)
    config.root = root

    search_paths: list[Path] = []
    entry_dir = target if target.is_dir() else target.parent
    # A file inside a package is a member of that package: `pkg/a.py` is
    # `pkg.a`, and its `from . import b` names `pkg.b`. Searching from its own
    # directory would call it `a` and lose the package. The search path is
    # the parent of the outermost package the directory sits in.
    search_paths.append(_package_root(entry_dir.resolve()))
    for source_root in config.source_roots:
        candidate = (root / source_root).resolve()
        if candidate.is_dir() and candidate not in search_paths:
            search_paths.append(candidate)

    store = CacheStore(config.cache_path)
    plugins = load_plugins(config)
    return Project(
        config=config, root=root, search_paths=search_paths, store=store, plugins=plugins
    )


def _package_root(directory: Path) -> Path:
    """The directory to import from so that `directory` keeps its package name."""
    while any((directory / f"__init__{suffix}").is_file() for suffix in _SOURCE_SUFFIXES):
        if directory.parent == directory:
            break
        directory = directory.parent
    return directory


def collect_sources(target: Path, *, ppy_only: bool = False) -> list[Path]:
    """Every source file a command should act on."""
    if target.is_file():
        return [target.resolve()]
    suffixes = (".ppy",) if ppy_only else _SOURCE_SUFFIXES
    return sorted(
        path.resolve()
        for suffix in suffixes
        for path in target.rglob(f"*{suffix}")
        if not any(part in {"__pycache__", ".ppy-cache", ".git", ".venv"} for part in path.parts)
    )


def analyze_paths(
    project: Project,
    entries: list[Path],
    *,
    backend: str = "python",
    follow_imports: bool = True,
    overlays: dict[Path, str] | None = None,
) -> AnalysisBundle:
    """Parse, resolve, type-check, and verify contracts for the given entries."""
    diagnostics = DiagnosticBag()
    for problem in project.plugins.problems:
        # A plugin the project asked for and cannot have: an error, not a
        # silently smaller compiler.
        diagnostics.add(Diagnostic("E1901", Severity.ERROR, problem))
    if not entries:
        diagnostics.add(Diagnostic("E1002", Severity.ERROR, "no PPY source files were found"))
        graph = ModuleGraph(root=project.root, search_paths=project.search_paths)
        symbols = ProjectSymbols(
            graph, diagnostics, strict=project.config.strict, generics=project.config.generics
        )
        analysis = ProjectAnalysis(symbols=symbols, diagnostics=diagnostics)
        return AnalysisBundle(project, graph, symbols, analysis, {}, diagnostics)

    graph = build_graph(
        entries,
        project.search_paths,
        diagnostics,
        root=project.root,
        follow_imports=follow_imports,
        overlays=overlays,
    )
    symbols = ProjectSymbols(
        graph, diagnostics, strict=project.config.strict, generics=project.config.generics
    )
    for qualname, display in stdlib.EXTERNAL_TYPES.items():
        symbols.register_external_type(qualname, display)
    for qualname, display in project.plugins.external_types().items():
        symbols.register_external_type(qualname, display)
    symbols.build()

    analysis = analyze(
        symbols,
        diagnostics,
        strict=project.config.strict,
        dynamic_policy=project.config.dynamic_boundaries,
        plugins=project.plugins,
    )
    reports = verify(analysis, diagnostics, backend=backend)
    return AnalysisBundle(
        project=project,
        graph=graph,
        symbols=symbols,
        analysis=analysis,
        reports=reports,
        diagnostics=diagnostics,
        entry=graph.entry,
    )


def module_cache_key(
    bundle: AnalysisBundle,
    module_name: str,
    *,
    target: str,
    opt_level: int,
    extra: tuple[object, ...] = (),
) -> CacheKey:
    """A content-addressed key covering source, config, deps, and plugins (spec 27.2)."""
    symbols = bundle.symbols.modules[module_name]
    dependency_hashes: list[str] = []
    for edge in symbols.module.imports:
        other = bundle.symbols.modules.get(edge.target)
        if other is None:
            dependency_hashes.append(f"external:{edge.target}")
            continue
        dependency_hashes.append(_public_abi_hash(bundle, edge.target))
    directives = [
        f"{info.qualname}:{d.name}:{sorted(d.options.items())}"
        for info in bundle.symbols.functions.values()
        if info.module == module_name
        for d in info.directives
    ]
    config = bundle.project.config
    if target == "llvm":
        from ..backend.builtin import builtin_backend

        # The LLVM under the objects: another llvmlite is other code generation.
        extra = (*extra, f"backend={builtin_backend('llvm').fingerprint()}")
        if config.llvm.sanitize:
            extra = (*extra, f"sanitize={','.join(sorted(config.llvm.sanitize))}")
        if config.llvm.instrument:
            extra = (*extra, "profile=instrument")
        if config.llvm.pgo:
            from .profile import profile_digest

            extra = (*extra, f"pgo={profile_digest(config.llvm.pgo)}")
    return CacheKey.build(
        target,
        source_digest=symbols.module.source.digest(),
        compiler_version=f"{COMPILER_VERSION}+{compiler_fingerprint()}",
        opt_level=opt_level,
        directives=directives,
        target=target,
        dependency_hashes=dependency_hashes,
        plugin_fingerprints=bundle.project.plugins.fingerprints(_imported_modules(symbols)),
        extra=(
            config.strict,
            config.dynamic_boundaries,
            config.inference.implicit_any,
            config.llvm.safeguards or "hoisted",
            _prover_descriptor(config),
            _target_descriptor(config),
            *extra,
        ),
    )


def backend_identity(  # type: ignore[no-untyped-def]
    bundle: AnalysisBundle, module_name: str, backend, opt_level: int
) -> CacheKey:
    """What identifies one module's artifact from an external backend: the
    module key (source, compiler, dependencies, plugins, options) with the
    backend's name, the distribution and version it came from, the interface
    version, its fingerprint, its configuration, its target, and the IR
    schema the backend reads.

    The distribution's version is in the key whether or not the backend's
    author thought to put it in `fingerprint()`: two releases of one package
    can carry the same module and class names and generate different code,
    and an artifact from the older one must not be served for the newer.
    """
    from ..ir import IR_SCHEMA_VERSION

    settings = bundle.project.config.backend(backend.name)
    distribution = backend.distribution or ("", "")
    return module_cache_key(
        bundle,
        module_name,
        target=f"backend:{backend.name}",
        opt_level=opt_level,
        extra=(
            f"ir-schema={IR_SCHEMA_VERSION}",
            f"backend={backend.fingerprint()}",
            f"backend-api={backend.api_version}",
            f"backend-dist={distribution[0]} {distribution[1]}",
            f"backend-config={settings.fingerprint()}",
            f"backend-target={settings.get('target', '')}",
        ),
    )


def backend_context(  # type: ignore[no-untyped-def]
    bundle: AnalysisBundle, backend, *, opt_level: int | None = None, entry: Path | None = None
):
    """Everything an external backend is told besides the IR (`BackendContext`)."""
    from ..backend.base import BackendContext

    config = bundle.project.config
    settings = config.backend(backend.name)
    level = opt_level if opt_level is not None else config.opt_level
    context = BackendContext(
        root=bundle.project.root,
        config=config,
        backend_config=settings,
        opt_level=level,
        target=str(settings.get("target", "")),
        registry=bundle.project.plugins.dialect_registry(),
        plugin_fingerprints=bundle.project.plugins.fingerprints(),
        entry=entry,
    )
    for name in bundle.symbols.modules:
        context.identity[name] = backend_identity(bundle, name, backend, level).hex()
    return context


def ir_file_backend_context(  # type: ignore[no-untyped-def]
    project: Project, backend, module, *, opt_level: int, entry: Path | None = None
):
    """A `BackendContext` for IR read from a `.ppyir` file.

    There is no analysis bundle behind it -- the file is the whole input --
    so the artifact's identity is the module's own encoded bytes with
    everything that identifies the compiler and the backend, which is what
    the bundle's key carries for a source build.
    """
    from ..backend.base import BackendContext
    from ..ir import IR_SCHEMA_VERSION, encode

    config = project.config
    settings = config.backend(backend.name)
    distribution = backend.distribution or ("", "")
    identity = digest(
        f"{COMPILER_VERSION}+{compiler_fingerprint()}",
        IR_SCHEMA_VERSION,
        encode(module),
        backend.name,
        backend.api_version,
        f"{distribution[0]} {distribution[1]}",
        backend.fingerprint(),
        settings.fingerprint(),
        settings.get("target", ""),
        opt_level,
        environment_fingerprint(),
    )
    return BackendContext(
        root=project.root,
        config=config,
        backend_config=settings,
        opt_level=opt_level,
        target=str(settings.get("target", "")),
        registry=project.plugins.dialect_registry(),
        plugin_fingerprints=project.plugins.fingerprints(),
        identity={module.name: identity},
        entry=entry,
    )


def _prover_descriptor(config) -> str:  # type: ignore[no-untyped-def]
    """Which prover shaped the lowering, and which version of it.

    A proof is the solver's word, and a different solver may keep a guard
    this one dropped; the artifact is keyed by both.
    """
    prover = config.llvm.prover or "off"
    if prover == "off":
        return "prover:off"
    from ..backend.llvm.prover import Prover

    return f"prover:{prover}:{Prover.version()}"


def _target_descriptor(config) -> str:  # type: ignore[no-untyped-def]
    """What the emitted object code was compiled for.

    A baseline object and a host-specific one are different artifacts, and a
    host object built on one machine must not be reused on another, so the
    CPU and its feature set are part of the key when `--host-cpu` is on.
    """
    if not config.llvm.host_cpu:
        return "baseline"
    try:
        from ..backend.llvm.jit import host_target

        cpu, features = host_target()
    except Exception:  # noqa: BLE001 - no LLVM means nothing host-specific ran
        return "host:unknown"
    return f"host:{cpu}:{hashlib.sha256(features.encode()).hexdigest()[:16]}"


def _imported_modules(symbols) -> set[str]:  # type: ignore[no-untyped-def]
    """Every module name this module imports, however it was spelled."""
    names = {binding.module for binding in symbols.imports.values()}
    names |= {binding.canonical for binding in symbols.imports.values()}
    return names


def _public_abi_hash(bundle: AnalysisBundle, module_name: str) -> str:
    """Separate type and effect hashes for each exported symbol (spec 27.5)."""
    from ..cache import digest

    symbols = bundle.symbols.modules.get(module_name)
    if symbols is None:
        return f"missing:{module_name}"
    parts: list[str] = []
    for name, info in sorted(symbols.functions.items()):
        parts.append(f"{name}|{info.signature()}|{info.effects}|{int(info.verified_pure)}")
    for name, cls in sorted(symbols.classes.items()):
        fields = ",".join(f"{f}:{t}" for f, t in sorted(cls.fields.items()))
        parts.append(f"class {name}|{fields}|{cls.mro}")
    for name, declared in sorted(symbols.globals.items()):
        parts.append(f"global {name}|{declared}")
    return digest(module_name, tuple(parts))


def build_python(
    bundle: AnalysisBundle,
    *,
    opt_level: int | None = None,
    target: str = "python",
    fusion: dict[str, dict[tuple[int, int], object]] | None = None,
    adjustments: dict[str, dict[tuple[int, int], object]] | None = None,
) -> BuildOutput:
    """Optimize every module and publish generated Python to the cache.

    `target` separates artifacts built for a different backend, because the
    LLVM path rewrites fused library expressions the Python path leaves alone.
    """
    from ..backend.python.emit import GeneratedModule, emit

    project = bundle.project
    level = opt_level if opt_level is not None else project.config.opt_level
    fusion = fusion or {}
    adjustments = adjustments or {}
    project.store.ensure()

    generated: dict[str, GeneratedModule] = {}
    remarks: list[Diagnostic] = []
    stats: dict[str, int] = {}

    for module in bundle.graph.order():
        module_analysis = bundle.analysis.modules.get(module.name)
        symbols = bundle.symbols.modules.get(module.name)
        if module_analysis is None or symbols is None:
            continue
        plan = fusion.get(module.name, {})
        tweaks = adjustments.get(module.name, {})
        key = module_cache_key(
            bundle,
            module.name,
            target=target,
            opt_level=level,
            extra=(
                tuple(sorted(str(k) for k in plan)),
                tuple(sorted(f"{k}:{v.reason}" for k, v in tweaks.items())),
            ),
        )
        cached = project.store.read_text(key)
        if cached is not None:
            line_map = _load_line_map(project.store, key)
            generated[module.name] = GeneratedModule(
                name=module.name,
                source_path=module.path,
                code=cached,
                artifact=project.store.get(key) or module.path,
                key=key.hex(),
                line_map=line_map,
                fused_symbols=_load_fused(project.store, key),
            )
            stats["cache_hits"] = stats.get("cache_hits", 0) + 1
            continue

        result: OptimizationResult = Optimizer(
            symbols,
            module_analysis,
            bundle.analysis,
            level=level,
            fusion=plan,
            adjustments=tweaks,
        ).run()
        dependencies = tuple(
            module_cache_key(bundle, edge.target, target=target, opt_level=level).hex()
            for edge in module.imports
            if edge.target in bundle.symbols.modules
        )
        generated[module.name] = emit(
            module.name, module.path, result, project.store, key, dependencies=dependencies
        )
        project.store.mark_root(key, module.name)
        remarks.extend(result.remarks)
        for name, value in result.stats.items():
            stats[name] = stats.get(name, 0) + value
        stats["cache_misses"] = stats.get("cache_misses", 0) + 1

    return BuildOutput(bundle=bundle, generated=generated, remarks=remarks, stats=stats)


def _load_line_map(store: CacheStore, key: CacheKey) -> dict[int, int]:
    return _load_metadata(store, key).get("lines", {})


def _load_fused(store: CacheStore, key: CacheKey) -> tuple[str, ...]:
    return tuple(_load_metadata(store, key).get("fused", ()))


def _load_metadata(store: CacheStore, key: CacheKey) -> dict:
    import json

    raw = store.read_text(f"{key.hex()}.map")
    if raw is None:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    lines = {int(k): int(v) for k, v in payload.get("lines", {}).items()}
    return {
        "lines": lines,
        "fused": payload.get("fused", []),
    }
