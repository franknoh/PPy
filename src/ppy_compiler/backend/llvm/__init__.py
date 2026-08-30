"""LLVM backend driver (spec 16)."""

from __future__ import annotations

import ast
import contextlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ...cache import CacheKey
from ...diagnostics import Diagnostic, Severity, Span
from ..binder import LibraryBinder
from .fusion import (
    FusedLoop,
    FusionCandidate,
    find_candidates,
    find_module_candidates,
    lower_candidate,
)
from .jit import JitEngine, LlvmUnavailable, available, llvm_status
from .link import (
    BuildArtifacts,
    ToolchainError,
    build_launcher,
    emit_object,
    link_shared_library,
    toolchain_status,
    write_manifest,
)
from .lowering import (
    LoweredFunction,
    LoweringResult,
    NativeSignature,
    lower_module,
    should_lower_native,
)
from .specialize import SpecializationPolicy, Specializer
from .wrapper_build import build_wrappers

__all__ = [
    "BuildArtifacts",
    "LlvmUnavailable",
    "NativeModule",
    "ToolchainError",
    "available",
    "compile_and_run",
    "compile_project",
    "emit_ir",
    "llvm_status",
    "toolchain_status",
]


@dataclass(slots=True)
class NativeModule:
    name: str
    ir: str
    functions: dict[str, LoweredFunction] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)
    sources: dict[str, tuple] = field(default_factory=dict)
    fused: dict[str, FusedLoop] = field(default_factory=dict)
    fusion_plan: dict[tuple[int, int], FusedLoop] = field(default_factory=dict)
    fusion_notes: list[tuple[int, str]] = field(default_factory=list)


#: Members that make attribute reads observable, so the class stays boxed.
_INTERCEPTORS = {"__getattr__", "__getattribute__", "__setattr__", "__init_subclass__"}


def _definitions(tree: ast.Module):  # type: ignore[no-untyped-def]
    """Every function the backend may lower, with its owning class if any."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield "", node
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield node.name, child


def _value_class_layouts(bundle) -> dict[str, tuple[tuple[str, str], ...]]:  # type: ignore[no-untyped-def]
    """Classes whose instances can be flattened into scalar arguments.

    A value class here is one with a fixed set of scalar fields and no base
    beyond `object`: nothing about it needs a Python object to represent
    (spec 13.2, 25.4).
    """
    from ...analysis import types as T

    scalars = {"int", "float", "bool"}
    layouts: dict[str, tuple[tuple[str, str], ...]] = {}
    for qualname, info in bundle.symbols.classes.items():
        if info.is_protocol or info.is_enum or info.is_pydantic:
            continue
        if tuple(entry for entry in info.mro if entry != "object") != (qualname,):
            continue
        # Reading a field must be a plain attribute read: anything that can
        # intercept it could observe the flattening.
        if _INTERCEPTORS & set(info.methods):
            continue
        if set(info.fields) & set(info.methods):
            continue
        fields: list[tuple[str, str]] = []
        for name, declared in info.fields.items():
            if name in info.class_vars:
                continue
            base = T.strip_literal(declared)
            if not isinstance(base, T.Instance) or base.name not in scalars:
                fields = []
                break
            fields.append((name, base.name))
        if fields:
            layouts[qualname] = tuple(fields)
    return layouts


def _collect(bundle, opt_level: int | None = None) -> dict[str, NativeModule]:  # type: ignore[no-untyped-def]
    """Lower every module in the project to LLVM IR, reusing a cached result."""
    modules: dict[str, NativeModule] = {}
    layouts = _value_class_layouts(bundle)
    for module in bundle.graph.order():
        analysis = bundle.analysis.modules.get(module.name)
        symbols = bundle.symbols.modules.get(module.name)
        if analysis is None or symbols is None:
            continue
        candidates: dict[str, tuple] = {}
        for owner, node in _definitions(module.tree):
            info = (
                symbols.classes[owner].methods.get(node.name)
                if owner and owner in symbols.classes
                else symbols.functions.get(node.name)
            )
            if info is None:
                continue
            function_analysis = analysis.functions.get(info.qualname)
            if function_analysis is None:
                continue
            candidates[info.qualname] = (info, function_analysis, node)

        reused = _cached_lowering(bundle, module.name, opt_level)
        if reused is not None:
            modules[module.name] = _module_from_cache(module.name, reused, candidates)
            continue

        fused, plan, notes = _fuse(symbols, analysis)
        if not candidates and not fused:
            continue
        result: LoweringResult = lower_module(
            analysis,
            candidates,
            layouts,
            safeguards=bundle.project.config.llvm.safeguards or "hoisted",
        )
        ir_text = result.ir
        if fused:
            ir_text = _append_fused(ir_text, module.name, fused)
        native = NativeModule(
            name=module.name,
            ir=ir_text,
            functions=result.functions,
            rejected=result.rejected,
            sources={
                qualname: (info, node)
                for qualname, (info, _analysis, node) in candidates.items()
                if qualname in result.functions
            },
            fused=fused,
            fusion_plan=plan,
            fusion_notes=notes,
        )
        modules[module.name] = native
        _store_lowering(bundle, module.name, opt_level, native)
    return modules


def _lowering_key(bundle, name: str, opt_level: int | None) -> str:  # type: ignore[no-untyped-def]
    from ...driver.pipeline import module_cache_key

    level = opt_level if opt_level is not None else bundle.project.config.opt_level
    return f"{module_cache_key(bundle, name, target='llvm', opt_level=level).hex()}.lowered"


def _cached_lowering(bundle, name: str, opt_level: int | None):  # type: ignore[no-untyped-def]
    """What lowering produced last time, when the key still describes it."""
    from .lowering_cache import decode

    try:
        text = bundle.project.store.read_text(_lowering_key(bundle, name, opt_level))
    except Exception:  # noqa: BLE001 - a cache miss must never fail a build
        return None
    return decode(text) if text is not None else None


def _store_lowering(bundle, name: str, opt_level: int | None, native: NativeModule) -> None:  # type: ignore[no-untyped-def]
    from .lowering_cache import encode

    # Caching is an optimization, not a contract; failing to store is fine.
    with contextlib.suppress(Exception):
        key = _lowering_key(bundle, name, opt_level)
        bundle.project.store.put(key, encode(native), kind="llvm", source=name, suffix=".json")
        bundle.project.store.mark_root(key, f"lowered:{name}")


def _module_from_cache(name: str, reused, candidates) -> NativeModule:  # type: ignore[no-untyped-def]
    """Rebuild a `NativeModule` from cached ABI decisions plus live symbols.

    The `FunctionInfo` and the AST are taken from this run's analysis rather
    than from the cache: they are cheap to have already, and reusing the live
    ones keeps diagnostics pointing at the current source.
    """
    functions: dict[str, LoweredFunction] = {}
    sources: dict[str, tuple] = {}
    for qualname, signature in reused.signatures.items():
        entry = candidates.get(qualname)
        if entry is None:
            # The cached module no longer matches the source in front of us.
            return NativeModule(name=name, ir="")
        info, _analysis, node = entry
        # Profitability is a pure function of today's source, so a cached
        # module answers it fresh rather than trusting yesterday's verdict.
        exposed, why = should_lower_native(info, _analysis)
        functions[qualname] = LoweredFunction(info, signature, exposed=exposed, exposure_reason=why)
        sources[qualname] = (info, node)
    return NativeModule(
        name=name,
        ir=reused.ir,
        functions=functions,
        rejected=dict(reused.rejected),
        sources=sources,
        fused=dict(reused.fused),
        fusion_plan=dict(reused.plan),
        fusion_notes=list(reused.notes),
    )


def _fuse(symbols, analysis):  # type: ignore[no-untyped-def]
    """Collect the fusible library expressions in one module (spec 19.4)."""
    loops: dict[str, FusedLoop] = {}
    plan: dict[tuple[int, int], FusedLoop] = {}
    notes: list[tuple[int, str]] = []
    functions = list(symbols.functions.values())
    for cls in symbols.classes.values():
        functions.extend(cls.methods.values())

    candidates = list(find_module_candidates(symbols.module.tree, analysis))
    for info in functions:
        candidates.extend(find_candidates(info, analysis))

    for candidate in candidates:
        loops[candidate.loop.symbol] = candidate.loop
        plan[(candidate.node.lineno, candidate.node.col_offset)] = candidate.loop
        notes.append(
            (
                candidate.node.lineno,
                f"NumPy expression fused into one strided loop: {', '.join(candidate.operations)}",
            )
        )
    return loops, plan, notes


def _append_fused(ir_text: str, module_name: str, fused: dict[str, FusedLoop]) -> str:
    """Emit the generated kernels into their own LLVM module."""
    from llvmlite import ir as llvm_ir

    kernels = llvm_ir.Module(name=f"{module_name}.fused")
    for loop in fused.values():
        lower_candidate(llvm_ir, kernels, FusionCandidate(function=None, node=None, loop=loop))
    if not ir_text.strip():
        return str(kernels)
    return ir_text + "\n" + _body_only(str(kernels))


def _body_only(text: str) -> str:
    """Drop the module header so two LLVM modules can be concatenated."""
    lines = [
        line
        for line in text.splitlines()
        if not line.startswith(
            ("; ModuleID", "source_filename", "target triple", "target datalayout")
        )
    ]
    return "\n".join(lines)


def emit_ir(bundle) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """Return optimized LLVM IR per module, for `ppy inspect --ir`."""
    if not available():
        raise LlvmUnavailable("llvmlite is not installed, so no LLVM IR can be produced")
    engine = JitEngine(opt_level=bundle.project.config.opt_level)
    return {name: engine.optimized_ir(module.ir) for name, module in _collect(bundle).items()}


def _library_key(objects: list[Path]) -> str:
    """A key over exactly the object files that go into the library."""
    from ...cache import digest

    return digest("ppy-library", *(o.read_bytes().hex() for o in sorted(objects))) + ".so"


def _link_and_cache(artifacts, store, key: str, destination: Path) -> None:  # type: ignore[no-untyped-def]
    try:
        artifacts.library = link_shared_library(artifacts.objects, destination)
    except ToolchainError as exc:
        artifacts.notes.append(str(exc))
        return
    try:
        store.put(key, artifacts.library.read_bytes(), kind="native", suffix=".so")
        store.mark_root(key, "library")
    except Exception:  # noqa: BLE001 - caching is an optimization
        return


def _object_key(key: CacheKey) -> str:
    """The key of the object file compiled from the module this key names."""
    return f"{key.hex()}.o"


def compile_project(  # type: ignore[no-untyped-def]
    bundle,
    reporter,
    *,
    opt_level: int | None = None,
    output: Path | None = None,
    entry: Path | None = None,
) -> BuildArtifacts:
    """Compile to LLVM, emit object code, link, and write the manifest (spec 4.2)."""
    from ...driver.pipeline import build_python, module_cache_key
    from ...opt.rewrites import adjustments_for_project

    level = opt_level if opt_level is not None else bundle.project.config.opt_level
    store = bundle.project.store
    store.ensure()

    # Opening the engine initializes LLVM, which is most of a warm build's
    # cost. Nothing needs it when every object comes from the cache.
    opened: list[JitEngine] = []

    def engine() -> JitEngine:
        if not opened:
            # Availability is asserted here rather than up front, so a build
            # that answers entirely from the cache needs no LLVM at all.
            if not available():
                raise LlvmUnavailable(
                    "llvmlite is not installed, so the LLVM backend is unavailable"
                )
            opened.append(JitEngine(opt_level=level).open())
        return opened[0]

    natives = _collect(bundle, level)
    build_directory = output or (bundle.project.config.cache_path / "native")
    artifacts = BuildArtifacts()
    signatures: dict[str, NativeSignature] = {}
    fused: dict[str, tuple[int, int]] = {}

    for name, native in natives.items():
        key: CacheKey = module_cache_key(bundle, name, target="llvm", opt_level=level)
        if store.get(key) is None:
            store.put(key, engine().optimized_ir(native.ir), kind="llvm", source=name, suffix=".ll")
        store.mark_root(key, f"llvm:{name}")
        _report(native, reporter, bundle)

        if not native.functions and not native.fused:
            continue
        destination = build_directory / f"{name.replace('.', '_')}.o"
        # The object depends on exactly what the module key covers, so a hit
        # means the previous one is still correct and running the optimizer and
        # the code generator again would produce the same bytes.
        cached = store.read(_object_key(key))
        if cached is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(cached)
            artifacts.objects.append(destination)
            artifacts.reused.append(name)
        else:
            try:
                emitted = emit_object(engine(), native.ir, destination)
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                artifacts.notes.append(f"could not emit object code for {name}: {exc}")
                continue
            store.put(
                _object_key(key), emitted.read_bytes(), kind="native", source=name, suffix=".o"
            )
            store.mark_root(_object_key(key), f"object:{name}")
            artifacts.objects.append(emitted)
        for lowered in native.functions.values():
            if lowered.exposed:
                signatures[lowered.signature.qualname] = lowered.signature
        for symbol, loop in native.fused.items():
            fused[symbol] = (len(loop.arrays), len(loop.scalars))

    # The generated Python is part of the build output: it is what the launcher
    # executes, and what binds the native entry points at import time.
    build_python(
        bundle,
        opt_level=level,
        target="llvm",
        fusion={name: native.fusion_plan for name, native in natives.items()},
        adjustments=adjustments_for_project(bundle),
    )

    if artifacts.objects:
        library_key = _library_key(artifacts.objects)
        destination = build_directory / f"libppy_{bundle.project.root.name}.so"
        cached_library = store.read(library_key)
        if cached_library is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(cached_library)
            artifacts.library = destination
            artifacts.reused.append("link")
        else:
            _link_and_cache(artifacts, store, library_key, destination)
    generated_dir = build_directory / "generated"
    program: dict | None = None
    entry_module = None
    if entry is not None:
        for name, symbols in bundle.symbols.modules.items():
            if symbols.path == entry.resolve():
                entry_module = name
                break
    if entry_module is not None:
        output = build_python(
            bundle,
            opt_level=level,
            target="llvm",
            fusion={name: native.fusion_plan for name, native in natives.items()},
            adjustments=adjustments_for_project(bundle),
        )
        generated_dir.mkdir(parents=True, exist_ok=True)
        listed: dict[str, str] = {}
        for name, module in output.generated.items():
            payload = {
                "name": module.name,
                "source": str(module.source_path),
                "code": module.code,
                "artifact": str(module.artifact),
                "key": module.key,
                "line_map": {str(k): v for k, v in module.line_map.items()},
                "fused_symbols": list(module.fused_symbols),
            }
            filename = f"{name}.json"
            (generated_dir / filename).write_text(json.dumps(payload, indent=1), encoding="utf-8")
            listed[name] = filename
        program = {
            "entry": entry_module,
            "modules": sorted(output.generated),
            "generated": listed,
            "search_paths": [str(path) for path in bundle.project.search_paths],
            "safeguards": bundle.project.config.llvm.safeguards or "hoisted",
        }
    wrapper_section = None
    if signatures:
        wrappers = build_wrappers(
            bundle.project.root.name,
            signatures,
            bundle.project.config.cache_path,
            notify=reporter.note,
        )
        if wrappers.ok and wrappers.path is not None:
            shipped = build_directory / wrappers.path.name
            if shipped != wrappers.path:
                shipped.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(wrappers.path, shipped)
            wrapper_section = {"library": shipped.name, "entries": dict(wrappers.entries)}
        elif wrappers.reason:
            artifacts.notes.append(f"the artifact uses the slower boundary: {wrappers.reason}")
    artifacts.manifest = write_manifest(
        build_directory / "ppy-bindings.json",
        signatures,
        library=artifacts.library,
        fused=fused,
        program=program,
        wrappers=wrapper_section,
    )

    if entry is not None:
        try:
            artifacts.launcher = build_launcher(
                entry,
                build_directory / entry.stem,
                bundle.project.search_paths,
                artifacts.manifest,
            )
        except ToolchainError as exc:
            artifacts.notes.append(f"no native launcher: {exc}")
    return artifacts


def compile_and_run(  # type: ignore[no-untyped-def]
    bundle, program_args, reporter, *, opt_level: int | None = None
) -> int:
    """`ppy run`: JIT-compile the native subset, then execute with CPython as host."""
    if not available():
        raise LlvmUnavailable("llvmlite is not installed, so the LLVM backend is unavailable")

    from ...backend.python.runner import execute, format_traceback
    from ...driver.pipeline import build_python
    from ...driver.staging import compile_torch_regions, stage_project
    from ...opt.rewrites import adjustments_for_project

    level = opt_level if opt_level is not None else bundle.project.config.opt_level
    natives = _collect(bundle, level)
    output = build_python(
        bundle,
        opt_level=level,
        target="llvm",
        fusion={name: native.fusion_plan for name, native in natives.items()},
        adjustments=adjustments_for_project(bundle),
    )

    engine = JitEngine(opt_level=level).open()
    for native in natives.values():
        if native.functions or native.fused:
            engine.add(native.ir)
    engine.finalize()

    binder = _Binder(
        threads=bundle.project.config.parallel.threads
        if bundle.project.config.parallel.enabled
        else 1
    )
    layouts = _value_class_layouts(bundle)
    for name, native in natives.items():
        analysis = bundle.analysis.modules.get(name)
        specializer = None
        if analysis is not None and native.sources:
            specializer = Specializer(
                engine=engine,
                module_analysis=analysis,
                cache=bundle.project.store,
                cache_directory=bundle.project.config.cache_path / "jit",
                layouts=layouts,
                safeguards=bundle.project.config.llvm.safeguards or "hoisted",
            )
            for info, node in native.sources.values():
                specializer.register(info, node)
        wrappers = build_wrappers(
            name,
            {q: lowered.signature for q, lowered in native.functions.items()},
            bundle.project.config.cache_path,
            notify=reporter.note,
        )
        if not wrappers.ok and wrappers.reason and native.functions:
            reporter.emit(
                Diagnostic(
                    "W2004",
                    Severity.WARNING,
                    f"module `{name}` uses the slower boundary: {wrappers.reason}",
                )
            )

        for qualname, lowered in native.functions.items():
            if not lowered.exposed:
                # Native callers reach the symbol directly; Python callers
                # keep the Python body, because the boundary would cost more
                # than the body saves (spec: native profitability).
                if bundle.project.config.diagnostics.optimization_remarks:
                    reporter.emit(
                        Diagnostic(
                            "R3004",
                            Severity.REMARK,
                            f"`{qualname}` stays on the Python boundary: {lowered.exposure_reason}",
                        )
                    )
                continue
            address = engine.address(lowered.signature.symbol)
            if not address:
                continue
            binder.add(
                name,
                _binding_name(lowered.info),
                lowered.signature,
                address,
                specializer=specializer,
                info=lowered.info,
                wrappers=wrappers,
                qualname=qualname,
                engine=engine,
            )
        for symbol, loop in native.fused.items():
            address = engine.address(symbol)
            if address:
                binder.add_fused(name, loop, address)
        _report(native, reporter, bundle)

    regions = compile_torch_regions(bundle, notify=reporter.note)
    for module_name, entries in regions.compiled.items():
        for function, entry_point in entries.items():
            binder.add_region(module_name, function, entry_point)

    staged = stage_project(bundle)
    for module_name, entries in staged.artifacts.items():
        for function, artifact in entries.items():
            binder.add_exported(module_name, function, artifact.payload)
    if bundle.project.config.diagnostics.optimization_remarks:
        for remark in (*regions.diagnostics, *staged.diagnostics):
            reporter.emit(remark)

    entry = output.generated.get(bundle.entry or "")
    if entry is None:
        reporter.emit(
            Diagnostic("E1002", Severity.ERROR, "no generated module for the entry point")
        )
        return 2

    result = execute(
        entry,
        output.generated,
        program_args,
        search_paths=bundle.project.search_paths,
        natives=binder,
        entry_name=bundle.entry,
    )
    if result.exception is not None:
        sys.stderr.write(format_traceback(result.exception))
    return result.exit_code


def _binding_name(info) -> str:  # type: ignore[no-untyped-def]
    """How a generated module names this entry point when it binds it."""
    if info.owner:
        return f"{info.owner.rpartition('.')[2]}.{info.name}"
    return info.name


class _Binder(LibraryBinder):
    """Serves guarded native entry points to generated modules as they load."""

    def __init__(self, threads: str | int = "auto") -> None:
        super().__init__()
        self.threads = threads
        self._entries: dict[str, dict[str, tuple]] = {}
        self._fused: dict[str, dict[str, tuple]] = {}
        self.bindings: list = []
        self.fused_bindings: list = []

    def add(  # type: ignore[no-untyped-def]
        self,
        module: str,
        function: str,
        signature,
        address: int,
        specializer=None,
        info=None,
        wrappers=None,
        qualname: str = "",
        engine=None,
    ) -> None:
        self._entries.setdefault(module, {})[function] = (
            signature,
            address,
            specializer,
            info,
            wrappers,
            qualname,
            engine,
        )

    def add_fused(self, module: str, loop, address: int) -> None:  # type: ignore[no-untyped-def]
        self._fused.setdefault(module, {})[loop.symbol] = (loop, address)

    def names(self, module: str) -> frozenset[str]:
        return frozenset(self._entries.get(module, {}))

    def fused(self, module: str, symbol: str, fallback):  # type: ignore[no-untyped-def]
        from .fused_runtime import bind_fused

        entry = self._fused.get(module, {}).get(symbol)
        if entry is None:
            return fallback
        loop, address = entry
        binding = bind_fused(loop, address, fallback, parallel=loop.parallel, threads=self.threads)
        self.fused_bindings.append(binding)
        return binding.wrapper

    def bind(self, module: str, function: str, fallback):  # type: ignore[no-untyped-def]
        from ppy_runtime.binding import adopt, observation_wanted, value_class_types

        from .runtime import bind as make_binding

        entry = self._entries.get(module, {}).get(function)
        if entry is None or not callable(fallback):
            return fallback
        signature, address, specializer, info, wrappers, qualname, engine = entry
        policy = SpecializationPolicy.of(info) if info is not None else None
        fast_entry = None
        register = None
        if wrappers is not None and wrappers.ok:
            types = value_class_types(signature, fallback)
            if types is not None:
                register = wrappers.registrar(qualname)
                if not (observation_wanted(specializer, policy, info) and register is not None):
                    # Nothing to watch for: the wrapper holds the fallback in C
                    # and no Python frame stands on the call path at all.
                    direct = wrappers.bind(qualname, address, types, fallback)
                    if direct is not None:
                        binding = adopt(signature, direct, fallback, owner=(engine, wrappers))
                        self.bindings.append(binding)
                        return direct
                fast_entry = wrappers.bind(qualname, address, types)
        binding = make_binding(
            signature,
            address,
            fallback,
            specializer=specializer,
            policy=policy,
            info=info,
            fast_entry=fast_entry,
            owner=(engine, wrappers),
            register=register,
        )
        self.bindings.append(binding)
        return binding.wrapper


def _report(native: NativeModule, reporter, bundle) -> None:  # type: ignore[no-untyped-def]
    if not bundle.project.config.diagnostics.optimization_remarks:
        return
    symbols = bundle.symbols.modules.get(native.name)
    path = symbols.path if symbols else Path(native.name)
    for qualname, lowered in sorted(native.functions.items()):
        reporter.emit(
            Diagnostic(
                "R3001",
                Severity.REMARK,
                f"`{qualname}` lowered natively as {lowered.signature}",
                Span(path, lowered.info.node.lineno, 0),
            )
        )
    for line, message in native.fusion_notes:
        reporter.emit(Diagnostic("R3001", Severity.REMARK, message, Span(path, line, 0)))
    for qualname, reason in sorted(native.rejected.items()):
        info = bundle.symbols.functions.get(qualname)
        reporter.emit(
            Diagnostic(
                "R3001",
                Severity.REMARK,
                f"`{qualname}` stays boxed: {reason}",
                Span(path, info.node.lineno if info else 1, 0),
            )
        )
