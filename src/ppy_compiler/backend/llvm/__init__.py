"""LLVM backend driver (spec 16)."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ...cache import CacheKey
from ...diagnostics import Diagnostic, Severity, Span
from .fusion import FusedLoop, FusionCandidate, find_candidates, lower_candidate
from .jit import JitEngine, LlvmUnavailable, available, llvm_status
from .lowering import LoweredFunction, LoweringResult, lower_module

__all__ = [
    "LlvmUnavailable",
    "llvm_status",
    "available",
    "compile_project",
    "compile_and_run",
    "emit_ir",
    "NativeModule",
]


@dataclass(slots=True)
class NativeModule:
    name: str
    ir: str
    functions: dict[str, LoweredFunction] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)
    fused: dict[str, FusedLoop] = field(default_factory=dict)
    fusion_plan: dict[tuple[int, int], FusedLoop] = field(default_factory=dict)
    fusion_notes: list[tuple[int, str]] = field(default_factory=list)


def _collect(bundle) -> dict[str, NativeModule]:  # type: ignore[no-untyped-def]
    """Lower every module in the project to LLVM IR."""
    modules: dict[str, NativeModule] = {}
    for module in bundle.graph.order():
        analysis = bundle.analysis.modules.get(module.name)
        symbols = bundle.symbols.modules.get(module.name)
        if analysis is None or symbols is None:
            continue
        candidates: dict[str, tuple] = {}
        for node in module.tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            info = symbols.functions.get(node.name)
            if info is None:
                continue
            function_analysis = analysis.functions.get(info.qualname)
            if function_analysis is None:
                continue
            candidates[info.qualname] = (info, function_analysis, node)
        fused, plan, notes = _fuse(symbols, analysis)
        if not candidates and not fused:
            continue
        result: LoweringResult = lower_module(analysis, candidates)
        ir_text = result.ir
        if fused:
            ir_text = _append_fused(ir_text, module.name, fused)
        modules[module.name] = NativeModule(
            name=module.name,
            ir=ir_text,
            functions=result.functions,
            rejected=result.rejected,
            fused=fused,
            fusion_plan=plan,
            fusion_notes=notes,
        )
    return modules


def _fuse(symbols, analysis):  # type: ignore[no-untyped-def]
    """Collect the fusible library expressions in one module (spec 19.4)."""
    loops: dict[str, FusedLoop] = {}
    plan: dict[tuple[int, int], FusedLoop] = {}
    notes: list[tuple[int, str]] = []
    functions = list(symbols.functions.values())
    for cls in symbols.classes.values():
        functions.extend(cls.methods.values())
    for info in functions:
        for candidate in find_candidates(info, analysis):
            loops[candidate.loop.symbol] = candidate.loop
            plan[(candidate.node.lineno, candidate.node.col_offset)] = candidate.loop
            notes.append(
                (
                    candidate.node.lineno,
                    f"NumPy expression fused into one strided loop: "
                    f"{', '.join(candidate.operations)}",
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
        line for line in text.splitlines()
        if not line.startswith(("; ModuleID", "source_filename", "target triple", "target datalayout"))
    ]
    return "\n".join(lines)


def emit_ir(bundle) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """Return optimized LLVM IR per module, for `ppy inspect --ir`."""
    if not available():
        raise LlvmUnavailable("llvmlite is not installed, so no LLVM IR can be produced")
    engine = JitEngine(opt_level=bundle.project.config.opt_level)
    return {name: engine.optimized_ir(module.ir) for name, module in _collect(bundle).items()}


def compile_project(bundle, reporter, *, opt_level: int | None = None) -> dict[str, Path]:  # type: ignore[no-untyped-def]
    """Compile to LLVM and publish IR artifacts to the cache (spec 4.2)."""
    if not available():
        raise LlvmUnavailable("llvmlite is not installed, so the LLVM backend is unavailable")
    from ...driver.pipeline import module_cache_key

    level = opt_level if opt_level is not None else bundle.project.config.opt_level
    store = bundle.project.store
    store.ensure()
    engine = JitEngine(opt_level=level)

    artifacts: dict[str, Path] = {}
    for name, native in _collect(bundle).items():
        key: CacheKey = module_cache_key(bundle, name, target="llvm", opt_level=level)
        optimized = engine.optimized_ir(native.ir)
        artifacts[name] = store.put(key, optimized, kind="llvm", source=name, suffix=".ll")
        store.mark_root(key, f"llvm:{name}")
        _report(native, reporter, bundle)
    return artifacts


def compile_and_run(bundle, program_args, reporter, *, opt_level: int | None = None) -> int:  # type: ignore[no-untyped-def]
    """`ppy run`: JIT-compile the native subset, then execute with CPython as host."""
    if not available():
        raise LlvmUnavailable("llvmlite is not installed, so the LLVM backend is unavailable")

    from ...backend.python.runner import execute, format_traceback
    from ...driver.pipeline import build_python
    from .runtime import bind

    level = opt_level if opt_level is not None else bundle.project.config.opt_level
    natives = _collect(bundle)
    output = build_python(
        bundle,
        opt_level=level,
        target="llvm",
        fusion={name: native.fusion_plan for name, native in natives.items()},
    )

    engine = JitEngine(opt_level=level).open()
    for native in natives.values():
        if native.functions or native.fused:
            engine.add(native.ir)
    engine.finalize()

    binder = _Binder()
    for name, native in natives.items():
        for qualname, lowered in native.functions.items():
            address = engine.address(lowered.signature.symbol)
            if not address:
                continue
            binder.add(name, qualname.rpartition(".")[2], lowered.signature, address)
        for symbol, loop in native.fused.items():
            address = engine.address(symbol)
            if address:
                binder.add_fused(name, loop, address)
        _report(native, reporter, bundle)

    entry = output.generated.get(bundle.entry or "")
    if entry is None:
        reporter.emit(Diagnostic("E1002", Severity.ERROR, "no generated module for the entry point"))
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


class _Binder:
    """Serves guarded native entry points to generated modules as they load."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, tuple]] = {}
        self._fused: dict[str, dict[str, tuple]] = {}
        self.bindings: list = []
        self.fused_bindings: list = []

    def add(self, module: str, function: str, signature, address: int) -> None:  # type: ignore[no-untyped-def]
        self._entries.setdefault(module, {})[function] = (signature, address)

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
        binding = bind_fused(loop, address, fallback)
        self.fused_bindings.append(binding)
        return binding.wrapper

    def bind(self, module: str, function: str, fallback):  # type: ignore[no-untyped-def]
        from .runtime import bind as make_binding

        entry = self._entries.get(module, {}).get(function)
        if entry is None or not callable(fallback):
            return fallback
        signature, address = entry
        binding = make_binding(signature, address, fallback)
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
