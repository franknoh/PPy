"""The LLVM backend's road: canonical IR -> shared passes -> LLVM IR.

`lower_module_via_ir` answers, for one module, which functions are native
-- an LLVM module for every eligible one, and a reason for every other --
by way of the canonical IR: the frontend in `ppy_compiler.lowering`, the
shared passes in `ppy_compiler.driver.ir_pipeline` (the same road every
backend takes), and `from_ir`, which reads the IR and nothing else. It is
the only road; the direct AST-to-LLVM lowering 0.2.0 started with was
removed once this one covered everything it did.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ...analysis.checker import FunctionAnalysis, ModuleAnalysis
from ...analysis.symbols import FunctionInfo
from ...driver.ir_pipeline import (
    canonical_ir_modules,
    optimize_shared_ir,
    parallel_pass,
)
from ...ir import encode
from .from_ir import emit_module
from .lowering import ClassLayouts, LoweredFunction, LoweringResult, Unsupported
from .prover import Prover

__all__ = [
    "ir_modules",
    "lower_module_via_ir",
    "lower_specialization_via_ir",
    "optimize",
    "parallel_pass",
]

#: The shared road under its old names, for the code that learned them here.
ir_modules = canonical_ir_modules
optimize = optimize_shared_ir


def lower_module_via_ir(
    module: ModuleAnalysis,
    functions: dict[str, tuple[FunctionInfo, FunctionAnalysis, ast.FunctionDef]],
    layouts: ClassLayouts | None = None,
    *,
    safeguards: str = "hoisted",
    standalone: bool = False,
    opt_level: int = 2,
    prover: Prover | None = None,
    root: Path | None = None,
    plugins=None,  # type: ignore[no-untyped-def]
    target=None,  # type: ignore[no-untyped-def]
    parallel=None,  # type: ignore[no-untyped-def]
    imports=None,  # type: ignore[no-untyped-def]
    sanitize=(),  # type: ignore[no-untyped-def]
    instrument: bool = False,
    profile=None,  # type: ignore[no-untyped-def]
) -> LoweringResult:
    from ...lowering import lower_module_to_ir

    lowered = lower_module_to_ir(
        module,
        functions,
        layouts,
        cpu_compatible=True,
        safeguards=safeguards,
        standalone=standalone,
        prover=prover,
        root=root,
        imports=imports,
        plugins=plugins,
    )
    ctx = optimize(
        lowered.module,
        opt_level,
        plugins,
        parallel,
        sanitize=sanitize,
        instrument=instrument,
        profile=profile,
    )
    _reject_external_operations(lowered)
    text = emit_module(lowered.module, target) if lowered.functions else ""
    libraries = lowered.module.attributes.get("ppy.libraries", ())
    exports = {
        str(f.attributes["ppy.export"]): str(f.attributes.get("ppy.qualname", name))
        for name, f in lowered.module.functions.items()
        if "ppy.export" in f.attributes
    }
    return LoweringResult(
        ir=text,
        ppyir=encode(lowered.module, ctx.registry) if lowered.functions else "",
        functions={
            name: LoweredFunction(
                entry.info,
                entry.signature.native,
                exposed=entry.exposed,
                exposure_reason=entry.exposure_reason,
            )
            for name, entry in lowered.functions.items()
            if entry.signature.native is not None
        },
        rejected=lowered.rejected,
        proved=lowered.proved,
        libraries=tuple(str(lib) for lib in libraries),  # type: ignore[union-attr]
        exports=exports,
        remarks=(*lowered.remarks, *ctx.remarks),
    )


def _reject_external_operations(lowered) -> None:  # type: ignore[no-untyped-def]
    """Keep unsupported plugin operations on Python after shared lowering.

    Plugin passes may convert their dialect to builtin operations. Surviving
    external operations have no LLVM emitter; reject those functions and all
    callers before emission, preserving unrelated native functions.
    """
    from ...ir import DialectRegistry, SymbolRef
    from ...ir.dialects import builtin_dialects

    registry = DialectRegistry()
    for dialect in builtin_dialects():
        registry.register(dialect)
    rejected: dict[str, str] = {}
    references: dict[str, set[str]] = {}

    def symbols(value: object) -> set[str]:
        if isinstance(value, SymbolRef):
            return {value.name}
        if isinstance(value, dict):
            return set().union(*(symbols(item) for item in value.values()))
        if isinstance(value, (tuple, list)):
            return set().union(*(symbols(item) for item in value))
        return set()

    for name, function in lowered.module.functions.items():
        references[name] = symbols(function.attributes)
        for op in function.operations():
            references[name].update(symbols(op.attributes))
            if registry.op_spec(op.name) is None:
                rejected[name] = f"operation `{op.name}` has no LLVM lowering"
    while True:
        blocked = {
            name: f"callee `{min(callees & rejected.keys())}` has no LLVM lowering"
            for name, callees in references.items()
            if name not in rejected and callees & rejected.keys()
        }
        if not blocked:
            break
        rejected.update(blocked)
    for qualname in list(lowered.functions):
        name = qualname.replace(".", "_")
        if name in rejected:
            lowered.rejected[qualname] = rejected[name]
            del lowered.functions[qualname]
    for name in rejected:
        lowered.module.functions.pop(name, None)


def lower_specialization_via_ir(
    module: ModuleAnalysis,
    info: FunctionInfo,
    node: ast.FunctionDef,
    constants: dict[str, object],
    symbol: str,
    layouts: ClassLayouts | None = None,
    safeguards: str = "hoisted",
    opt_level: int = 2,
    prover: Prover | None = None,
) -> str:
    """One specialized copy, the generic ABI, constants pinned in the body."""
    from ...lowering import lower_function

    try:
        ir_module = lower_function(
            module,
            info,
            node,
            constants=constants,
            cpu_compatible=True,
            symbol=symbol,
            layouts=layouts,
            safeguards=safeguards,
            prover=prover,
        )
    except Unsupported as error:
        raise Unsupported(str(error)) from error
    optimize(ir_module, opt_level)
    return emit_module(ir_module)
