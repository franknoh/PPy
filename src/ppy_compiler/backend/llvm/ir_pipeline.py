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
from .lowering import ClassLayouts, LoweringResult, Unsupported
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
        safeguards=safeguards,
        standalone=standalone,
        prover=prover,
        root=root,
        imports=imports,
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
    text = emit_module(lowered.module, target) if lowered.functions else ""
    libraries = lowered.module.attributes.get("ppy.libraries", ())
    exports = {
        str(f.attributes["ppy.export"]): str(f.attributes.get("ppy.qualname", name))
        for name, f in lowered.module.functions.items()
        if "ppy.export" in f.attributes
    }
    return LoweringResult(
        ir=text,
        ppyir=encode(lowered.module) if lowered.functions else "",
        functions=lowered.functions,
        rejected=lowered.rejected,
        proved=lowered.proved,
        libraries=tuple(str(lib) for lib in libraries),  # type: ignore[union-attr]
        exports=exports,
        remarks=(*lowered.remarks, *ctx.remarks),
    )


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
            symbol=symbol,
            layouts=layouts,
            safeguards=safeguards,
            prover=prover,
        )
    except Unsupported as error:
        raise Unsupported(str(error)) from error
    optimize(ir_module, opt_level)
    return emit_module(ir_module)
