"""The IR road through the LLVM backend: AST -> canonical IR -> passes -> LLVM.

`lower_module_via_ir` answers the same question `lowering.lower_module`
does -- an LLVM module for every eligible function, and a reason for every
other -- by way of the canonical IR. The two roads run side by side while
the IR one proves itself function family by function family; the
`pipeline` setting under `[tool.ppy.llvm]` (or `PPY_LOWERING` in the
environment) picks the road, and a differential run compares them.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

from ...analysis.checker import FunctionAnalysis, ModuleAnalysis
from ...analysis.symbols import FunctionInfo
from ...driver.config import PIPELINES, selected_pipeline
from ...ir import IRModule, PassContext, verify_or_raise
from ...ir.transforms import default_pipeline
from .from_ir import emit_module
from .lowering import ClassLayouts, LoweringResult, Unsupported
from .prover import Prover

__all__ = [
    "PIPELINES",
    "ir_modules",
    "lower_module_via_ir",
    "lower_specialization_via_ir",
    "optimize",
    "selected_pipeline",
]


def _verify_between_passes() -> bool:
    return bool(os.environ.get("PPY_IR_VERIFY"))


def optimize(module: IRModule, level: int) -> PassContext:
    """Verify the frontend's IR, run the shared passes, verify again."""
    verify_or_raise(module)
    ctx = PassContext(verify_after_each=_verify_between_passes())
    default_pipeline(level, ctx).run(module)
    verify_or_raise(module)
    return ctx


def ir_modules(bundle) -> dict[str, IRModule]:  # type: ignore[no-untyped-def]
    """The canonical IR of every module in the project, after the passes."""
    from ...lowering import lower_module_to_ir
    from . import _definitions, _value_class_layouts, prover_for

    config = bundle.project.config
    layouts = _value_class_layouts(bundle)
    modules: dict[str, IRModule] = {}
    for module in bundle.graph.order():
        analysis = bundle.analysis.modules.get(module.name)
        symbols = bundle.symbols.modules.get(module.name)
        if analysis is None or symbols is None:
            continue
        candidates: dict = {}
        for owner, node in _definitions(module.tree):
            info = (
                symbols.classes[owner].methods.get(node.name)
                if owner and owner in symbols.classes
                else symbols.functions.get(node.name)
            )
            if info is None:
                continue
            function_analysis = analysis.functions.get(info.qualname)
            if function_analysis is not None:
                candidates[info.qualname] = (info, function_analysis, node)
        if not candidates:
            continue
        lowered = lower_module_to_ir(
            analysis,
            candidates,
            layouts,
            safeguards=config.llvm.safeguards or "hoisted",
            prover=prover_for(config),
            root=bundle.project.root,
        )
        if not lowered.functions:
            continue
        optimize(lowered.module, config.opt_level)
        modules[module.name] = lowered.module
    return modules


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
    )
    optimize(lowered.module, opt_level)
    text = emit_module(lowered.module) if lowered.functions else ""
    return LoweringResult(
        ir=text, functions=lowered.functions, rejected=lowered.rejected, proved=lowered.proved
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
