"""`ppy inspect --stage`: the program as each stage of the compiler holds it (spec 80).

`analysis` is what the checker knows of every function; `ir` the frontend's
module before any pass; `canonical` after canonicalization; `tensor` after
fusion and before the tensor dialect lowers to loops (`columnar` the same
point, for the modules that hold columnar operations); `optimized` the
module a backend receives; `gpu` its device code alone; `stablehlo` and
`llvm` what those backends write.
"""

from __future__ import annotations

STAGES = (
    "analysis",
    "ir",
    "canonical",
    "optimized",
    "tensor",
    "columnar",
    "gpu",
    "stablehlo",
    "llvm",
)
#: Where the pipeline stops for a stage that is a point in it.
_STOPS = {"ir": "none", "canonical": "after-canonicalization", "tensor": "after-fusion"}
_STOPS["columnar"] = "after-fusion"


def stage_texts(bundle, stage: str) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """Each module's text at `stage`, by module name."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; the stages are {', '.join(STAGES)}")
    if stage == "analysis":
        return _analysis_texts(bundle)
    if stage == "llvm":
        from ..backend.llvm import emit_ir

        return emit_ir(bundle)
    if stage == "stablehlo":
        from .emit import _stablehlo_texts

        return _stablehlo_texts(bundle)
    from ..ir import encode, print_function
    from ..ir.dialects.gpu import kind_of
    from .ir_pipeline import canonical_ir_modules as ir_modules

    modules = ir_modules(bundle, launches=True, until=_STOPS.get(stage))
    texts: dict[str, str] = {}
    for name, module in modules.items():
        if stage == "gpu":
            device = [
                f
                for f in module.functions.values()
                if not f.is_declaration and kind_of(f) != "host"
            ]
            if device:
                texts[name] = "\n\n".join(print_function(f) for f in device) + "\n"
            continue
        if stage == "columnar" and not any(
            op.dialect in {"columnar", "arrow"}
            for f in module.functions.values()
            for op in f.operations()
        ):
            continue
        texts[name] = encode(module)
    return texts


def _analysis_texts(bundle) -> dict[str, str]:  # type: ignore[no-untyped-def]
    from ..backend.llvm.lowering import eligible, should_lower_native

    texts: dict[str, str] = {}
    for name, analysis in bundle.analysis.modules.items():
        symbols = bundle.symbols.modules.get(name)
        if symbols is None:
            continue
        lines = [f"module {name}"]
        for info in symbols.functions.values():
            function_analysis = analysis.functions.get(info.qualname)
            if function_analysis is None:
                continue
            ok, reason = eligible(info, function_analysis, allow_async=True, allow_launch=True)
            exposed, why = should_lower_native(info, function_analysis)
            lines.append(f"  {info.qualname}: {info.signature()}")
            lines.append(f"    effects: {function_analysis.effects}")
            lines.append(f"    native: {'eligible' if ok else 'stays in Python: ' + reason}")
            if ok:
                lines.append(
                    f"    boundary: {'bound' if exposed else 'native callers only'} ({why})"
                )
            gpu = next(
                (
                    f"{api} {kind}"
                    for api in ("cuda", "hip")
                    for kind in ("kernel", "device")
                    if info.directive(f"{api}.{kind}") is not None
                ),
                None,
            )
            if gpu:
                lines.append(f"    gpu: {gpu}")
            if info.directive("xla.jit") is not None:
                lines.append("    xla: marked for XLA")
            if info.is_async:
                lines.append("    async: a coroutine")
        texts[name] = "\n".join(lines) + "\n"
    return texts
