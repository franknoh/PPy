"""`ppy emit KIND TARGET`: a compiler stage as text, and `ppy build X.ppyir`.

One rule for every kind: a single file with no `-o` goes to standard
output, `-o FILE` writes that file, and a directory target writes one file
per module into the directory `-o` names. `ir` is the canonical IR after
the shared passes; `llvm-ir` is what the LLVM backend makes of it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..diagnostics import Diagnostic, Severity
from .pipeline import analyze_paths, collect_sources, open_project
from .reporting import Reporter

__all__ = ["KINDS", "build_ir_file", "run_emit"]

KINDS = ("ir", "llvm-ir")
_SUFFIXES = {"ir": ".ppyir", "llvm-ir": ".ll"}


def run_emit(options: argparse.Namespace, reporter: Reporter) -> int:
    target: Path = options.target
    if not target.exists():
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{target} does not exist"))
        return 2
    output: Path | None = options.output
    if target.is_dir() and output is None:
        reporter.emit(
            Diagnostic("E1002", Severity.ERROR, "emitting a directory needs `-o DIR` to write into")
        )
        return 2
    project = open_project(target)
    bundle = analyze_paths(project, collect_sources(target), backend="llvm")
    errors = reporter.report(bundle.diagnostics)
    if errors:
        reporter.summary(errors, 0)
        return 1
    try:
        texts = _texts(options.kind, bundle)
    except Exception as error:  # noqa: BLE001 - the backend's refusal is the message
        reporter.emit(Diagnostic("E1801", Severity.ERROR, str(error)))
        return 2
    if not texts:
        reporter.emit(Diagnostic("E1002", Severity.ERROR, "nothing to emit: no module lowered"))
        return 1
    suffix = _SUFFIXES[options.kind]
    if target.is_file():
        text = "\n".join(texts.values())
        if output is None:
            print(text, end="" if text.endswith("\n") else "\n")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
            reporter.note(f"wrote {output}")
        return 0
    assert output is not None
    output.mkdir(parents=True, exist_ok=True)
    for name, text in sorted(texts.items()):
        path = output / f"{name}{suffix}"
        path.write_text(text, encoding="utf-8")
        reporter.note(f"wrote {path}")
    return 0


def _texts(kind: str, bundle) -> dict[str, str]:  # type: ignore[no-untyped-def]
    from ..backend.llvm import emit_ir
    from ..backend.llvm.ir_pipeline import ir_modules
    from ..ir import encode

    if kind == "ir":
        return {name: encode(module) for name, module in ir_modules(bundle).items()}
    return emit_ir(bundle)


def build_ir_file(path: Path, options: argparse.Namespace, reporter: Reporter) -> int:
    """`ppy build foo.ppyir`: objects, a library, and a manifest from IR alone."""
    from ..backend.llvm import LlvmUnavailable
    from ..backend.llvm.from_ir import emit_module
    from ..backend.llvm.ir_pipeline import optimize
    from ..backend.llvm.jit import JitEngine, available
    from ..backend.llvm.link import (
        ToolchainError,
        emit_object,
        link_shared_library,
        write_manifest,
    )
    from ..ir import CodecError, read
    from ..lowering.abi import signature_from_ir

    try:
        module = read(path)
    except CodecError as error:
        reporter.emit(Diagnostic("E1801", Severity.ERROR, str(error)))
        return 2
    if not available():
        reporter.emit(Diagnostic("E1801", Severity.ERROR, "llvmlite is not installed"))
        return 2
    project = open_project(path)
    level = getattr(options, "opt_level", None) or project.config.opt_level
    try:
        optimize(module, level)
        text = emit_module(module)
    except Exception as error:  # noqa: BLE001 - the verifier's or the backend's refusal
        reporter.emit(Diagnostic("E1801", Severity.ERROR, str(error)))
        return 2
    output: Path = options.output or (project.config.cache_path / "native")
    output.mkdir(parents=True, exist_ok=True)
    signatures = {
        f.attributes.get("ppy.qualname", name): signature_from_ir(f)
        for name, f in module.functions.items()
        if not f.is_declaration
    }
    try:
        engine = JitEngine(opt_level=level).open()
        object_path = emit_object(
            engine, text, output / f"{module.name}.o", host_cpu=project.config.llvm.host_cpu
        )
        library = link_shared_library([object_path], output / f"libppy_{module.name}.so")
    except (LlvmUnavailable, ToolchainError) as error:
        reporter.emit(Diagnostic("E1801", Severity.ERROR, str(error)))
        return 2
    manifest = write_manifest(
        output / "ppy-bindings.json",
        signatures,  # type: ignore[arg-type]
        library=library,
        program={
            "entry": module.name,
            "modules": [module.name],
            "generated": {},
            "search_paths": [],
            "safeguards": project.config.llvm.safeguards or "off",
        },
    )
    reporter.note("objects:  1")
    reporter.note(f"library:  {library}")
    reporter.note(f"manifest: {manifest}")
    return 0
