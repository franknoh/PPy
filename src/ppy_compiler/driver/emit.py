"""`ppy emit KIND TARGET`: a compiler stage as text, and `ppy build X.ppyir`.

One rule for every kind: a single file with no `-o` goes to standard
output, `-o FILE` writes that file, and a directory target writes one file
per module into the directory `-o` names. `ir` is the canonical IR after
the shared passes; `llvm-ir` is what the LLVM backend makes of it; `c` and
`cpp` are what the C backend makes of it, a translation unit each, or with
`--header-only` a header that carries the functions inline; `header` is
the C declarations of the module's exports. Any other kind is a format an
installed backend registers (`ppy_compiler.backend`): the backend is
loaded, handed the canonical IR after the shared passes and its own, and
asked for the format, text or bytes, under the same rule.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..diagnostics import Diagnostic, Severity
from .pipeline import analyze_paths, collect_sources, open_project
from .reporting import Reporter

__all__ = ["KINDS", "build_ir_file", "run_emit"]

KINDS = (
    "ir",
    "linked-ir",
    "llvm-ir",
    "c",
    "cpp",
    "header",
    "stablehlo",
    "cuda",
    "hip",
    "nvvm-ir",
    "ptx",
)
_SUFFIXES = {
    "ir": ".ppyir",
    "llvm-ir": ".ll",
    "c": ".c",
    "cpp": ".cpp",
    "header": ".h",
    "stablehlo": ".mlir",
    "cuda": ".cu",
    "hip": ".hip",
    "nvvm-ir": ".nvvm.ll",
    "ptx": ".ptx",
    "linked-ir": ".ppyir",
}
_HEADER_ONLY_SUFFIXES = {"c": ".h", "cpp": ".hpp"}
#: The kinds `--format` runs through clang-format.
_FORMATTED = frozenset({"c", "cpp", "cuda", "hip", "header"})


def run_emit(options: argparse.Namespace, reporter: Reporter) -> int:
    target: Path = options.target
    header_only = bool(getattr(options, "header_only", False))
    if not target.exists():
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{target} does not exist"))
        return 2
    standalone = bool(getattr(options, "standalone", False))
    for flag, given in (("--header-only", header_only), ("--standalone", standalone)):
        if given and options.kind not in _HEADER_ONLY_SUFFIXES:
            reporter.emit(
                Diagnostic("E1002", Severity.ERROR, f"`{flag}` applies to `emit c` and `emit cpp`")
            )
            return 2
    if standalone and not target.is_file():
        reporter.emit(
            Diagnostic("E1002", Severity.ERROR, "`--standalone` takes the program's entry file")
        )
        return 2
    output: Path | None = options.output
    if target.is_dir() and output is None:
        reporter.emit(
            Diagnostic("E1002", Severity.ERROR, "emitting a directory needs `-o DIR` to write into")
        )
        return 2
    formatting = bool(getattr(options, "format", False))
    if options.kind not in KINDS:
        for flag, given in (
            ("--header-only", header_only),
            ("--standalone", standalone),
            ("--format", formatting),
        ):
            if given:
                reporter.emit(
                    Diagnostic(
                        "E1002",
                        Severity.ERROR,
                        f"`{flag}` applies to the builtin kinds, not to a backend's format",
                    )
                )
                return 2
        return _emit_with_backend(options.kind, target, output, reporter)
    if formatting and options.kind not in _FORMATTED:
        reporter.emit(
            Diagnostic(
                "E1002",
                Severity.ERROR,
                "`--format` applies to `emit c`, `cpp`, `cuda`, `hip`, and `header`",
            )
        )
        return 2
    project = open_project(target)
    bundle = analyze_paths(project, collect_sources(target), backend="llvm")
    errors = reporter.report(bundle.diagnostics)
    if errors:
        reporter.summary(errors, 0)
        return 1
    try:
        if standalone:
            texts = _standalone_text(options.kind, bundle, reporter, target, header_only)
            if isinstance(texts, int):
                return texts
        else:
            texts = _texts(options.kind, bundle, header_only)
    except Exception as error:  # noqa: BLE001 - the backend's refusal is the message
        reporter.emit(Diagnostic(_refusal_code(error), Severity.ERROR, str(error)))
        return 2
    if not texts:
        reporter.emit(Diagnostic("E1002", Severity.ERROR, "nothing to emit: no module lowered"))
        return 1
    suffix = _HEADER_ONLY_SUFFIXES[options.kind] if header_only else _SUFFIXES[options.kind]
    if formatting:
        from ..backend.c.format import ClangFormatError, clang_format

        try:
            texts = {name: clang_format(text, suffix, project.root) for name, text in texts.items()}
        except ClangFormatError as error:
            reporter.emit(Diagnostic("E1802", Severity.ERROR, str(error), help=error.help))
            return 2
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


def _emit_with_backend(kind: str, target: Path, output: Path | None, reporter: Reporter) -> int:
    """`ppy emit <format>` for a format an installed backend registers.

    The backend is found by its format and loaded; its toolchain must be
    here; the modules go through the shared passes and the backend's own,
    then its validation, then `emit`, one module at a time in name order.
    A refusal is a diagnostic with the backend's reason: `E1903` when no
    backend can be used for the format, `E1801` when its toolchain is
    missing, `E1904` when one of its passes broke the IR, `E1802` when it
    cannot take the IR or the format.
    """
    from ..backend import (
        BackendError,
        BackendLoadError,
        BackendUnavailable,
        BackendValidationError,
        emit_format_owner,
    )
    from ..ir import PassVerificationError
    from .ir_pipeline import canonical_ir_modules, prepare_for_backend
    from .pipeline import backend_context

    project = open_project(target)
    try:
        owner = emit_format_owner(kind, project.config.backend)
    except BackendLoadError as error:
        reporter.emit(Diagnostic("E1903", Severity.ERROR, str(error)))
        return 2
    backend, spec = owner.backend, owner.format
    status = backend.toolchain_status()
    if not status.available:
        reporter.emit(
            Diagnostic(
                "E1801",
                Severity.ERROR,
                f"backend {backend.name!r} is unavailable here: {status.detail}",
            )
        )
        return 2
    bundle = analyze_paths(project, collect_sources(target), backend="llvm")
    errors = reporter.report(bundle.diagnostics)
    if errors:
        reporter.summary(errors, 0)
        return 1
    outputs: dict[str, str | bytes] = {}
    try:
        modules = canonical_ir_modules(bundle, launches=True, backend=backend)
        context = backend_context(bundle, backend)
        for name, module in sorted(modules.items()):
            prepare_for_backend(module, backend, context)
            outputs[name] = backend.emit(module, spec.name, context)
    except PassVerificationError as error:
        reporter.emit(Diagnostic("E1904", Severity.ERROR, f"backend {backend.name!r}: {error}"))
        return 2
    except BackendUnavailable as error:
        reporter.emit(Diagnostic("E1801", Severity.ERROR, str(error)))
        return 2
    except (BackendValidationError, BackendError) as error:
        reporter.emit(Diagnostic("E1802", Severity.ERROR, str(error)))
        return 2
    for note in context.notes:
        reporter.note(note)
    if not outputs:
        reporter.emit(Diagnostic("E1002", Severity.ERROR, "nothing to emit: no module lowered"))
        return 1
    for text in outputs.values():
        if isinstance(text, bytes) != spec.binary:
            reporter.emit(
                Diagnostic(
                    "E1802",
                    Severity.ERROR,
                    f"backend {backend.name!r} answered {type(text).__name__} for "
                    f"{spec.name!r}, a {'binary' if spec.binary else 'text'} format",
                )
            )
            return 2
    if target.is_file():
        if spec.binary:
            data = b"".join(outputs.values())  # type: ignore[arg-type]
            if output is None:
                sys.stdout.buffer.write(data)
                sys.stdout.flush()
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(data)
                reporter.note(f"wrote {output}")
            return 0
        joined = "\n".join(outputs.values())  # type: ignore[arg-type]
        if output is None:
            print(joined, end="" if joined.endswith("\n") else "\n")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(joined, encoding="utf-8")
            reporter.note(f"wrote {output}")
        return 0
    assert output is not None
    output.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        path = output / f"{name}{spec.suffix}"
        if isinstance(text, bytes):
            path.write_bytes(text)
        else:
            path.write_text(text, encoding="utf-8")
        reporter.note(f"wrote {path}")
    return 0


def _standalone_text(kind: str, bundle, reporter: Reporter, entry: Path, header_only: bool):  # type: ignore[no-untyped-def]
    """The whole program from `main` as one unit, or the exit status."""
    from ..backend.c import Language, emit_module
    from ..backend.llvm.standalone import standalone_ir

    module = standalone_ir(bundle, reporter, entry)
    if isinstance(module, int):
        return module
    from ..target import configured_target

    entry_symbol = str(module.attributes["ppy.entry"])
    machine = configured_target(bundle.project.config.llvm.target)
    return {
        module.name: emit_module(
            module, Language(kind), header_only=header_only, entry=entry_symbol, target=machine
        )
    }


def _refusal_code(error: Exception) -> str:
    from ..backend.c import EmitError, HeaderOnlyError

    if isinstance(error, HeaderOnlyError):
        return "E1804"
    if isinstance(error, EmitError):
        return "E1802"
    return "E1801"


def _texts(kind: str, bundle, header_only: bool) -> dict[str, str]:  # type: ignore[no-untyped-def]
    from ..backend.llvm import emit_ir
    from ..ir import encode
    from .ir_pipeline import canonical_ir_modules as ir_modules

    if kind == "ir":
        return {name: encode(module) for name, module in ir_modules(bundle, launches=True).items()}
    if kind == "linked-ir":
        from ..ir.linker import link
        from ..ir.transforms import whole_program

        modules = ir_modules(bundle, launches=True)
        if not modules:
            return {}
        linked = link(list(modules.values()), bundle.project.root.name)
        whole_program(linked.module, set(linked.module.functions), bundle.project.config.opt_level)
        return {linked.module.name: encode(linked.module)}
    if kind == "llvm-ir":
        return emit_ir(bundle)
    if kind == "stablehlo":
        return _stablehlo_texts(bundle)
    if kind in {"nvvm-ir", "ptx"}:
        return _device_texts(bundle, kind)
    if kind == "header":
        from ..backend.llvm.link import header_text
        from ..lowering.abi import signature_from_ir

        texts: dict[str, str] = {}
        for name, module in ir_modules(bundle).items():
            exports = {
                str(f.attributes["ppy.export"]): signature_from_ir(f)
                for f in module.functions.values()
                if not f.is_declaration and "ppy.export" in f.attributes
            }
            if exports:
                texts[name] = header_text(name, exports)
        return texts
    from ..backend.c import Language, emit_module
    from ..target import configured_target

    language = Language(kind)
    machine = configured_target(bundle.project.config.llvm.target)
    return {
        name: emit_module(module, language, header_only=header_only, target=machine)
        for name, module in ir_modules(bundle, launches=kind in {"cuda", "hip"}).items()
    }


def _device_texts(bundle, kind: str) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """Each module's kernels and device functions as NVVM IR, or as PTX; none where it has none."""
    from ..backend.c import EmitError
    from ..backend.nvvm import NvvmError, emit_module, ptx_from_ir
    from .ir_pipeline import canonical_ir_modules as ir_modules

    texts: dict[str, str] = {}
    for name, module in ir_modules(bundle).items():
        try:
            text = emit_module(module)
            if text:
                texts[name] = text if kind == "nvvm-ir" else ptx_from_ir(text)
        except NvvmError as error:
            raise EmitError(str(error)) from error
    return texts


def _stablehlo_texts(bundle) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """Each module's `@ppy.xla.jit` functions as StableHLO; every function where none is marked.

    A function XLA cannot take is left out and named in the reporter's notes;
    a module with nothing XLA takes produces no text.
    """
    from ..backend.stablehlo import emit_module, prepare, supports
    from .ir_pipeline import canonical_ir_modules as ir_modules

    texts: dict[str, str] = {}
    for name, module in ir_modules(bundle).items():
        prepare(module)
        symbols = bundle.symbols.modules.get(name)
        marked = {
            info.qualname
            for info in (symbols.functions.values() if symbols is not None else ())
            if info.directive("xla.jit") is not None
        }
        chosen = []
        for function in module.functions.values():
            if function.is_declaration:
                continue
            qualname = str(function.attributes.get("ppy.qualname", function.name))
            if marked and qualname not in marked:
                continue
            if supports(function) is None:
                chosen.append(function.name)
        if chosen:
            texts[name] = emit_module(module, tuple(chosen))
    return texts


def build_ir_file(path: Path, options: argparse.Namespace, reporter: Reporter) -> int:
    """`ppy build foo.ppyir`: objects, a library, and a manifest from IR alone."""
    from ..backend.llvm import LlvmUnavailable
    from ..backend.llvm.from_ir import emit_module
    from ..backend.llvm.jit import JitEngine, available
    from ..backend.llvm.link import (
        ToolchainError,
        emit_object,
        link_shared_library,
        write_manifest,
    )
    from ..ir import CodecError, read
    from ..lowering.abi import signature_from_ir
    from .ir_pipeline import optimize_shared_ir as optimize

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
        optimize(module, level, parallel=project.config.parallel)
        text = emit_module(module)
    except Exception as error:  # noqa: BLE001 - the verifier's or the backend's refusal
        reporter.emit(Diagnostic("E1801", Severity.ERROR, str(error)))
        return 2
    output: Path = options.output or (project.config.cache_path / "native")
    output.mkdir(parents=True, exist_ok=True)
    signatures = {
        f.attributes.get("ppy.qualname", name): signature_from_ir(f)
        for name, f in module.functions.items()
        if not f.is_declaration and "ppy.synthesized" not in f.attributes
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
