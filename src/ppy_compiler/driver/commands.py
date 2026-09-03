"""Implementations of the `ppy` subcommands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..diagnostics import Diagnostic, Severity
from .pipeline import (
    COMPILER_VERSION,
    AnalysisBundle,
    analyze_paths,
    build_python,
    collect_sources,
    open_project,
)
from .reporting import Reporter

__all__ = [
    "build",
    "cache",
    "check",
    "clean",
    "convert",
    "differential_test",
    "doctor",
    "explain",
    "fmt",
    "inspect",
    "language_server",
    "run_llvm_backend",
    "run_python_backend",
]


def _overrides(options: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if getattr(options, "opt_level", None) is not None:
        overrides["opt_level"] = options.opt_level
    if getattr(options, "no_strict", False):
        overrides["strict"] = False
    return overrides


def _resolve_host_cpu(options: argparse.Namespace, project) -> None:  # type: ignore[no-untyped-def]
    """Settle host targeting before any cache key reads it."""
    if getattr(options, "host_cpu", False):
        project.config.llvm.host_cpu = True


def _resolve_safeguards(options: argparse.Namespace, project, command: str) -> None:  # type: ignore[no-untyped-def]
    """Settle the guard mode before any cache key reads it.

    Priority: an explicit `--safeguards`, then the `--unsafe`/`--safe`
    flags, then the project's `[tool.ppy.llvm] safeguards`, then the
    command's own default -- `run` keeps Python-integer semantics, `build`
    produces a wrap-semantics artifact like every native compiler.
    """
    from .warm import resolved_safeguards

    project.config.llvm.safeguards = resolved_safeguards(
        options, project.config.llvm.safeguards, command
    )


def _prepare(
    target: Path, options: argparse.Namespace, *, backend: str = "python"
) -> AnalysisBundle:
    project = open_project(target, config_overrides=_overrides(options))
    entries = collect_sources(target)
    return analyze_paths(project, entries, backend=backend)


def check(options: argparse.Namespace, reporter: Reporter) -> int:
    target: Path = options.path
    if not target.exists():
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{target} does not exist"))
        return 2
    bundle = _prepare(target, options)
    errors = reporter.report(bundle.diagnostics, show_remarks=options.remarks)
    warnings = sum(1 for d in bundle.diagnostics if d.severity is Severity.WARNING)
    modules = len(bundle.graph.modules)
    if errors:
        reporter.summary(errors, warnings, subject=f" in {modules} module(s)")
        return 1
    reporter.note(
        f"checked {modules} module(s): no errors" + (f", {warnings} warning(s)" if warnings else "")
    )
    return 0


def run_python_backend(
    file: Path,
    program_args: list[str],
    options: argparse.Namespace,
    reporter: Reporter,
) -> int:
    """`ppy foo.ppy`: validate, optimize, cache, and execute generated Python."""
    from ..backend.binder import LibraryBinder
    from ..backend.python.runner import execute, format_traceback
    from ..opt.rewrites import adjustments_for_project
    from .staging import stage_project

    if not file.is_file():
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{file} is not a file"))
        return 2
    bundle = _prepare(file, options, backend="python")
    errors = reporter.report(bundle.diagnostics)
    if errors:
        reporter.summary(errors, 0)
        return 1

    output = build_python(
        bundle,
        opt_level=_overrides(options).get("opt_level"),  # type: ignore[arg-type]
        adjustments=adjustments_for_project(bundle),
    )
    if bundle.project.config.diagnostics.optimization_remarks:
        for remark in output.remarks:
            reporter.emit(remark)

    entry_name = bundle.entry
    entry = output.generated.get(entry_name or "")
    if entry is None:
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"no generated module for {file}"))
        return 2

    binder = LibraryBinder()
    _install_library_regions(bundle, binder, reporter)

    staged = stage_project(bundle)
    for module_name, entries in staged.artifacts.items():
        for function, artifact in entries.items():
            binder.add_exported(module_name, function, artifact.payload)
    if bundle.project.config.diagnostics.optimization_remarks:
        for remark in staged.diagnostics:
            reporter.emit(remark)

    result = execute(
        entry,
        output.generated,
        program_args,
        search_paths=bundle.project.search_paths,
        natives=binder,
        entry_name=entry_name,
    )
    if result.exception is not None:
        sys.stderr.write(format_traceback(result.exception))
    return result.exit_code


def run_llvm_backend(
    file: Path,
    program_args: list[str],
    options: argparse.Namespace,
    reporter: Reporter,
) -> int:
    """`ppy run foo.ppy`: compile through LLVM and execute.

    The compile is a build into the cache, and the execute is the launcher:
    the next run of the same program under the same compiler and
    configuration finds the artifact by its key and never gets here (see
    `warm.py`). Programs the launcher cannot serve -- runtime specialization,
    fused NumPy kernels, torch or JAX plugin work -- take the in-process path
    and leave a marker so the next run knows without analyzing.
    """
    from ppy_runtime.launch import main as launch

    from ..backend.llvm import LlvmUnavailable, compile_and_run, compile_for_run
    from .warm import locate

    prebuilt = getattr(options, "prebuilt", None)
    if prebuilt is not None:
        # A built artifact runs through the runtime alone. No project
        # discovery, no analysis, no LLVM -- the manifest is the whole
        # contract, and validation stays as cheap as reading it.
        return launch(prebuilt, program_args)
    if not file.is_file():
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{file} is not a file"))
        return 2
    warm = locate(file, options)
    project = open_project(file, config_overrides=_overrides(options))
    _resolve_safeguards(options, project, "run")
    bundle = analyze_paths(project, collect_sources(file), backend="llvm")
    errors = reporter.report(bundle.diagnostics)
    if errors:
        reporter.summary(errors, 0)
        return 1
    level = _overrides(options).get("opt_level")
    try:
        if not warm.needs_jit:
            manifest = compile_for_run(
                bundle,
                reporter,
                warm.directory,
                file.resolve(),
                opt_level=level,  # type: ignore[arg-type]
            )
            if manifest is not None:
                return launch(manifest, program_args)
        return compile_and_run(
            bundle,
            program_args,
            reporter,
            opt_level=level,  # type: ignore[arg-type]
        )
    except LlvmUnavailable as exc:
        reporter.emit(
            Diagnostic(
                "E1801",
                Severity.ERROR,
                str(exc),
                help="install the LLVM extra: `uv pip install 'ppy[llvm]'`",
            )
        )
        return 2


def build(options: argparse.Namespace, reporter: Reporter) -> int:
    """`ppy build TARGET`: compile without running."""
    target: Path = options.target
    if not target.exists():
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{target} does not exist"))
        return 2
    backend = options.backend
    project = open_project(target, config_overrides=_overrides(options))
    if backend == "llvm":
        _resolve_safeguards(options, project, "build")
        _resolve_host_cpu(options, project)
    bundle = analyze_paths(project, collect_sources(target), backend=backend)
    errors = reporter.report(bundle.diagnostics)
    if errors:
        reporter.summary(errors, 0)
        return 1

    if backend == "python":
        output = build_python(bundle)
        reporter.note(
            f"built {len(output.generated)} module(s) into {bundle.project.config.cache_path}"
        )
        return 0

    from ..backend.llvm import LlvmUnavailable, compile_project

    entry = target.resolve() if target.is_file() else None
    if getattr(options, "standalone", False):
        from ..backend.llvm.standalone import build_standalone

        if entry is None:
            reporter.emit(
                Diagnostic("E1803", Severity.ERROR, "a standalone build needs an entry file")
            )
            return 2
        try:
            return build_standalone(
                bundle,
                reporter,
                entry,
                options.output,
                opt_level=_overrides(options).get("opt_level"),
            )  # type: ignore[arg-type]
        except LlvmUnavailable as exc:
            reporter.emit(Diagnostic("E1801", Severity.ERROR, str(exc)))
            return 2
    try:
        artifacts = compile_project(
            bundle,
            reporter,
            opt_level=_overrides(options).get("opt_level"),  # type: ignore[arg-type]
            output=options.output,
            entry=entry,
        )
    except LlvmUnavailable as exc:
        reporter.emit(Diagnostic("E1801", Severity.ERROR, str(exc)))
        return 2

    reporter.note(f"objects:  {len(artifacts.objects)}")
    if artifacts.library:
        reporter.note(f"library:  {artifacts.library}")
    if artifacts.manifest:
        reporter.note(f"manifest: {artifacts.manifest}")
    if artifacts.launcher:
        reporter.note(f"launcher: {artifacts.launcher}")
    for note in artifacts.notes:
        reporter.emit(Diagnostic("W2004", Severity.WARNING, note))
    return 0


def convert(options: argparse.Namespace, reporter: Reporter) -> int:
    from .convert import run_convert

    return run_convert(options, reporter)


def migrate(options: argparse.Namespace, reporter: Reporter) -> int:
    from .convert import run_migrate

    return run_migrate(options, reporter)


def fmt(options: argparse.Namespace, reporter: Reporter) -> int:
    from .formatting import run_fmt

    return run_fmt(options, reporter)


def explain(options: argparse.Namespace, reporter: Reporter) -> int:
    from .explain import run_explain

    return run_explain(options, reporter)


def inspect(options: argparse.Namespace, reporter: Reporter) -> int:
    """Show the generated artifact for a target (spec 4.2)."""
    target: Path = options.target
    if not target.exists():
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{target} does not exist"))
        return 2
    bundle = _prepare(target, options, backend=options.backend)
    errors = reporter.report(bundle.diagnostics)
    if errors:
        return 1

    if options.backend == "llvm" or options.ir:
        from ..backend.llvm import LlvmUnavailable, emit_ir

        try:
            for name, ir in emit_ir(bundle).items():
                print(f"; ---- {name} ----")
                print(ir)
        except LlvmUnavailable as exc:
            reporter.emit(Diagnostic("E1801", Severity.ERROR, str(exc)))
            return 2
        # The native path is not only LLVM IR: the boundary back into CPython
        # and any library region are C and C++ that this compiler wrote too.
        for label, source in _generated_native_sources(bundle).items():
            print(f"/* ---- {label} ---- */")
            print(source)
        return 0

    from ..opt.rewrites import adjustments_for_project

    # Inspect must show what a real run executes, plugin rewrites included.
    output = build_python(bundle, adjustments=adjustments_for_project(bundle))
    for name, generated in output.generated.items():
        if target.is_file() and generated.source_path != target.resolve():
            continue
        print(f"# ---- {name} -> {generated.artifact} ----")
        print(generated.code)
    return 0


def _generated_native_sources(bundle) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """The C and C++ the native path compiles alongside the IR."""
    found: dict[str, str] = {}
    from ..backend.llvm import _collect
    from ..backend.llvm.wrapper import generate

    for name, module in _collect(bundle).items():
        signatures = {function: lowered.signature for function, lowered in module.functions.items()}
        if signatures:
            found[f"{name} (CPython ABI wrappers, C)"] = generate(name, signatures).source

    try:
        from ..plugins.torch_region import find_regions
    except ImportError:  # pragma: no cover - the plugin is optional
        return found
    for name, symbols in bundle.symbols.modules.items():
        analysis = bundle.analysis.modules.get(name)
        if analysis is None:
            continue
        for region in find_regions(symbols, analysis):
            if region.body:
                found[f"{name}.{region.name} (ATen region, C++)"] = region.source()
    return found


def cache(options: argparse.Namespace, reporter: Reporter) -> int:
    project = open_project(Path.cwd())
    store = project.store
    match options.cache_command:
        case "status":
            stats = store.stats()
            print(f"cache: {stats.root}")
            print(f"entries: {stats.entries}   size: {stats.human_total()}")
            for kind, (count, size) in sorted(stats.by_kind.items()):
                print(f"  {kind:<10} {count:>6} entries  {size:>10} bytes")
            return 0
        case "clean":
            store.clean()
            reporter.note(f"cleaned {store.root}")
            return 0
        case "gc":
            removed, freed = store.gc(
                max_age_days=options.max_age_days, max_bytes=options.max_bytes
            )
            reporter.note(f"removed {removed} artifact(s), freed {freed} bytes")
            return 0
    return 2


def clean(options: argparse.Namespace, reporter: Reporter) -> int:
    import shutil

    project = open_project(Path.cwd())
    path = project.config.cache_path
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        reporter.note(f"removed {path}")
    else:
        reporter.note(f"nothing to remove at {path}")
    return 0


def doctor(options: argparse.Namespace, reporter: Reporter) -> int:
    import platform

    project = open_project(Path.cwd())
    print(f"ppy               {COMPILER_VERSION}")
    print(f"python            {platform.python_version()} ({sys.implementation.name})")
    print(f"platform          {platform.system()} {platform.machine()}")
    print(f"project root      {project.root}")
    print(f"cache             {project.config.cache_path}")
    print(f"strict            {project.config.strict}")
    print(f"opt-level         {project.config.opt_level}")
    print(f"dynamic           {project.config.dynamic_boundaries}")
    print(f"build-execution   {project.config.build_execution}")

    from ..backend.llvm import llvm_status, toolchain_status

    status, detail = llvm_status()
    print(f"llvm backend      {status}" + (f" ({detail})" if detail else ""))
    usable, toolchain = toolchain_status()
    print(f"native toolchain  {'available' if usable else 'unavailable'} ({toolchain})")

    from ..backend.llvm.wrapper_build import wrapper_toolchain

    ready, detail = wrapper_toolchain()
    boundary = "generated CPython ABI" if ready else "ctypes"
    print(f"python boundary   {boundary} ({detail})")

    print("plugins:")
    for plugin in project.plugins:
        print(f"  {plugin.name:<10} {plugin.fingerprint()}")

    from ..plugins.torch_build import toolchain_ready

    ready, detail = toolchain_ready()
    print(f"aten regions      {'available' if ready else 'unavailable'} ({detail})")
    return 0


def differential_test(options: argparse.Namespace, reporter: Reporter) -> int:
    """Compare plain CPython and Python-backend execution (spec 33.1)."""
    from ..testing.differential import run_suite

    return run_suite(options.path, reporter)


def language_server(options: argparse.Namespace, reporter: Reporter) -> int:
    """`ppy lsp`: serve the analysis daemon to an editor over stdio."""
    from ..lsp.server import serve

    return serve(options.root.resolve())


def _install_library_regions(bundle, binder, reporter) -> None:  # type: ignore[no-untyped-def]
    """Compile and install the ATen regions this project declares (spec 20.3)."""
    from .staging import compile_torch_regions

    result = compile_torch_regions(bundle, notify=reporter.note)
    for module_name, entries in result.compiled.items():
        for function, entry_point in entries.items():
            binder.add_region(module_name, function, entry_point)
    if bundle.project.config.diagnostics.optimization_remarks:
        for remark in result.diagnostics:
            reporter.emit(remark)
