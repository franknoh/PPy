"""Implementations of the `ppy` subcommands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..analysis import types as T
from ..diagnostics import Diagnostic, Severity, Span, describe
from .pipeline import (
    COMPILER_VERSION,
    AnalysisBundle,
    analyze_paths,
    build_python,
    collect_sources,
    module_cache_key,
    open_project,
)
from .reporting import Reporter

__all__ = [
    "check", "convert", "fmt", "explain", "inspect", "cache", "clean",
    "doctor", "run_python_backend", "run_llvm_backend", "build",
    "differential_test",
]


def _overrides(options: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if getattr(options, "opt_level", None) is not None:
        overrides["opt_level"] = options.opt_level
    if getattr(options, "no_strict", False):
        overrides["strict"] = False
    return overrides


def _prepare(target: Path, options: argparse.Namespace, *, backend: str = "python") -> AnalysisBundle:
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
    reporter.note(f"checked {modules} module(s): no errors" + (f", {warnings} warning(s)" if warnings else ""))
    return 0



def run_python_backend(
    file: Path,
    program_args: list[str],
    options: argparse.Namespace,
    reporter: Reporter,
) -> int:
    """`ppy foo.ppy`: validate, optimize, cache, and execute generated Python."""
    from ..backend.python.runner import execute, format_traceback

    if not file.is_file():
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{file} is not a file"))
        return 2
    bundle = _prepare(file, options, backend="python")
    errors = reporter.report(bundle.diagnostics)
    if errors:
        reporter.summary(errors, 0)
        return 1

    output = build_python(bundle, opt_level=_overrides(options).get("opt_level"))  # type: ignore[arg-type]
    if bundle.project.config.diagnostics.optimization_remarks:
        for remark in output.remarks:
            reporter.emit(remark)

    entry_name = bundle.entry
    entry = output.generated.get(entry_name or "")
    if entry is None:
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"no generated module for {file}"))
        return 2

    result = execute(entry, output.generated, program_args, search_paths=bundle.project.search_paths)
    if result.exception is not None:
        sys.stderr.write(format_traceback(result.exception))
    return result.exit_code


def run_llvm_backend(
    file: Path,
    program_args: list[str],
    options: argparse.Namespace,
    reporter: Reporter,
) -> int:
    """`ppy run foo.ppy`: compile through LLVM and execute."""
    from ..backend.llvm import LlvmUnavailable, compile_and_run

    if not file.is_file():
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{file} is not a file"))
        return 2
    bundle = _prepare(file, options, backend="llvm")
    errors = reporter.report(bundle.diagnostics)
    if errors:
        reporter.summary(errors, 0)
        return 1
    try:
        return compile_and_run(bundle, program_args, reporter, opt_level=_overrides(options).get("opt_level"))  # type: ignore[arg-type]
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
    bundle = _prepare(target, options, backend=backend)
    errors = reporter.report(bundle.diagnostics)
    if errors:
        reporter.summary(errors, 0)
        return 1

    if backend == "python":
        output = build_python(bundle)
        reporter.note(f"built {len(output.generated)} module(s) into {bundle.project.config.cache_path}")
        return 0

    from ..backend.llvm import LlvmUnavailable, compile_project

    try:
        artifacts = compile_project(bundle, reporter, opt_level=_overrides(options).get("opt_level"))  # type: ignore[arg-type]
    except LlvmUnavailable as exc:
        reporter.emit(Diagnostic("E1801", Severity.ERROR, str(exc)))
        return 2
    reporter.note(f"built {len(artifacts)} native artifact(s) into {bundle.project.config.cache_path / 'llvm'}")
    return 0



def convert(options: argparse.Namespace, reporter: Reporter) -> int:
    from .convert import run_convert

    return run_convert(options, reporter)


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
        return 0

    output = build_python(bundle)
    for name, generated in output.generated.items():
        if target.is_file() and generated.source_path != target.resolve():
            continue
        print(f"# ---- {name} -> {generated.artifact} ----")
        print(generated.code)
    return 0



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
            removed, freed = store.gc(max_age_days=options.max_age_days, max_bytes=options.max_bytes)
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

    from ..backend.llvm import llvm_status

    status, detail = llvm_status()
    print(f"llvm backend      {status}" + (f" ({detail})" if detail else ""))

    print("plugins:")
    for plugin in project.plugins:
        print(f"  {plugin.name:<10} {plugin.fingerprint()}")
    return 0



def differential_test(options: argparse.Namespace, reporter: Reporter) -> int:
    """Compare plain CPython and Python-backend execution (spec 33.1)."""
    from ..testing.differential import run_suite

    return run_suite(options.path, reporter)
