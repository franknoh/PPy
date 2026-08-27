"""`ppy explain`: why a function was or was not optimized (spec 29.3)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..analysis.checker import FunctionAnalysis
from ..analysis.contracts import ContractReport
from ..analysis.representation import select
from ..analysis.symbols import FunctionInfo
from ..diagnostics import Diagnostic, Severity, describe
from .pipeline import AnalysisBundle, analyze_paths, collect_sources, open_project
from .reporting import Reporter

__all__ = ["run_explain"]

_LOCATION = re.compile(r"^(?P<path>.+?):(?P<line>\d+)(?::(?P<column>\d+))?$")
_CODE = re.compile(r"^[EWR]\d{4}$", re.IGNORECASE)


def run_explain(options: argparse.Namespace, reporter: Reporter) -> int:
    location: str = options.location

    if _CODE.match(location):
        return _explain_code(location, reporter)

    match = _LOCATION.match(location)
    if match is not None:
        path = Path(match.group("path"))
        if not path.exists():
            reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{path} does not exist"))
            return 2
        bundle = _analyze(path, options)
        return _explain_location(bundle, path, int(match.group("line")), reporter)

    target = Path.cwd()
    bundle = _analyze(target, options)
    return _explain_qualname(bundle, location, reporter)


def _analyze(target: Path, options: argparse.Namespace) -> AnalysisBundle:
    project = open_project(target)
    entries = collect_sources(target)
    return analyze_paths(project, entries, backend="llvm")


def _explain_code(code: str, reporter: Reporter) -> int:
    description = describe(code)
    if description is None:
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"unknown diagnostic code {code!r}"))
        return 2
    print(f"{code.upper()}: {description}")
    return 0


def _explain_location(bundle: AnalysisBundle, path: Path, line: int, reporter: Reporter) -> int:
    resolved = path.resolve()
    for module in bundle.symbols.modules.values():
        if module.path != resolved:
            continue
        found = _function_at(bundle, module.name, line)
        if found is not None:
            _print_function(bundle, *found)
            return 0
        _print_module(bundle, module.name, line)
        return 0
    reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{path} is not part of the analyzed project"))
    return 2


def _explain_qualname(bundle: AnalysisBundle, name: str, reporter: Reporter) -> int:
    for qualname, info in bundle.symbols.functions.items():
        if qualname == name or info.name == name:
            analysis = bundle.analysis.function(qualname)
            if analysis is not None:
                _print_function(bundle, info, analysis, bundle.reports.get(qualname))
                return 0
    reporter.emit(Diagnostic("E1002", Severity.ERROR, f"no function named {name!r} was found"))
    return 2


def _function_at(
    bundle: AnalysisBundle, module_name: str, line: int
) -> tuple[FunctionInfo, FunctionAnalysis, ContractReport | None] | None:
    best: tuple[int, FunctionInfo] | None = None
    for qualname, info in bundle.symbols.functions.items():
        if info.module != module_name:
            continue
        start = info.node.lineno
        end = getattr(info.node, "end_lineno", start) or start
        if start <= line <= end and (best is None or start > best[0]):
            best = (start, info)
    if best is None:
        return None
    info = best[1]
    analysis = bundle.analysis.function(info.qualname)
    if analysis is None:
        return None
    return info, analysis, bundle.reports.get(info.qualname)


def _print_function(
    bundle: AnalysisBundle,
    info: FunctionInfo,
    analysis: FunctionAnalysis,
    report: ContractReport | None,
) -> None:
    level = info.opt_level if info.opt_level is not None else bundle.project.config.opt_level
    print(f"function: {info.name}")
    print(f"qualname: {info.qualname}")
    print(f"semantic type: {info.signature()}")
    print(f"effects: {analysis.effects}")
    print(f"purity: {_purity(info, analysis)}")
    print(f"optimization: O{level}")
    print(f"python backend: {_python_backend(analysis)}")
    print(f"llvm backend: {_llvm_backend(report)}")
    print(f"jit: {'eligible' if report and report.jit_eligible else 'not eligible'}")
    print(f"parallel: {'accepted' if report and report.parallel_ok else 'rejected'}")
    if report and not report.parallel_ok and report.parallel_reason:
        print(f"reason: {report.parallel_reason}")
    if report and report.reductions:
        print(f"reductions: {', '.join(report.reductions)}")

    representations = []
    for param in info.params:
        chosen = select(param.type, param.facts, escapes=param.name in analysis.escaping)
        representations.append(f"{param.name}: {param.type} -> {chosen}")
    if representations:
        print("representation:")
        for line in representations:
            print(f"  {line}")
    returned = select(info.ret, info.ret_facts, escapes=True)
    print(f"  return: {info.ret} -> {returned}")

    facts = info.ret_facts.describe()
    if facts:
        print(f"refinements: {', '.join(facts)}")
    if analysis.dynamic:
        print("dynamic: this function contains a ppy.dynamic boundary")

    module = bundle.analysis.modules.get(info.module)
    if module is not None:
        notes = [n for n in module.lowerings.values() if info.node.lineno <= n.line <= (info.node.end_lineno or n.line)]
        if notes:
            print("library lowering:")
            for note in notes:
                print(f"  line {note.line}: {note.qualname} -> {note.lowering} ({note.reason})")
                for guard in note.guards:
                    print(f"      guard: {guard}")


def _purity(info: FunctionInfo, analysis: FunctionAnalysis) -> str:
    if info.declared_pure:
        return "declared and verified" if analysis.verified_pure else "declared but NOT verified"
    return "inferred pure" if analysis.verified_pure else "impure"


def _python_backend(analysis: FunctionAnalysis) -> str:
    if analysis.dynamic:
        return "boxed: contains a dynamic boundary"
    return "optimized"


def _llvm_backend(report: ContractReport | None) -> str:
    if report is None:
        return "unknown"
    if report.native_ok:
        return "native"
    return f"boxed: {report.native_reason}"


def _print_module(bundle: AnalysisBundle, module_name: str, line: int) -> None:
    module = bundle.analysis.modules.get(module_name)
    print(f"module: {module_name}")
    if module is None:
        return
    print(f"module effects: {module.module_effects}")
    for note in module.lowerings.values():
        if note.line == line:
            print(f"line {line}: {note.qualname} -> {note.lowering} ({note.reason})")
    for start, end in module.dynamic_spans:
        if start <= line <= end:
            print(f"line {line} is inside a ppy.dynamic boundary (lines {start}-{end})")
