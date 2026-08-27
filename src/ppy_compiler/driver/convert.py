"""`ppy convert`: source-preserving `.py` to `.ppy` conversion (spec 4.1)."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from ..analysis import types as T
from ..analysis.render import render_annotation
from ..diagnostics import Diagnostic, DiagnosticBag, Severity
from ..frontend.source import span_of
from .pipeline import analyze_paths, collect_sources, open_project
from .reporting import Reporter

__all__ = ["run_convert", "convert_source", "ConversionPlan"]

_DYNAMIC_CALLS = {"eval", "exec", "compile", "globals", "locals", "vars", "__import__"}


@dataclass(slots=True)
class ConversionPlan:
    """Annotations to insert into one module, keyed by source position."""

    params: dict[tuple[int, str], str] = field(default_factory=dict)
    returns: dict[int, str] = field(default_factory=dict)
    assignments: dict[tuple[int, str], str] = field(default_factory=dict)
    #: Instance fields, annotated where `__init__` first assigns them.
    fields: dict[tuple[int, str], str] = field(default_factory=dict)
    #: Class names that are not yet bound where an annotation mentioning them
    #: is written, and so have to be quoted.
    forward: dict[str, int] = field(default_factory=dict)
    typing_imports: set[str] = field(default_factory=set)
    ppy_imports: set[str] = field(default_factory=set)
    needs_ppy: bool = False

    @property
    def is_empty(self) -> bool:
        return not (
            self.params or self.returns or self.assignments or self.fields or self.needs_ppy
        )


def run_convert(options: argparse.Namespace, reporter: Reporter) -> int:
    target: Path = options.path
    if not target.exists():
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{target} does not exist"))
        return 2

    project = open_project(target)
    project.config.strict = False
    sources = [p for p in collect_sources(target) if p.suffix == ".py"]
    if not sources:
        reporter.note(f"no .py sources found under {target}")
        return 0

    # A project conversion analyzes the whole module and call graph at once.
    bundle = analyze_paths(project, sources, backend="python")
    observed = refine_with_call_sites(bundle)

    written: list[Path] = []
    failures = 0
    for path in sources:
        module_name = _module_for(bundle, path)
        if module_name is None:
            continue
        plan, diagnostics = build_plan(bundle, module_name, observed)
        for diagnostic in diagnostics:
            reporter.emit(diagnostic)
            if diagnostic.severity is Severity.ERROR:
                failures += 1

        original = path.read_text(encoding="utf-8")
        converted = convert_source(original, plan)
        destination = path if options.in_place else path.with_suffix(".ppy")

        if options.dry_run:
            print(f"# ---- {destination} ----")
            print(converted)
            continue
        if destination.exists() and destination != path and not options.force:
            reporter.emit(
                Diagnostic(
                    "E1002",
                    Severity.ERROR,
                    f"{destination} already exists",
                    help="pass --force to overwrite it",
                )
            )
            failures += 1
            continue
        destination.write_text(converted, encoding="utf-8")
        written.append(destination)

    if written:
        reporter.note(f"converted {len(written)} file(s): " + ", ".join(str(p) for p in written))
    return 1 if failures else 0


def _module_for(bundle, path: Path) -> str | None:  # type: ignore[no-untyped-def]
    resolved = path.resolve()
    for name, symbols in bundle.symbols.modules.items():
        if symbols.path == resolved:
            return name
    return None


#: Call-site evidence propagates along the call graph, so one round only
#: reaches functions called from already-typed code. Repeating it walks the
#: chain; a handful of rounds settles any realistic project.
_REFINEMENT_ROUNDS = 6


def refine_with_call_sites(bundle) -> dict[tuple[str, int], T.Type]:  # type: ignore[no-untyped-def]
    """Adopt call-site argument types and re-infer, until nothing new appears.

    Without this a function whose parameters were only inferred would still
    return `<unknown>`, and nothing downstream of it could be annotated.
    """
    from ..analysis.checker import analyze
    from ..diagnostics import DiagnosticBag

    observed: dict[tuple[str, int], T.Type] = {}
    for _round in range(_REFINEMENT_ROUNDS):
        observed = _observed_arguments(bundle)
        changed = False
        for (qualname, index), inferred in observed.items():
            info = bundle.symbols.functions.get(qualname)
            if info is None or index >= len(info.params):
                continue
            param = info.params[index]
            if param.annotated or isinstance(inferred, (T.UnknownType, T.AnyType)):
                continue
            param.type = inferred
            param.annotated = True
            changed = True
        changed |= _infer_fields(bundle)
        changed |= _infer_from_usage(bundle)
        if not changed:
            break
        bundle.analysis = analyze(
            bundle.symbols,
            DiagnosticBag(),
            strict=False,
            dynamic_policy=bundle.project.config.dynamic_boundaries,
            plugins=bundle.project.plugins,
        )
    return observed


def build_plan(  # type: ignore[no-untyped-def]
    bundle,
    module_name: str,
    observed: dict[tuple[str, int], T.Type] | None = None,
) -> tuple[ConversionPlan, list[Diagnostic]]:
    """Decide which annotations to insert, using interprocedural evidence."""
    plan = ConversionPlan(needs_ppy=True)
    diagnostics: list[Diagnostic] = []
    symbols = bundle.symbols.modules[module_name]
    plan.forward = _forward_references(symbols)
    analysis = bundle.analysis.modules.get(module_name)
    if observed is None:
        observed = refine_with_call_sites(bundle)

    functions = list(symbols.functions.values())
    for cls in symbols.classes.values():
        functions.extend(cls.methods.values())

    for info in functions:
        line = info.node.lineno
        for index, param in enumerate(info.params):
            if param.kind in {"var_positional", "var_keyword"}:
                continue
            if index == 0 and info.is_method and not info.is_static:
                # The receiver's type is the class it is defined in.
                continue
            if _has_source_annotation(info, param.name):
                continue
            candidate = observed.get((info.qualname, index))
            if candidate is None and param.annotated:
                candidate = param.type
            rendered = render_annotation(candidate, local_module=module_name) if candidate is not None else None
            if rendered is None:
                diagnostics.append(
                    Diagnostic(
                        "E1304",
                        Severity.WARNING,
                        f"cannot infer a stable type for parameter `{param.name}`",
                        span_of(info.path, info.node),
                        help="annotate it explicitly, split the function, or isolate the dynamic operation",
                    )
                )
                continue
            plan.params[(line, param.name)] = rendered.text
            plan.typing_imports |= rendered.typing_imports
            plan.ppy_imports |= rendered.ppy_imports

        if not info.ret_annotated:
            rendered = render_annotation(info.ret, info.ret_facts, local_module=module_name)
            if rendered is not None:
                plan.returns[line] = rendered.text
                plan.typing_imports |= rendered.typing_imports
                plan.ppy_imports |= rendered.ppy_imports

    if bundle.project.config.inference.write_local_annotations and analysis is not None:
        _plan_module_globals(symbols, analysis, plan, module_name)
    _plan_fields(symbols, plan, module_name)

    diagnostics.extend(_dynamic_findings(symbols))
    return plan, diagnostics


def _observed_arguments(bundle) -> dict[tuple[str, int], T.Type]:  # type: ignore[no-untyped-def]
    """Join the argument types seen at every call site of each function."""
    observed: dict[tuple[str, int], T.Type] = {}
    for module_name, module in bundle.analysis.modules.items():
        symbols = bundle.symbols.modules.get(module_name)
        if symbols is None:
            continue
        for node in ast.walk(symbols.module.tree):
            if not isinstance(node, ast.Call):
                continue
            qualname = _callee_qualname(bundle, symbols, node)
            if qualname is None:
                continue
            info = bundle.symbols.functions.get(qualname)
            if info is None:
                continue
            offset = 1 if info.is_method and not info.is_static else 0
            for index, argument in enumerate(node.args):
                observed_type = T.strip_literal(module.type_of(argument))
                if isinstance(observed_type, (T.UnknownType, T.AnyType, T.NeverType)):
                    continue
                key = (qualname, index + offset)
                existing = observed.get(key)
                observed[key] = observed_type if existing is None else T.join(existing, observed_type)
    for qualname, info in bundle.symbols.functions.items():
        for index, param in enumerate(info.params):
            if param.annotated and not isinstance(param.type, T.UnknownType):
                observed.setdefault((qualname, index), param.type)
    return observed


def _callee_qualname(bundle, symbols, node: ast.Call) -> str | None:  # type: ignore[no-untyped-def]
    """The function a call reaches, following a constructor to `__init__`."""
    resolver = bundle.symbols.resolver(symbols)
    direct = resolver.canonical(node.func)
    if direct is not None and direct in bundle.symbols.functions:
        return direct
    if direct is not None and direct in bundle.symbols.classes:
        initializer = f"{direct}.__init__"
        if initializer in bundle.symbols.functions:
            return initializer
    if isinstance(node.func, ast.Name):
        local = f"{symbols.name}.{node.func.id}"
        if local in bundle.symbols.functions:
            return local
        if local in bundle.symbols.classes:
            initializer = f"{local}.__init__"
            if initializer in bundle.symbols.functions:
                return initializer
    if isinstance(node.func, ast.Attribute):
        # A method called on a value of a known class.
        analysis = bundle.analysis.modules.get(symbols.name)
        if analysis is not None:
            owner = T.strip_literal(analysis.type_of(node.func.value))
            if isinstance(owner, T.Instance):
                method = f"{owner.name}.{node.func.attr}"
                if method in bundle.symbols.functions:
                    return method
    return None


def _plan_module_globals(symbols, analysis, plan: ConversionPlan, module_name: str) -> None:  # type: ignore[no-untyped-def]
    for node in symbols.module.tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id.startswith("__"):
            continue
        value_type = T.strip_literal(analysis.type_of(node.value))
        rendered = render_annotation(value_type, analysis.facts_of(node.value), local_module=module_name)
        if rendered is None:
            continue
        plan.assignments[(node.lineno, target.id)] = rendered.text
        plan.typing_imports |= rendered.typing_imports
        plan.ppy_imports |= rendered.ppy_imports


def _dynamic_findings(symbols) -> list[Diagnostic]:  # type: ignore[no-untyped-def]
    """Report the dynamic features that block conversion (spec 9)."""
    found: list[Diagnostic] = []
    for node in ast.walk(symbols.module.tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _DYNAMIC_CALLS:
            found.append(
                Diagnostic(
                    "E1504",
                    Severity.WARNING,
                    f"`{node.func.id}` is a dynamic feature the converted module must isolate",
                    span_of(symbols.path, node),
                    help="wrap it in `with ppy.dynamic:` or mark the function `@ppy.dynamic`",
                )
            )
    return found



class _Annotator(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, plan: ConversionPlan) -> None:
        self.plan = plan
        self._function_lines: list[int] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self._function_lines.append(self.get_metadata(PositionProvider, node).start.line)
        return True

    def leave_FunctionDef(
        self, original: cst.FunctionDef, updated: cst.FunctionDef
    ) -> cst.FunctionDef:
        line = self._function_lines.pop()
        returns = self.plan.returns.get(line)
        if returns is not None and updated.returns is None:
            text = _quote_if_forward(returns, line, self.plan.forward)
            updated = updated.with_changes(returns=cst.Annotation(cst.parse_expression(text)))
        return updated

    def leave_Param(self, original: cst.Param, updated: cst.Param) -> cst.Param:
        if updated.annotation is not None or not self._function_lines:
            return updated
        line = self._function_lines[-1]
        annotation = self.plan.params.get((line, updated.name.value))
        if annotation is None:
            return updated
        text = _quote_if_forward(annotation, line, self.plan.forward)
        return updated.with_changes(annotation=cst.Annotation(cst.parse_expression(text)))

    def leave_SimpleStatementLine(
        self, original: cst.SimpleStatementLine, updated: cst.SimpleStatementLine
    ) -> cst.SimpleStatementLine:
        if len(updated.body) != 1:
            return updated
        statement = updated.body[0]
        if not isinstance(statement, cst.Assign) or len(statement.targets) != 1:
            return updated
        target = statement.targets[0].target
        line = self.get_metadata(PositionProvider, original).start.line

        if isinstance(target, cst.Attribute) and isinstance(target.value, cst.Name):
            annotation = self.plan.fields.get((line, target.attr.value))
            if annotation is None:
                return updated
            text = _quote_if_forward(annotation, line, self.plan.forward)
            return updated.with_changes(
                body=[
                    cst.AnnAssign(
                        target=target,
                        annotation=cst.Annotation(cst.parse_expression(text)),
                        value=statement.value,
                    )
                ]
            )

        if self._function_lines or not isinstance(target, cst.Name):
            return updated
        annotation = self.plan.assignments.get((line, target.value))
        if annotation is None:
            return updated
        return updated.with_changes(
            body=[
                cst.AnnAssign(
                    target=target,
                    annotation=cst.Annotation(cst.parse_expression(annotation)),
                    value=statement.value,
                )
            ]
        )


def convert_source(source: str, plan: ConversionPlan) -> str:
    """Rewrite `source` through a concrete syntax tree, preserving trivia."""
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
    annotated = wrapper.visit(_Annotator(plan))
    return _insert_imports(annotated, plan).code


def _insert_imports(module: cst.Module, plan: ConversionPlan) -> cst.Module:
    """Insert `import ppy` after the docstring and `__future__` imports."""
    existing = _existing_imports(module)
    additions: list[cst.SimpleStatementLine] = []

    ppy_names = sorted(plan.ppy_imports - existing)
    if plan.needs_ppy and "ppy" not in existing:
        additions.append(cst.parse_statement("import ppy"))
    if ppy_names:
        additions.append(cst.parse_statement(f"from ppy import {', '.join(ppy_names)}"))
    typing_names = sorted(plan.typing_imports - existing)
    if typing_names:
        additions.append(cst.parse_statement(f"from typing import {', '.join(typing_names)}"))

    if not additions:
        return module
    index = _insert_index(module)
    body = list(module.body)
    return module.with_changes(body=[*body[:index], *additions, *body[index:]])


def _existing_imports(module: cst.Module) -> set[str]:
    found: set[str] = set()
    for statement in module.body:
        if not isinstance(statement, cst.SimpleStatementLine):
            continue
        for small in statement.body:
            if isinstance(small, cst.Import):
                for alias in small.names:
                    found.add(_dotted(alias.name))
            elif isinstance(small, cst.ImportFrom) and small.module is not None:
                found.add(_dotted(small.module))
                if not isinstance(small.names, cst.ImportStar):
                    for alias in small.names:
                        found.add(_dotted(alias.name))
    return found


def _dotted(node: cst.BaseExpression) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return f"{_dotted(node.value)}.{node.attr.value}"
    return ""


def _insert_index(module: cst.Module) -> int:
    index = 0
    for position, statement in enumerate(module.body):
        if not isinstance(statement, cst.SimpleStatementLine):
            break
        first = statement.body[0]
        is_docstring = (
            position == 0
            and isinstance(first, cst.Expr)
            and isinstance(first.value, cst.SimpleString)
        )
        is_future = isinstance(first, cst.ImportFrom) and _dotted(first.module or cst.Name("")) == "__future__"
        if is_docstring or is_future:
            index = position + 1
            continue
        break
    return index


def _has_source_annotation(info, name: str) -> bool:  # type: ignore[no-untyped-def]
    """Was the parameter already annotated in the original source?"""
    arguments = info.node.args
    for group in (arguments.posonlyargs, arguments.args, arguments.kwonlyargs):
        for argument in group:
            if argument.arg == name:
                return argument.annotation is not None
    for argument in (arguments.vararg, arguments.kwarg):
        if argument is not None and argument.arg == name:
            return argument.annotation is not None
    return False


def _infer_fields(bundle) -> bool:  # type: ignore[no-untyped-def]
    """Give each instance field the type `__init__` assigns to it.

    `self.width = width` says as much about `width` as an annotation would,
    and without it nothing that reads the field can be typed.
    """
    changed = False
    for qualname, info in bundle.symbols.classes.items():
        initializer = info.methods.get("__init__")
        analysis = bundle.analysis.modules.get(info.module)
        if initializer is None or analysis is None:
            continue
        for node in ast.walk(initializer.node):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not _is_self_attribute(target, initializer):
                continue
            name = target.attr  # type: ignore[union-attr]
            if not isinstance(info.fields.get(name, T.UNKNOWN), T.UnknownType):
                continue
            assigned = T.strip_literal(analysis.type_of(node.value))
            if isinstance(assigned, (T.UnknownType, T.AnyType, T.NeverType)):
                continue
            info.fields[name] = assigned
            changed = True
    return changed


def _infer_from_usage(bundle) -> bool:  # type: ignore[no-untyped-def]
    """Type a parameter nothing calls, from the arithmetic it takes part in.

    A helper that is only ever called from outside the module has no call-site
    evidence, but `self.width * factor` still says `factor` is a number.
    """
    changed = False
    for qualname, info in bundle.symbols.functions.items():
        analysis = bundle.analysis.modules.get(info.module)
        if analysis is None:
            continue
        pending = {p.name: p for p in info.params if not p.annotated}
        if not pending:
            continue
        for node in ast.walk(info.node):
            if not isinstance(node, ast.BinOp):
                continue
            for side, other in ((node.left, node.right), (node.right, node.left)):
                if not isinstance(side, ast.Name) or side.id not in pending:
                    continue
                partner = T.strip_literal(analysis.type_of(other))
                if partner not in (T.INT, T.FLOAT):
                    continue
                param = pending.pop(side.id)
                param.type = partner
                param.annotated = True
                changed = True
    return changed


def _is_self_attribute(target: ast.expr, info) -> bool:  # type: ignore[no-untyped-def]
    receiver = info.params[0].name if info.params else "self"
    return (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == receiver
    )


def _plan_fields(symbols, plan: ConversionPlan, module_name: str) -> None:  # type: ignore[no-untyped-def]
    """Write each inferred instance field where `__init__` first assigns it.

    A converted module has to stand on its own: a field type discovered during
    conversion is of no use to anything reading the result unless it is
    written down.
    """
    for info in symbols.classes.values():
        initializer = info.methods.get("__init__")
        if initializer is None:
            continue
        declared = {
            node.target.id
            for node in info.node.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        written: set[str] = set()
        for node in ast.walk(initializer.node):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not _is_self_attribute(target, initializer):
                continue
            name = target.attr  # type: ignore[union-attr]
            if name in declared or name in written:
                continue
            rendered = render_annotation(
                info.fields.get(name, T.UNKNOWN),
                info.field_facts.get(name),
                local_module=module_name,
            )
            if rendered is None:
                continue
            written.add(name)
            plan.fields[(node.lineno, name)] = rendered.text
            plan.typing_imports |= rendered.typing_imports
            plan.ppy_imports |= rendered.ppy_imports


def _forward_references(symbols) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Where each class in this module becomes usable as a runtime name."""
    return {
        info.name: (info.node.end_lineno or info.node.lineno)
        for info in symbols.classes.values()
    }


def _quote_if_forward(text: str, line: int, forward: dict[str, int]) -> str:
    """Quote an annotation that names a class not yet defined at `line`.

    Annotations are evaluated eagerly before Python 3.14, so a bare reference
    to the enclosing class would raise at import.
    """
    for name, defined_at in forward.items():
        if line <= defined_at and _mentions(text, name):
            return repr(text)
    return text


def _mentions(text: str, name: str) -> bool:
    import re

    return re.search(rf"\b{re.escape(name)}\b", text) is not None
