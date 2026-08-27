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
    #: `ppy` decorators to attach, keyed by the line the `def` starts on.
    decorators: dict[int, tuple[str, ...]] = field(default_factory=dict)
    #: Module-level list constructions to wrap in `array.array`, by line.
    buffers: dict[int, str] = field(default_factory=dict)
    needs_array: bool = False
    typing_imports: set[str] = field(default_factory=set)
    ppy_imports: set[str] = field(default_factory=set)
    needs_ppy: bool = False

    @property
    def is_empty(self) -> bool:
        return not (
            self.params or self.returns or self.assignments or self.fields
            or self.decorators or self.buffers or self.needs_ppy
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
    promotions: list[BufferPromotion] = []
    blocked: list[tuple[object, str, str]] = []
    if getattr(options, "promote_buffers", False):
        promotions, blocked = plan_buffer_promotions(bundle)

    written: list[Path] = []
    failures = 0
    for path in sources:
        module_name = _module_for(bundle, path)
        if module_name is None:
            continue
        plan, diagnostics = build_plan(bundle, module_name, observed, promotions, blocked)
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
    promotions: "list[BufferPromotion] | None" = None,
    blocked: "list[tuple[object, str, str]] | None" = None,
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
    if analysis is not None:
        _plan_purity(functions, analysis, plan)
    if promotions:
        _apply_promotions(promotions, module_name, plan, diagnostics, bundle)
    for info, name, reason in blocked or ():
        if getattr(info, "module", None) != module_name:
            continue
        diagnostics.append(
            Diagnostic(
                "R3003",
                Severity.REMARK,
                f"`{name}` could be a borrowed buffer, but {reason}",
                span_of(info.path, info.node),
                help="a borrowed buffer is what lets this loop lower to native code",
            )
        )

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
        wanted = self.plan.decorators.get(line, ())
        if wanted:
            present = {_dotted(d.decorator) for d in updated.decorators}
            added = [
                cst.Decorator(decorator=cst.parse_expression(text))
                for text in wanted
                if text.partition("(")[0] not in present
            ]
            if added:
                updated = updated.with_changes(decorators=[*added, *updated.decorators])
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

        code = self.plan.buffers.get(line)
        if code is not None and isinstance(target, cst.Name):
            wrapped = cst.parse_expression(
                f'array.array("{code}", {cst.Module(body=[]).code_for_node(statement.value)})'
            )
            return updated.with_changes(
                body=[cst.Assign(targets=list(statement.targets), value=wrapped)]
            )

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

    if plan.needs_array and "array" not in existing:
        additions.append(cst.parse_statement("import array"))
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


def _apply_promotions(  # type: ignore[no-untyped-def]
    promotions, module_name: str, plan: ConversionPlan, diagnostics, bundle
) -> None:
    """Write the promoted signatures and the `array.array` calls that feed them."""
    for promotion in promotions:
        info = bundle.symbols.functions.get(promotion.qualname)
        if info is not None and info.module == module_name:
            plan.params[(promotion.line, promotion.param)] = f"Buffer[{promotion.element}]"
            plan.ppy_imports.add("Buffer")
            diagnostics.append(
                Diagnostic(
                    "R3002",
                    Severity.REMARK,
                    f"`{promotion.param}` is only read by index, so it is declared "
                    f"`Buffer[{promotion.element}]` and borrowed instead of copied",
                    span_of(info.path, info.node),
                    help="the values feeding it become `array.array` so the memory can be lent",
                )
            )
        for source_module, line, name in promotion.sources:
            if source_module != module_name:
                continue
            plan.buffers[line] = promotion.code
            plan.needs_array = True


def _plan_purity(functions, analysis, plan: ConversionPlan) -> None:  # type: ignore[no-untyped-def]
    """Attach `@ppy.pure` where the checker already proved the contract holds.

    The decorator returns the function unchanged, so this cannot alter what
    plain CPython does; what it adds is a contract the compiler will keep
    verifying as the code changes, and which unlocks the optimizations that
    depend on purity.
    """
    for info in functions:
        if info.directives or info.decorators:
            # Leave a function that already carries decorators alone; ordering
            # against an unknown transform is not ours to decide.
            continue
        function = analysis.functions.get(info.qualname)
        if function is None or not function.verified_pure:
            continue
        if not info.node.body or info.is_async:
            continue
        plan.decorators[info.node.lineno] = ("ppy.pure",)


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


#: Buffer element codes `array.array` uses, by PPY scalar name.
_ARRAY_CODES = {"float": "d", "int": "q"}

#: Ways of using a name that a borrowed buffer cannot serve. `array.array`
#: does have `append`, but growing it reallocates, which is exactly what
#: borrowing rules out.
_LIST_ONLY_METHODS = frozenset({
    "append", "extend", "insert", "pop", "remove", "clear", "sort", "copy", "count", "index",
})


@dataclass
class BufferPromotion:
    """One `list[T]` parameter to declare as `Buffer[T]`, with its call sites."""

    qualname: str
    line: int
    param: str
    element: str
    code: str
    #: `(module, line, name)` of each list construction that has to become an
    #: `array.array` for the promoted signature to still type-check.
    sources: tuple[tuple[str, int, str], ...] = ()


def plan_buffer_promotions(bundle):  # type: ignore[no-untyped-def]
    """Find `list[float]` parameters that would lower natively as buffers.

    A promotion only pays off if the caller passes real buffer memory, so a
    parameter is promoted only when every construction feeding it can be
    rewritten too. Anything the converter cannot follow is left alone.
    """
    found: list[BufferPromotion] = []
    blocked: list[tuple[object, str, str]] = []
    for qualname, info in bundle.symbols.functions.items():
        analysis = bundle.analysis.modules.get(info.module)
        if analysis is None or info.is_method:
            continue
        exported = bundle.symbols.modules.get(info.module)
        if exported is not None and exported.all_exports and info.name in exported.all_exports:
            continue
        for index, param in enumerate(info.params):
            element = _buffer_scalar(param.type)
            if element is None or _has_source_annotation(info, param.name):
                continue
            blocker = _buffer_blocker(info.node, param.name)
            if blocker is not None:
                blocked.append((info, param.name, blocker))
                continue
            sources = _list_sources(bundle, qualname, index, element)
            if sources is None:
                blocked.append((
                    info, param.name,
                    f"the values passed as `{param.name}` are not all traceable to a "
                    "single construction this converter can rewrite",
                ))
                continue
            found.append(BufferPromotion(
                qualname, info.node.lineno, param.name,
                element, _ARRAY_CODES[element], sources,
            ))
    return found, blocked


def _buffer_scalar(t: T.Type) -> str | None:
    """The element name of a `list[int]` or `list[float]`, if it is one."""
    base = T.strip_literal(t)
    if not isinstance(base, T.Instance) or base.name != "list" or len(base.args) != 1:
        return None
    element = T.strip_literal(base.args[0])
    if not isinstance(element, T.Instance) or element.name not in _ARRAY_CODES:
        return None
    return element.name


def _buffer_blocker(node: ast.AST, name: str) -> str | None:
    """What stops `name` from being served by a buffer, if anything."""
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            if isinstance(child.value, ast.Name) and child.value.id == name:
                if child.attr in _LIST_ONLY_METHODS:
                    return f"`{name}.{child.attr}()` needs a list that can grow or reorder"
        if isinstance(child, ast.Subscript):
            if isinstance(child.value, ast.Name) and child.value.id == name:
                if isinstance(child.slice, ast.Slice):
                    return (
                        f"`{name}` is sliced, which copies; indexing it element by "
                        "element instead would let the memory be borrowed"
                    )
        if isinstance(child, ast.BinOp):
            for side in (child.left, child.right):
                if isinstance(side, ast.Name) and side.id == name:
                    return f"`{name}` is used with `+` or `*`, which are list operations"
        if isinstance(child, (ast.Return, ast.Yield)):
            value = child.value
            if isinstance(value, ast.Name) and value.id == name:
                return f"`{name}` is returned, so the caller would hold the buffer"
    return None


def _list_sources(  # type: ignore[no-untyped-def]
    bundle, qualname: str, index: int, element: str
) -> tuple[tuple[str, int, str], ...] | None:
    """Every list construction feeding this parameter, or None if any is opaque."""
    sources: list[tuple[str, int, str]] = []
    seen_call = False
    for module_name, symbols in bundle.symbols.modules.items():
        analysis = bundle.analysis.modules.get(module_name)
        if analysis is None:
            continue
        assignments = _module_list_assignments(symbols)
        for node in ast.walk(symbols.module.tree):
            if not isinstance(node, ast.Call):
                continue
            if _callee_qualname(bundle, symbols, node) != qualname:
                continue
            if index >= len(node.args):
                return None
            seen_call = True
            argument = node.args[index]
            if not isinstance(argument, ast.Name):
                return None
            line = assignments.get(argument.id)
            if line is None:
                return None
            observed = T.strip_literal(analysis.type_of(argument))
            if _buffer_scalar(observed) != element:
                return None
            sources.append((module_name, line, argument.id))
    if not seen_call or not sources:
        return None
    return tuple(dict.fromkeys(sources))


def _module_list_assignments(symbols) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Module-level names bound exactly once to a list the converter can rewrite."""
    counts: dict[str, int] = {}
    lines: dict[str, int] = {}
    for statement in symbols.module.tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(statement, ast.AnnAssign) and statement.value is not None:
            target, value = statement.target, statement.value
        elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        if not isinstance(target, ast.Name) or value is None:
            continue
        counts[target.id] = counts.get(target.id, 0) + 1
        # `array.array(code, ...)` accepts any iterable of the right scalars, so
        # the construction does not have to be a literal -- only unambiguous.
        lines[target.id] = statement.lineno
    return {name: line for name, line in lines.items() if counts.get(name) == 1}





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
