"""`ppy convert`: source-preserving `.py` to `.ppy` conversion (spec 4.1)."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path

import libcst as cst
import libcst.matchers as m
from libcst.metadata import MetadataWrapper, PositionProvider

from ..analysis import types as T
from ..analysis.binding import bind_ast_call
from ..analysis.inference import (
    callee_qualname,
    has_source_annotation,
    is_self_attribute,
    refine_with_call_sites,
)
from ..analysis.render import render_annotation
from ..diagnostics import Diagnostic, Severity
from ..frontend.source import span_of
from .formatting import FormatterFailed, format_source, normalize_source
from .pipeline import analyze_paths, collect_sources, open_project
from .reporting import Reporter

__all__ = ["ConversionPlan", "convert_source", "run_convert"]

_DYNAMIC_CALLS = {"eval", "exec", "compile", "globals", "locals", "vars", "__import__"}

#: Names PEP 585 moved out of `typing`.
_COLLECTIONS_ABC = frozenset(
    {
        "Sequence",
        "Iterable",
        "Iterator",
        "Mapping",
        "MutableMapping",
        "MutableSequence",
        "Callable",
        "Generator",
        "Coroutine",
        "Awaitable",
        "AsyncIterable",
        "AsyncIterator",
        "Container",
        "Collection",
        "Set",
    }
)


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
    #: Modules of this project that the file imports. `import ppy` installs the
    #: loader those need, so it has to be placed ahead of them.
    local_imports: set[str] = field(default_factory=set)
    #: Classes proven safe to move, or None to move any (aggressive mode).
    hoistable: frozenset[str] | None = frozenset()
    #: Top-level definitions a hoist may cross (safe mode only): a crossed
    #: decorator that observes module state would see the moved class early.
    reorder_safe: frozenset[str] | None = None
    typing_imports: set[str] = field(default_factory=set)
    ppy_imports: set[str] = field(default_factory=set)
    needs_ppy: bool = False

    @property
    def is_empty(self) -> bool:
        return not (
            self.params
            or self.returns
            or self.assignments
            or self.fields
            or self.decorators
            or self.buffers
            or self.needs_ppy
        )


def run_convert(options: argparse.Namespace, reporter: Reporter) -> int:
    target: Path = options.path
    if not target.exists():
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{target} does not exist"))
        return 2

    project = open_project(target)
    project.config.strict = False
    if getattr(options, "hoist_classes", None):
        project.config.convert.hoist_classes = options.hoist_classes
    sources = [p for p in collect_sources(target) if p.suffix == ".py"]
    if not sources:
        reporter.note(f"no .py sources found under {target}")
        return 0

    # A project conversion analyzes the whole module and call graph at once.
    bundle = analyze_paths(project, sources, backend="python")
    observed = refine_with_call_sites(bundle, bundle.diagnostics)

    # A plan built over broken analysis writes broken contracts; an error
    # anywhere is a reason to write nothing anywhere. The verdict comes from
    # a fresh analysis *after* inference -- the first pass over untyped input
    # is full of unknowns that inference exists to resolve -- and dynamic
    # features are exempt: converting them faithfully is this command's job,
    # and `ppy check` will still demand their `ppy.dynamic` boundary later.
    from ..analysis.global_writes import build_write_index
    from ..analysis.reflection import build_reflection_index

    # `Final` needs the whole project's word, not just the files being
    # converted: a reverse dependency assigning `foo.NAME` is invisible to
    # the bundle and disqualifies the name all the same. The same goes for
    # reflection: whoever reads `f.__annotations__` may live anywhere.
    bundle.global_writes = build_write_index(project.root, project.config.source_roots)
    bundle.reflection = build_reflection_index(project.root, project.config.source_roots)

    fatal = _fatal_findings(bundle, reporter)
    if fatal and not options.dry_run:
        reporter.note(f"{fatal} error(s); nothing was converted")
        return 1

    promotions: list[BufferPromotion] = []
    blocked: list[tuple[object, str, str]] = []
    if getattr(options, "promote_buffers", False):
        promotions, blocked = plan_buffer_promotions(bundle)

    ready: list[tuple[Path, Path, str]] = []
    failures = fatal
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
        # Conversion itself stays deterministic, so the same input gives the
        # same output everywhere. The project's own formatter is applied on top
        # only when asked for, because it is not.
        converted = convert_source(original, plan)
        if _wants_formatting(options, project):
            try:
                # The plan resolved the imports, so it knows exactly which of
                # them are first-party; the formatter must not have to guess.
                converted = format_source(converted, path, frozenset(plan.local_imports))
            except FormatterFailed as failure:
                reporter.emit(
                    Diagnostic(
                        "E1802",
                        Severity.ERROR,
                        f"`{failure.tool}` failed while formatting {path}",
                        help=failure.detail or "run it directly to see what it objects to",
                    )
                )
                failures += 1
                continue
        # `--in-place` replaces the source: the module becomes `.ppy` and the
        # `.py` goes, which is what leaves a project importable afterwards.
        destination = path.with_suffix(".ppy")

        if options.dry_run:
            print(f"# ---- {destination} ----")
            print(converted)
            continue
        if destination.exists() and not options.force:
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
        ready.append((path, destination, converted))

    if options.dry_run:
        return 1 if failures else 0
    if failures:
        # Atomic: a project conversion either lands whole or not at all. A
        # tree left half `.py` and half `.ppy` by `--in-place` would not even
        # import, and there is no good half of that outcome to keep.
        reporter.note(f"{failures} error(s); nothing was converted")
        return 1

    written: list[Path] = []
    for path, destination, converted in ready:
        destination.write_text(converted, encoding="utf-8")
        if options.in_place:
            path.unlink()
        written.append(destination)

    if written:
        reporter.note(f"converted {len(written)} file(s): " + ", ".join(str(p) for p in written))
        _warn_about_shadowed_sources(written, reporter)
    return 0


#: Source problems the frontend reports before analysis can even start.
_STRUCTURAL = ("E1001", "E1002", "E1003", "E9001")


def _fatal_findings(bundle, reporter: Reporter) -> int:  # type: ignore[no-untyped-def]
    """Report what blocks this conversion, and say how much did."""
    from ..analysis.checker import analyze
    from ..diagnostics import DiagnosticBag

    fatal = 0
    for diagnostic in bundle.diagnostics:
        if diagnostic.severity is Severity.ERROR and diagnostic.code in _STRUCTURAL:
            reporter.emit(diagnostic)
            fatal += 1
    settled = DiagnosticBag()
    bundle.analysis = analyze(
        bundle.symbols,
        settled,
        strict=False,
        dynamic_policy=bundle.project.config.dynamic_boundaries,
        plugins=bundle.project.plugins,
    )
    for diagnostic in settled:
        if diagnostic.severity is not Severity.ERROR:
            continue
        if diagnostic.code.startswith("E15"):
            # A dynamic feature converts fine; it is `ppy check` that will
            # insist on its boundary.
            continue
        reporter.emit(diagnostic)
        fatal += 1
    return fatal


def _may_materialize(info, plugins, reflection=None) -> bool:  # type: ignore[no-untyped-def]
    """May inferred annotations be written into this function's source?

    "The analysis knows the type" and "the type is safe to write down" are
    different claims. A decorator that reads `__annotations__`, or one nobody
    can vouch for, makes the second one false; so does any code anywhere in
    the project inspecting this function's annotations at runtime -- it saw
    an untyped function, and the conversion must hand it the same one.
    """
    from ..analysis.decorators import semantics_of

    if info.directive("reflective") is not None:
        # `@ppy.reflective` is the author saying the annotations, as written,
        # are runtime-visible state; the toolchain leaves them alone.
        return False
    if reflection is not None and reflection.blocks_function(info.name, info.qualname):
        return False
    for name in info.decorators:
        known = semantics_of(name, plugins)
        if known is None or known.reads_annotations:
            return False
    return True


def _reflection_index(bundle):  # type: ignore[no-untyped-def]
    if bundle is None:
        return None
    existing = getattr(bundle, "reflection", None)
    if existing is not None:
        return existing
    from ..analysis.reflection import build_reflection_index

    index = build_reflection_index(bundle.project.root, bundle.project.config.source_roots)
    bundle.reflection = index
    return index


def _hoistable_classes(symbols, bundle):  # type: ignore[no-untyped-def]
    """Which classes may move, and which definitions they may move across.

    Defining a class runs its decorators, bases, keywords, and body; so does
    every `def` it would cross. A hoist is meaning-preserving only when both
    sides are reorder-safe -- a crossed decorator that probes `globals()`
    would otherwise observe the moved class ahead of time. Aggressive mode
    waives all of it, explicitly.
    """
    from ..analysis.decorators import definition_time_reorder_safe

    mode = bundle.project.config.convert.hoist_classes
    if mode == "off":
        return frozenset(), frozenset()
    if mode == "aggressive":
        return None, None
    identify = bundle.symbols.resolver(symbols).decorator_identity
    safe = frozenset(
        node.name
        for node in symbols.module.tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and definition_time_reorder_safe(node, bundle.project.plugins, identify)
    )
    return (
        frozenset(n.name for n in symbols.module.tree.body if isinstance(n, ast.ClassDef)) & safe,
        safe,
    )


def _wants_formatting(options: argparse.Namespace, project) -> bool:  # type: ignore[no-untyped-def]
    """`--format`, or `[tool.ppy.convert] format = true`."""
    if getattr(options, "apply_format", False):
        return True
    return bool(project.config.convert.format)


def _warn_about_shadowed_sources(written: list[Path], reporter: Reporter) -> None:
    """Both `foo.py` and `foo.ppy` now exist, which a project may not contain.

    Conversion leaves the original in place on purpose, but until it is gone
    the module is ambiguous and `ppy check` refuses it, so say what to do next
    rather than let the next command report a surprise.
    """
    shadowed = [path for path in written if path.with_suffix(".py").exists()]
    if not shadowed:
        return
    listed = ", ".join(str(path.with_suffix(".py")) for path in shadowed[:3])
    if len(shadowed) > 3:
        listed += f", and {len(shadowed) - 3} more"
    reporter.emit(
        Diagnostic(
            "W2005",
            Severity.WARNING,
            f"{len(shadowed)} module(s) now have both a .py and a .ppy source: {listed}",
            help="remove the .py sources, or convert with --in-place, before "
            "running `ppy check`; a module may not be provided by both",
        )
    )


def _module_for(bundle, path: Path) -> str | None:  # type: ignore[no-untyped-def]
    resolved = path.resolve()
    for name, symbols in bundle.symbols.modules.items():
        if symbols.path == resolved:
            return name
    return None


def build_plan(  # type: ignore[no-untyped-def]
    bundle,
    module_name: str,
    observed: dict[tuple[str, int], T.Type] | None = None,
    promotions: list[BufferPromotion] | None = None,
    blocked: list[tuple[object, str, str]] | None = None,
) -> tuple[ConversionPlan, list[Diagnostic]]:
    """Decide which annotations to insert, using interprocedural evidence."""
    plan = ConversionPlan(needs_ppy=True)
    diagnostics: list[Diagnostic] = []
    symbols = bundle.symbols.modules[module_name]
    plan.forward = _forward_references(symbols)
    analysis = bundle.analysis.modules.get(module_name)
    if observed is None:
        observed = refine_with_call_sites(bundle, bundle.diagnostics)

    functions = list(symbols.functions.values())
    for cls in symbols.classes.values():
        functions.extend(cls.methods.values())

    reflection = _reflection_index(bundle)
    for info in functions:
        line = info.node.lineno
        if not _may_materialize(info, bundle.project.plugins, reflection):
            # The analysis still knows the types; writing them down is a
            # separate decision, and an unknown decorator may be reading
            # `__annotations__` -- new annotations would change its input.
            continue
        for index, param in enumerate(info.params):
            if param.kind in {"var_positional", "var_keyword"}:
                continue
            if index == 0 and info.is_method and not info.is_static:
                # The receiver's type is the class it is defined in.
                continue
            if has_source_annotation(info, param.name):
                continue
            # Inference has the last word: it started from the observed types
            # and then settled them, widening a read-only container to the
            # protocol its body needs.
            candidate = param.type if param.known else observed.get((info.qualname, index))
            rendered = (
                render_annotation(candidate, local_module=module_name)
                if candidate is not None
                else None
            )
            if rendered is None:
                diagnostics.append(
                    Diagnostic(
                        "E1304",
                        Severity.WARNING,
                        f"cannot infer a stable type for parameter `{param.name}`",
                        span_of(info.path, info.node),
                        help=(
                            "annotate it explicitly, split the function, "
                            "or isolate the dynamic operation"
                        ),
                    )
                )
                continue
            plan.params[(line, param.name)] = rendered.text
            plan.typing_imports |= rendered.typing_imports
            plan.ppy_imports |= rendered.ppy_imports

        if not info.ret_annotated:
            # A body is the whole evidence for what it returns, so a finite set
            # of literals is a contract rather than a sample and is kept.
            rendered = render_annotation(
                info.ret, info.ret_facts, local_module=module_name, closed_literals=True
            )
            if rendered is not None:
                plan.returns[line] = rendered.text
                plan.typing_imports |= rendered.typing_imports
                plan.ppy_imports |= rendered.ppy_imports

    if bundle.project.config.inference.write_local_annotations and analysis is not None:
        _plan_module_globals(symbols, analysis, plan, module_name, bundle)
        _plan_empty_containers(symbols, analysis, plan, module_name)
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

    plan.hoistable, plan.reorder_safe = _hoistable_classes(symbols, bundle)
    plan.local_imports = {
        binding.module for binding in symbols.imports.values() if not binding.external
    }
    _settle_imports(plan)
    plan.needs_ppy = _uses_ppy(plan, symbols)
    diagnostics.extend(_dynamic_findings(symbols))
    return plan, diagnostics


def _settle_imports(plan: ConversionPlan) -> None:
    """Recompute the imports from the annotations that will actually be written.

    A later decision can replace an annotation -- a buffer promotion overwrites
    the widened `Sequence` form -- and the import it needed would otherwise be
    left behind unused.
    """
    written = (
        list(plan.params.values())
        + list(plan.returns.values())
        + list(plan.assignments.values())
        + list(plan.fields.values())
    )
    plan.typing_imports = {
        name for name in plan.typing_imports if any(_mentions(text, name) for text in written)
    }
    plan.ppy_imports = {
        name for name in plan.ppy_imports if any(_mentions(text, name) for text in written)
    }


def _uses_ppy(plan: ConversionPlan, symbols) -> bool:  # type: ignore[no-untyped-def]
    """Is `import ppy` actually needed, rather than merely conventional?

    The runtime import hook matters to a module that imports a sibling `.ppy`,
    and the decorators and markers matter to one that uses them. A module that
    does neither would carry an unused import.
    """
    if plan.decorators:
        return True
    annotations = list(plan.params.values()) + list(plan.returns.values())
    annotations += list(plan.assignments.values()) + list(plan.fields.values())
    if any("ppy." in text for text in annotations):
        return True
    return any(not binding.external for binding in symbols.imports.values())


#: `Final` is a contract about a module's interface, not a note that the
#: compiler happened to see one assignment. It is written where a programmer
#: would write it: on a name that already announces itself as a constant.
def _reads_as_a_constant(name: str) -> bool:
    return name.upper() == name and any(c.isalpha() for c in name)


def _plan_module_globals(  # type: ignore[no-untyped-def]
    symbols, analysis, plan: ConversionPlan, module_name: str, bundle=None
) -> None:
    reflection = _reflection_index(bundle)
    if reflection is not None and reflection.blocks_module_globals(module_name):
        # Someone prints this module's `__annotations__`; adding entries to
        # it would change what they see.
        return
    rebound = _rebound_globals(symbols)
    index = _write_index(bundle)
    annotated: set[str] = set()
    for node in symbols.module.tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id.startswith("__"):
            continue
        if target.id in annotated:
            # A name is declared where it is first bound; annotating a later
            # assignment of it would be a redeclaration.
            continue
        annotated.add(target.id)
        value_type = T.strip_literal(analysis.type_of(node.value))
        rendered = render_annotation(
            value_type, analysis.facts_of(node.value), local_module=module_name
        )
        if rendered is None:
            continue
        text = rendered.text
        if (
            target.id not in rebound
            and _reads_as_a_constant(target.id)
            and (index is None or index.can_emit_final(module_name, target.id))
        ):
            # Bound once, nowhere rebound in the project, and named as a
            # constant -- which is when `Final` says what the author meant.
            text = f"Final[{text}]"
            plan.typing_imports = plan.typing_imports | {"Final"}
        plan.assignments[(node.lineno, target.id)] = text
        plan.typing_imports |= rendered.typing_imports
        plan.ppy_imports |= rendered.ppy_imports


def _rebound_globals(symbols) -> set[str]:  # type: ignore[no-untyped-def]
    """Module-level names that are bound more than once, or unbound again.

    `Final` is a claim about the whole module, so a second binding anywhere --
    a `global` statement in some function, a `with ... as`, a `def` that
    shadows an assignment -- disqualifies the name. Python has more ways to
    bind a name than assignment, and each one that is missed here is a `Final`
    the module goes on to contradict.
    """
    counts: dict[str, int] = {}
    rebound: set[str] = set()

    def count(target: ast.expr | None) -> None:
        for name in ast.walk(target) if target is not None else ():
            if isinstance(name, ast.Name):
                counts[name.id] = counts.get(name.id, 0) + 1

    for node in ast.walk(symbols.module.tree):
        if isinstance(node, ast.Global):
            rebound.update(node.names)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    rebound.add(target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                count(target)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor, ast.NamedExpr)):
            count(node.target)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            counts[node.name] = counts.get(node.name, 0) + 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name.partition(".")[0]
                counts[bound] = counts.get(bound, 0) + 1
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                count(item.optional_vars)
        elif (
            (isinstance(node, ast.ExceptHandler) and node.name)
            or (isinstance(node, ast.MatchAs) and node.name)
            or (isinstance(node, ast.MatchStar) and node.name)
        ):
            counts[node.name] = counts.get(node.name, 0) + 1
        elif isinstance(node, ast.MatchMapping) and node.rest:
            counts[node.rest] = counts.get(node.rest, 0) + 1
    return rebound | {name for name, seen in counts.items() if seen > 1}


def _write_index(bundle):  # type: ignore[no-untyped-def]
    """The project-wide write index, built on demand for direct callers."""
    if bundle is None:
        return None
    existing = getattr(bundle, "global_writes", None)
    if existing is not None:
        return existing
    from ..analysis.global_writes import build_write_index

    index = build_write_index(bundle.project.root, bundle.project.config.source_roots)
    bundle.global_writes = index
    return index


def _plan_empty_containers(symbols, analysis, plan: ConversionPlan, module_name: str) -> None:  # type: ignore[no-untyped-def]
    """Annotate a local that starts empty and gets its element type later.

    `out = []` says nothing on its own; the element type comes from what is
    appended further down. Writing it where the binding happens is the only
    place another reader -- a person or a type checker -- can see it.
    """
    for info in symbols.functions.values():
        function = analysis.functions.get(info.qualname)
        if function is None:
            continue
        for node in ast.walk(info.node):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or not _is_empty_container(node.value):
                continue
            # At the assignment the container is still empty; the element type
            # is whatever it holds by the end of the function.
            bound = T.strip_literal(function.locals.get(target.id, T.UNKNOWN))
            if not isinstance(bound, T.Instance) or not bound.args:
                continue
            if any(isinstance(arg, (T.NeverType, T.UnknownType, T.AnyType)) for arg in bound.args):
                continue
            rendered = render_annotation(bound, local_module=module_name)
            if rendered is None:
                continue
            plan.assignments[(node.lineno, target.id)] = rendered.text
            plan.typing_imports |= rendered.typing_imports
            plan.ppy_imports |= rendered.ppy_imports


def _is_empty_container(value: ast.expr) -> bool:
    if isinstance(value, ast.List | ast.Dict) and not (
        getattr(value, "elts", None) or getattr(value, "keys", None)
    ):
        return True
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in {"list", "dict", "set"}
        and not value.args
    )


def _dynamic_findings(symbols) -> list[Diagnostic]:  # type: ignore[no-untyped-def]
    """Report the dynamic features that block conversion (spec 9)."""
    found: list[Diagnostic] = []
    found.extend(
        Diagnostic(
            "E1504",
            Severity.WARNING,
            f"`{node.func.id}` is a dynamic feature the converted module must isolate",
            span_of(symbols.path, node),
            help="wrap it in `with ppy.dynamic:` or mark the function `@ppy.dynamic`",
        )
        for node in ast.walk(symbols.module.tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _DYNAMIC_CALLS
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

        # Both module-level bindings and the locals that start empty are
        # planned, so being inside a function is no longer a reason to skip.
        if not isinstance(target, cst.Name):
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
    """Rewrite `source` through a concrete syntax tree, preserving trivia.

    Inserting annotations and imports disturbs the blank lines around what it
    touches, so the result is normalized before it is written. The built-in
    normalizer is used rather than an installed formatter, because the output
    has to be the same on every machine.
    """
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
    annotated = wrapper.visit(_Annotator(plan))
    imported = _insert_imports(annotated, plan)
    # A quoted annotation only exists because the class was not bound yet.
    # Moving the class above its first use removes the reason for the quotes.
    reordered = _unquote_resolved(_hoist_classes(imported, plan.hoistable, plan.reorder_safe))
    return normalize_source(reordered.code, frozenset(plan.local_imports))


def _insert_imports(module: cst.Module, plan: ConversionPlan) -> cst.Module:
    """Insert `import ppy` after the docstring and `__future__` imports."""
    existing = _existing_imports(module)
    # PEP 8 groups imports, and pylint checks the grouping, so a standard
    # library addition goes at the top of the block and the rest at the end.
    standard: list[cst.SimpleStatementLine] = []
    trailing: list[cst.SimpleStatementLine] = []

    if plan.needs_array and "array" not in existing:
        standard.append(cst.parse_statement("import array"))
    wanted = plan.typing_imports - existing
    # PEP 585 moved the container protocols to `collections.abc`; importing
    # them from `typing` is deprecated.
    abc_names = sorted(wanted & _COLLECTIONS_ABC)
    if abc_names:
        standard.append(cst.parse_statement(f"from collections.abc import {', '.join(abc_names)}"))
    typing_names = sorted(wanted - _COLLECTIONS_ABC)
    if typing_names:
        standard.append(cst.parse_statement(f"from typing import {', '.join(typing_names)}"))
    if plan.needs_ppy and "ppy" not in existing:
        trailing.append(cst.parse_statement("import ppy"))
    ppy_names = sorted(plan.ppy_imports - existing)
    if ppy_names:
        trailing.append(cst.parse_statement(f"from ppy import {', '.join(ppy_names)}"))

    if not standard and not trailing:
        return module
    body = list(module.body)
    head = _insert_index(module)
    tail = _import_block_end(module, head, plan.local_imports)
    return module.with_changes(
        body=[*body[:head], *standard, *body[head:tail], *trailing, *body[tail:]]
    )


def _import_block_end(module: cst.Module, start: int, local: set[str]) -> int:
    """Where a `ppy` import belongs: after the third-party ones, before the local.

    `import ppy` installs the loader that a sibling `.ppy` module needs, so it
    cannot follow one. That is also where PEP 8 puts it, third-party coming
    before first-party.
    """
    end = start
    for position in range(start, len(module.body)):
        statement = module.body[position]
        if not isinstance(statement, cst.SimpleStatementLine):
            break
        first = statement.body[0]
        if not isinstance(first, (cst.Import, cst.ImportFrom)):
            break
        if _imports_any(first, local):
            break
        end = position + 1
    return end


def _imports_any(statement: cst.BaseSmallStatement, local: set[str]) -> bool:
    """Does this import statement bring in one of the project's own modules?"""
    if isinstance(statement, cst.ImportFrom):
        module = statement.module
        return module is not None and _dotted(module).partition(".")[0] in local
    if isinstance(statement, cst.Import):
        return any(_dotted(alias.name).partition(".")[0] in local for alias in statement.names)
    return False


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
        is_future = (
            isinstance(first, cst.ImportFrom)
            and _dotted(first.module or cst.Name("")) == "__future__"
        )
        if is_docstring or is_future:
            index = position + 1
            continue
        break
    return index


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
        for source_module, line, _name in promotion.sources:
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
            if not is_self_attribute(target, initializer):
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
_LIST_ONLY_METHODS = frozenset(
    {
        "append",
        "extend",
        "insert",
        "pop",
        "remove",
        "clear",
        "sort",
        "copy",
        "count",
        "index",
    }
)


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
            if element is None or has_source_annotation(info, param.name):
                continue
            function = analysis.functions.get(qualname)
            blocker = _buffer_blocker(info.node, param.name, function.aliases if function else None)
            if blocker is not None:
                blocked.append((info, param.name, blocker))
                continue
            sources = _list_sources(bundle, qualname, index, element)
            if sources is None:
                blocked.append(
                    (
                        info,
                        param.name,
                        (
                            f"the values passed as `{param.name}` are not all traceable to a "
                            "single construction this converter can rewrite"
                        ),
                    )
                )
                continue
            found.append(
                BufferPromotion(
                    qualname,
                    info.node.lineno,
                    param.name,
                    element,
                    _ARRAY_CODES[element],
                    sources,
                )
            )
    return _consistent(found, bundle, blocked), blocked


def _consistent(found, bundle, blocked):  # type: ignore[no-untyped-def]
    """Keep only promotions whose rewritten values reach nothing that stayed a list.

    Rewriting a construction into `array.array` changes what every reader of
    that name receives. If one function that reads it was not promoted too, the
    rewrite would hand it a buffer where it declared a list, so the whole group
    has to be abandoned.
    """
    promoted = {(p.qualname, p.param) for p in found}
    while True:
        readers = _readers_of(bundle, {name for p in found for _m, _l, name in p.sources})
        conflicted = {name for name, users in readers.items() if not users <= promoted}
        if not conflicted:
            return found
        kept = []
        for promotion in found:
            names = {name for _m, _l, name in promotion.sources}
            if names & conflicted:
                blocked.append(
                    (
                        bundle.symbols.functions[promotion.qualname],
                        promotion.param,
                        "a value it receives is also read by a parameter that stayed a list",
                    )
                )
                continue
            kept.append(promotion)
        if len(kept) == len(found):
            return kept
        found = kept
        promoted = {(p.qualname, p.param) for p in found}


def _readers_of(bundle, names: set[str]) -> dict[str, set[tuple[str, str]]]:  # type: ignore[no-untyped-def]
    """Every `(function, parameter)` each of these module-level names reaches."""
    readers: dict[str, set[tuple[str, str]]] = {name: set() for name in names}
    for symbols in bundle.symbols.modules.values():
        for node in ast.walk(symbols.module.tree):
            if not isinstance(node, ast.Call):
                continue
            qualname = callee_qualname(bundle, symbols, node)
            info = bundle.symbols.functions.get(qualname or "")
            if info is None:
                continue
            for bound in bind_ast_call(info, node):
                argument = bound.value
                if isinstance(argument, ast.Name) and argument.id in readers:
                    readers[argument.id].add((qualname, bound.param.name))
    return readers


def _buffer_scalar(t: T.Type) -> str | None:
    """The element name of a `list[int]` or `list[float]`, if it is one."""
    base = T.strip_literal(t)
    # `Sequence` as well as `list`: a parameter that was widened to the protocol
    # is read-only by construction, which is what a borrowed buffer wants.
    if not isinstance(base, T.Instance) or base.name not in {"list", "Sequence"}:
        return None
    if len(base.args) != 1:
        return None
    element = T.strip_literal(base.args[0])
    if not isinstance(element, T.Instance) or element.name not in _ARRAY_CODES:
        return None
    return element.name


def _buffer_blocker(node: ast.AST, name: str, aliases=None) -> str | None:
    """What stops `name` from being served by a buffer, if anything.

    Uses are matched through the alias map: `view = values; view.append(0.0)`
    grows the parameter as surely as `values.append` would, and a borrowed
    buffer must not grow -- reallocation is exactly what borrowing rules out.
    """

    def is_use(candidate) -> bool:  # type: ignore[no-untyped-def]
        if not isinstance(candidate, ast.Name):
            return False
        if candidate.id == name:
            return True
        return aliases is not None and name in aliases.roots_at(candidate, candidate.id)

    for child in ast.walk(node):
        if (
            isinstance(child, ast.Attribute)
            and is_use(child.value)
            and child.attr in _LIST_ONLY_METHODS
        ):
            return f"`{name}.{child.attr}()` needs a list that can grow or reorder"
        if (
            isinstance(child, ast.Subscript)
            and is_use(child.value)
            and isinstance(child.slice, ast.Slice)
        ):
            return (
                f"`{name}` is sliced, which copies; indexing it element by "
                "element instead would let the memory be borrowed"
            )
        if isinstance(child, ast.BinOp) and (is_use(child.left) or is_use(child.right)):
            return f"`{name}` is used with `+` or `*`, which are list operations"
        if isinstance(child, ast.AugAssign) and (is_use(child.target) or is_use(child.value)):
            return f"`{name}` is grown or concatenated in place, which reallocates"
        if isinstance(child, ast.Delete):
            for target in child.targets:
                if isinstance(target, (ast.Subscript, ast.Attribute)) and is_use(target.value):
                    return f"`{name}` has items deleted, which a borrowed buffer cannot"
        if isinstance(child, (ast.Return, ast.Yield)) and is_use(child.value):
            return f"`{name}` is returned, so the caller would hold the buffer"
    return None


def _list_sources(  # type: ignore[no-untyped-def]
    bundle, qualname: str, index: int, element: str
) -> tuple[tuple[str, int, str], ...] | None:
    """Every list construction feeding this parameter, or None if any is opaque."""
    info = bundle.symbols.functions[qualname]
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
            if callee_qualname(bundle, symbols, node) != qualname:
                continue
            seen_call = True
            # Positional order, keywords by name -- the binder's rules, so a
            # call written `f(data=xs)` traces the same list `f(xs)` would,
            # and `f(other, data=xs)` never traces the wrong argument.
            reached = [b.value for b in bind_ast_call(info, node) if b.index == index]
            if len(reached) != 1 or not isinstance(reached[0], ast.Name):
                return None
            argument = reached[0]
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
    """Names bound exactly once to a list the converter can rewrite.

    Script code belongs in a `main()` rather than at module level, so the
    bodies of top-level functions are searched as well. A name is only usable
    when it is bound once across everything searched, which keeps two scopes
    that happen to share a name out of each other's way.
    """
    counts: dict[str, int] = {}
    lines: dict[str, int] = {}
    scopes = [symbols.module.tree.body]
    scopes += [
        statement.body
        for statement in symbols.module.tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for statement in [s for scope in scopes for s in scope]:
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


def _definition_time_names(node: cst.CSTNode) -> set[str]:
    """Names a class needs bound the moment its `class` statement executes.

    The body of a method does not run at class creation, but its decorators,
    defaults, and annotations do, and so do the bases and the class-level
    statements. Anything reachable that way has to already exist.
    """
    found: set[str] = set()

    def walk(target: cst.CSTNode) -> None:
        for child in target.children:
            if isinstance(child, cst.Name):
                found.add(child.value)
            walk(child)

    if isinstance(node, cst.ClassDef):
        for decorator in node.decorators:
            walk(decorator)
        for base in node.bases:
            walk(base)
        for keyword in node.keywords:
            walk(keyword)
        body = node.body.body if isinstance(node.body, cst.IndentedBlock) else []
        for statement in body:
            if isinstance(statement, cst.FunctionDef):
                for decorator in statement.decorators:
                    walk(decorator)
                walk(statement.params)
                if statement.returns is not None:
                    walk(statement.returns)
                continue
            walk(statement)
    else:
        walk(node)
    return found


def _hoist_classes(
    module: cst.Module,
    hoistable: frozenset[str] | None,
    reorder_safe: frozenset[str] | None = None,
) -> cst.Module:
    """Move a class above the definitions that annotate against it.

    A quoted annotation is only needed because the class is not bound yet.
    Moving the class up removes the reason for the quotes. Which classes may
    move at all was decided against the analysis (`hoistable`): a definition
    with observable effects stays put, and the annotation stays quoted.
    """
    body = list(module.body)
    moved = True
    passes = 0
    while moved and passes < len(body):
        moved = False
        passes += 1
        for index, statement in enumerate(body):
            if not isinstance(statement, cst.ClassDef):
                continue
            if hoistable is not None and statement.name.value not in hoistable:
                continue
            target = _earliest_position(body, index, statement, reorder_safe)
            if target is None or target >= index:
                continue
            body.insert(target, body.pop(index))
            moved = True
            break
    return module.with_changes(body=body)


def _earliest_position(
    body: list[cst.BaseStatement],
    index: int,
    statement: cst.ClassDef,
    reorder_safe: frozenset[str] | None = None,
) -> int | None:
    """The first slot this class can occupy without breaking a dependency."""
    needed = _definition_time_names(statement)
    position = index
    while position > 0:
        previous = body[position - 1]
        if not isinstance(previous, (cst.FunctionDef, cst.ClassDef)):
            break
        if isinstance(previous, cst.ClassDef) and previous.name.value in needed:
            break
        if isinstance(previous, cst.FunctionDef) and previous.name.value in needed:
            break
        if reorder_safe is not None and previous.name.value not in reorder_safe:
            # Crossing this definition would run its decorators and defaults
            # with the moved class already bound -- observable.
            break
        position -= 1
    if position == index:
        return None
    # Only worth moving when something ahead of it actually names this class.
    name = statement.name.value
    crossed = body[position:index]
    if not any(name in _annotation_names(other) for other in crossed):
        return None
    return position


def _annotation_names(node: cst.CSTNode) -> set[str]:
    """Every name appearing in an annotation, quoted or not."""
    found: set[str] = set()
    for annotation in m.findall(node, m.Annotation()):
        expression = annotation.annotation
        if isinstance(expression, cst.SimpleString):
            text = expression.raw_value
            try:
                expression = cst.parse_expression(text)
            except cst.ParserSyntaxError:
                continue
        for name in m.findall(expression, m.Name()):
            found.add(name.value)
    return found


def _unquote_resolved(module: cst.Module) -> cst.Module:
    """Drop quotes from an annotation whose names are all bound before it.

    Only a top-level statement can be judged this way: a method annotating
    against its own enclosing class still needs the quotes, because the class
    is not bound until its body has finished executing.
    """
    body = list(module.body)
    defined: set[str] = set()
    rewritten: list[cst.BaseStatement] = []
    for original in body:
        statement = original
        if isinstance(statement, (cst.FunctionDef, cst.ClassDef)):
            inner = statement.name.value
            statement = statement.with_changes(
                **_unquoted_signature(
                    statement, defined - {inner} if isinstance(statement, cst.ClassDef) else defined
                )
            )
            defined.add(inner)
        rewritten.append(statement)
    return module.with_changes(body=rewritten)


def _unquoted_signature(statement: cst.BaseStatement, defined: set[str]) -> dict:
    """Replacement fields for a definition whose annotations can lose quotes."""
    if not isinstance(statement, cst.FunctionDef):
        return {}
    changes: dict = {}
    params = statement.params
    updated = [
        param.with_changes(annotation=_unquote(param.annotation, defined))
        for param in params.params
    ]
    if any(new is not old for new, old in zip(updated, params.params, strict=False)):
        changes["params"] = params.with_changes(params=updated)
    returns = _unquote(statement.returns, defined)
    if returns is not statement.returns:
        changes["returns"] = returns
    return changes


def _unquote(annotation, defined: set[str]):  # type: ignore[no-untyped-def]
    if annotation is None or not isinstance(annotation.annotation, cst.SimpleString):
        return annotation
    text = annotation.annotation.raw_value
    try:
        expression = cst.parse_expression(text)
    except cst.ParserSyntaxError:
        return annotation
    names = {name.value for name in m.findall(expression, m.Name())}
    if not names <= defined | _ALWAYS_BOUND:
        return annotation
    return annotation.with_changes(annotation=expression)


#: Names an annotation may use that are never module-level definitions here.
_ALWAYS_BOUND = {
    "int",
    "float",
    "str",
    "bytes",
    "bool",
    "complex",
    "None",
    "object",
    "list",
    "dict",
    "set",
    "tuple",
    "frozenset",
    "type",
    "Optional",
    "Union",
    "Any",
    "Annotated",
    "Literal",
    "Callable",
    "Sequence",
    "Iterable",
    "Mapping",
    "Buffer",
    "ppy",
}


def _forward_references(symbols) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Where each class in this module becomes usable as a runtime name."""
    return {
        info.name: (info.node.end_lineno or info.node.lineno) for info in symbols.classes.values()
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
