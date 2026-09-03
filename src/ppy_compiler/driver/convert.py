"""`ppy convert`: source-preserving `.py` to `.ppy` conversion (spec 4.1)."""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

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
from .formatting import FormatterFailed, format_source
from .pipeline import analyze_paths, collect_sources, open_project
from .plan import ConversionPlan, mentions
from .reporting import Reporter
from .rewrite import convert_source

__all__ = ["ConversionPlan", "convert_source", "run_convert"]

_DYNAMIC_CALLS = {"eval", "exec", "compile", "globals", "locals", "vars", "__import__"}


def run_convert(options: argparse.Namespace, reporter: Reporter) -> int:
    """`ppy convert`: strict staticization. The output must be valid strict PPY.

    There is deliberately no escape hatch here: a convert that can be asked
    not to be strict is two pipelines wearing one name, and `ppy migrate` is
    already the permissive one.
    """
    return _run_conversion(options, reporter, strict=True)


def run_migrate(options: argparse.Namespace, reporter: Reporter) -> int:
    """`ppy migrate`: permissive rewriting of normal Python toward strict PPY."""
    return _run_conversion(options, reporter, strict=False, command="migrate")


def _run_conversion(
    options: argparse.Namespace, reporter: Reporter, *, strict: bool, command: str = "convert"
) -> int:
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

    # Migration rewrites first, then staticizes what the rewrites produced:
    # a `setattr` that became `obj.attr = value` is an attribute write by the
    # time the checker looks at it.
    from ..migration import MigrationReport, apply_passes

    report = MigrationReport(root=project.root)
    rewritten: dict[Path, str] = {}
    if command == "migrate":
        for path in sources:
            text = path.read_text(encoding="utf-8")
            fixed, fixes = apply_passes(text)
            if fixes:
                rewritten[path] = fixed
                rewritten[path.resolve()] = fixed
                report.add_rewrites(path, fixes)

    # A project conversion analyzes the whole module and call graph at once.
    bundle = analyze_paths(project, sources, backend="python", overlays=rewritten or None)
    observed = refine_with_call_sites(bundle, bundle.diagnostics)

    # A plan built over broken analysis writes broken contracts; an error
    # anywhere is a reason to write nothing anywhere. The verdict comes from
    # a fresh analysis *after* inference -- the first pass over untyped input
    # is full of unknowns that inference exists to resolve -- and dynamic
    # features are exempt here: `migrate` converts them faithfully on purpose,
    # and for `convert` the strict gate at the end is the stage that rejects
    # them, with the checker's own explanations.
    from ..analysis.global_writes import build_write_index
    from ..analysis.project_scan import scan_project
    from ..analysis.reflection import build_reflection_index

    # `Final` needs the whole project's word, not just the files being
    # converted: a reverse dependency assigning `foo.NAME` is invisible to
    # the bundle and disqualifies the name all the same. The same goes for
    # reflection: whoever reads `f.__annotations__` may live anywhere. Both
    # read the same files, so both read them from one scan.
    scan = scan_project(project.root, project.config.source_roots)
    bundle.global_writes = build_write_index(project.root, project.config.source_roots, scan=scan)
    bundle.reflection = build_reflection_index(project.root, project.config.source_roots, scan=scan)

    fatal = _fatal_findings(bundle, reporter, strict=strict)
    if fatal and not options.dry_run:
        reporter.note(f"{fatal} error(s); nothing was converted")
        return 1

    promotions: list[BufferPromotion] = []
    blocked: list[tuple[object, str, str]] = []
    if getattr(options, "promote_buffers", False):
        promotions, blocked = plan_buffer_promotions(bundle)

    produced: list[tuple[Path, Path, str]] = []
    failures = fatal
    pinned: dict[tuple[Path, int], str] = {}
    for path in sources:
        module_name = _module_for(bundle, path)
        if module_name is None:
            continue
        report.files_scanned += 1
        plan, diagnostics = build_plan(bundle, module_name, observed, promotions, blocked)
        if strict:
            pinned.update(_blocked_materializations(bundle, module_name))
        if command == "migrate":
            for key, count in plan.annotation_counts().items():
                report.annotations[key] = report.annotations.get(key, 0) + count
        for diagnostic in diagnostics:
            if strict and diagnostic.code == "E1504":
                # The strict gate is about to report the same dynamic feature
                # as the error it is; the migration-facing advisory would say
                # it twice.
                continue
            reporter.emit(diagnostic)
            if diagnostic.severity is Severity.ERROR:
                failures += 1
            if command == "migrate" and diagnostic.code == "R3003":
                # Opportunities come from planning; everything unresolved is
                # classified from the strict check of the final output below.
                report.add_finding(diagnostic)

        original = rewritten.get(path) or path.read_text(encoding="utf-8")
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
        produced.append((path, path.with_suffix(".ppy"), converted))
        if converted != path.read_text(encoding="utf-8"):
            report.files_changed += 1

    if command == "migrate" and not failures:
        # The report is about the *final* output: what was written is
        # re-analyzed in strict mode, and whatever the strict language still
        # rejects is what the migration has left to do. A strict failure is
        # not a migration failure -- the files are written either way.
        for diagnostic in _strict_findings(project, sources, produced):
            if diagnostic.severity is Severity.ERROR:
                report.strict_errors += 1
            report.add_finding(diagnostic)
        for line in report.summary_lines():
            reporter.note(line)
        if getattr(options, "report", None):
            options.report.write_text(report.to_json(), encoding="utf-8")

    if strict and not failures:
        failures += _strict_gate(project, sources, produced, pinned, reporter)

    if getattr(options, "diff", False):
        import difflib

        for path, destination, converted in produced:
            disk = path.read_text(encoding="utf-8")
            sys.stdout.writelines(
                difflib.unified_diff(
                    disk.splitlines(keepends=True),
                    converted.splitlines(keepends=True),
                    fromfile=str(path),
                    tofile=str(destination),
                )
            )
        return 1 if failures else 0

    if options.dry_run:
        for _path, destination, converted in produced:
            print(f"# ---- {destination} ----")
            print(converted)
        return 1 if failures else 0

    # `--in-place` replaces the source: the module becomes `.ppy` and the
    # `.py` goes, which is what leaves a project importable afterwards.
    ready: list[tuple[Path, Path, str]] = []
    for path, destination, converted in produced:
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
        verb = "migrated" if command == "migrate" else "converted"
        reporter.note(f"{verb} {len(written)} file(s): " + ", ".join(str(p) for p in written))
        _warn_about_shadowed_sources(written, reporter)
    return 0


#: Findings about the state of the tree rather than the produced source; the
#: overlay analysis sees a half-converted world the write step never leaves.
_TREE_STATE = frozenset({"E1003"})


def _strict_findings(project, sources, produced) -> list[Diagnostic]:  # type: ignore[no-untyped-def]
    """What the strict language says about the text about to be written.

    The produced text is re-analyzed in place of the originals, in strict
    mode with dynamic-feature enforcement on. `ppy convert` turns the errors
    into a refusal; `ppy migrate` turns them into the report's account of
    what remains.
    """
    overlays: dict[Path, str] = {}
    for path, _destination, text in produced:
        overlays[path] = text
        overlays[path.resolve()] = text
    previous = project.config.strict
    project.config.strict = True
    try:
        checked = analyze_paths(project, sources, backend="python", overlays=overlays)
    finally:
        project.config.strict = previous
    return [d for d in checked.diagnostics if d.code not in _TREE_STATE]


def _strict_gate(project, sources, produced, pinned, reporter: Reporter) -> int:  # type: ignore[no-untyped-def]
    """Strict conversion may not write invalid strict PPY, so it refuses to."""
    errors = 0
    for diagnostic in _strict_findings(project, sources, produced):
        if diagnostic.severity is not Severity.ERROR:
            continue
        _explain_pinned(diagnostic, pinned)
        reporter.emit(diagnostic)
        errors += 1
    return errors


def _blocked_materializations(bundle, module_name: str) -> dict[tuple[Path, int], str]:  # type: ignore[no-untyped-def]
    """Why each skipped function was skipped, for the strict gate's messages."""
    from ..analysis.decorators import semantics_of

    symbols = bundle.symbols.modules[module_name]
    functions = list(symbols.functions.values())
    for cls in symbols.classes.values():
        functions.extend(cls.methods.values())
    reflection = _reflection_index(bundle)
    found: dict[tuple[Path, int], str] = {}
    for info in functions:
        if info.directive("reflective") is not None:
            # The author pinned the annotations deliberately; the plain
            # `annotate it yourself` guidance is already the right one.
            continue
        if _may_materialize(info, bundle.project.plugins, reflection):
            continue
        reason = "materialization was blocked"
        if reflection is not None and reflection.blocks_function(info.name, info.qualname):
            reason = "the project reads this function's annotations at runtime"
        else:
            for name in info.decorators:
                known = semantics_of(name, bundle.project.plugins)
                if known is None:
                    reason = f"decorator `@{name}` has semantics nobody vouches for"
                    break
                if known.reads_annotations:
                    reason = f"decorator `@{name}` reads `__annotations__` at definition time"
                    break
        found[(info.path, info.node.lineno)] = reason
    return found


def _explain_pinned(diagnostic: Diagnostic, pinned: dict[tuple[Path, int], str]) -> None:
    """Replace a type-gap error's help when conversion knew and could not write.

    The default help says `annotate or run ppy convert`, which is circular
    advice coming from `ppy convert` itself; what the reader needs is why the
    inferred annotation was withheld and which ways out exist.
    """
    if diagnostic.code not in {"E1201", "E1204", "E1304"} or diagnostic.span is None:
        return
    reason = pinned.get((Path(diagnostic.span.path).resolve(), diagnostic.span.line))
    if reason is None:
        if diagnostic.code in {"E1201", "E1304"}:
            # The checker's default help suggests running `ppy convert`,
            # which is exactly what produced this message.
            diagnostic.help = (
                "annotate it explicitly; strict conversion could not infer a stable type to insert"
            )
        return
    diagnostic.help = (
        f"{reason}, so the inferred annotations were not written; annotate the "
        "function yourself, mark it `@ppy.reflective`, or run `ppy migrate`"
    )


#: Source problems the frontend reports before analysis can even start.
_STRUCTURAL = ("E1001", "E1002", "E1003", "E9001")


def _fatal_findings(bundle, reporter: Reporter, strict: bool = False) -> int:  # type: ignore[no-untyped-def]
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
        if diagnostic.code == "W2006":
            # The one warning that belongs with the errors: it says how many
            # more there were and why they are not here.
            reporter.emit(diagnostic)
            continue
        if diagnostic.severity is not Severity.ERROR:
            continue
        if diagnostic.code.startswith("E15") and not strict:
            # `ppy migrate` converts a dynamic feature faithfully on purpose;
            # the strict pass over its final output classifies it for the
            # report, and `ppy check` will insist on its boundary.
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
        _plan_function_locals(symbols, analysis, plan, module_name)
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
    plan.rewrite_input = _reads_with_input(symbols.module.tree)
    plan.needs_ppy = _uses_ppy(plan, symbols) or plan.rewrite_input
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
        name for name in plan.typing_imports if any(mentions(text, name) for text in written)
    }
    plan.ppy_imports = {
        name for name in plan.ppy_imports if any(mentions(text, name) for text in written)
    }


def _reads_with_input(tree: ast.AST) -> bool:
    """Does this module read with `input`, and only with `input`?

    The typed reader owns the file descriptor and buffers it itself, so a
    module that also touches `sys.stdin` keeps the `input` it has rather than
    ending up with two readers of the same stream.
    """
    reads = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "stdin":
            return False
        called = isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        if called and node.func.id == "input":  # type: ignore[attr-defined]
            reads = True
    return reads


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


def _scope_statements(function: ast.AST):  # type: ignore[no-untyped-def]
    """The nodes of one function's own scope: nested `def`s keep their own."""
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _unannotatable_locals(info) -> set[str]:  # type: ignore[no-untyped-def]
    """Names bound by anything other than a plain single-target assignment.

    A `for` target, a `with ... as`, an `except ... as`, a match capture, a
    walrus, or a `global` declaration binds without a statement an annotation
    could sit on, and proving the annotation against every such binding is
    more cleverness than the reader gains from it.
    """
    found: set[str] = {p.name for p in info.params}
    for node in _scope_statements(info.node):
        targets: list[ast.expr | None] = []
        if isinstance(node, (ast.For, ast.AsyncFor)):
            targets = [node.target]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            targets = [item.optional_vars for item in node.items]
        elif isinstance(node, ast.ExceptHandler) and node.name:
            found.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            found.update(node.names)
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            found.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            found.update((a.asname or a.name).partition(".")[0] for a in node.names)
        elif isinstance(node, ast.Assign) and (
            len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name)
        ):
            for target in ast.walk(node):
                if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                    found.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.add(node.target.id)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            found.add(node.name)
        for target in targets:
            if target is None:
                continue
            for name in ast.walk(target):
                if isinstance(name, ast.Name):
                    found.add(name.id)
    return found


def _plan_function_locals(symbols, analysis, plan: ConversionPlan, module_name: str) -> None:  # type: ignore[no-untyped-def]
    """Annotate a function-local where it is first bound, like a global.

    Module constants and empty containers already get annotations; a plain
    `total = 0.0` accumulator deserves the same treatment, and for the same
    reader. The annotation goes on the first binding, and only when every
    later assignment provably fits the settled type -- a local that changes
    its mind keeps its silence.
    """
    functions = list(symbols.functions.values())
    for cls in symbols.classes.values():
        functions.extend(cls.methods.values())
    for info in functions:
        function = analysis.functions.get(info.qualname)
        if function is None:
            continue
        skip = _unannotatable_locals(info)
        assigns: dict[str, list[ast.Assign]] = {}
        for node in _scope_statements(info.node):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assigns.setdefault(target.id, []).append(node)
        for name, nodes in assigns.items():
            if name in skip or name.startswith("_"):
                continue
            first = min(nodes, key=lambda n: n.lineno)
            if _is_empty_container(first.value):
                # `out = []` is the empty-container pass's business.
                continue
            bound = T.strip_literal(function.locals.get(name, T.UNKNOWN))
            if isinstance(bound, (T.UnknownType, T.AnyType, T.NeverType)):
                continue
            if not _resolves_once_written(bound, module_name):
                continue
            observed = [T.strip_literal(analysis.type_of(n.value)) for n in nodes]
            if any(
                isinstance(t, (T.UnknownType, T.NeverType)) or not T.is_assignable(t, bound)
                for t in observed
            ):
                continue
            rendered = render_annotation(bound, local_module=module_name)
            if rendered is None:
                continue
            plan.assignments[(first.lineno, name)] = rendered.text
            plan.typing_imports |= rendered.typing_imports
            plan.ppy_imports |= rendered.ppy_imports


def _resolves_once_written(t: T.Type, module_name: str) -> bool:
    """Would every name in this annotation still resolve once written down?

    Rendering shortens `pkg.Class` to its tail unless the package is one
    whose types are conventionally spelled whole; a shortened tail from
    anywhere else names something this module may never have imported, and
    the annotation would be the one line in the file that does not check.
    """
    from ..analysis.render import KEEPS_QUALNAME

    base = T.strip_literal(t)
    if isinstance(base, T.Union_):
        return all(_resolves_once_written(m, module_name) for m in base.members)
    if isinstance(base, T.Tuple_):
        return all(_resolves_once_written(i, module_name) for i in base.items)
    if isinstance(base, T.Instance):
        resolvable = (
            "." not in base.name
            or base.name.startswith(module_name + ".")
            or base.name.startswith(KEEPS_QUALNAME)
        )
        return resolvable and all(_resolves_once_written(a, module_name) for a in base.args)
    return not isinstance(base, (T.UnknownType, T.AnyType, T.NeverType, T.Callable_))


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


def _forward_references(symbols) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Where each class in this module becomes usable as a runtime name."""
    return {
        info.name: (info.node.end_lineno or info.node.lineno) for info in symbols.classes.values()
    }
