"""Verification of PPY directive contracts (spec 6.2, 11.3, 17.1)."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from difflib import get_close_matches

from ..diagnostics import Diagnostic, DiagnosticBag, Severity
from ..frontend.source import span_of
from . import representation
from . import types as T
from .checker import FunctionAnalysis, ProjectAnalysis
from .effects import Effect
from .refinements import Facts
from .representation import Repr
from .symbols import DIRECTIVE_NAMES, FunctionInfo

__all__ = ["ContractReport", "native_report", "parallel_report", "verify"]

#: Effects that force an opaque CPython call inside the function body.
BOXED_EFFECTS = (
    Effect.IO,
    Effect.RANDOM,
    Effect.TIME,
    Effect.THREAD,
    Effect.PROCESS,
    Effect.SYNC,
    Effect.PYTHON_CALLBACK,
    Effect.WRITE_GLOBAL,
    Effect.READ_GLOBAL,
    Effect.EXTERNAL_UNKNOWN,
)

_REDUCTION_OPS = {
    ast.Add: "+",
    ast.Mult: "*",
    ast.BitOr: "or",
    ast.BitAnd: "and",
}


@dataclass(slots=True)
class ContractReport:
    """Per-function conclusions used by `ppy explain` and the backends."""

    qualname: str
    pure: bool = False
    pure_declared: bool = False
    parallel_ok: bool = False
    parallel_reason: str = ""
    native_ok: bool = False
    native_reason: str = ""
    reductions: tuple[str, ...] = ()
    jit_eligible: bool = False
    dynamic: bool = False
    blockers: tuple[str, ...] = field(default=())


def verify(
    analysis: ProjectAnalysis, diagnostics: DiagnosticBag, *, backend: str = "python"
) -> dict[str, ContractReport]:
    """Check every declared contract and return per-function reports."""
    reports: dict[str, ContractReport] = {}
    for module in analysis.modules.values():
        for info in module.symbols.classes.values():
            _check_directive_names(info.path, info.node, info.directives, diagnostics)
        for qualname, function in module.functions.items():
            info = function.info
            _check_directive_names(info.path, info.node, info.directives, diagnostics)
            fused = _fused_regions(module, function)
            report = _verify_one(function, diagnostics, backend=backend, fused=fused)
            reports[qualname] = report
    return reports


def _check_directive_names(path, node, directives, diagnostics: DiagnosticBag) -> None:  # type: ignore[no-untyped-def]
    """A `ppy.` decorator the runtime does not export never reaches the runtime.

    Plain CPython raises AttributeError on the decorator line, so a typo here
    is a hard error rather than an unknown directive to ignore.
    """
    for directive in directives:
        if directive.name in DIRECTIVE_NAMES:
            continue
        suggestion = get_close_matches(directive.name, sorted(DIRECTIVE_NAMES), n=1)
        help = f"did you mean `@ppy.{suggestion[0]}`?" if suggestion else None
        diagnostics.add(
            Diagnostic(
                "E1205",
                Severity.ERROR,
                f"`ppy` has no directive `{directive.name}`",
                span_of(path, directive.node or node),
                help=help,
            )
        )


def _fused_regions(module, function: FunctionAnalysis) -> int:
    """How many operations inside this function a plugin lowers to a kernel."""
    node = function.info.node
    start = node.lineno
    end = getattr(node, "end_lineno", start) or start
    return sum(
        1
        for note in module.lowerings.values()
        if note.lowering == "Intrinsic" and start <= note.line <= end
    )


def _verify_one(
    function: FunctionAnalysis,
    diagnostics: DiagnosticBag,
    *,
    backend: str,
    fused: int = 0,
) -> ContractReport:
    info = function.info
    parallel_ok, parallel_reason, reductions = parallel_report(
        function, backend=backend, fused=fused
    )
    native_ok, native_reason = native_report(function)
    report = ContractReport(
        qualname=info.qualname,
        pure=function.verified_pure,
        pure_declared=info.declared_pure,
        parallel_ok=parallel_ok,
        parallel_reason=parallel_reason,
        native_ok=native_ok,
        native_reason=native_reason,
        reductions=reductions,
        jit_eligible=native_ok and info.directive("jit") is not None,
        dynamic=function.dynamic,
        blockers=function.purity_blockers,
    )

    if info.declared_pure:
        _check_purity(function, diagnostics)
    if (
        (directive := info.directive("parallel")) is not None
        and directive.require
        and not parallel_ok
    ):
        diagnostics.add(
            Diagnostic(
                "E1701",
                Severity.ERROR,
                f"`@ppy.parallel(require=True)` cannot be satisfied "
                f"for `{info.name}` on the {backend} backend",
                span_of(info.path, directive.node or info.node),
                help=parallel_reason,
            )
        )
    if (directive := info.directive("native")) is not None and directive.require and not native_ok:
        diagnostics.add(
            Diagnostic(
                "E1702",
                Severity.ERROR,
                f"`@ppy.native(require=True)` cannot be satisfied for `{info.name}`",
                span_of(info.path, directive.node or info.node),
                help=native_reason,
            )
        )
    if (directive := info.directive("opt")) is not None and directive.level is None:
        diagnostics.add(
            Diagnostic(
                "E1301",
                Severity.ERROR,
                "`@ppy.opt` requires an integer level in 0..3",
                span_of(info.path, directive.node or info.node),
            )
        )
    return report


def _check_purity(function: FunctionAnalysis, diagnostics: DiagnosticBag) -> None:
    info = function.info
    violations = function.effects.violations()
    directive = info.directive("pure")
    span = span_of(info.path, directive.node if directive and directive.node else info.node)

    if function.unknown_callees:
        names = ", ".join(f"`{n}`" for n in function.unknown_callees[:3])
        diagnostic = Diagnostic(
            "E1602",
            Severity.ERROR,
            f"`{info.name}` is declared `@ppy.pure` but calls {names} with unknown effects",
            span,
            help="give the callee a stub or plugin effect summary, or drop the purity contract",
        )
        diagnostics.add(diagnostic)
        return
    # Spec 11.2: allocating and filling a local is allowed inside a pure
    # function, as long as nothing else could see the object before it returns.
    if function.writes_only_locals:
        violations = violations - {Effect.WRITE_OBJECT}
    if violations:
        listed = ", ".join(sorted(str(v) for v in violations))
        diagnostic = Diagnostic(
            "E1601",
            Severity.ERROR,
            f"`{info.name}` is declared `@ppy.pure` but has forbidden effect(s): {listed}",
            span,
        )
        for blocker in function.purity_blockers[:4]:
            diagnostic.with_note(blocker)
        diagnostics.add(diagnostic)


def native_report(function: FunctionAnalysis) -> tuple[bool, str]:
    """Can the function be lowered without an opaque Python call in its body?"""
    info = function.info
    if function.dynamic:
        return False, "the function body contains a dynamic Python boundary"
    if function.native_blockers:
        return False, function.native_blockers[0]
    if Effect.EXTERNAL_UNKNOWN in function.effects:
        callee = function.unknown_callees[0] if function.unknown_callees else "an external call"
        return False, f"`{callee}` has no native lowering"
    for effect in BOXED_EFFECTS:
        if effect in function.effects:
            return (
                False,
                f"the body has the `{effect}` effect, which requires an opaque Python call",
            )
    if info.is_generator:
        return False, "generators are lowered through the boxed runtime"
    if info.is_async:
        return False, "coroutines are lowered through the boxed runtime"
    # A clean body is not enough: the signature has to have a native form too,
    # or `@ppy.native(require=True)` would accept a function that stays boxed.
    for param in info.params:
        if param.kind in {"var_positional", "var_keyword"}:
            return False, f"`*{param.name}` has no native ABI"
        if not _has_native_abi(param.type, param.facts):
            return False, f"parameter `{param.name}` is `{param.type}`, which has no native ABI"
    if not _has_native_abi(info.ret, info.ret_facts):
        return False, f"returns `{info.ret}`, which has no native ABI"
    return True, ""


def _has_native_abi(t: T.Type, facts: Facts) -> bool:
    """Does this type reach the native ABI, as a scalar or a described buffer?

    Mirrors what the backend accepts. A `PyObject*` representation is the boxed
    one, and a list of scalars is passed as a buffer rather than an object.
    """
    chosen = representation.select(t, facts)
    if chosen.repr not in {Repr.PY_OBJECT, Repr.PY_LONG, Repr.PY_LIST, Repr.PY_ARRAY}:
        return True
    return _is_buffer_like(t) or _is_flat_class(t)


def _is_buffer_like(t: T.Type) -> bool:
    base = T.strip_literal(t)
    if not isinstance(base, T.Instance) or len(base.args) != 1:
        return False
    if base.name not in {"list", "Sequence", "Buffer", "memoryview", "array"}:
        return False
    element = T.strip_literal(base.args[0])
    return isinstance(element, T.Instance) and element.name in {"int", "float", "bool"}


def _is_flat_class(t: T.Type) -> bool:
    """A fixed tuple of scalars, or a class whose fields are all scalars."""
    base = T.strip_literal(t)
    if isinstance(base, T.Tuple_) and not base.homogeneous and base.items:
        return all(_has_native_abi(item, Facts()) for item in base.items)
    return False


def parallel_report(
    function: FunctionAnalysis, *, backend: str, fused: int = 0
) -> tuple[bool, str, tuple[str, ...]]:
    """Decide whether the function's loops can be parallelized safely."""
    info = function.info
    loops = _top_level_loops(info)
    reductions = tuple(dict.fromkeys(_reductions(info)))

    if backend == "python":
        return (
            False,
            (
                "the Python backend cannot bypass the GIL for CPU-bound loops; "
                "use `ppy run` for the LLVM backend"
            ),
            reductions,
        )
    if fused and not loops:
        # A fused library kernel is one loop over independent elements, and it
        # runs with the GIL released.
        return True, "", reductions
    if not loops:
        return False, "the function contains no parallelizable loop", reductions
    if Effect.IO in function.effects:
        return False, "the loop body performs I/O, so ordering is observable", reductions
    if Effect.PYTHON_CALLBACK in function.effects:
        return (
            False,
            "the loop body invokes a Python callback, which must keep its order",
            reductions,
        )
    if Effect.EXTERNAL_UNKNOWN in function.effects:
        callee = function.unknown_callees[0] if function.unknown_callees else "an external call"
        return (
            False,
            f"`{callee}` has unknown effects, so iteration independence is unproven",
            reductions,
        )
    if Effect.WRITE_GLOBAL in function.effects:
        return False, "the loop body writes a global, so iterations are not independent", reductions

    aliasing = _possible_aliases(info)
    if aliasing:
        a, b = aliasing
        return False, f"possible alias between `{a}` and `{b}`", reductions

    carried = _loop_carried(loops, reductions)
    if carried:
        return False, f"`{carried}` carries a dependency across iterations", reductions

    return True, "", reductions


def _top_level_loops(info: FunctionInfo) -> list[ast.For | ast.While | ast.ListComp]:
    return [
        node  # type: ignore[misc]
        for node in ast.walk(info.node)
        if isinstance(node, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp))
    ]


def _reductions(info: FunctionInfo) -> list[str]:
    found: list[str] = []
    for node in ast.walk(info.node):
        if isinstance(node, ast.AugAssign) and type(node.op) in _REDUCTION_OPS:
            found.append(_REDUCTION_OPS[type(node.op)])
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"min", "max", "sum", "all", "any"}
        ):
            found.append(node.func.id)
    return found


def _loop_carried(loops: list, reductions: tuple[str, ...]) -> str | None:
    """Find a variable updated across iterations that is not a recognized reduction."""
    for loop in loops:
        if not isinstance(loop, (ast.For, ast.While)):
            continue
        targets = {
            t.id
            for node in ast.walk(loop)
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Name)
        }
        read_before_write: set[str] = set()
        for node in ast.walk(loop):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in targets:
                read_before_write.add(node.id)
        for node in ast.walk(loop):
            if (
                isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Name)
                and type(node.op) not in _REDUCTION_OPS
            ):
                return node.target.id
        if read_before_write:
            return min(read_before_write)
    return None


def _possible_aliases(info: FunctionInfo) -> tuple[str, str] | None:
    """Mutable container parameters are assumed to alias unless declared otherwise."""
    mutable = [p for p in info.params if _is_mutable_container(p.type) and not p.facts.no_alias]
    if len(mutable) < 2:
        return None
    mutated = {
        node.value.id
        for node in ast.walk(info.node)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
    }
    if not mutated:
        return None
    names = [p.name for p in mutable]
    if any(name in mutated for name in names):
        return names[0], names[1]
    return None


def _is_mutable_container(t: object) -> bool:
    base = T.strip_literal(t) if isinstance(t, T.Type) else None
    return isinstance(base, T.Instance) and base.name in {
        "list",
        "dict",
        "set",
        "bytearray",
        "ndarray",
    }
