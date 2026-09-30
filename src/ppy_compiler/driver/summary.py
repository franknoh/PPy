"""`ppy explain --summary`: how much of a project goes native, and what keeps the rest out.

Every function the analysis saw lands in one of three places:

- native, called from Python through its boundary;
- native, called only by other native code (its boundary is not worth crossing,
  or it has none: it returns nothing, takes a handle, or is too small);
- Python: the optimized Python backend runs it (or plain CPython, where it
  holds a `ppy.dynamic` boundary), and the lowering says why.

The reasons the lowering gives are sentences about one function (`parameter
`xs` is `dict[str, Any]`, which has no native ABI`). The summary groups them:
each reason is matched against a table of known shapes, which names the
category, says what a user can do, and points at the guide page; a reason no
shape matches is grouped by its text with the names in backticks taken out.
Categories are ordered by how much code they keep in Python, counted in
statements, since a blocker on one long function matters more than one on
ten one-liners.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Blocker",
    "FunctionOutcome",
    "Summary",
    "categorize",
    "render_summary",
    "summarize",
    "summary_json",
]

#: Where the guide explains each kind of blocker.
_DOCS = "https://ppy.franknoh.dev/latest"


@dataclass(frozen=True, slots=True)
class _Shape:
    pattern: re.Pattern[str]
    #: The category, with `{0}` ... for the pattern's groups.
    category: str
    hint: str
    page: str


def _shape(pattern: str, category: str, hint: str, page: str) -> _Shape:
    return _Shape(re.compile(pattern), category, hint, page)


#: Known reasons, most specific first. The groups a category names are
#: normalized (see `_normal`) so `dict[str, int]` and `dict[str, str]` stay apart
#: while two parameters of one type are one category.
_SHAPES: tuple[_Shape, ...] = (
    _shape(
        r"parameter `[^`]+` is `([^`]+)`, which has no native ABI",
        "a parameter of type `{0}`",
        "take a type native code holds (numbers, str, tuples, lists, dicts, sets, ppy "
        "collections, project classes)",
        "guide/native-lowering/",
    ),
    _shape(
        r"returns `([^`]+)`, which has no native ABI",
        "a result of type `{0}`",
        "return a type native code holds, or a tuple of them",
        "guide/native-lowering/",
    ),
    _shape(
        r"calls `([^`]+)` with unknown effects|`([^`]+)` has no native lowering",
        "a call to `{0}`",
        "a library or builtin with no native form keeps its caller in Python; call it outside "
        "the hot function",
        "reference/compatibility/",
    ),
    _shape(
        r"generators use the boxed runtime",
        "a generator that is stored, returned, or stepped by hand",
        "consume the generator where it is made: in a `for`, `sum`, `min`, `max`, `sorted`, or "
        "a comprehension",
        "guide/exceptions-and-generators/",
    ),
    _shape(
        r"a coroutine",
        "a coroutine",
        "native coroutines run under `ppy.aio`",
        "guide/aio/",
    ),
    _shape(
        r"variadic parameters",
        "`*args` or `**kwargs`",
        "name the parameters",
        "guide/native-lowering/",
    ),
    _shape(
        r"mutates `[^`]+`, which is not a borrowed buffer",
        "writes to a parameter native code copies",
        "return the new value, or take a `Buffer`, a list, or a ppy collection",
        "guide/native.md".removesuffix(".md") + "/",
    ),
    _shape(
        r"field `([^`]+)` of `([^`]+)` has no native form",
        "a class field native code cannot hold (`{1}.{0}`)",
        "give the field a type native code holds, or keep the class out of the hot function",
        "guide/classes/",
    ),
    _shape(
        r"writes through a target the compiler cannot identify",
        "writes through a name the compiler cannot follow",
        "write through a parameter, a local, or `self` directly",
        "guide/native-lowering/",
    ),
    _shape(
        r"the module did not lower",
        "the module failed to lower (a compiler bug; please report it)",
        "run `ppy run` on the file to see the error, and open an issue with it",
        "reference/compatibility/",
    ),
    _shape(
        r"dynamic",
        "a `ppy.dynamic` boundary",
        "a dynamic region runs on CPython by design",
        "guide/subset/",
    ),
    _shape(
        r"generic function is specialized",
        "a generic function (specialized where it is called)",
        "nothing to do: each native caller gets its own instance",
        "guide/generics/",
    ),
    _shape(
        r"a caller of a function that did not lower|calls .* which did not lower|"
        r"has no native lowering, so this call",
        "calls a function that stays in Python",
        "fix the callee first; its caller follows",
        "guide/native-lowering/",
    ),
)

#: Not blockers: the function is fine where it is.
_DESIGNED = (
    "a generic function",
    "a C binding",
    "device code",
)


def _normal(text: str) -> str:
    """A type or name as a category word: whitespace folded, a module path kept."""
    return " ".join(text.split())


#: Each effect that keeps a function on CPython, in a user's words, and what to do.
_EFFECTS = {
    "IO": ("does I/O (`print`, `input`, files)", "read and print in `main`; pass the data in"),
    "ReadGlobal": (
        "reads a module global that can change",
        "pass the value in, or make it a constant (`X: Final = ...`)",
    ),
    "WriteGlobal": ("writes a module global", "return the new value instead"),
    "ExternalUnknown": (
        "calls code whose effects are unknown (a library, or a call the checker cannot type)",
        "annotate it, add a stub or plugin, or call it outside the hot function",
    ),
    "PythonCallback": (
        "calls back into Python (a function passed in, or a callable object)",
        "call a function of the module by name",
    ),
    "PythonDynamic": ("uses dynamic Python features", "keep them behind `ppy.dynamic`"),
    "WriteObject": (
        "writes to an object native code does not own",
        "return the new value, or take a list, a dict, or a ppy collection",
    ),
    "Network": ("uses the network", "native networking runs under `ppy.aio`"),
    "Random": ("draws random numbers", "pass the numbers in, or seed a generator per call"),
    "Time": ("reads the clock or sleeps", "time the call from outside"),
    "Process": ("starts or controls processes", "keep it out of the hot function"),
    "GpuLaunch": ("launches a GPU kernel", "a kernel launches from the CPU side natively"),
}


def categorize(reason: str) -> list[tuple[str, str, str]]:
    """A lowering reason as (category, hint, guide page) entries: one, or one
    per effect when the reason is a list of effects."""
    effects = re.match(r"has effects that must run on CPython: (.+)", reason)
    if effects is not None:
        found = []
        for name in (part.strip() for part in effects.group(1).split(",")):
            words, hint = _EFFECTS.get(name, (f"has the effect {name}", ""))
            found.append((words, hint, f"{_DOCS}/guide/effects/"))
        return found
    return [_categorize_one(reason)]


def _categorize_one(reason: str) -> tuple[str, str, str]:
    for shape in _SHAPES:
        found = shape.pattern.search(reason)
        if found is None:
            continue
        groups = [_normal(g) for g in found.groups() if g is not None]
        if groups == ["<unknown>"] and "type" in shape.category:
            # A type the checker could not tell: nothing wrong with the type,
            # it is the missing annotation that is the blocker.
            return (
                "a parameter or result with no annotation the checker could infer",
                "annotate it, or run `ppy convert` to write the inferred annotations",
                f"{_DOCS}/guide/subset/",
            )
        return shape.category.format(*groups), shape.hint, f"{_DOCS}/{shape.page}"
    # An unknown shape: its text, names taken out, is its category.
    general = re.sub(r"`[^`]*`", "`…`", reason).strip()
    return general, "", ""


@dataclass(slots=True)
class FunctionOutcome:
    qualname: str
    module: str
    path: str
    line: int
    #: Statements in the body, the weight of the function.
    statements: int
    #: "native", "internal" (native, no Python boundary), or "python".
    tier: str
    #: Why it stays in Python, or why its boundary is not used.
    reason: str = ""


@dataclass(slots=True)
class Blocker:
    category: str
    hint: str
    page: str
    functions: int = 0
    statements: int = 0
    where: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Summary:
    outcomes: list[FunctionOutcome]
    #: Modules that could not be analyzed, with why.
    failed: dict[str, str]

    def totals(
        self, outcomes: Iterable[FunctionOutcome] | None = None
    ) -> dict[str, dict[str, int]]:
        chosen = list(self.outcomes if outcomes is None else outcomes)
        found: dict[str, dict[str, int]] = {}
        for tier in ("native", "internal", "python"):
            picked = [o for o in chosen if o.tier == tier]
            found[tier] = {
                "functions": len(picked),
                "statements": sum(o.statements for o in picked),
            }
        return found

    def blockers(self, tier: str = "python") -> list[Blocker]:
        grouped: dict[str, Blocker] = {}
        for outcome in self.outcomes:
            if outcome.tier != tier:
                continue
            for category, hint, page in categorize(outcome.reason or "no reason given"):
                entry = grouped.setdefault(category, Blocker(category, hint, page))
                entry.functions += 1
                entry.statements += outcome.statements
                if len(entry.where) < 3:
                    entry.where.append(f"{outcome.path}:{outcome.line} {outcome.qualname}")
        return sorted(grouped.values(), key=lambda b: (-b.statements, -b.functions, b.category))


def _statements(node: ast.AST) -> int:
    return sum(isinstance(child, ast.stmt) for child in ast.walk(node)) - 1


def summarize(bundle, lowered, sources=None, failures=None) -> Summary:  # type: ignore[no-untyped-def]
    """Each function of the given sources (all analyzed modules, without them),
    and where it runs. A module the analysis followed an import into is not
    what was asked about.

    `lowered` is the LLVM backend's per-module result (`backend.llvm._collect`).
    """
    root = bundle.project.root
    wanted = {Path(p).resolve() for p in sources} if sources is not None else None
    outcomes: list[FunctionOutcome] = []
    for name, module in bundle.analysis.modules.items():
        result = lowered.get(name)
        for qualname, analysis in module.functions.items():
            info = bundle.symbols.functions.get(qualname)
            if info is None or info.node is None:
                continue
            if wanted is not None and Path(info.path).resolve() not in wanted:
                continue
            path = _shown(Path(info.path), root)
            outcome = FunctionOutcome(
                qualname=qualname,
                module=name,
                path=path,
                line=info.node.lineno,
                statements=max(_statements(info.node), 1),
                tier="python",
            )
            native = result.functions.get(qualname) if result is not None else None
            if failures and name in failures:
                outcome.reason = f"the module did not lower: {failures[name]}"
            if native is not None:
                if native.exposed:
                    outcome.tier = "native"
                else:
                    outcome.tier = "internal"
                    outcome.reason = native.exposure_reason
            elif analysis.dynamic:
                outcome.reason = "contains a ppy.dynamic boundary"
            elif failures and name in failures:
                pass
            elif result is not None and qualname in result.rejected:
                outcome.reason = result.rejected[qualname]
            else:
                outcome.reason = "not lowered"
            outcomes.append(outcome)
    failed = {
        _shown(Path(d.span.path), root) if d.span is not None else "?": d.message
        for d in bundle.diagnostics.sorted()
        if d.code in {"E1001", "E1002"}
    }
    for name, why in (failures or {}).items():
        module = bundle.analysis.modules.get(name)
        path = getattr(getattr(module, "symbols", None), "path", None)
        failed[_shown(Path(path), root) if path else name] = f"lowering failed: {why}"
    outcomes.sort(key=lambda o: (o.path, o.line))
    return Summary(outcomes, failed)


def _shown(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _share(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "-"


def render_summary(summary: Summary, *, limit: int = 10, modules: bool = False) -> str:
    """The summary as text: totals, the blockers by weight, then per module."""
    lines: list[str] = []
    totals = summary.totals()
    every = sum(t["functions"] for t in totals.values())
    weight = sum(t["statements"] for t in totals.values())
    lines.append(f"{every} functions, {weight} statements")
    labels = {
        "native": "native, called from Python",
        "internal": "native, called from native code",
        "python": "Python",
    }
    for tier, label in labels.items():
        counts = totals[tier]
        lines.append(
            f"  {label:<34} {counts['functions']:>6} functions "
            f"({_share(counts['functions'], every):>4})  "
            f"{counts['statements']:>7} statements ({_share(counts['statements'], weight):>4})"
        )
    blockers = summary.blockers()
    if blockers:
        lines.append("")
        lines.append(
            "what keeps functions in Python, by statements kept out "
            "(a function can count under more than one):"
        )
        for blocker in blockers[:limit]:
            lines.append(
                f"  {blocker.statements:>7} statements  {blocker.functions:>5} functions  "
                f"{blocker.category}"
            )
            if blocker.hint:
                lines.append(f"      {blocker.hint}")
            if blocker.page:
                lines.append(f"      see {blocker.page}")
            lines.extend(f"      {where}" for where in blocker.where)
        if len(blockers) > limit:
            rest = blockers[limit:]
            lines.append(
                f"  ... {len(rest)} more reasons, {sum(b.functions for b in rest)} functions "
                "(--limit to see more, --json for all)"
            )
    internal = summary.blockers("internal")
    if internal:
        lines.append("")
        lines.append("native, but Python calls the Python body (why its boundary is not used):")
        lines.extend(
            f"  {blocker.functions:>5} functions  {blocker.category}"
            for blocker in internal[:limit]
        )
    if modules:
        lines.append("")
        lines.append("by module (native / native from native / Python, functions):")
        by_module: dict[str, list[FunctionOutcome]] = {}
        for outcome in summary.outcomes:
            by_module.setdefault(outcome.path, []).append(outcome)
        for path, found in sorted(by_module.items()):
            counts = summary.totals(found)
            lines.append(
                f"  {counts['native']['functions']:>4} {counts['internal']['functions']:>4} "
                f"{counts['python']['functions']:>4}  {path}"
            )
    if summary.failed:
        lines.append("")
        lines.append(f"{len(summary.failed)} files could not be analyzed:")
        for path, message in sorted(summary.failed.items())[:limit]:
            lines.append(f"  {path}: {message}")
    return "\n".join(lines) + "\n"


def summary_json(summary: Summary) -> str:
    """Everything, for a tool: totals, every blocker, every function."""
    payload = {
        "totals": summary.totals(),
        "blockers": [
            {
                "category": b.category,
                "hint": b.hint,
                "page": b.page,
                "functions": b.functions,
                "statements": b.statements,
                "where": b.where,
            }
            for b in summary.blockers()
        ],
        "internal": [
            {"reason": b.category, "functions": b.functions} for b in summary.blockers("internal")
        ],
        "functions": [
            {
                "qualname": o.qualname,
                "path": o.path,
                "line": o.line,
                "statements": o.statements,
                "tier": o.tier,
                "reason": o.reason,
            }
            for o in summary.outcomes
        ],
        "failed": summary.failed,
    }
    return json.dumps(payload, indent=2) + "\n"
