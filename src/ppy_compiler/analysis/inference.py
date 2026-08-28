"""Settling the types a conversion will write down.

`ppy convert` cannot annotate what it has not inferred, and inference here is
interprocedural: a parameter's type comes from the call sites that reach it,
which are only themselves typed once their callers are. So this runs as a
fixpoint, joining evidence rather than taking the first answer, and generalizes
only once nothing new is arriving.

Keeping it apart from the rewriter matters: everything here reasons about
`ast` and the symbol table, and nothing here knows that the output is text.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from . import types as T
from .binding import bind_ast_call

__all__ = [
    "callee_qualname",
    "has_source_annotation",
    "is_self_attribute",
    "observed_arguments",
    "refine_with_call_sites",
]


#: Call-site evidence propagates along the call graph, so one round only
#: reaches functions called from already-typed code. Repeating it walks the
#: chain; a handful of rounds settles any realistic project.
_REFINEMENT_ROUNDS = 6


def refine_with_call_sites(bundle) -> dict[tuple[str, int], T.Type]:  # type: ignore[no-untyped-def]
    """Adopt call-site argument types and re-infer, until nothing new appears.

    Without this a function whose parameters were only inferred would still
    return `<unknown>`, and nothing downstream of it could be annotated.

    Evidence is joined across rounds rather than taken from the first round
    that produced any. A call reached through another function is only typed
    once that function is, so the second caller of a helper routinely shows up
    a round after the first -- and a signature frozen on round one would
    describe just the callers that happened to be easy to type.
    """
    from ..diagnostics import DiagnosticBag
    from .checker import analyze

    evidence: dict[tuple[str, int], T.Type] = {}
    for _round in range(_REFINEMENT_ROUNDS):
        changed = False
        fresh: set[tuple[str, int]] = set()
        for key, seen in observed_arguments(bundle).items():
            joined = seen if key not in evidence else T.join(evidence[key], seen)
            if evidence.get(key) != joined:
                evidence[key] = joined
                fresh.add(key)
        for qualname, index in fresh:
            info = bundle.symbols.functions.get(qualname)
            if info is None or index >= len(info.params):
                continue
            param = info.params[index]
            inferred = evidence[(qualname, index)]
            if param.annotated or isinstance(inferred, (T.UnknownType, T.AnyType)):
                continue
            # Only evidence that grew this round is written back, so a type a
            # later pass generalized is not reset to the raw observation on
            # every round -- which would keep the fixpoint from settling.
            param.type = inferred
            param.inferred = True
            changed = True
        changed |= _infer_fields(bundle)
        changed |= _infer_from_usage(bundle)
        changed |= _widen_read_only_params(bundle)
        if not changed:
            break
        bundle.analysis = analyze(
            bundle.symbols,
            DiagnosticBag(),
            strict=False,
            dynamic_policy=bundle.project.config.dynamic_boundaries,
            plugins=bundle.project.plugins,
        )
    return evidence


def observed_arguments(bundle) -> dict[tuple[str, int], T.Type]:  # type: ignore[no-untyped-def]
    """Join the argument types seen at every call site of each function."""
    observed: dict[tuple[str, int], T.Type] = {}
    for module_name, module in bundle.analysis.modules.items():
        symbols = bundle.symbols.modules.get(module_name)
        if symbols is None:
            continue
        for node in ast.walk(symbols.module.tree):
            if not isinstance(node, ast.Call):
                continue
            qualname = callee_qualname(bundle, symbols, node)
            if qualname is None:
                continue
            info = bundle.symbols.functions.get(qualname)
            if info is None:
                continue
            for argument in bind_ast_call(info, node):
                observed_type = T.strip_literal(module.type_of(argument.value))
                if isinstance(observed_type, (T.UnknownType, T.AnyType, T.NeverType)):
                    continue
                key = (qualname, argument.index)
                existing = observed.get(key)
                observed[key] = (
                    observed_type if existing is None else T.join(existing, observed_type)
                )
    for qualname, info in bundle.symbols.functions.items():
        for index, param in enumerate(info.params):
            # A declared type is evidence too, and unlike an inferred one it is
            # not something this pass is entitled to revise.
            if param.annotated and not isinstance(param.type, T.UnknownType):
                observed.setdefault((qualname, index), param.type)
    return observed


def callee_qualname(bundle, symbols, node: ast.Call) -> str | None:  # type: ignore[no-untyped-def]
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


def has_source_annotation(info, name: str) -> bool:  # type: ignore[no-untyped-def]
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
    for info in bundle.symbols.classes.values():
        initializer = info.methods.get("__init__")
        analysis = bundle.analysis.modules.get(info.module)
        if initializer is None or analysis is None:
            continue
        for node in ast.walk(initializer.node):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not is_self_attribute(target, initializer):
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
    for info in bundle.symbols.functions.values():
        analysis = bundle.analysis.modules.get(info.module)
        if analysis is None:
            continue
        pending = {p.name: p for p in info.params if not p.known}
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
                param.inferred = True
                changed = True
    return changed


def is_self_attribute(target: ast.expr, info) -> bool:  # type: ignore[no-untyped-def]
    receiver = info.params[0].name if info.params else "self"
    return (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == receiver
    )


@dataclass(frozen=True, slots=True)
class _Widening:
    """A concrete container and the protocol it may be declared as instead."""

    protocol: str
    # Not `mro`: `dataclasses` reads class attributes to find defaults, and
    # `type.mro` would look like one.
    bases: tuple[str, ...]
    allowed: frozenset[str]
    #: Builtins this protocol can be handed. Not every protocol offers the
    #: same ones, so this belongs to the widening rather than being shared.
    inspectors: frozenset[str]


#: A protocol offers fewer operations than the concrete type, and some of the
#: ones it drops would change what a function returns rather than merely
#: failing: `xs[:]` of a tuple is a tuple, and `xs + ys` of a tuple is a tuple.
#: So this is an allowlist of uses, not a test for mutation.
_READ_ONLY_USES = frozenset({"len", "index", "iterate", "contains", "inspecting-call"})

#: Builtins that read a container without keeping it or depending on its exact
#: type: `sorted(xs)` is a list whatever `xs` was. Anything needing more than
#: iteration is named by the protocol that actually offers it.
_INSPECTS_ANY_ITERABLE = frozenset(
    {
        "len",
        "sum",
        "min",
        "max",
        "sorted",
        "any",
        "all",
        "list",
        "tuple",
        "set",
        "dict",
        "enumerate",
        "zip",
        "iter",
        "print",
        "repr",
        "str",
    }
)

_WIDENINGS: dict[str, _Widening] = {
    "list": _Widening(
        "Sequence",
        ("Sequence", "Iterable", "object"),
        _READ_ONLY_USES | {"method:count", "method:index"},
        # A `Sequence` is reversible; `reversed()` is part of its contract.
        _INSPECTS_ANY_ITERABLE | {"reversed"},
    ),
    "dict": _Widening(
        "Mapping",
        ("Mapping", "Iterable", "object"),
        _READ_ONLY_USES | {"method:get", "method:keys", "method:values", "method:items"},
        # `dict` is reversible but `Mapping` is not: it declares no
        # `__reversed__`, and neither `__len__` with `__getitem__` by position.
        _INSPECTS_ANY_ITERABLE,
    ),
}


def _widen_read_only_params(bundle) -> bool:  # type: ignore[no-untyped-def]
    """Declare each read-only container parameter as the protocol it needs.

    This happens inside the inference fixpoint rather than at render time so
    that the rest of the analysis sees the widened type: what a `Sequence`
    yields when iterated is what decides the function's own return type.
    """
    changed = False
    for info in bundle.symbols.functions.values():
        analysis = bundle.analysis.modules.get(info.module)
        if analysis is None:
            continue
        callees = bundle.symbols.modules[info.module].functions
        for param in info.params:
            if has_source_annotation(info, param.name):
                continue
            widened = _read_only_view(param.type, info, param.name, analysis, callees)
            if widened != param.type:
                param.type = widened
                param.inferred = True
                changed = True
    return changed


def _read_only_view(t: T.Type, info, name: str, analysis, callees=None):  # type: ignore[no-untyped-def]
    """Declare a container parameter as the protocol its body actually needs.

    A function that only reads its argument should not demand a `list`, which
    rejects the tuple a caller already has. Being read-only is not enough on
    its own: `xs.copy()`, `xs + ys`, and `xs[:]` are all reads, and each either
    does not exist on the protocol or returns something else through it. What
    decides is whether every use the body makes is one the protocol offers.
    """
    if analysis is None:
        return t
    function = analysis.functions.get(info.qualname)
    if function is None or name in function.mutated_params or function.foreign_writes:
        return t
    base = T.strip_literal(t)
    protocol = _shared_protocol(base)
    if protocol is None:
        return t
    widening, args = protocol
    protocol_type = T.Instance(widening.protocol, args, widening.bases)
    uses = _parameter_uses(info.node, name, protocol_type, callees, widening)
    if uses is None or not uses <= widening.allowed:
        return t
    return protocol_type


def _shared_protocol(t: T.Type) -> tuple[_Widening, tuple[T.Type, ...]] | None:
    """The protocol that describes this type, or the one every member shares.

    Call sites that pass a list from one place and a tuple from another infer a
    union of concrete containers. Naming the protocol they have in common is
    both the honest annotation and the only usable one.
    """
    members = t.members if isinstance(t, T.Union_) else (t,)
    found: set[str] = set()
    element: tuple[T.Type, ...] | None = None
    for member in members:
        described = _as_container(T.strip_literal(member))
        if described is None:
            return None
        protocol, args = described
        found.add(protocol)
        if element is None:
            element = args
        elif element != args:
            return None
    if len(found) != 1 or element is None:
        return None
    return _WIDENINGS[found.pop()], element


def _as_container(t: T.Type) -> tuple[str, tuple[T.Type, ...]] | None:
    """Which widening this concrete container belongs to, and its arguments."""
    if isinstance(t, T.Tuple_):
        if not t.items:
            return None
        first = t.items[0]
        if any(item != first for item in t.items):
            return None
        return "list", (first,)
    if isinstance(t, T.Instance) and t.args:
        if t.name == "tuple":
            return "list", t.args[:1]
        if t.name in _WIDENINGS:
            return t.name, t.args
    return None


def _parameter_uses(  # type: ignore[no-untyped-def]
    node: ast.AST, name: str, protocol=None, callees=None, widening=None
) -> set[str] | None:
    """Every way the body uses this parameter, or None if one is unrecognized.

    An unrecognized use is not taken to be harmless; the parameter keeps its
    concrete type rather than the analysis guessing about it.
    """
    uses: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript) and _is_name(child.value, name):
            if isinstance(child.slice, ast.Slice):
                return None
            uses.add("index")
        elif isinstance(child, ast.Attribute) and _is_name(child.value, name):
            uses.add(f"method:{child.attr}")
        elif isinstance(child, (ast.For, ast.AsyncFor, ast.comprehension)):
            if _is_name(child.iter, name):
                uses.add("iterate")
        elif isinstance(child, ast.Compare):
            for operator, comparator in zip(child.ops, child.comparators, strict=False):
                if isinstance(operator, (ast.In, ast.NotIn)) and _is_name(comparator, name):
                    uses.add("contains")
                elif _is_name(comparator, name) or _is_name(child.left, name):
                    return None
        elif isinstance(child, ast.BinOp):
            if _is_name(child.left, name) or _is_name(child.right, name):
                return None
        elif isinstance(child, ast.Call):
            written = [*child.args, *(k.value for k in child.keywords)]
            for argument in written:
                if not _is_name(argument, name):
                    continue
                inspectors = widening.inspectors if widening is not None else frozenset()
                if isinstance(child.func, ast.Name) and child.func.id in inspectors:
                    uses.add("len" if child.func.id == "len" else "inspecting-call")
                    continue
                if not _accepts(child, argument, protocol, callees):
                    return None
                uses.add("inspecting-call")
        elif isinstance(child, ast.Return) and _is_name(child.value, name):
            return None
        elif isinstance(child, (ast.Starred, ast.Await, ast.Yield, ast.YieldFrom)):
            if _is_name(getattr(child, "value", None), name):
                return None
    return uses


def _accepts(call: ast.Call, argument: ast.expr, protocol, callees) -> bool:  # type: ignore[no-untyped-def]
    """Would the callee still accept this argument once it is the protocol?

    Passing the parameter on is safe only when whatever receives it declares
    something the protocol satisfies. The inference fixpoint re-runs, so a
    callee that widens on one round lets its callers widen on the next.
    """
    if protocol is None or callees is None or not isinstance(call.func, ast.Name):
        return False
    info = callees.get(call.func.id)
    if info is None:
        return False
    reached = [b for b in bind_ast_call(info, call) if b.value is argument]
    if not reached:
        # Written as a `**splat`, or naming a parameter that does not exist.
        # Either way this pass cannot say where the value lands.
        return False
    return all(b.param.known and T.is_assignable(protocol, b.param.type) for b in reached)


def _is_name(node, name: str) -> bool:  # type: ignore[no-untyped-def]
    return isinstance(node, ast.Name) and node.id == name
