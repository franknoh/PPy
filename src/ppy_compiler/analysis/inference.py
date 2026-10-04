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
    "infer_fields",
    "is_self_attribute",
    "modules_with_unannotated_fields",
    "observed_arguments",
    "refine_with_call_sites",
]


#: A hard ceiling, not a tuning knob: the passes below are monotone over
#: finite lattices, so a run that reaches it is a compiler bug and says so.
_MAX_ITERATIONS = 50


def refine_with_call_sites(bundle, diagnostics=None) -> dict[tuple[str, int], T.Type]:  # type: ignore[no-untyped-def]
    """Run inference to a real fixpoint, then generalize, then settle again.

    The two stages are different kinds of decision and must not interleave:

    * **Evidence** is monotone -- call-site types only join, a field or usage
      inference only fills an unknown -- and runs until nothing changes. A
      caller that only becomes typeable late still reaches its callee.
    * **Generalization** (a read-only `list[T]` presented as `Sequence[T]`)
      is a presentation choice about settled facts. Running it against a
      half-analyzed function decides from stale effects: a mutation through
      an alias is only visible once the concrete types are in.

    "What the analysis knows" and "what is safe to write into the source" are
    separate questions; this function answers the first, and hands the second
    settled facts to decide from.
    """
    evidence: dict[tuple[str, int], T.Type] = {}

    def evidence_round() -> bool:
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
            param.type = inferred
            param.inferred = True
            changed = True
        changed |= _infer_fields(bundle)
        changed |= _infer_from_usage(bundle)
        return changed

    _run_to_fixpoint(bundle, evidence_round, "evidence", diagnostics)
    _run_to_fixpoint(bundle, lambda: _widen_read_only_params(bundle), "generalization", diagnostics)
    return evidence


def _run_to_fixpoint(bundle, step, stage: str, diagnostics) -> None:  # type: ignore[no-untyped-def]
    from ..diagnostics import Diagnostic, DiagnosticBag, Severity
    from .checker import analyze

    for _iteration in range(_MAX_ITERATIONS):
        if not step():
            return
        # These rounds discard their diagnostics, which is what makes handing
        # over the previous analysis safe: a module nothing moved keeps it.
        bundle.analysis = analyze(
            bundle.symbols,
            DiagnosticBag(),
            strict=False,
            dynamic_policy=bundle.project.config.dynamic_boundaries,
            plugins=bundle.project.plugins,
            previous=bundle.analysis,
        )
    # Refusing to settle is an internal error; the caller must hear about it
    # rather than receive whatever types the last round happened to hold.
    message = f"type inference did not converge in {_MAX_ITERATIONS} {stage} rounds"
    if diagnostics is not None:
        diagnostics.add(Diagnostic("E9001", Severity.ERROR, message))
    else:
        raise RuntimeError(message)


def observed_arguments(bundle) -> dict[tuple[str, int], T.Type]:  # type: ignore[no-untyped-def]
    """Join the argument types seen at every call site of each function."""
    observed: dict[tuple[str, int], T.Type] = {}
    for module_name, module in bundle.analysis.modules.items():
        symbols = bundle.symbols.modules.get(module_name)
        if symbols is None:
            continue
        for node in symbols.module.nodes:
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
    return infer_fields(bundle.symbols, bundle.analysis.modules)


def modules_with_unannotated_fields(symbols) -> set[str]:  # type: ignore[no-untyped-def]
    """The modules with a class that assigns a `self.x` nothing annotated.

    These are the modules whose methods the seed pass records, so that the
    fields can be typed from the assignments; a module of dataclasses and
    annotated bodies has nothing to learn and is not recorded.
    """
    found: set[str] = set()
    for info in symbols.classes.values():
        if info.module in found:
            continue
        for method in info.methods.values():
            for node in method.nodes:
                if isinstance(node, ast.Assign):
                    pairs = [
                        pair
                        for each in node.targets
                        for pair in _field_assignments(each, node.value)
                    ]
                    for target, _value in pairs:
                        if (
                            is_self_attribute(target, method) and target.attr not in info.fields  # type: ignore[union-attr]
                        ):
                            found.add(info.module)
                            break
                if info.module in found:
                    break
            if info.module in found:
                break
    return found


def infer_fields(symbols, modules) -> bool:  # type: ignore[no-untyped-def]
    """Give each instance field the join of everything its class assigns to it.

    `self.width = width` says as much about `width` as an annotation would,
    and without it nothing that reads the field can be typed. The assignment
    that types a field is not only the first one in `__init__`: a field set
    to `None` there and to a tensor in `setup()` is `Tensor | None`, and
    fixing it as `None` from the first line alone made every later
    assignment an error. The join only grows, so inference still settles.
    """
    changed = False
    outside = _outside_evidence(symbols, modules)
    for info in symbols.classes.values():
        module_analysis = modules.get(info.module)
        if module_analysis is None or not info.methods:
            continue
        # `__init__` first, so the order of evidence is the order of the
        # object's life; the join does not depend on it, the remarks might.
        methods = sorted(info.methods.values(), key=lambda m: m.name != "__init__")
        seen: dict[str, T.Type] = {}
        for method in methods:
            for node in method.nodes:
                if not isinstance(node, ast.Assign):
                    continue
                # `self.left = self.right = self` assigns each target the value.
                pairs = [
                    pair for each in node.targets for pair in _field_assignments(each, node.value)
                ]
                for target, value in pairs:
                    if not is_self_attribute(target, method):
                        continue
                    assigned = T.strip_literal(module_analysis.type_of(value))
                    if isinstance(assigned, (T.UnknownType, T.AnyType, T.NeverType)):
                        continue
                    name = target.attr  # type: ignore[union-attr]
                    seen[name] = assigned if name not in seen else _merge(seen[name], assigned)
        for name, assigned in outside.get(info.qualname, {}).items():
            if name in info.annotated_fields:
                continue
            seen[name] = assigned if name not in seen else _merge(seen[name], assigned)
        for name, assigned in seen.items():
            current = info.fields.get(name, T.UNKNOWN)
            if name in info.declared_fields:
                # A field the class body annotated is what its author said.
                continue
            joined = _by_bases(
                assigned if isinstance(current, T.UnknownType) else _merge(current, assigned)
            )
            if joined != current:
                info.fields[name] = joined
                changed = True
    return changed


#: Methods whose argument becomes an element of the container they are
#: called on, by the argument's position (the last one).
_STORES = {"append": 1, "add": 1, "appendleft": 1, "insert": 2}


def _outside_evidence(symbols, modules) -> dict[str, dict[str, T.Type]]:  # type: ignore[no-untyped-def]
    """What the whole program stores into each class's fields, beyond what
    its own methods assign to `self.x`.

    `node.left = Node(v)` in another method or function is as much a value
    of `Node.left` as `self.left = None` in `__init__`, and a field set to
    `None` there and linked later is `Node | None`. An empty container a
    field starts as (`self.queue = []`) is typed by what is stored into it:
    `self.queue.append(item)`, `self.index[key] = i`. Only a field the class
    already has gains evidence here (a store that would add one stays
    Python's). A value the checker could not type is no evidence, as in
    `infer_fields`: native code is only handed an object whose fields hold
    what the class says, which the boundary checks as the object crosses.
    """
    found: dict[str, dict[str, T.Type]] = {}

    def owner_of(receiver: T.Type, attr: str) -> str | None:
        base = T.strip_literal(receiver)
        if isinstance(base, T.Union_):
            members = [m for m in base.members if m != T.NONE]
            if len(members) != 1:
                return None
            base = T.strip_literal(members[0])
        if not isinstance(base, T.Instance):
            return None
        for entry in (base.name, *base.mro):
            info = symbols.classes.get(entry)
            if info is not None and attr in info.fields and attr not in info.class_vars:
                return info.qualname
        return None

    def note(owner: str, attr: str, value: T.Type) -> None:
        value = T.strip_literal(value)
        if isinstance(value, (T.UnknownType, T.AnyType, T.NeverType)) or _unknown_inside(value):
            return
        fields = found.setdefault(owner, {})
        fields[attr] = _by_bases(value if attr not in fields else _merge(fields[attr], value))

    analysis = None

    def stored_in(container: ast.expr, key: ast.expr, value: T.Type) -> T.Type | None:
        """The container `container[key] = value` makes of `container`'s type."""
        held = T.strip_literal(analysis.type_of(container))  # type: ignore[union-attr]
        if not isinstance(held, T.Instance):
            return None
        if held.name == "dict":
            return T.instance("dict", T.strip_literal(analysis.type_of(key)), value)  # type: ignore[union-attr]
        if held.name == "list" and not isinstance(key, ast.Slice):
            return T.instance("list", value)
        return None

    def lift(node: ast.expr, filled: T.Type) -> None:
        """`filled` is what `node` holds: the field it is, or is inside of,
        holds as much."""
        while isinstance(node, ast.Subscript):
            parent = stored_in(node.value, node.slice, filled)
            if parent is None:
                return
            node, filled = node.value, parent
        if isinstance(node, ast.Attribute):
            owner = owner_of(analysis.type_of(node.value), node.attr)  # type: ignore[union-attr]
            # Only a field whose element nothing has told yet learns it here;
            # one already typed is checked against what is stored.
            if owner is not None and _never_inside(symbols.classes[owner].fields[node.attr]):
                note(owner, node.attr, filled)

    for module_name, analysis in modules.items():
        module_symbols = getattr(analysis, "symbols", None)
        if module_symbols is None:
            continue
        for node in module_symbols.module.nodes:
            if isinstance(node, ast.Assign):
                pairs = [
                    pair for each in node.targets for pair in _field_assignments(each, node.value)
                ]
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                pairs = [(node.target, node.value)]
            else:
                pairs = []
            for target, value in pairs:
                if isinstance(target, ast.Attribute):
                    owner = owner_of(analysis.type_of(target.value), target.attr)
                    if owner is not None:
                        note(owner, target.attr, analysis.type_of(value))
                elif isinstance(target, ast.Subscript):
                    # `self.index[key] = i`, `self.grid[r][c] = 0`: an entry
                    # of the field's container, or of one inside it.
                    entry = stored_in(target.value, target.slice, analysis.type_of(value))
                    if entry is not None:
                        lift(target.value, entry)
            if not isinstance(node, ast.Call) or node.keywords:
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in _STORES
                and len(node.args) == _STORES[func.attr]
            ):
                # `self.queue.append(item)`, `self.adj[u].append(v)`.
                receiver = T.strip_literal(analysis.type_of(func.value))
                element = analysis.type_of(node.args[-1])
                if isinstance(receiver, T.Instance) and receiver.name == "list" and func.attr in {
                    "append",
                    "insert",
                }:
                    lift(func.value, T.instance("list", element))
                elif isinstance(receiver, T.Instance) and receiver.name == "set" and func.attr == "add":
                    lift(func.value, T.instance("set", element))
            elif len(node.args) == 2 and _spelled(func) in _HEAP_PUSHES:
                # `heapq.heappush(self.heap, (priority, item))`.
                receiver = T.strip_literal(analysis.type_of(node.args[0]))
                if isinstance(receiver, T.Instance) and receiver.name == "list":
                    lift(node.args[0], T.instance("list", analysis.type_of(node.args[1])))
    return found


#: `heapq.heappush` as a module may spell it: its second argument becomes an
#: element of its first.
_HEAP_PUSHES = frozenset({"heapq.heappush", "heappush"})


def _spelled(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    return ""


def _by_bases(t: T.Type) -> T.Type:
    """A union of classes with an instance of a subclass in it as the base
    class: `Node | Special | None` is `Node | None` where `Special(Node)`."""
    if not isinstance(t, T.Union_):
        return t
    names = {m.name for m in t.members if isinstance(m, T.Instance)}
    kept = [
        m
        for m in t.members
        if not (isinstance(m, T.Instance) and any(b in names for b in m.mro[1:] if b != m.name))
    ]
    return T.union(*kept) if len(kept) != len(t.members) else t


def _merge(seen: T.Type, more: T.Type) -> T.Type:
    """Two values stored into one field, as one type. Where one is the other
    with parts the checker could not tell (`list[<unknown>]` beside
    `list[list[int]]`), the told one is the evidence: the untold part says
    nothing against it, and the boundary checks what crosses."""
    if _told(seen, more):
        return seen
    if _told(more, seen):
        return more
    return T.join(seen, more)


def _told(known: T.Type, partial: T.Type) -> bool:
    """Whether `known` is `partial` with its unknown parts told."""
    known, partial = T.strip_literal(known), T.strip_literal(partial)
    if isinstance(partial, (T.UnknownType, T.AnyType)):
        return not _unknown_inside(known)
    if isinstance(known, T.Instance) and isinstance(partial, T.Instance):
        return (
            known.name == partial.name
            and len(known.args) == len(partial.args)
            and all(
                mine == theirs or isinstance(theirs, T.NeverType) or _told(mine, theirs)
                for mine, theirs in zip(known.args, partial.args, strict=True)
            )
            and known != partial
        )
    if isinstance(known, T.Tuple_) and isinstance(partial, T.Tuple_):
        return (
            known.homogeneous == partial.homogeneous
            and len(known.items) == len(partial.items)
            and all(
                mine == theirs or _told(mine, theirs)
                for mine, theirs in zip(known.items, partial.items, strict=True)
            )
            and known != partial
        )
    return False


def _never_inside(t: T.Type) -> bool:
    """An empty container's type: `list[Never]`, `dict[str, list[Never]]`."""
    t = T.strip_literal(t)
    if isinstance(t, T.NeverType):
        return True
    if isinstance(t, T.Instance):
        return any(_never_inside(a) for a in t.args)
    if isinstance(t, T.Union_):
        return any(_never_inside(m) for m in t.members)
    return False


def _unknown_inside(t: T.Type) -> bool:
    """An element the checker could not type: `list[<unknown>]` says nothing."""
    if isinstance(t, (T.UnknownType, T.AnyType)):
        return True
    if isinstance(t, T.Instance):
        return any(_unknown_inside(a) for a in t.args)
    if isinstance(t, T.Union_):
        return any(_unknown_inside(m) for m in t.members)
    return False


def _field_assignments(target: ast.expr, value: ast.expr) -> list[tuple[ast.expr, ast.expr]]:
    """The (target, value) pairs one assignment makes: `self.a, self.b = x, y`
    is two, each with the value that lands in it."""
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
    ):
        return list(zip(target.elts, value.elts, strict=True))
    return [(target, value)]


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
        for node in info.nodes:
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
    uses = _parameter_uses(info.node, name, protocol_type, callees, widening, function.aliases)
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
    node: ast.AST, name: str, protocol=None, callees=None, widening=None, aliases=None
) -> set[str] | None:
    """Every way the body uses this parameter, or None if one is unrecognized.

    An unrecognized use is not taken to be harmless; the parameter keeps its
    concrete type rather than the analysis guessing about it. Uses are found
    through the alias map: `ys = xs; ys.append(1)` is a use of `xs`, made by
    a different name, and a scan that only matched the spelling would declare
    a `Sequence` the body goes on to mutate.
    """

    def is_use(candidate) -> bool:  # type: ignore[no-untyped-def]
        if not isinstance(candidate, ast.Name):
            return False
        if candidate.id == name:
            return True
        if aliases is None:
            return False
        return name in aliases.roots_at(candidate, candidate.id)

    uses: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript) and is_use(child.value):
            if isinstance(child.slice, ast.Slice):
                return None
            uses.add("index")
        elif isinstance(child, ast.Attribute) and is_use(child.value):
            uses.add(f"method:{child.attr}")
        elif isinstance(child, (ast.For, ast.AsyncFor, ast.comprehension)):
            if is_use(child.iter):
                uses.add("iterate")
        elif isinstance(child, ast.Compare):
            for operator, comparator in zip(child.ops, child.comparators, strict=False):
                if isinstance(operator, (ast.In, ast.NotIn)) and is_use(comparator):
                    uses.add("contains")
                elif is_use(comparator) or is_use(child.left):
                    return None
        elif isinstance(child, ast.BinOp):
            if is_use(child.left) or is_use(child.right):
                return None
        elif isinstance(child, ast.AugAssign):
            # `xs += ys` mutates in place, and `other += xs` concatenates;
            # neither exists on the protocol.
            if is_use(child.target) or is_use(child.value):
                return None
        elif isinstance(child, ast.Delete):
            for target in child.targets:
                if isinstance(target, (ast.Subscript, ast.Attribute)) and is_use(target.value):
                    return None
        elif isinstance(child, ast.Call):
            written = [*child.args, *(k.value for k in child.keywords)]
            for argument in written:
                if not is_use(argument):
                    continue
                inspectors = widening.inspectors if widening is not None else frozenset()
                if isinstance(child.func, ast.Name) and child.func.id in inspectors:
                    uses.add("len" if child.func.id == "len" else "inspecting-call")
                    continue
                if not _accepts(child, argument, protocol, callees):
                    return None
                uses.add("inspecting-call")
        elif isinstance(child, ast.Return) and is_use(child.value):
            return None
        elif isinstance(child, (ast.Starred, ast.Await, ast.Yield, ast.YieldFrom)):
            if is_use(getattr(child, "value", None)):
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
