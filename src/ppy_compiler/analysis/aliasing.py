"""Flow-sensitive local alias analysis.

`ys = xs; zs = ys; zs.append(1)` mutates `xs`, and every analysis that asks
"is this parameter mutated?" has to see that. Names are not objects: this pass
tracks, at each point in a function, which *roots* a name may refer to — a
parameter, a local allocation site, or something from outside — so that a
mutation or escape recorded against a name can be charged to the objects it
could actually reach.

The analysis is deliberately local and conservative: reassignment kills an
alias, branches join by union, loops run to a (finite, monotone) fixpoint, and
anything it cannot see — an attribute read, an unknown call — is the external
root, which consumers must treat as "could be anything".
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

__all__ = ["EXTERNAL", "AliasInfo", "analyze_aliases"]

#: The root standing for every object this function did not create and was not
#: handed as a parameter: globals, attribute reads, results of unknown calls.
EXTERNAL = "<external>"

#: Builtins whose result is a fresh container holding the argument's elements.
_FRESH_FROM_ELEMENTS = frozenset(
    {"list", "sorted", "reversed", "set", "tuple", "frozenset", "dict", "bytearray"}
)

#: Builtins whose result shares nothing with the argument.
_FRESH_SCALAR = frozenset(
    {
        "len",
        "sum",
        "min",
        "max",
        "any",
        "all",
        "abs",
        "round",
        "int",
        "float",
        "str",
        "bool",
        "repr",
        "hash",
        "id",
        "ord",
        "chr",
        "range",
        "enumerate",
        "zip",
    }
)

_State = dict[str, frozenset[str]]


@dataclass(slots=True)
class AliasInfo:
    """What each name may refer to, at each statement of one function."""

    params: frozenset[str]
    #: id(node) -> name -> roots, snapshotted before the enclosing statement.
    _at: dict[int, _State] = field(default_factory=dict)
    #: root -> roots of values stored into it (flow-insensitive, grows only).
    holds: dict[str, frozenset[str]] = field(default_factory=dict)

    def roots_at(self, node: ast.AST, name: str) -> frozenset[str]:
        """The objects `name` may refer to where `node` executes.

        A point this pass never saw answers conservatively: a parameter is at
        least itself, and anything else could be anything.
        """
        state = self._at.get(id(node))
        if state is not None and name in state:
            return state[name]
        if name in self.params:
            return frozenset({name, EXTERNAL})
        return frozenset({EXTERNAL})

    def param_roots(self, roots: frozenset[str]) -> frozenset[str]:
        return roots & self.params

    def only_local(self, roots: frozenset[str]) -> bool:
        """Do these roots name only objects this function created?"""
        return bool(roots) and EXTERNAL not in roots and not (roots & self.params)


class _Analyzer:
    def __init__(self, params: frozenset[str], immutable: frozenset[str]) -> None:
        self.params = params
        self.immutable = immutable
        self.at: dict[int, _State] = {}
        self.holds: dict[str, set[str]] = {}
        self._alloc = 0

    def fresh(self) -> frozenset[str]:
        self._alloc += 1
        return frozenset({f"@{self._alloc}"})

    def run(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> AliasInfo:
        state: _State = {name: frozenset({name}) for name in self.params}
        self.flow(node.body, state)
        return AliasInfo(
            self.params, self.at, {root: frozenset(held) for root, held in self.holds.items()}
        )

    # -- statements ---------------------------------------------------------

    def flow(self, body: list[ast.stmt], state: _State) -> _State:
        for stmt in body:
            self.snapshot(stmt, state)
            state = self.stmt(stmt, state)
        return state

    def snapshot(self, stmt: ast.stmt, state: _State) -> None:
        """Attach the current state to this statement's own expressions.

        Nested statements stop the walk: they are snapshotted by their own
        `flow` visit, with the state that actually reaches them. A loop body
        is visited more than once, so states union -- may-alias means every
        iteration's answer counts.
        """
        frozen = dict(state)
        stack: list[ast.AST] = [stmt]
        while stack:
            sub = stack.pop()
            existing = self.at.get(id(sub))
            if existing is None:
                self.at[id(sub)] = frozen
            elif existing is not frozen:
                self.at[id(sub)] = self.join(existing, frozen)
            for child in ast.iter_child_nodes(sub):
                if isinstance(child, (ast.stmt, ast.Lambda)):
                    continue
                stack.append(child)

    def stmt(self, node: ast.stmt, state: _State) -> _State:
        if isinstance(node, ast.Assign):
            roots = self.eval(node.value, state)
            for target in node.targets:
                state = self.bind(target, roots, node.value, state)
            return state
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            return self.bind(node.target, self.eval(node.value, state), node.value, state)
        if isinstance(node, ast.AugAssign):
            # `xs += ...` keeps the same object for a list -- but for a value
            # known to be immutable (a tuple-typed parameter), `+=` builds a
            # new object and the alias is over. Anything uncertain keeps its
            # roots, which is the conservative direction.
            self.eval(node.value, state)
            if isinstance(node.target, ast.Name):
                roots = state.get(node.target.id, frozenset())
                if roots and roots <= self.immutable:
                    state = dict(state)
                    state[node.target.id] = self.fresh()
            elif isinstance(node.target, (ast.Subscript, ast.Attribute)):
                self.store_into(self.eval(node.target.value, state), self.eval(node.value, state))
            return state
        if isinstance(node, ast.If):
            self.eval(node.test, state)
            then = self.flow(node.body, dict(state))
            other = self.flow(node.orelse, dict(state))
            return self.join(then, other)
        if isinstance(node, (ast.For, ast.AsyncFor)):
            elements = self.elements_of(self.eval(node.iter, state))
            return self.loop(
                node,
                lambda s: self.flow(node.body, self.bind(node.target, elements, None, s)),
                state,
            )
        if isinstance(node, ast.While):
            self.eval(node.test, state)
            return self.loop(node, lambda s: self.flow(node.body, s), state)
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                self.eval(item.context_expr, state)
                if item.optional_vars is not None:
                    # `__enter__` may return anything, including the context
                    # object itself.
                    entered = self.eval(item.context_expr, state) | {EXTERNAL}
                    state = self.bind(item.optional_vars, entered, None, state)
            return self.flow(node.body, state)
        if isinstance(node, (ast.Try, ast.TryStar)):
            merged = self.flow(node.body, dict(state))
            for handler in node.handlers:
                inner = dict(state)
                if handler.name:
                    inner[handler.name] = frozenset({EXTERNAL})
                merged = self.join(merged, self.flow(handler.body, inner))
            merged = self.flow(node.orelse, merged)
            return self.flow(node.finalbody, merged)
        if isinstance(node, ast.Match):
            self.eval(node.subject, state)
            merged: _State | None = None
            subject = self.eval(node.subject, state)
            for case in node.cases:
                inner = dict(state)
                for name in _capture_names(case.pattern):
                    # A capture may bind the subject itself or a piece of it.
                    inner[name] = subject | self.elements_of(subject)
                after = self.flow(case.body, inner)
                merged = after if merged is None else self.join(merged, after)
            return merged if merged is not None else state
        if isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    state.pop(target.id, None)
            return state
        if isinstance(node, (ast.Expr, ast.Return, ast.Raise, ast.Assert)):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self.eval(child, state)
            return state
        return state

    def loop(self, node: ast.stmt, body, state: _State):  # type: ignore[no-untyped-def]
        """Iterate a loop body until the may-alias state stops growing.

        The join is a union over a finite set of roots, so this terminates;
        the guard is only against a pathological function costing real time.
        """
        del node
        current = dict(state)
        for _ in range(10):
            after = self.join(current, body(dict(current)))
            if after == current:
                return current
            current = after
        for name in current:
            current[name] = current[name] | {EXTERNAL}
        return current

    def join(self, left: _State, right: _State) -> _State:
        merged: _State = {}
        for name in left.keys() | right.keys():
            merged[name] = left.get(name, frozenset()) | right.get(name, frozenset())
        return merged

    def bind(
        self, target: ast.expr, roots: frozenset[str], value: ast.expr | None, state: _State
    ) -> _State:
        if isinstance(target, ast.Name):
            state = dict(state)
            state[target.id] = roots
            return state
        if isinstance(target, ast.Starred):
            return self.bind(target.value, self.elements_of(roots), None, state)
        if isinstance(target, (ast.Tuple, ast.List)):
            literal = (
                isinstance(value, (ast.Tuple, ast.List))
                and len(value.elts) == len(target.elts)
                and not any(isinstance(e, ast.Starred) for e in target.elts)
            )
            for index, element in enumerate(target.elts):
                piece = (
                    self.eval(value.elts[index], state)  # type: ignore[union-attr]
                    if literal
                    else self.elements_of(roots)
                )
                state = self.bind(element, piece, None, state)
            return state
        if isinstance(target, (ast.Subscript, ast.Attribute)):
            self.store_into(self.eval(target.value, state), roots)
            return state
        return state

    def store_into(self, container: frozenset[str], stored: frozenset[str]) -> None:
        for root in container:
            self.holds.setdefault(root, set()).update(stored)

    def elements_of(self, roots: frozenset[str]) -> frozenset[str]:
        """What reading an element out of these objects can yield."""
        found: set[str] = set()
        for root in roots:
            if root == EXTERNAL or root in self.params:
                # What a parameter or external container holds came from
                # outside this function.
                found.add(EXTERNAL)
            found.update(self.holds.get(root, ()))
        return frozenset(found) if found else frozenset({EXTERNAL})

    # -- expressions --------------------------------------------------------

    def eval(self, node: ast.expr, state: _State) -> frozenset[str]:
        if isinstance(node, ast.Name):
            if node.id in state:
                return state[node.id]
            return frozenset({EXTERNAL})
        if isinstance(node, ast.NamedExpr):
            roots = self.eval(node.value, state)
            if isinstance(node.target, ast.Name):
                # In-place so the walrus is visible to the rest of the
                # statement; the snapshot was taken before it ran.
                state[node.target.id] = roots
            return roots
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            alloc = self.fresh()
            for element in node.elts:
                inner = element.value if isinstance(element, ast.Starred) else element
                piece = self.eval(inner, state)
                if isinstance(element, ast.Starred):
                    piece = self.elements_of(piece)
                self.store_into(alloc, piece)
            return alloc
        if isinstance(node, ast.Dict):
            alloc = self.fresh()
            for key, value in zip(node.keys, node.values, strict=False):
                if key is not None:
                    self.store_into(alloc, self.eval(key, state))
                    self.store_into(alloc, self.eval(value, state))
                else:
                    self.store_into(alloc, self.elements_of(self.eval(value, state)))
            return alloc
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for generator in node.generators:
                self.eval(generator.iter, state)
            return self.fresh()
        if isinstance(node, ast.Call):
            for argument in node.args:
                self.eval(argument.value if isinstance(argument, ast.Starred) else argument, state)
            for keyword in node.keywords:
                self.eval(keyword.value, state)
            if isinstance(node.func, ast.Name) and node.func.id not in state:
                if node.func.id in _FRESH_FROM_ELEMENTS:
                    alloc = self.fresh()
                    if node.args:
                        self.store_into(alloc, self.elements_of(self.eval(node.args[0], state)))
                    return alloc
                if node.func.id in _FRESH_SCALAR:
                    return frozenset()
            return frozenset({EXTERNAL})
        if isinstance(node, ast.Subscript):
            base = self.eval(node.value, state)
            self.eval(node.slice, state)
            if isinstance(node.slice, ast.Slice):
                # A slice of a list is a fresh list over the same elements.
                alloc = self.fresh()
                self.store_into(alloc, self.elements_of(base))
                return alloc
            return self.elements_of(base)
        if isinstance(node, ast.IfExp):
            self.eval(node.test, state)
            return self.eval(node.body, state) | self.eval(node.orelse, state)
        if isinstance(node, ast.BoolOp):
            merged: set[str] = set()
            for value in node.values:
                merged.update(self.eval(value, state))
            return frozenset(merged)
        if isinstance(node, ast.Starred):
            return self.eval(node.value, state)
        if isinstance(node, ast.Await):
            return self.eval(node.value, state) | {EXTERNAL}
        if isinstance(node, ast.Constant):
            return frozenset()
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare)):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self.eval(child, state)
            # `xs + ys` allocates; it holds the operands' elements, not them.
            return frozenset()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.eval(child, state)
        return frozenset({EXTERNAL})


def _capture_names(pattern: ast.pattern) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(pattern):
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
    return names


def analyze_aliases(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    immutable_params: frozenset[str] = frozenset(),
) -> AliasInfo:
    """Analyze one function. Nested function bodies are left out: a name they
    capture is not re-bound here, and what they do with it is the effect
    analysis's problem, not the alias map's.

    `immutable_params` names parameters whose declared type cannot be mutated
    in place; an augmented assignment through one of those is a rebinding.
    """
    params = frozenset(
        arg.arg
        for arg in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            *([node.args.vararg] if node.args.vararg else []),
            *([node.args.kwarg] if node.args.kwarg else []),
        ]
    )
    return _Analyzer(params, immutable_params & params).run(node)
