"""Generators in native code, inlined where they are consumed.

A generator function called in a `for` loop, or a generator expression, is
lowered into the loop that consumes it: its body runs in its own scope, and
at each `yield` the consumer's step runs with the yielded value, then the
body carries on after the `yield`. `continue` in the consumer goes back
into the generator; `break` closes it. Nothing is allocated for the
generator itself, so nothing is left to free but its locals, which go when
it is exhausted, when the consumer breaks, or, where an exception leaves,
with the function's own.

The consumers are `for`, `sum`, `min`, `max`, `sorted`, `any`, `all`,
`next`, and a collection built from one (`Vec[int](gen())`). A generator
held in a name and stepped by the statements after it (`it = gen()`, then
`x = next(it)` and `for x in it:` at the same level) is inlined once, with
those statements as its consumer. A generator that escapes (returned,
passed on, stored in a collection), one that calls itself, one with a
`try`, and `x = yield` (which `send` answers) keep the function in Python.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field

from ..analysis import types as T
from ..backend.llvm.lowering import Unsupported
from ..ir import BOOL, I64, Block, Successor, Value
from ..ir.dialects import core
from .collections import HANDLE, Kind, Shape, shape_of

__all__ = ["GeneratorLowering"]

#: What a function's lowering keeps by local name: swapped whole when a
#: generator's body runs inside its consumer.
_SCOPE = (
    "slots",
    "buffers",
    "_buffer_origins",
    "tuples",
    "objects",
    "matches",
    "collections",
    "caught",
    "_stable",
    "_entry_loads",
    "constants",
    "_induction",
    "info",
    "bindings",
    "_loops",
)

#: What a step gets: the value, and whether the step owns it (a handle).
Visit = Callable[[Value, bool], None]


@dataclass(slots=True)
class _Scope:
    values: dict[str, object]


@dataclass(slots=True)
class _Inline:
    """One generator being lowered into its consumer."""

    shape: Shape
    consumer: _Scope
    visit: Visit
    #: Where the generator's end goes, and where a `break` in the consumer goes.
    exhausted: Block
    broken: Block
    #: Whether the consumer's step is a loop body `break` and `continue` reach.
    loops: bool
    qualname: str
    #: The scopes of the generator's own locals, let go of at its end.
    owned: list[Value] = field(default_factory=list)


class GeneratorLowering:  # pylint: disable=attribute-defined-outside-init
    """The generator half of lowering one function; mixed into `_FunctionLowering`."""

    # -- scopes -----------------------------------------------------------------

    def _scope(self) -> _Scope:
        return _Scope({name: getattr(self, name) for name in _SCOPE if hasattr(self, name)})

    def _enter(self, scope: _Scope) -> None:
        for name, value in scope.values.items():
            setattr(self, name, value)

    def _fresh_scope(self, info: object) -> _Scope:
        """An empty scope for a generator function's own locals."""
        fresh: dict[str, object] = {}
        for name in _SCOPE:
            if not hasattr(self, name):
                continue
            current = getattr(self, name)
            if isinstance(current, dict):
                fresh[name] = {}
            elif isinstance(current, set):
                fresh[name] = set()
            elif isinstance(current, list):
                fresh[name] = []
            else:
                fresh[name] = current
        fresh["info"] = info
        return _Scope(fresh)

    def _layered_scope(self, hidden: set[str]) -> _Scope:
        """The consumer's scope as a generator expression sees it: every name,
        but for its own loop variables, which are new."""
        layered: dict[str, object] = {}
        for name in _SCOPE:
            if not hasattr(self, name):
                continue
            current = getattr(self, name)
            if isinstance(current, dict):
                layered[name] = {k: v for k, v in current.items() if k not in hidden}
            elif isinstance(current, set):
                layered[name] = {k for k in current if k not in hidden}
            elif isinstance(current, list):
                layered[name] = []
            else:
                layered[name] = current
        return _Scope(layered)

    # -- what is a generator ------------------------------------------------------------

    def _inlines_stack(self) -> list[_Inline]:
        found = getattr(self, "_inlines", None)
        if found is None:
            found = []
            self._inlines = found
        return found

    def _generator_function(self, node: ast.expr):  # type: ignore[no-untyped-def]
        """The generator function `node` calls, with its analysis, or None."""
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return None
        called = T.strip_literal(self._type_of(node.func))  # type: ignore[attr-defined]
        qualname = getattr(called, "qualname", None)
        if not qualname:
            return None
        found = self.frontend.analysis.functions.get(qualname)  # type: ignore[attr-defined]
        if found is None or not found.info.is_generator:
            return None
        return found

    def _is_generator(self, node: ast.expr) -> bool:
        return isinstance(node, ast.GeneratorExp) or self._generator_function(node) is not None

    def _element_shape(self, node: ast.expr) -> Shape:
        """What a generator yields, from what the checker says it is an iterator of."""
        made = T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]
        element = made.args[0] if isinstance(made, T.Instance) and made.args else None
        shape = shape_of(element, self._records()) if element is not None else None  # type: ignore[attr-defined]
        if shape is None:
            raise Unsupported(f"`{ast.unparse(node)}` yields values with no native form")
        return shape

    # -- lowering a generator into its consumer ----------------------------------------

    def _inline(
        self, node: ast.expr, visit: Visit, exhausted: Block, broken: Block, *, loops: bool
    ) -> None:
        """Lower the generator `node` into the code here: `visit` runs at each
        `yield`, in the consumer's scope; its end goes to `exhausted`, and a
        `break` in a looping consumer to `broken`."""
        shape = self._element_shape(node)
        hoist = getattr(self, "hoist", False)
        self.hoist = False  # type: ignore[attr-defined]
        consumer = self._scope()
        at_end = self._block("gen.end")  # type: ignore[attr-defined]
        at_break = self._block("gen.break")  # type: ignore[attr-defined]
        if isinstance(node, ast.GeneratorExp):
            if any(g.is_async for g in node.generators):
                raise Unsupported("an async generator expression has no native lowering")
            hidden = {
                n.id for g in node.generators for n in ast.walk(g.target) if isinstance(n, ast.Name)
            }
            inline = _Inline(shape, consumer, visit, at_end, at_break, loops, "<genexpr>")
            self._enter(self._layered_scope(hidden))
            body: list[ast.stmt] = [_expression_loops(node)]
        else:
            found = self._generator_function(node)
            info = found.info
            if any(i.qualname == info.qualname for i in self._inlines_stack()):
                raise Unsupported(f"`{info.name}` yields from itself")
            _check_generator(info)
            assert isinstance(node, ast.Call)
            if node.keywords or len(node.args) != len(info.params):
                raise Unsupported(f"`{info.name}` is called with its parameters in order natively")
            arguments = [
                self._argument(param, arg)
                for param, arg in zip(info.params, node.args, strict=True)
            ]
            inline = _Inline(shape, consumer, visit, at_end, at_break, loops, info.qualname)
            self._enter(self._fresh_scope(info))
            for param, (value, owned, reference) in zip(info.params, arguments, strict=True):
                if reference is not None:
                    self._bind(param.name, reference, value, owned)  # type: ignore[attr-defined]
                else:
                    self._store(ast.Name(id=param.name, ctx=ast.Store()), value)  # type: ignore[attr-defined]
            body = list(info.node.body)
        self._inlines_stack().append(inline)
        try:
            self._body(body)  # type: ignore[attr-defined]
            if self._open():  # type: ignore[attr-defined]
                core.br(self.b, Successor(at_end))  # type: ignore[attr-defined]
            owned = [held.slot for held in self.collections.values()]  # type: ignore[attr-defined]
        finally:
            self._inlines_stack().pop()
            self._enter(consumer)
            self.hoist = hoist  # type: ignore[attr-defined]
        # The generator's own locals go at its end and where it is closed; an
        # exception that leaves lets them go with the function's.
        mine = [
            slot for slot in owned if all(slot is not h.slot for h in self.collections.values())
        ]  # type: ignore[attr-defined]
        self._exception_slots.extend(mine)  # type: ignore[attr-defined]
        for block, onward in ((at_end, exhausted), (at_break, broken)):
            if self._seal(block):  # type: ignore[attr-defined]
                continue
            self.b.at_end(block)  # type: ignore[attr-defined]
            for slot in mine:
                self._release(core.load(self.b, slot))  # type: ignore[attr-defined]
                core.store(self.b, self._rt("ppy_coll_none", (), HANDLE), slot)  # type: ignore[attr-defined]
            core.br(self.b, Successor(onward))  # type: ignore[attr-defined]

    def _argument(self, param, argument: ast.expr) -> tuple[Value, bool, object]:  # type: ignore[no-untyped-def]
        """A generator's argument, made in the caller's scope: its value, whether it
        is an owned handle, and what it is held as where it is a handle."""
        reference = self._reference_of_type(param.type)  # type: ignore[attr-defined]
        if reference is not None:
            value, owned = self._handle(argument)  # type: ignore[attr-defined]
            return value, owned, reference
        shape = shape_of(param.type, self._records())  # type: ignore[attr-defined]
        if shape is None:
            raise Unsupported(f"`{param.name}` has no native form")
        value, owned = self._value(argument, shape)  # type: ignore[attr-defined]
        return value, owned, None

    def _yield_statement(self, node: ast.Yield | ast.YieldFrom) -> None:
        stack = self._inlines_stack()
        if not stack:
            raise Unsupported("a generator runs natively where it is consumed")
        inline = stack[-1]
        if isinstance(node, ast.YieldFrom):
            self._yield_from(inline, node.value)
            return
        if node.value is None:
            raise Unsupported("a generator that yields None has no native form")
        value, owned = self._value(node.value, inline.shape)  # type: ignore[attr-defined]
        self._step(inline, value, owned)

    def _yield_from(self, inline: _Inline, source: ast.expr) -> None:
        """`yield from` another generator, or from a collection: each of its values
        is one of this generator's."""
        if self._is_generator(source):
            after = self._block("yield.from.end")  # type: ignore[attr-defined]
            self._inline(
                source,
                lambda value, owned: self._step(inline, value, owned),
                after,
                after,
                loops=False,
            )
            self.b.at_end(after)  # type: ignore[attr-defined]
            return
        walked = self._source(source) if self._is_walk(source) else None  # type: ignore[attr-defined]
        if walked is None:
            raise Unsupported(f"`yield from {ast.unparse(source)}` has no native lowering")
        self._walk(walked, lambda items: self._step(inline, items[0][1], False))  # type: ignore[attr-defined]

    def _step(self, inline: _Inline, value: Value, owned: bool) -> None:
        """One `yield`: the consumer's step in its scope, then on in the generator."""
        resume = self._block("gen.resume")  # type: ignore[attr-defined]
        generator = self._scope()
        stack = self._inlines_stack()
        stack.remove(inline)
        self._enter(inline.consumer)
        if inline.loops:
            self._loops.append((resume, inline.broken))  # type: ignore[attr-defined]
        try:
            inline.visit(value, owned)
        finally:
            if inline.loops:
                self._loops.pop()  # type: ignore[attr-defined]
            inline.consumer = self._scope()
            self._enter(generator)
            stack.append(inline)
        reached = self._open()  # type: ignore[attr-defined]
        if reached:
            core.br(self.b, Successor(resume))  # type: ignore[attr-defined]
        self.b.at_end(resume)  # type: ignore[attr-defined]
        if not reached:
            # The step never comes back (`next` took the first value): nothing
            # after this `yield` runs.
            core.unreachable(self.b)  # type: ignore[attr-defined]
            self._dead.add(id(resume))  # type: ignore[attr-defined]

    def _generator_return(self, node: ast.Return) -> bool:
        """`return` in a generator's body: it is exhausted."""
        stack = self._inlines_stack()
        if not stack:
            return False
        if node.value is not None and not (
            isinstance(node.value, ast.Constant) and node.value.value is None
        ):
            raise Unsupported("a generator's return value has no native lowering")
        core.br(self.b, Successor(stack[-1].exhausted))  # type: ignore[attr-defined]
        return True

    # -- a generator held in a name and stepped ------------------------------------

    def _stepped_generator(self, statement: ast.stmt, rest: list[ast.stmt]) -> bool:
        """`it = gen(...)` followed by statements that step it: `x = next(it)`,
        `x = next(it, default)`, and `for x in it:`, each a statement of the
        same list. Lowers the assignment and every statement after it; False
        where `statement` is no such assignment.

        The generator is inlined once, and its consumer is the rest of the list:
        a value goes to the site waiting for it (a word says which), and the
        statements between two sites run there before the generator resumes.
        Its end goes to the same site with no value: `next` gives its default
        or raises `StopIteration`, and a `for` runs its `else`."""
        found = _stepped_target(statement)
        if found is None:
            return False
        name, source = found
        if not self._is_generator(source):
            return False
        sites = [_Site.of(index, entry, name) for index, entry in enumerate(rest)]
        steps = [site for site in sites if site is not None]
        if not steps:
            return False
        for index, entry in enumerate(rest):
            _check_segment_use(entry, name, sites[index] is not None)
        shape = self._element_shape(source)
        if shape.reference and any(site.kind == "next" for site in steps):
            raise Unsupported("`next` of strings or objects has no native lowering")
        segments = _segments(rest, steps)
        # The first runs before the generator starts, the last after it closes;
        # the ones between run inside it.
        for segment in segments[1:-1]:
            _check_segment(segment)
        waiting = self._alloca(I64, f"{name}.site")  # type: ignore[attr-defined]
        after = self._block("stepped.after")  # type: ignore[attr-defined]
        exhausted = self._block("stepped.end")  # type: ignore[attr-defined]
        closed = self._block("stepped.closed")  # type: ignore[attr-defined]
        self._body(segments[0])  # type: ignore[attr-defined]
        if not self._open():  # type: ignore[attr-defined]
            return True
        core.store(self.b, self._word(0), waiting)  # type: ignore[attr-defined]

        def visit(value: Value, owned: bool) -> None:
            resume = self._loops[-1][0]  # type: ignore[attr-defined]
            joined = self._block("stepped.resume")  # type: ignore[attr-defined]
            for k, site in enumerate(steps):
                here = self._block(f"stepped.site{k}")  # type: ignore[attr-defined]
                other = self._block(f"stepped.not{k}")  # type: ignore[attr-defined]
                current = core.load(self.b, waiting)  # type: ignore[attr-defined]
                matched = core.cmp(self.b, "eq", current, self._word(k))  # type: ignore[attr-defined]
                core.cond_br(self.b, matched, Successor(here), Successor(other))  # type: ignore[attr-defined]
                self.b.at_end(here)  # type: ignore[attr-defined]
                self._site_value(
                    site, k, steps, segments, shape, value, owned, waiting, resume, closed
                )
                if self._open():  # type: ignore[attr-defined]
                    core.br(self.b, Successor(joined))  # type: ignore[attr-defined]
                self.b.at_end(other)  # type: ignore[attr-defined]
            core.unreachable(self.b)  # type: ignore[attr-defined]
            if not self._seal(joined):  # type: ignore[attr-defined]
                # Left open: the generator resumes from here.
                self.b.at_end(joined)  # type: ignore[attr-defined]

        self._inline(source, visit, exhausted, closed, loops=True)
        self.b.at_end(exhausted)  # type: ignore[attr-defined]
        for k in range(len(steps)):
            here = self._block(f"stepped.ended{k}")  # type: ignore[attr-defined]
            other = self._block(f"stepped.notended{k}")  # type: ignore[attr-defined]
            current = core.load(self.b, waiting)  # type: ignore[attr-defined]
            matched = core.cmp(self.b, "eq", current, self._word(k))  # type: ignore[attr-defined]
            core.cond_br(self.b, matched, Successor(here), Successor(other))  # type: ignore[attr-defined]
            self.b.at_end(here)  # type: ignore[attr-defined]
            self._drained(k, steps, segments, shape, after)
            self.b.at_end(other)  # type: ignore[attr-defined]
        core.unreachable(self.b)  # type: ignore[attr-defined]
        if not self._seal(closed):  # type: ignore[attr-defined]
            self.b.at_end(closed)  # type: ignore[attr-defined]
            core.br(self.b, Successor(after))  # type: ignore[attr-defined]
        if self._seal(after):  # type: ignore[attr-defined]
            return True
        self.b.at_end(after)  # type: ignore[attr-defined]
        self._body(segments[-1])  # type: ignore[attr-defined]
        return True

    def _site_value(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        site: _Site,
        k: int,
        steps: list[_Site],
        segments: list[list[ast.stmt]],
        shape: Shape,
        value: Value,
        owned: bool,
        waiting: Value,
        resume: Block,
        closed: Block,
    ) -> None:
        """A value for the site waiting, `k`: bound where it asks, then on to the
        next site, which waits for the next value, or, past the last one, the
        generator closes."""
        if site.kind == "next":
            self._store(site.target, value)  # type: ignore[attr-defined]
            self._after_site(k, steps, segments, waiting, closed)
            return
        assert isinstance(site.node, ast.For)
        left = self._block(f"stepped.for{k}.break")  # type: ignore[attr-defined]
        self._bind_item(site.node.target, [(shape, value)])  # type: ignore[attr-defined]
        if owned:
            self._release(value)  # type: ignore[attr-defined]
        self._loops.append((resume, left))  # type: ignore[attr-defined]
        try:
            self._body(site.node.body)  # type: ignore[attr-defined]
        finally:
            self._loops.pop()  # type: ignore[attr-defined]
        if self._open():  # type: ignore[attr-defined]
            core.br(self.b, Successor(resume))  # type: ignore[attr-defined]
        if self._seal(left):  # type: ignore[attr-defined]
            # No `break`: this loop ends only where the generator does.
            self.b.at_end(self._block(f"stepped.for{k}.dead"))  # type: ignore[attr-defined]
            core.unreachable(self.b)  # type: ignore[attr-defined]
            return
        self.b.at_end(left)  # type: ignore[attr-defined]
        self._after_site(k, steps, segments, waiting, closed)

    def _after_site(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        k: int,
        steps: list[_Site],
        segments: list[list[ast.stmt]],
        waiting: Value,
        closed: Block,
    ) -> None:
        """What runs once site `k` is done: past the last site, the generator
        closes (what follows runs after it); otherwise the statements up to the
        next site, which then waits for a value while the generator resumes."""
        if k + 1 == len(steps):
            core.br(self.b, Successor(closed))  # type: ignore[attr-defined]
            return
        self._body(segments[k + 1])  # type: ignore[attr-defined]
        if self._open():  # type: ignore[attr-defined]
            core.store(self.b, self._word(k + 1), waiting)  # type: ignore[attr-defined]

    def _drained(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        k: int,
        steps: list[_Site],
        segments: list[list[ast.stmt]],
        shape: Shape,
        after: Block,
    ) -> None:
        """The generator ended with site `k` waiting: it and every site after it
        find it empty, in order, with the statements between them."""
        for j in range(k, len(steps)):
            site = steps[j]
            if site.kind == "next":
                if site.default is None:
                    self._raise_made(  # type: ignore[attr-defined]
                        "StopIteration",
                        _stop_tag(),
                        self._string_literal(""),  # type: ignore[attr-defined]
                        self._word(1),  # type: ignore[attr-defined]
                    )
                    return
                default, _ = self._value(site.default, shape)  # type: ignore[attr-defined]
                self._store(site.target, default)  # type: ignore[attr-defined]
            else:
                assert isinstance(site.node, ast.For)
                self._body(site.node.orelse)  # type: ignore[attr-defined]
            if not self._open():  # type: ignore[attr-defined]
                return
            if j + 1 < len(steps):
                self._body(segments[j + 1])  # type: ignore[attr-defined]
                if not self._open():  # type: ignore[attr-defined]
                    return
        core.br(self.b, Successor(after))  # type: ignore[attr-defined]

    # -- consumers ----------------------------------------------------------------

    def _for_generator(self, node: ast.For) -> bool:
        """`for x in gen(...)`: the loop's body is the generator's step."""
        if not self._is_generator(node.iter):
            return False
        shape = self._element_shape(node.iter)
        done = self._block("for.gen.end")  # type: ignore[attr-defined]
        exhausted = self._block("for.gen.else") if node.orelse else done  # type: ignore[attr-defined]

        def visit(value: Value, owned: bool) -> None:
            self._bind_item(node.target, [(shape, value)])  # type: ignore[attr-defined]
            if owned:
                self._release(value)  # type: ignore[attr-defined]
            self._body(node.body)  # type: ignore[attr-defined]

        self._inline(node.iter, visit, exhausted, done, loops=True)
        if node.orelse:
            self.b.at_end(exhausted)  # type: ignore[attr-defined]
            self._body(node.orelse)  # type: ignore[attr-defined]
            if self._open():  # type: ignore[attr-defined]
                core.br(self.b, Successor(done))  # type: ignore[attr-defined]
        self.b.at_end(done)  # type: ignore[attr-defined]
        return True

    def _generator_consumer(self, name: str, node: ast.Call) -> Value | None:
        """`any`, `all`, and `next` of a generator."""
        if node.keywords or not node.args or not self._is_generator(node.args[0]):
            return None
        source = node.args[0]
        shape = self._element_shape(source)
        if name in {"any", "all"} and len(node.args) == 1:
            if shape.reference:
                raise Unsupported(f"`{name}` of strings or objects has no native lowering")
            found = self._alloca(BOOL, name)  # type: ignore[attr-defined]
            core.store(self.b, core.const(self.b, name == "all", BOOL), found)  # type: ignore[attr-defined]
            done = self._block(f"{name}.end")  # type: ignore[attr-defined]

            def decide(value: Value, _owned: bool) -> None:
                truth = self._truth(value)  # type: ignore[attr-defined]
                if name == "all":
                    true = core.const(self.b, True, BOOL)  # type: ignore[attr-defined]
                    truth = core.bitwise(self.b, "xor", truth, true)  # type: ignore[attr-defined]
                decided = self._block(f"{name}.decided")  # type: ignore[attr-defined]
                onward = self._block(f"{name}.on")  # type: ignore[attr-defined]
                core.cond_br(self.b, truth, Successor(decided), Successor(onward))  # type: ignore[attr-defined]
                self.b.at_end(decided)  # type: ignore[attr-defined]
                core.store(self.b, core.const(self.b, name == "any", BOOL), found)  # type: ignore[attr-defined]
                core.br(self.b, Successor(self._loops[-1][1]))  # type: ignore[attr-defined]
                self.b.at_end(onward)  # type: ignore[attr-defined]

            self._inline(source, decide, done, done, loops=True)
            self.b.at_end(done)  # type: ignore[attr-defined]
            return core.load(self.b, found)  # type: ignore[attr-defined]
        if name == "next" and len(node.args) in {1, 2}:
            if shape.reference:
                raise Unsupported("`next` of strings or objects has no native lowering")
            got = self._alloca(shape.ir_type(), "next.value")  # type: ignore[attr-defined]
            done = self._block("next.end")  # type: ignore[attr-defined]
            empty = self._block("next.empty")  # type: ignore[attr-defined]

            def first(value: Value, _owned: bool) -> None:
                core.store(self.b, value, got)  # type: ignore[attr-defined]
                core.br(self.b, Successor(self._loops[-1][1]))  # type: ignore[attr-defined]

            self._inline(source, first, empty, done, loops=True)
            self.b.at_end(empty)  # type: ignore[attr-defined]
            if len(node.args) == 2:
                default, _ = self._value(node.args[1], shape)  # type: ignore[attr-defined]
                core.store(self.b, default, got)  # type: ignore[attr-defined]
                core.br(self.b, Successor(done))  # type: ignore[attr-defined]
            else:
                self._raise_made(  # type: ignore[attr-defined]
                    "StopIteration",
                    _stop_tag(),
                    self._string_literal(""),  # type: ignore[attr-defined]
                    self._word(1),  # type: ignore[attr-defined]
                )
            self.b.at_end(done)  # type: ignore[attr-defined]
            return core.load(self.b, got)  # type: ignore[attr-defined]
        return None

    # -- the collections' walks, over a generator ----------------------------------------

    def _is_walk(self, node: ast.expr) -> bool:
        return self._is_generator(node) or super()._is_walk(node)  # type: ignore[misc]

    def _source(self, node: ast.expr):  # type: ignore[no-untyped-def]
        if not self._is_generator(node):
            return super()._source(node)  # type: ignore[misc]
        from .collection_api import _Source  # pylint: disable=import-outside-toplevel

        made = _Source(Kind("Vec", self._element_shape(node)), None)  # type: ignore[arg-type]
        self.__dict__.setdefault("_generator_sources", {})[id(made)] = (made, node)
        return made

    def _generator_of(self, source: object) -> ast.expr | None:
        found = self.__dict__.get("_generator_sources", {}).get(id(source))
        return found[1] if found is not None and found[0] is source else None

    def _walk(self, source, visit) -> None:  # type: ignore[no-untyped-def]
        node = self._generator_of(source)
        if node is None:
            super()._walk(source, visit)  # type: ignore[misc]
            return
        shape = source.kind.value
        done = self._block("walk.gen.end")  # type: ignore[attr-defined]

        def step(value: Value, owned: bool) -> None:
            visit([(shape, value)])
            if owned:
                self._release(value)  # type: ignore[attr-defined]

        self._inline(node, step, done, done, loops=False)
        self.b.at_end(done)  # type: ignore[attr-defined]

    def _fill(self, kind: Kind, handle: Value, node: ast.expr, front: bool = False) -> None:
        """A collection built from a generator: each value added as it is yielded."""
        if not self._is_generator(node):
            super()._fill(kind, handle, node, front)  # type: ignore[misc]
            return
        source = self._source(node)
        self._walk(
            source,
            lambda items: self._add_value(kind, handle, items[0][1], front),  # type: ignore[attr-defined]
        )

    def _start(self, source):  # type: ignore[no-untyped-def]
        if self._generator_of(source) is not None:
            raise Unsupported("a generator is walked by `for`, not stepped in `zip` or `enumerate`")
        return super()._start(source)  # type: ignore[misc]


@dataclass(slots=True)
class _Site:
    """A statement that steps a held generator: `x = next(it[, default])` or `for`."""

    index: int
    kind: str
    node: ast.stmt
    target: ast.expr | None = None
    default: ast.expr | None = None

    @staticmethod
    def of(index: int, node: ast.stmt, name: str) -> _Site | None:
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id == name:
            return _Site(index, "for", node)
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "next"
            and not value.keywords
            and value.args
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id == name
            and len(value.args) <= 2
            and len(targets) == 1
            and isinstance(targets[0], ast.Name)
        ):
            default = value.args[1] if len(value.args) == 2 else None
            return _Site(index, "next", node, targets[0], default)
        return None


def _stepped_target(statement: ast.stmt) -> tuple[str, ast.expr] | None:
    """`it = gen(...)` or `it: Iterator[int] = (x for ...)`: the name and the generator."""
    if isinstance(statement, ast.Assign):
        if len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            return statement.targets[0].id, statement.value
        return None
    if (
        isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.value is not None
    ):
        return statement.target.id, statement.value
    return None


def _check_segment_use(statement: ast.stmt, name: str, site: bool) -> None:
    """The held generator is named only by its sites, and not bound again."""
    if site:
        assert isinstance(statement, (ast.For, ast.Assign, ast.AnnAssign))
        inside = statement.body + statement.orelse if isinstance(statement, ast.For) else []
        named = [n for part in inside for n in ast.walk(part)]
        if isinstance(statement, (ast.Assign, ast.AnnAssign)) and statement.value is not None:
            call = statement.value
            assert isinstance(call, ast.Call)
            named.extend(n for extra in call.args[1:] for n in ast.walk(extra))
    else:
        named = list(ast.walk(statement))
    if any(isinstance(n, ast.Name) and n.id == name for n in named):
        raise Unsupported(f"`{name}` is stepped natively only by `next` and `for` statements")


def _segments(rest: list[ast.stmt], steps: list[_Site]) -> list[list[ast.stmt]]:
    """The statements before the first site, between each two, and after the last."""
    bounds = [site.index for site in steps]
    segments = [rest[: bounds[0]]]
    for here, following in zip(bounds, [*bounds[1:], len(rest)], strict=True):
        segments.append(rest[here + 1 : following])
    return segments


def _check_segment(segment: list[ast.stmt]) -> None:
    """Statements that run between steps run inside the generator's lowering: a
    `return`, or a `break` or `continue` of a loop around them, would leave it
    without closing it."""
    for statement in segment:
        for inner in _outside_loops(statement):
            if isinstance(inner, (ast.Return, ast.Break, ast.Continue)):
                raise Unsupported(
                    "a `return`, `break`, or `continue` between a held generator's steps "
                    "has no native lowering"
                )


def _outside_loops(node: ast.AST):  # type: ignore[no-untyped-def]
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.For, ast.While)):
            # A `break` or `continue` in there is that loop's; a `return` is not.
            yield from (n for n in ast.walk(child) if isinstance(n, ast.Return))
            continue
        if isinstance(child, (ast.FunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield from _outside_loops(child)


def _stop_tag() -> int:
    from .exceptions import exception_tag  # pylint: disable=import-outside-toplevel

    return exception_tag("StopIteration")


def _check_generator(info) -> None:  # type: ignore[no-untyped-def]
    """A generator function native code can inline: a plain function, no `try`,
    and each `yield` a statement of its own."""
    if info.owner or info.type_params or info.is_async:
        raise Unsupported(f"`{info.name}` is a method, generic, or async generator")
    for inner in ast.walk(info.node):
        if inner is info.node:
            continue
        if isinstance(inner, (ast.Try, ast.FunctionDef, ast.Lambda, ast.ClassDef, ast.With)):
            raise Unsupported(f"`{info.name}` holds a `{type(inner).__name__}` natively unlowered")
    yields = {id(n) for n in ast.walk(info.node) if isinstance(n, (ast.Yield, ast.YieldFrom))}
    statements = {
        id(n.value)
        for n in ast.walk(info.node)
        if isinstance(n, ast.Expr) and isinstance(n.value, (ast.Yield, ast.YieldFrom))
    }
    if yields - statements:
        raise Unsupported(f"`{info.name}` uses a `yield`'s value, which `send` gives")


def _expression_loops(node: ast.GeneratorExp) -> ast.stmt:
    """A generator expression as the loops it stands for: `for ...: if ...: yield elt`."""
    yielded = ast.Expr(ast.Yield(node.elt))
    ast.copy_location(yielded, node.elt)
    inner: list[ast.stmt] = [yielded]
    for generator in reversed(node.generators):
        for condition in reversed(generator.ifs):
            inner = [ast.copy_location(ast.If(test=condition, body=inner, orelse=[]), condition)]
        inner = [
            ast.copy_location(
                ast.For(target=generator.target, iter=generator.iter, body=inner, orelse=[]),
                generator.iter,
            )
        ]
    return inner[0]
