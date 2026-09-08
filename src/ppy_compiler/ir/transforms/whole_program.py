"""Whole-program optimization over a linked module (spec 10, 11).

Once the modules are one program the optimizer can see across them.
`internalize` makes every function the program does not hand out private:
not bound to Python, not a C export, not a kernel or a resume function the
runtime reaches by address. `Inline` copies a small callee -- or one marked
`ppy.inline` -- into its caller, so a call across modules costs what a
call within one did, and the constants of one side reach the other.
`GlobalDCE` then drops the private functions and globals nothing reaches.
`whole_program` runs them in that order, with the ordinary cleanups between.
"""

from __future__ import annotations

from ..dialects import core
from ..model import (
    Block,
    Builder,
    IRFunction,
    IRModule,
    Operation,
    Successor,
    Symbol,
    SymbolRef,
    Value,
)
from ..passes import Pass, PassContext, PassManager
from .canonicalize import Canonicalize
from .dce import DeadCodeElimination
from .simplify_cfg import SimplifyCFG

__all__ = ["GlobalDCE", "Inline", "internalize", "references", "whole_program"]

#: A callee this many operations or fewer is inlined without being asked.
INLINE_BUDGET = 24
#: Functions the program hands out by address: they stay whatever their callers.
_KEPT_BY_ATTRIBUTE = ("ppy.export", "ppy.entry", "ppy.async.resume", "gpu.kind")


def internalize(module: IRModule, keep: set[str]) -> int:
    """Every definition not in `keep` and not handed out by address becomes private."""
    changed = 0
    for name, function in module.functions.items():
        if function.is_declaration or name in keep or function.symbol.visibility == "private":
            continue
        if any(attribute in function.attributes for attribute in _KEPT_BY_ATTRIBUTE):
            continue
        function.symbol = Symbol(name, "private")
        changed += 1
    return changed


def references(function: IRFunction) -> set[str]:
    """The symbols `function`'s operations name: callees, spawned frames, string constants."""
    found: set[str] = set()
    for op in function.operations():
        for value in op.attributes.values():
            if isinstance(value, SymbolRef):
                found.add(value.name)
        symbol = op.attributes.get("symbol")
        if isinstance(symbol, str):
            found.add(symbol)
    return found


class GlobalDCE(Pass):
    """Private functions and globals nothing public reaches go away."""

    name = "global-dce"

    def run(self, module: IRModule, ctx: PassContext) -> bool:
        reachable: set[str] = set()
        pending = [
            name
            for name, function in module.functions.items()
            if function.symbol.visibility != "private"
        ]
        pending.extend(
            name for name, item in module.globals.items() if item.symbol.visibility != "private"
        )
        while pending:
            name = pending.pop()
            if name in reachable:
                continue
            reachable.add(name)
            function = module.functions.get(name)
            if function is not None:
                pending.extend(references(function) - reachable)
        dead_functions = [name for name in module.functions if name not in reachable]
        dead_globals = [name for name in module.globals if name not in reachable]
        for name in dead_functions:
            del module.functions[name]
        for name in dead_globals:
            del module.globals[name]
        if dead_functions or dead_globals:
            ctx.remark(
                f"global-dce: {len(dead_functions)} function(s) and {len(dead_globals)} "
                "global(s) nothing reached were removed"
            )
        return bool(dead_functions or dead_globals)


class Inline(Pass):
    """A small callee, or one marked `ppy.inline`, is copied into its callers."""

    name = "inline"

    def __init__(self, budget: int = INLINE_BUDGET) -> None:
        self.budget = budget
        self._counter = 0

    def run(self, module: IRModule, ctx: PassContext) -> bool:
        changed = False
        for function in list(module.functions.values()):
            if function.is_declaration:
                continue
            inlined = 0
            for op in list(function.operations()):
                if (
                    op.name != "core.call"
                    or op.parent is None
                    or op.attributes.get("capture_status")
                ):
                    continue
                callee = module.functions.get(op.attributes["callee"].name)  # type: ignore[union-attr]
                if callee is None or not self._inlinable(callee, function):
                    continue
                self._inline(function, op, callee)
                inlined += 1
            if inlined:
                changed = True
                ctx.invalidate(function)
                ctx.remark(f"@{function.name}: {inlined} call(s) inlined")
        return changed

    def _inlinable(self, callee: IRFunction, caller: IRFunction) -> bool:
        if callee.is_declaration or callee is caller:
            return False
        attributes = callee.attributes
        if any(
            attributes.get(key)
            for key in ("ppy.noinline", "ppy.async", "ppy.async.resume", "gpu.kind", "ppy.export")
        ):
            return False
        if attributes.get("ppy.abi") == "resume":
            return False
        operations = list(callee.operations())
        if any(op.regions for op in operations):
            return False
        for op in operations:
            callee_name = op.attributes.get("callee")
            if isinstance(callee_name, SymbolRef) and callee_name.name == callee.name:
                return False  # recursion
        if attributes.get("ppy.inline"):
            return True
        return len(operations) <= self.budget

    def _inline(self, caller: IRFunction, call: Operation, callee: IRFunction) -> None:
        self._counter += 1
        tag = self._counter
        block = call.parent
        assert block is not None
        index = block.operations.index(call)
        tail = block.operations[index + 1 :]
        continuation = caller.body.add_block(
            f"{block.name}.inlined{tag}", [(r.name, r.type) for r in call.results]
        )
        for op in tail:
            block.remove(op)
            continuation.append(op)
        for result, argument in zip(call.results, continuation.arguments, strict=True):
            result.replace_all_uses_with(argument)
        mapping: dict[int, Value] = {}
        clones: dict[int, Block] = {}
        for original in callee.body.blocks:
            clone = caller.body.add_block(
                f"{callee.name}.{original.name}{tag}",
                [(a.name, a.type) for a in original.arguments],
            )
            clones[id(original)] = clone
            for old, new in zip(original.arguments, clone.arguments, strict=True):
                mapping[id(old)] = new
        for original in callee.body.blocks:
            clone = clones[id(original)]
            for op in original.operations:
                if op.name == "core.ret":
                    returned = [mapping.get(id(v), v) for v in op.operands]
                    clone.append(
                        Operation(
                            "core.br", (), (), {}, [Successor(continuation, returned)], op.location
                        )
                    )
                    continue
                operands = [mapping.get(id(v), v) for v in op.operands]
                successors = [
                    Successor(clones[id(s.block)], [mapping.get(id(v), v) for v in s.arguments])
                    for s in op.successors
                ]
                copied = Operation(
                    op.name,
                    operands,
                    [r.type for r in op.results],
                    dict(op.attributes),
                    successors,
                    op.location,
                    [r.name for r in op.results],
                )
                for old, new in zip(op.results, copied.results, strict=True):
                    mapping[id(old)] = new
                clone.append(copied)
        entry = clones[id(callee.body.blocks[0])]
        b = Builder().before(call)
        core.br(b, Successor(entry, list(call.operands)))
        call.erase()


def whole_program(module: IRModule, keep: set[str], level: int = 2) -> PassContext:
    """Internalize, inline, clean up, and drop the dead: the program as one unit."""
    internalize(module, keep)
    ctx = PassContext()
    manager = PassManager(ctx)
    if level >= 1:
        manager.add(Inline())
        manager.add(Canonicalize())
        manager.add(SimplifyCFG())
        manager.add(DeadCodeElimination())
    manager.add(GlobalDCE())
    manager.run(module)
    return ctx
