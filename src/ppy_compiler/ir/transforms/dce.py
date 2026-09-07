"""Dead code elimination: unused pure operations and unreachable blocks."""

from __future__ import annotations

from ..model import Block, IRFunction, Region
from ..passes import FunctionPass, PassContext

__all__ = ["DeadCodeElimination", "reachable_blocks"]


def reachable_blocks(region: Region) -> set[Block]:
    entry = region.entry
    if entry is None:
        return set()
    seen = {entry}
    pending = [entry]
    while pending:
        block = pending.pop()
        for successor in block.successors:
            if successor not in seen:
                seen.add(successor)
                pending.append(successor)
    return seen


class DeadCodeElimination(FunctionPass):
    name = "dce"
    #: Removing an unreachable block changes no reachable block's dominators.
    preserves = ("dominators",)

    def run_on_function(self, function: IRFunction, ctx: PassContext) -> bool:
        changed = False
        pending: list[Region] = [function.body]
        while pending:
            region = pending.pop()
            for block in region.blocks:
                for op in block.operations:
                    pending.extend(op.regions)
            changed |= self._drop_unreachable(region, ctx)
            changed |= self._drop_unused(region, ctx)
        return changed

    @staticmethod
    def _drop_unreachable(region: Region, ctx: PassContext) -> bool:
        live = reachable_blocks(region)
        dead = [block for block in region.blocks if block not in live]
        if not dead:
            return False
        for block in dead:
            for op in list(block.operations):
                op.erase()
            region.blocks.remove(block)
            ctx.remark(f"dce: unreachable block ^{block.name} removed")
        return True

    @staticmethod
    def _drop_unused(region: Region, ctx: PassContext) -> bool:
        changed = False
        while True:
            removed = False
            for block in region.blocks:
                for op in list(reversed(block.operations)):
                    spec = ctx.registry.op_spec(op.name)
                    if spec is None or not spec.pure or op.regions:
                        continue
                    if any(result.uses for result in op.results):
                        continue
                    op.erase()
                    ctx.remark(f"dce: unused {op.name} removed")
                    removed = changed = True
            if not removed:
                return changed
