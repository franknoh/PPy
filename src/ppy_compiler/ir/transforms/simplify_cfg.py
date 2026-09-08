"""Control-flow simplification: constant branches, merged blocks, dead blocks."""

from __future__ import annotations

from ..model import Block, IRFunction, Operation, Region, Successor
from ..passes import FunctionPass, PassContext
from .dce import reachable_blocks

__all__ = ["SimplifyCFG"]


def _constant_bool(op: Operation) -> bool | None:
    owner = op.operands[0].owner
    if isinstance(owner, Operation) and owner.name == "core.const":
        value = owner.attributes.get("value")
        if isinstance(value, bool):
            return value
    return None


class SimplifyCFG(FunctionPass):
    name = "simplify-cfg"
    invalidates = ("dominators",)

    def run_on_function(self, function: IRFunction, ctx: PassContext) -> bool:
        changed = False
        pending: list[Region] = [function.body]
        while pending:
            region = pending.pop()
            for block in region.blocks:
                for op in block.operations:
                    pending.extend(op.regions)
            while True:
                step = (
                    self._fold_branches(region, ctx)
                    or self._drop_unreachable(region, ctx)
                    or self._merge_chains(region, ctx)
                )
                if not step:
                    break
                changed = True
        return changed

    @staticmethod
    def _fold_branches(region: Region, ctx: PassContext) -> bool:
        for block in region.blocks:
            terminator = block.terminator
            if terminator is None or terminator.name != "core.cond_br":
                continue
            then, otherwise = terminator.successors
            chosen: Successor | None = None
            reason = ""
            constant = _constant_bool(terminator)
            if constant is not None:
                chosen = then if constant else otherwise
                reason = "constant condition"
            elif then.block is otherwise.block and [id(v) for v in then.arguments] == [
                id(v) for v in otherwise.arguments
            ]:
                chosen = then
                reason = "both targets are one block"
            if chosen is None:
                continue
            target = Successor(chosen.block, list(chosen.arguments))
            terminator.erase()
            block.append(Operation("core.br", successors=(target,)))
            ctx.remark(f"simplify-cfg: ^{block.name} branch folded ({reason})")
            return True
        return False

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
            ctx.remark(f"simplify-cfg: unreachable block ^{block.name} removed")
        return True

    @staticmethod
    def _merge_chains(region: Region, ctx: PassContext) -> bool:
        """`A: ... br B(args)` with B's only predecessor A: B's body joins A."""
        predecessors: dict[Block, list[Block]] = {b: [] for b in region.blocks}
        for block in region.blocks:
            for successor in block.successors:
                predecessors[successor].append(block)
        for block in region.blocks:
            terminator = block.terminator
            if terminator is None or terminator.name != "core.br":
                continue
            target = terminator.successors[0].block
            if target is block or target is region.entry or predecessors[target] != [block]:
                continue
            arguments = list(terminator.successors[0].arguments)
            for parameter, value in zip(target.arguments, arguments, strict=True):
                parameter.replace_all_uses_with(value)
            terminator.erase()
            for op in list(target.operations):
                target.remove(op)
                block.append(op)
            region.blocks.remove(target)
            ctx.remark(f"simplify-cfg: ^{target.name} merged into ^{block.name}")
            return True
        return False
