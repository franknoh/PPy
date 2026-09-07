"""Analyses over the IR that passes ask for by name and a pass manager caches.

An analysis is a pure function of a function's IR. The pass manager keeps
the result until a pass that does not preserve it runs, so the same
dominator tree serves every pass between two changes to the CFG.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .model import Block, IRFunction, Region

__all__ = ["ANALYSES", "Dominators", "dominators", "register_analysis"]


@dataclass(frozen=True, slots=True)
class Dominators:
    """Which blocks dominate which, per region of one function."""

    sets: dict[Block, frozenset[Block]]

    def dominates(self, a: Block, b: Block) -> bool:
        return a in self.sets.get(b, frozenset())

    def immediate(self, block: Block) -> Block | None:
        """The nearest strict dominator, or None for an entry."""
        strict = self.sets.get(block, frozenset()) - {block}
        best: Block | None = None
        for candidate in strict:
            if best is None or self.dominates(best, candidate):
                best = candidate
        return best


def region_dominators(region: Region) -> dict[Block, frozenset[Block]]:
    """Dominator sets by the classic iteration; regions are small."""
    blocks = region.blocks
    if not blocks:
        return {}
    entry = blocks[0]
    predecessors: dict[Block, list[Block]] = {b: [] for b in blocks}
    for block in blocks:
        for successor in block.successors:
            if successor in predecessors:
                predecessors[successor].append(block)
    everything = set(blocks)
    sets: dict[Block, set[Block]] = {b: set(everything) for b in blocks}
    sets[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for block in blocks[1:]:
            incoming = [sets[p] for p in predecessors[block]]
            new = set.intersection(*incoming) if incoming else set()
            new.add(block)
            if new != sets[block]:
                sets[block] = new
                changed = True
    return {block: frozenset(dominating) for block, dominating in sets.items()}


def dominators(function: IRFunction) -> Dominators:
    sets: dict[Block, frozenset[Block]] = {}
    pending: list[Region] = [function.body]
    while pending:
        region = pending.pop()
        sets.update(region_dominators(region))
        for block in region.blocks:
            for op in block.operations:
                pending.extend(op.regions)
    return Dominators(sets)


#: Analyses by name; a pass names what it requires and preserves.
ANALYSES: dict[str, Callable[[IRFunction], object]] = {"dominators": dominators}


def register_analysis(name: str, compute: Callable[[IRFunction], object]) -> None:
    ANALYSES[name] = compute
