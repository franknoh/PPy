"""`promote-slots`: stack slots read and written in one block become values.

The frontend gives every local a `core.alloca` and spells assignment as a
store and every read as a load. Where all of a slot's traffic is in one
block and a store comes first, the loads are the stored values and the
slot is nothing: each load reads the value most recently stored above it,
and the memory operations go away. A slot touched from two blocks would
need block arguments to carry the value, and is left as it is; so is a
slot whose pointer does anything but load and store.
"""

from __future__ import annotations

from ..model import Block, IRFunction, Operation, Value
from ..passes import FunctionPass, PassContext

__all__ = ["PromoteSlots", "promote_slots"]


class PromoteSlots(FunctionPass):
    name = "promote-slots"
    #: Loads and stores go; no branch changes.
    preserves = ("dominators",)

    def run_on_function(self, function: IRFunction, ctx: PassContext) -> bool:
        promoted = promote_slots(function)
        if promoted:
            ctx.remark(f"@{function.name}: {promoted} stack slot(s) promoted to values")
        return promoted > 0


def promote_slots(function: IRFunction) -> int:
    """Promote what can be promoted; the number of slots that were."""
    promoted = 0
    for block in list(function.body.blocks):
        for op in list(block.operations):
            if op.name == "core.alloca" and op.attributes.get("count", 1) == 1:
                promoted += int(_promote(op))
    return promoted


def _promote(alloca: Operation) -> bool:
    slot = alloca.results[0]
    loads: list[Operation] = []
    stores: list[Operation] = []
    for user, index in slot.uses:
        if not isinstance(user, Operation):
            return False
        if user.name == "core.load" and index == 0:
            loads.append(user)
        elif user.name == "core.store" and index == 1:
            stores.append(user)
        else:
            return False
    traffic = loads + stores
    if not traffic:
        alloca.erase()
        return True
    block: Block | None = traffic[0].parent
    if block is None or any(op.parent is not block for op in traffic):
        return False
    accesses = {id(op) for op in traffic}
    current: Value | None = None
    replacements: list[tuple[Operation, Value]] = []
    for op in block.operations:
        if id(op) not in accesses:
            continue
        if op.name == "core.store":
            current = op.operands[0]
        elif current is None:
            # Read before any write: whatever the slot held, it is not ours to say.
            return False
        else:
            replacements.append((op, current))
    for load, value in replacements:
        load.results[0].replace_all_uses_with(value)
        load.erase()
    for store in stores:
        store.erase()
    alloca.erase()
    return True
