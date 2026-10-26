"""A function's graph, made ready for structured C: a private copy when it must change.

Two shapes keep the C writer from expressing a graph with `while` and `if`:

- An empty block that only branches on -- a generator's resume point, a
  branch's empty arm -- splits one join into several. Its predecessors are
  sent straight to where it goes.
- A cycle entered at more than one block is irreducible: no `while` has
  two tops. Python's own control flow never makes one, but a generated
  matcher can -- a backtracking regex re-enters its forward path from its
  dispatcher. Each extra entry is split off: the predecessors from outside
  the cycle get their own copy of the entry and of every block it
  dominates, so the cycle keeps one entry. A value defined in a copied
  block is read only by blocks that block dominates, and those are copied
  with it, so the copy needs no new block arguments.

The module the function belongs to is never changed: when there is work
to do, it is done on a copy, and the C writer reads the copy.
"""

from __future__ import annotations

from ...ir import Block, IRFunction, Operation, Successor, Value
from ...ir.model import Region

__all__ = ["prepare"]

#: How far splitting may grow a function before the writer is left the original.
_GROWTH = 3
_SLACK = 64


def prepare(function: IRFunction) -> IRFunction:
    """`function`, or a copy of it with trampolines bypassed and every cycle given one entry."""
    blocks = list(function.body.blocks)
    if not blocks or not (_trampolines(blocks) or _irreducible(blocks)):
        return function
    copy = _clone(function)
    region = copy.body
    _bypass(region)
    limit = _GROWTH * sum(len(b.operations) for b in region.blocks) + _SLACK
    while True:
        entry = _extra_entry(list(region.blocks))
        if entry is None:
            return copy
        _split(region, *entry)
        if sum(len(b.operations) for b in region.blocks) > limit:
            return function


# -- the graph ---------------------------------------------------------------------


def _successors(block: Block) -> list[Block]:
    terminator = block.terminator
    return [s.block for s in terminator.successors] if terminator is not None else []


def _predecessors(blocks: list[Block]) -> dict[int, list[Block]]:
    preds: dict[int, list[Block]] = {id(b): [] for b in blocks}
    for block in blocks:
        for successor in _successors(block):
            if id(successor) in preds:
                preds[id(successor)].append(block)
    return preds


def _reverse_postorder(blocks: list[Block]) -> list[Block]:
    order: list[Block] = []
    seen = {id(blocks[0])}
    stack = [(blocks[0], iter(_successors(blocks[0])))]
    while stack:
        block, successors = stack[-1]
        for successor in successors:
            if id(successor) not in seen:
                seen.add(id(successor))
                stack.append((successor, iter(_successors(successor))))
                break
        else:
            stack.pop()
            order.append(block)
    return order[::-1]


def _dominators(blocks: list[Block]) -> dict[int, Block | None]:
    """Each reachable block's immediate dominator (Cooper, Harvey, Kennedy)."""
    rpo = _reverse_postorder(blocks)
    position = {id(b): i for i, b in enumerate(rpo)}
    preds = _predecessors(blocks)
    idom: dict[int, Block] = {id(rpo[0]): rpo[0]}

    def intersect(a: Block, b: Block) -> Block:
        while a is not b:
            while position[id(a)] > position[id(b)]:
                a = idom[id(a)]
            while position[id(b)] > position[id(a)]:
                b = idom[id(b)]
        return a

    changed = True
    while changed:
        changed = False
        for block in rpo[1:]:
            candidate: Block | None = None
            for pred in preds[id(block)]:
                if id(pred) in idom and id(pred) in position:
                    candidate = pred if candidate is None else intersect(pred, candidate)
            if candidate is not None and idom.get(id(block)) is not candidate:
                idom[id(block)] = candidate
                changed = True
    result: dict[int, Block | None] = dict(idom)
    result[id(rpo[0])] = None
    return result


def _dominates(idom: dict[int, Block | None], a: Block, b: Block | None) -> bool:
    while b is not None:
        if b is a:
            return True
        b = idom.get(id(b))
    return False


def _trampolines(blocks: list[Block]) -> bool:
    return any(_is_trampoline(b, blocks[0]) for b in blocks[1:])


def _is_trampoline(block: Block, entry: Block) -> bool:
    terminator = block.terminator
    return (
        block is not entry
        and not block.arguments
        and len(block.operations) == 1
        and terminator is not None
        and terminator.name == "core.br"
        and terminator.successors[0].block is not block
    )


def _irreducible(blocks: list[Block]) -> bool:
    """Whether some edge closes a cycle at a block that does not dominate its source."""
    idom = _dominators(blocks)
    rpo = _reverse_postorder(blocks)
    position = {id(b): i for i, b in enumerate(rpo)}
    for block in rpo:
        for successor in _successors(block):
            if position[id(successor)] <= position[id(block)] and not _dominates(
                idom, successor, block
            ):
                return True
    return False


def _components(nodes: list[Block], inside: set[int]) -> list[list[Block]]:
    """The strongly connected components of the graph on `inside` (Tarjan), cycles only."""
    index: dict[int, int] = {}
    low: dict[int, int] = {}
    on_stack: set[int] = set()
    stack: list[Block] = []
    found: list[list[Block]] = []
    counter = 0
    for root in nodes:
        if id(root) in index:
            continue
        work = [(root, iter([s for s in _successors(root) if id(s) in inside]))]
        index[id(root)] = low[id(root)] = counter
        counter += 1
        stack.append(root)
        on_stack.add(id(root))
        while work:
            block, successors = work[-1]
            advanced = False
            for successor in successors:
                if id(successor) not in index:
                    index[id(successor)] = low[id(successor)] = counter
                    counter += 1
                    stack.append(successor)
                    on_stack.add(id(successor))
                    work.append(
                        (successor, iter([s for s in _successors(successor) if id(s) in inside]))
                    )
                    advanced = True
                    break
                if id(successor) in on_stack:
                    low[id(block)] = min(low[id(block)], index[id(successor)])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[id(parent)] = min(low[id(parent)], low[id(block)])
            if low[id(block)] == index[id(block)]:
                component: list[Block] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(id(member))
                    component.append(member)
                    if member is block:
                        break
                cyclic = len(component) > 1 or block in _successors(block)
                if cyclic:
                    found.append(component)
    return found


def _extra_entry(blocks: list[Block]) -> tuple[Block, list[Block]] | None:
    """A cycle's second entry and the predecessors that enter there, or None.

    Cycles nest: once a cycle has one entry, the cycles inside it are those
    of its blocks without that entry.
    """
    rpo = _reverse_postorder(blocks)
    position = {id(b): i for i, b in enumerate(rpo)}
    preds = _predecessors(rpo)
    idom = _dominators(blocks)

    def cost(head: Block) -> int:
        """What splitting `head` copies: it and every block it dominates."""
        return sum(len(b.operations) for b in rpo if _dominates(idom, head, b))

    pending = [rpo]
    while pending:
        nodes = pending.pop()
        inside = {id(b) for b in nodes}
        for component in _components(nodes, inside):
            members = {id(b) for b in component}
            entries: dict[int, list[Block]] = {}
            for block in component:
                outside = [p for p in preds[id(block)] if id(p) not in members]
                if outside:
                    entries[id(block)] = outside
            ordered = sorted(component, key=lambda b: position[id(b)])
            heads = [b for b in ordered if id(b) in entries]
            if len(heads) > 1:
                # The entry that is cheapest to copy splits off; the rest stay.
                split = min(heads, key=lambda b: (cost(b), -position[id(b)]))
                return split, entries[id(split)]
            header = heads[0] if heads else ordered[0]
            pending.append([b for b in ordered if b is not header])
    return None


# -- changing the copy -------------------------------------------------------------


def _retarget(op: Operation, index: int, target: Successor) -> None:
    old = op.successors[index]
    old.drop_uses()
    op.successors[index] = Successor(target.block, list(target.arguments))


def _bypass(region: Region) -> None:
    """Send every branch to an empty forwarding block on to where it goes."""
    entry = region.blocks[0]
    for block in region.blocks:
        terminator = block.terminator
        if terminator is None:
            continue
        for index, successor in enumerate(list(terminator.successors)):
            final = successor
            seen: set[int] = set()
            while _is_trampoline(final.block, entry) and id(final.block) not in seen:
                seen.add(id(final.block))
                hop = final.block.terminator
                assert hop is not None
                final = hop.successors[0]
            if final is successor:
                continue
            others = [s for i, s in enumerate(terminator.successors) if i != index]
            if any(o.block is final.block for o in others):
                # Both arms would name one block; the empty one keeps them apart.
                continue
            _retarget(terminator, index, final)
    live = {id(b) for b in _reverse_postorder(region.blocks)}
    for block in list(region.blocks):
        if id(block) not in live:
            for op in list(block.operations):
                op.erase()
            region.blocks.remove(block)


def _split(region: Region, entry: Block, outside: list[Block]) -> None:
    """Give the predecessors in `outside` their own copy of `entry` and all it dominates."""
    blocks = list(region.blocks)
    idom = _dominators(blocks)
    owned = [b for b in blocks if _dominates(idom, entry, b)]
    mapping: dict[int, Value] = {}
    copies: dict[int, Block] = {}
    for block in owned:
        copy = region.add_block(
            f"{block.name}.again", [(a.name, a.type) for a in block.arguments]
        )
        copies[id(block)] = copy
        for old, new in zip(block.arguments, copy.arguments, strict=True):
            mapping[id(old)] = new
    late: list[tuple[Operation, int, Value]] = []
    for block in owned:
        for op in block.operations:
            copies[id(block)].append(_copy_operation(op, mapping, copies, late))
    _settle(late, mapping)
    for pred in outside:
        terminator = pred.terminator
        assert terminator is not None
        for index, successor in enumerate(terminator.successors):
            if successor.block is entry:
                _retarget(terminator, index, Successor(copies[id(entry)], successor.arguments))


def _copy_operation(
    op: Operation,
    mapping: dict[int, Value],
    blocks: dict[int, Block],
    late: list[tuple[Operation, int, Value]],
) -> Operation:
    """`op` again, reading the copies of what it reads where they exist.

    An operand defined later in block order is fixed up by `_settle`, once
    every copy exists; `late` collects those.
    """
    successors = [
        Successor(
            blocks.get(id(s.block), s.block),
            [mapping.get(id(v), v) for v in s.arguments],
        )
        for s in op.successors
    ]
    copied = Operation(
        op.name,
        [mapping.get(id(v), v) for v in op.operands],
        [r.type for r in op.results],
        dict(op.attributes),
        successors,
        op.location,
        [r.name for r in op.results],
    )
    for index, value in enumerate(op.operands):
        if id(value) not in mapping:
            late.append((copied, index, value))
    for s_index, s in enumerate(op.successors):
        for a_index, value in enumerate(s.arguments):
            if id(value) not in mapping:
                late.append((copied, -1 - (s_index * 1024 + a_index), value))
    for old, new in zip(op.results, copied.results, strict=True):
        mapping[id(old)] = new
    for inner in op.regions:
        region = copied.add_region()
        _clone_region(inner, region, mapping, late)
    return copied


def _clone_region(
    source: Region,
    target: Region,
    mapping: dict[int, Value],
    late: list[tuple[Operation, int, Value]],
) -> None:
    blocks: dict[int, Block] = {}
    for block in source.blocks:
        copy = target.add_block(block.name, [(a.name, a.type) for a in block.arguments])
        blocks[id(block)] = copy
        for old, new in zip(block.arguments, copy.arguments, strict=True):
            mapping[id(old)] = new
    for block in source.blocks:
        for op in block.operations:
            blocks[id(block)].append(_copy_operation(op, mapping, blocks, late))


def _settle(late: list[tuple[Operation, int, Value]], mapping: dict[int, Value]) -> None:
    """Point the operands read before their copy existed at the copy."""
    for op, index, value in late:
        new = mapping.get(id(value))
        if new is None:
            continue
        if index >= 0:
            op.set_operand(index, new)
        else:
            s_index, a_index = divmod(-1 - index, 1024)
            op.successors[s_index].set_argument(a_index, new)


def _clone(function: IRFunction) -> IRFunction:
    copy = IRFunction(
        function.symbol,
        function.params,
        function.results,
        function.attributes,
        function.location,
    )
    copy.param_attributes = function.param_attributes
    copy.module = function.module
    mapping: dict[int, Value] = {}
    late: list[tuple[Operation, int, Value]] = []
    _clone_region(function.body, copy.body, mapping, late)
    _settle(late, mapping)
    return copy
