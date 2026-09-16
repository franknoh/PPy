"""Conservative integer bounds for eliminating unnecessary floor corrections.

Bounds follow acyclic control flow and nonescaping local slots. A function
containing a cycle is left alone: loop induction needs a separate fixed-point
analysis. Arithmetic whose interval can overflow loses its bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...ir import Block, IntType, IRFunction, Operation, Value

type Bounds = tuple[int, int]


@dataclass
class _State:
    bounds: dict[Value, Bounds] = field(default_factory=dict)
    aliases: dict[Value, Value] = field(default_factory=dict)

    def copy(self) -> _State:
        return _State(dict(self.bounds), dict(self.aliases))

    def get(self, value: Value) -> Bounds | None:
        if value in self.bounds:
            return self.bounds[value]
        if not isinstance(value.type, IntType):
            return None
        producer = value.owner
        if isinstance(producer, Operation) and producer.name == "core.const":
            number = producer.attributes["value"]
            assert isinstance(number, int)
            return number, number
        return value.type.min, value.type.max

    def constrain(self, value: Value, bounds: Bounds) -> None:
        old = self.get(value)
        if old is None:
            return
        clipped = max(old[0], bounds[0]), min(old[1], bounds[1])
        if clipped[0] > clipped[1]:
            return  # An unreachable edge need not contribute a proof.
        self.bounds[value] = clipped
        slot = self.aliases.get(value)
        if slot is not None:
            self.bounds[slot] = clipped
            for other, home in self.aliases.items():
                if home is slot:
                    self.bounds[other] = clipped


def _merge(states: list[_State]) -> _State:
    if not states:
        return _State()
    result = states[0].copy()
    for state in states[1:]:
        result.bounds = {
            value: (min(bounds[0], state.bounds[value][0]), max(bounds[1], state.bounds[value][1]))
            for value, bounds in result.bounds.items()
            if value in state.bounds
        }
        result.aliases = {
            value: slot
            for value, slot in result.aliases.items()
            if state.aliases.get(value) is slot
        }
    return result


def _branch(state: _State, condition: Value, truth: bool) -> None:
    comparison = condition.owner
    if not isinstance(comparison, Operation) or comparison.name != "core.cmp":
        return
    left, right = comparison.operands
    a, b = state.get(left), state.get(right)
    if a is None or b is None:
        return
    predicate = str(comparison.attributes["predicate"])
    if not truth:
        predicate = {"eq": "ne", "ne": "eq", "lt": "ge", "le": "gt", "gt": "le", "ge": "lt"}[
            predicate
        ]
    if predicate in {"gt", "ge"}:
        left, right, a, b = right, left, b, a
        predicate = "lt" if predicate == "gt" else "le"
    if predicate in {"lt", "le"}:
        delta = int(predicate == "lt")
        state.constrain(left, (a[0], b[1] - delta))
        state.constrain(right, (a[0] + delta, b[1]))
    elif predicate == "eq":
        common = max(a[0], b[0]), min(a[1], b[1])
        state.constrain(left, common)
        state.constrain(right, common)


def _arithmetic(op: Operation, state: _State) -> Bounds | None:
    if op.name == "core.cast":
        return state.get(op.operands[0])
    if op.name == "core.select":
        a, b = (state.get(value) for value in op.operands[1:])
        return (min(a[0], b[0]), max(a[1], b[1])) if a and b else None
    if op.name not in {"core.add", "core.sub", "core.mul"}:
        return None
    a, b = (state.get(value) for value in op.operands)
    if a is None or b is None:
        return None
    if op.name == "core.add":
        return a[0] + b[0], a[1] + b[1]
    if op.name == "core.sub":
        return a[0] - b[1], a[1] - b[0]
    products = [x * y for x in a for y in b]
    return min(products), max(products)


def truncating_divisions(function: IRFunction) -> set[int]:
    """Operations for which floor and truncation provably give the same result."""
    if any(op.regions for op in function.operations()):
        return set()
    blocks = list(function.blocks())
    incoming: dict[Block, list[_State]] = {block: [] for block in blocks}
    predecessors: dict[Block, int] = dict.fromkeys(blocks, 0)
    for block in blocks:
        for successor in block.successors:
            predecessors[successor] += 1
    ready = [block for block in blocks if predecessors[block] == 0]
    ordered = []
    while ready:
        block = ready.pop()
        ordered.append(block)
        for successor in block.successors:
            predecessors[successor] -= 1
            if predecessors[successor] == 0:
                ready.append(successor)
    if len(ordered) != len(blocks):
        return set()
    slots = {
        op.result
        for op in function.operations()
        if op.name == "core.alloca"
        and op.attributes.get("count", 1) == 1
        and all(
            isinstance(user, Operation)
            and (user.name, index) in {("core.load", 0), ("core.store", 1)}
            for user, index in op.result.uses
        )
    }
    proven: set[int] = set()
    for block in ordered:
        state = _merge(incoming[block])
        for op in block.operations:
            if op.name == "core.store" and op.operands[1] in slots:
                value, slot = op.operands
                state.aliases = {v: p for v, p in state.aliases.items() if p is not slot}
                state.bounds.pop(slot, None)
                bounds = state.get(value)
                if bounds is not None:
                    state.bounds[slot] = bounds
                    state.aliases[value] = slot
            elif op.name == "core.load" and op.operands[0] in slots:
                slot = op.operands[0]
                state.aliases[op.result] = slot
                if slot in state.bounds:
                    state.bounds[op.result] = state.bounds[slot]
            elif len(op.results) == 1 and isinstance(op.result.type, IntType):
                bounds = _arithmetic(op, state)
                if bounds and op.result.type.fits(bounds[0]) and op.result.type.fits(bounds[1]):
                    state.bounds[op.result] = bounds
                if op.name in {"core.div", "core.mod"}:
                    a, b = (state.get(value) for value in op.operands)
                    if a and b and ((a[0] >= 0 and b[0] > 0) or (a[1] <= 0 and b[1] < 0)):
                        proven.add(id(op))
        terminator = block.terminator
        if terminator is None:
            continue
        for index, successor in enumerate(terminator.successors):
            edge = state.copy()
            if terminator.name == "core.cond_br":
                _branch(edge, terminator.operands[0], index == 0)
            for argument, value in zip(successor.block.arguments, successor.arguments, strict=True):
                bounds = edge.get(value)
                if bounds is not None:
                    edge.bounds[argument] = bounds
            incoming[successor.block].append(edge)
    return proven
