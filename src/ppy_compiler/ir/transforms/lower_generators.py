"""`lower-generators`: a generator function becomes a starter and a resume function.

A generator function the lowering marks `ppy.generator` is lowered like any
other, but for its `yield`s, each a `ppy.gen_yield` intrinsic. What comes out
is two functions. The starter keeps the function's name and parameters: it
makes the generator's frame, a runtime object of `words` words that the
collector sees and that counts its references like a collection does, stores
its arguments there, and returns the frame's handle. The resume function
takes the frame's handle, reads its state, and jumps to the code after the
`yield` that state names; it runs to the next `yield`, puts the value in the
frame, records the state, and returns 1, or runs off the end (or returns) and
returns 0. An exception or a failed guard leaves it as it leaves any native
function, so a consumer's call to it answers the way a call does.

The frame's first word is the resume function's address, the second its
state (-1 once the body has ended), then the yielded value's words. Locals
are the frame's: every stack slot becomes words of it, and every value read
after a `yield` it was made before is spilled to a word of its own. The slots
that own a reference (the lowering marks them `ppy.owns`), and the
parameters that are handles, which the starter takes a reference to, come
first, and the frame's handle mask names them: freeing the frame lets go of
what they hold. The lowering sets each such slot to `None` when it lets go
of it itself, so a generator run to its end frees nothing twice.
"""

from __future__ import annotations

from ..analysis import region_dominators
from ..dialect import DialectRegistry
from ..dialects import core
from ..model import Block, Builder, IRFunction, IRModule, Operation, Successor, Value
from ..passes import Pass, PassContext
from ..types import I64, IntType, IRType, PtrType
from .lower_async import _generic, _index, _regeneralize, _set, _user_op, _words

__all__ = ["GENERATOR_YIELD", "HANDLE_WORDS", "LowerGenerators", "lower_generators"]

#: The intrinsic a `yield` is until this pass.
GENERATOR_YIELD = "ppy.gen_yield"
#: How many words of a frame its handle mask can name.
HANDLE_WORDS = 32

SLOT_RESUME = 0
SLOT_STATE = 1
SLOT_VALUE = 2

_HANDLE = PtrType(IntType(8))


class GeneratorLoweringError(ValueError):
    """A generator the frame cannot hold."""


class LowerGenerators(Pass):
    name = "lower-generators"

    def run(self, module: IRModule, ctx: PassContext) -> bool:
        lowered = lower_generators(module, ctx.registry)
        if lowered:
            for function in module.functions.values():
                if not function.is_declaration:
                    ctx.invalidate(function)
        return bool(lowered)


def lower_generators(module: IRModule, registry: DialectRegistry | None = None) -> int:
    """Lower every generator function of `module`; how many there were."""
    generators = [
        f
        for f in module.functions.values()
        if not f.is_declaration
        and f.attributes.get("ppy.generator")
        and not f.attributes.get("ppy.generator.lowered")
    ]
    for function in generators:
        _Plan(module, function, registry).build()
    return len(generators)


class _Plan:
    """One generator function's lowering."""

    def __init__(
        self, module: IRModule, function: IRFunction, registry: DialectRegistry | None
    ) -> None:
        self.module = module
        self.function = function
        self.registry = registry
        symbol = str(function.attributes.get("ppy.symbol", function.name))
        self.resume = module.add_function(
            f"{function.name}.resume",
            [("generator", _HANDLE)],
            [I64],
            attributes={
                "ppy.symbol": f"{symbol}_resume",
                "ppy.qualname": function.attributes.get("ppy.qualname", function.name),
                "ppy.generator.resume": function.name,
                "ppy.abi": "canonical",
                "ppy.releases_gil": False,
                "effects": function.attributes.get("effects", ()),
            },
            location=function.location,
        )
        yielded = function.attributes.get("ppy.generator.value_words", 1)
        self.slots = SLOT_VALUE + max(int(yielded), 1)  # type: ignore[call-overload]
        self.handles = 0
        self.param_slots: list[int] = []
        self.segments: list[Block] = []
        self.labels = 0
        self.frame: Value | None = None

    def allocate(self, t: IRType, count: int = 1) -> int:
        first = self.slots
        self.slots += max(count, 1) * _words(t)
        return first

    def own(self, k: int) -> None:
        if k >= HANDLE_WORDS:
            raise GeneratorLoweringError(
                f"@{self.function.name}: a generator holds more references than its frame names"
            )
        self.handles |= 1 << k

    def slot(self, b: Builder, k: int, t: IRType, frame: Value | None = None) -> Value:
        """The address of word `k` of the frame, as a pointer to `t`."""
        base = frame if frame is not None else self.frame
        assert base is not None
        pointer = core.ptr_offset(b, base, core.const(b, k, I64))
        held = _generic(t)
        if held != I64:
            pointer = core.cast(b, pointer, PtrType(held))
        return pointer

    def fresh(self, label: str) -> str:
        self.labels += 1
        return f"{label}{self.labels}"

    # -- the resume function ------------------------------------------------------

    def build(self) -> None:
        function = self.function
        entry = self.resume.add_entry_block()
        b = Builder(entry)
        record = core.call_extern(b, "ppy_gen_frame", (entry.arguments[0],), (_HANDLE,))
        self.frame = core.cast(b, record.results[0], PtrType(I64))
        moved = list(function.body.blocks)
        function.body.blocks = []
        for block in moved:
            block.region = self.resume.body
            self.resume.body.blocks.append(block)
        original_entry = moved[0]
        original_entry.name = self.fresh("begin")
        parameters = list(original_entry.arguments)
        handle_params = [a for a in parameters if a.type == _HANDLE]
        others = [a for a in parameters if a.type != _HANDLE]
        slots_of: dict[int, int] = {}
        for argument in [*handle_params, *others]:
            k = self.allocate(argument.type)
            slots_of[id(argument)] = k
            if argument.type == _HANDLE:
                self.own(k)
        for argument in parameters:
            k = slots_of[id(argument)]
            self.param_slots.append(k)
            for user, index in list(argument.uses):
                ub = Builder().before(_user_op(user))
                _set(user, index, core.load(ub, self.slot(ub, k, argument.type)))
        original_entry.arguments = []
        allocas = [op for op in self.resume.operations() if op.name == "core.alloca"]
        owning = [op for op in allocas if op.attributes.get("ppy.owns")]
        for op in [*owning, *(op for op in allocas if not op.attributes.get("ppy.owns"))]:
            pointer = op.results[0].type
            assert isinstance(pointer, PtrType)
            count = int(op.attributes.get("count", 1))  # type: ignore[call-overload]
            k = self.allocate(pointer.pointee, count)
            if op.attributes.get("ppy.owns"):
                self.own(k)
            for user, index in list(op.results[0].uses):
                ub = Builder().before(_user_op(user))
                _set(user, index, self.slot(ub, k, pointer.pointee))
                if isinstance(user, Operation):
                    _regeneralize(user)
            op.erase()
        for op in list(self.resume.operations()):
            if op.name == "core.ret":
                rb = Builder().before(op)
                core.store(rb, core.const(rb, -1, I64), self.slot(rb, SLOT_STATE, I64))
                core.ret(rb, core.const(rb, 0, I64))
                op.erase()
        for op in list(self.resume.operations()):
            if (
                op.name == "core.call_intrinsic"
                and op.attributes.get("intrinsic") == GENERATOR_YIELD
            ):
                self._split_yield(op)
        self._dispatch(entry, original_entry)
        self._spill()
        self._reorder()
        self._starter()
        function.attributes["ppy.generator.lowered"] = True

    def _tail_into(self, op: Operation, label: str) -> Block:
        block = op.parent
        assert block is not None
        index = block.operations.index(op)
        tail = block.operations[index + 1 :]
        target = self.resume.body.add_block(self.fresh(f"{block.name}.{label}"))
        for following in tail:
            block.remove(following)
            target.append(following)
        return target

    def _split_yield(self, op: Operation) -> None:
        value = op.operands[0]
        resumed = self._tail_into(op, "resume")
        state = len(self.segments) + 1
        self.segments.append(resumed)
        b = Builder().before(op)
        core.store(b, value, self.slot(b, SLOT_VALUE, value.type))
        core.store(b, core.const(b, state, I64), self.slot(b, SLOT_STATE, I64))
        core.ret(b, core.const(b, 1, I64))
        op.erase()

    def _dispatch(self, entry: Block, original_entry: Block) -> None:
        b = Builder(entry)
        state = core.load(b, self.slot(b, SLOT_STATE, I64), name="state")
        ended = self.resume.body.add_block(self.fresh("ended"))
        eb = Builder(ended)
        core.ret(eb, core.const(eb, 0, I64))
        following = self.resume.body.add_block(self.fresh("dispatch"))
        core.cond_br(
            b,
            core.cmp(b, "lt", state, core.const(b, 0, I64)),
            Successor(ended),
            Successor(following),
        )
        b = Builder(following)
        for number, target in enumerate(self.segments, start=1):
            following = self.resume.body.add_block(self.fresh("dispatch"))
            matches = core.cmp(b, "eq", state, core.const(b, number, I64))
            core.cond_br(b, matches, Successor(target), Successor(following))
            b = Builder(following)
        core.br(b, Successor(original_entry))

    def _spill(self) -> None:
        """Every value read where its definition no longer dominates goes through the frame."""
        region = self.resume.body
        dominators = region_dominators(region, self.registry)
        owners: dict[int, Operation] = {}
        for block in region.blocks:
            terminator = block.terminator_for(self.registry)
            if terminator is not None:
                for successor in terminator.successors:
                    owners[id(successor)] = terminator
        for block in list(region.blocks):
            defined: list[tuple[Value, Operation | None]] = [(a, None) for a in block.arguments]
            defined.extend((r, op) for op in block.operations for r in op.results)
            for value, definition in defined:
                if value is self.frame:
                    continue
                broken: list[tuple[object, int, Operation]] = []
                for user, index in list(value.uses):
                    use_op = user if isinstance(user, Operation) else owners[id(user)]
                    use_block = use_op.parent
                    assert use_block is not None
                    if use_block is block:
                        if definition is None or _index(block, definition) < _index(block, use_op):
                            continue
                    elif block in dominators.get(use_block, frozenset()):
                        continue
                    broken.append((user, index, use_op))
                if not broken:
                    continue
                if definition is not None and definition.name == "core.const":
                    for user, index, use_op in broken:
                        cb = Builder().before(use_op)
                        _set(
                            user, index, core.const(cb, definition.attributes["value"], value.type)
                        )
                    continue
                k = self.allocate(value.type)
                if definition is None:
                    sb = Builder().before(block.operations[0])
                else:
                    following = block.operations[_index(block, definition) + 1]
                    sb = Builder().before(following)
                core.store(sb, value, self.slot(sb, k, value.type))
                for user, index, use_op in broken:
                    lb = Builder().before(use_op)
                    _set(user, index, core.load(lb, self.slot(lb, k, value.type)))

    def _reorder(self) -> None:
        region = self.resume.body
        order: list[Block] = []
        seen: set[int] = set()

        def visit(block: Block) -> None:
            if id(block) in seen:
                return
            seen.add(id(block))
            for successor in block.successors_for(self.registry):
                visit(successor)
            order.append(block)

        visit(region.blocks[0])
        ordered = list(reversed(order))
        ordered.extend(block for block in region.blocks if id(block) not in seen)
        region.blocks = ordered

    # -- the starter ------------------------------------------------------------------

    def _starter(self) -> None:
        function = self.function
        entry = function.add_entry_block()
        b = Builder(entry)
        address = core.function_address(b, self.resume.name)
        made = core.call_extern(
            b,
            "ppy_gen_new",
            (core.const(b, self.slots, I64), core.const(b, self.handles, I64), address),
            (_HANDLE,),
        ).results[0]
        record = core.call_extern(b, "ppy_gen_frame", (made,), (_HANDLE,)).results[0]
        frame = core.cast(b, record, PtrType(I64))
        for argument, k in zip(entry.arguments, self.param_slots, strict=True):
            if argument.type == _HANDLE:
                # The frame outlives the call: it holds its own reference.
                core.call_extern(b, "ppy_coll_retain", (argument,), ())
            core.store(b, argument, self.slot(b, k, argument.type, frame))
        core.ret(b, made)
