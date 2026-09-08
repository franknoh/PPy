"""`lower-async`: a coroutine becomes a starter and a state machine (spec 76).

An `async` function -- `ppy.async = true`, awaiting through `async.await` --
is split at each await. What comes out is two functions. The starter keeps
the coroutine's name and parameters: it takes a frame of `slots` words from
the runtime, stores its arguments in it, spawns the frame, and returns the
future the runtime will complete. The resume function takes the frame,
reads its state, and jumps to the segment after the await that state
names; a segment runs to the next await, records the new state, suspends on
the awaited future, and returns to the runtime, which calls the resume
function again once that future completes. A return completes the frame's
future; a guard that fails fails it, since a coroutine that has already
run cannot hand itself back to Python.

Locals are the frame's: every stack slot becomes words of the frame, and
every value defined before an await and read after it is spilled to a
word of its own around the suspension, so the resume function's blocks
are dominated the way SSA needs. The frame's first three words are the
runtime's -- its own future, the future it awaits, its state.
"""

from __future__ import annotations

from ..analysis import region_dominators
from ..dialects import aio as async_dialect
from ..dialects import core
from ..model import Block, Builder, IRFunction, IRModule, Operation, Successor, Value
from ..passes import Pass, PassContext
from ..types import (
    I64,
    VOID,
    BufferType,
    FutureType,
    IRType,
    PtrType,
    StructType,
    TupleType,
    VectorType,
)

__all__ = ["AsyncLoweringError", "LowerAsync", "lower_async"]

SLOT_STATE = 2


class AsyncLoweringError(ValueError):
    """A coroutine the state machine cannot hold: a buffer, a vector."""


class LowerAsync(Pass):
    name = "lower-async"

    def run(self, module: IRModule, ctx: PassContext) -> bool:
        lowered = lower_async(module)
        if lowered:
            for function in module.functions.values():
                if not function.is_declaration:
                    ctx.invalidate(function)
        return bool(lowered)


def lower_async(module: IRModule) -> int:
    """Lower every coroutine of `module`; how many there were."""
    coroutines = [
        f
        for f in module.functions.values()
        if not f.is_declaration
        and async_dialect.is_async(f)
        and not f.attributes.get("ppy.async.lowered")
    ]
    for function in module.functions.values():
        if (
            function.is_declaration
            and async_dialect.is_async(function)
            and not function.attributes.get("ppy.async.lowered")
        ):
            # Another module's coroutine: its starter hands back a future.
            inner = function.results[0] if function.results else VOID
            function.results = (FutureType(inner),)
            function.attributes["ppy.async.lowered"] = True
    if not coroutines:
        return 0
    module.require("async", 1)
    plans = [_Plan(module, f) for f in coroutines]
    for plan in plans:
        plan.build_resume()
    for plan in plans:
        plan.build_starter()
    for function in list(module.functions.values()):
        for op in list(function.operations()):
            if op.name != "async.create":
                continue
            callee = module.functions.get(op.attributes["callee"].name)  # type: ignore[union-attr]
            if callee is None:
                continue
            b = Builder().before(op)
            call = core.call(b, callee.name, tuple(op.operands), (op.results[0].type,))
            op.results[0].replace_all_uses_with(call.results[0])
            op.erase()
    return len(plans)


def _words(t: IRType) -> int:
    """How many 8-byte words of a frame a value of `t` takes."""
    if isinstance(t, TupleType):
        return sum(_words(item) for item in t.items) or 1
    if isinstance(t, StructType):
        return sum(_words(item) for _name, item in t.fields) or 1
    if isinstance(t, (BufferType, VectorType)):
        raise AsyncLoweringError(f"a coroutine cannot hold a {t} across an await")
    return 1


class _Plan:
    """One coroutine's lowering: its resume function, its frame, its starter."""

    def __init__(self, module: IRModule, function: IRFunction) -> None:
        self.module = module
        self.function = function
        self.inner: IRType = function.results[0] if function.results else VOID
        if len(function.results) > 1:
            raise AsyncLoweringError(f"@{function.name}: a coroutine returns one value or none")
        symbol = str(function.attributes.get("ppy.symbol", function.name))
        self.resume = module.add_function(
            f"{function.name}_resume",
            [("frame", PtrType(I64))],
            [],
            attributes={
                "ppy.symbol": f"{symbol}_resume",
                "ppy.abi": "resume",
                "ppy.async.resume": function.name,
            },
            location=function.location,
        )
        self.slots = async_dialect.HEADER_SLOTS
        self.param_slots: list[int] = []
        self.segments: list[Block] = []
        self.fails = 0
        self.labels = 0
        self.frame: Value | None = None

    # -- frame words --------------------------------------------------------------

    def allocate(self, t: IRType, count: int = 1) -> int:
        first = self.slots
        self.slots += max(count, 1) * _words(t)
        return first

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

    def build_resume(self) -> None:
        function = self.function
        entry = self.resume.add_entry_block()
        self.frame = entry.arguments[0]
        moved = list(function.body.blocks)
        function.body.blocks = []
        for block in moved:
            block.region = self.resume.body
            self.resume.body.blocks.append(block)
        original_entry = moved[0]
        original_entry.name = self.fresh("begin")
        for argument in list(original_entry.arguments):
            k = self.allocate(argument.type)
            self.param_slots.append(k)
            for user, index in list(argument.uses):
                b = Builder().before(_user_op(user))
                _set(user, index, core.load(b, self.slot(b, k, argument.type)))
        original_entry.arguments = []
        for op in list(self.resume.operations()):
            if op.name == "core.alloca":
                pointer = op.results[0].type
                assert isinstance(pointer, PtrType)
                count = int(op.attributes.get("count", 1))  # type: ignore[call-overload]
                k = self.allocate(pointer.pointee, count)
                for user, index in list(op.results[0].uses):
                    b = Builder().before(_user_op(user))
                    _set(user, index, self.slot(b, k, pointer.pointee))
                    if isinstance(user, Operation):
                        _regeneralize(user)
                op.erase()
        for op in list(self.resume.operations()):
            if op.name == "core.ret":
                b = Builder().before(op)
                async_dialect.complete(b, self.frame, op.operands[0] if op.operands else None)
                core.ret(b)
                op.erase()
            elif op.name == "core.guard":
                self._split_guard(op)
        for op in list(self.resume.operations()):
            if op.name == "async.await":
                self._split_await(op)
        self._dispatch(entry, original_entry)
        self._spill()
        self._reorder()
        function.attributes["ppy.async.lowered"] = True

    def _reorder(self) -> None:
        """Blocks in reverse postorder from the entry, so a definition precedes its uses.

        A backend that walks the blocks in order needs every dominator ahead
        of what it dominates; the split put resumed segments after the blocks
        they dominate.
        """
        region = self.resume.body
        order: list[Block] = []
        seen: set[int] = set()

        def visit(block: Block) -> None:
            if id(block) in seen:
                return
            seen.add(id(block))
            for successor in block.successors:
                visit(successor)
            order.append(block)

        visit(region.blocks[0])
        ordered = list(reversed(order))
        ordered.extend(block for block in region.blocks if id(block) not in seen)
        region.blocks = ordered

    def _tail_into(self, op: Operation, label: str) -> Block:
        """A new block holding everything after `op` in its block."""
        block = op.parent
        assert block is not None
        index = block.operations.index(op)
        tail = block.operations[index + 1 :]
        target = self.resume.body.add_block(self.fresh(f"{block.name}.{label}"))
        for following in tail:
            block.remove(following)
            target.append(following)
        return target

    def _split_await(self, op: Operation) -> None:
        assert self.frame is not None
        future = op.operands[0]
        result = op.results[0] if op.results else None
        resumed = self._tail_into(op, "resume")
        state = len(self.segments) + 1
        self.segments.append(resumed)
        b = Builder().before(op)
        core.store(b, core.const(b, state, I64), self.slot(b, SLOT_STATE, I64))
        async_dialect.suspend(b, self.frame, future)
        core.ret(b)
        op.erase()
        if result is not None:
            rb = Builder().before(resumed.operations[0])
            value = async_dialect.result(rb, self.frame, result.type)
            result.replace_all_uses_with(value)

    def _split_guard(self, op: Operation) -> None:
        assert self.frame is not None
        label = str(op.attributes.get("label") or op.attributes.get("kind") or "guard")
        ok = self._tail_into(op, "ok")
        failed = self.resume.body.add_block(self.fresh("fail"))
        b = Builder().before(op)
        core.cond_br(b, op.operands[0], Successor(ok), Successor(failed))
        op.erase()
        fb = Builder(failed)
        self.fails += 1
        async_dialect.fail(fb, self.frame, label, self.fails)
        core.ret(fb)

    def _dispatch(self, entry: Block, original_entry: Block) -> None:
        b = Builder(entry)
        state = core.load(b, self.slot(b, SLOT_STATE, I64), name="state")
        for number, target in enumerate(self.segments, start=1):
            following = self.resume.body.add_block(self.fresh("dispatch"))
            matches = core.cmp(b, "eq", state, core.const(b, number, I64))
            core.cond_br(b, matches, Successor(target), Successor(following))
            b = Builder(following)
        core.br(b, Successor(original_entry))

    def _spill(self) -> None:
        """Every value read where its definition no longer dominates goes through the frame."""
        region = self.resume.body
        dominators = region_dominators(region)
        owners: dict[int, Operation] = {}
        for block in region.blocks:
            terminator = block.terminator
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
                        b = Builder().before(use_op)
                        _set(user, index, core.const(b, definition.attributes["value"], value.type))
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

    # -- the starter --------------------------------------------------------------

    def build_starter(self) -> None:
        function = self.function
        entry = function.add_entry_block()
        b = Builder(entry)
        frame = async_dialect.frame_new(b, self.slots, name="frame")
        for argument, k in zip(entry.arguments, self.param_slots, strict=True):
            core.store(b, argument, self.slot(b, k, argument.type, frame))
        future = async_dialect.spawn(b, self.resume.name, frame, self.inner, name="future")
        core.ret(b, future)
        function.results = (FutureType(self.inner),)


def _generic(t: IRType) -> IRType:
    """`t` with every stack pointer in it made generic: the frame is heap memory."""
    if isinstance(t, PtrType):
        return PtrType(
            _generic(t.pointee),
            "generic" if t.address_space == "stack" else t.address_space,
            t.mutable,
        )
    return t


def _regeneralize(op: Operation) -> None:
    """Pointers derived from a former stack slot are generic now: the frame is heap."""
    for result in op.results:
        t = result.type
        if isinstance(t, PtrType) and t.address_space == "stack":
            result.type = PtrType(t.pointee, "generic", t.mutable)
            for user, _index in list(result.uses):
                if isinstance(user, Operation):
                    _regeneralize(user)


def _user_op(user: object) -> Operation:
    if isinstance(user, Operation):
        return user
    assert isinstance(user, Successor)
    raise AsyncLoweringError("a coroutine's parameter or slot is passed to a block; not yet")


def _set(user: object, index: int, value: Value) -> None:
    if isinstance(user, Operation):
        user.set_operand(index, value)
    else:
        assert isinstance(user, Successor)
        user.set_argument(index, value)


def _index(block: Block, op: Operation) -> int:
    return block.operations.index(op)
