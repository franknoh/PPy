"""LLVM lowerings of the simd, cpu, atomic, and concurrency dialects.

Each function takes the function emitter of `from_ir` and one operation,
and leaves the operation's results set on the emitter. What the dialects
promise is kept here to the letter: a reduction walks the lanes in order,
a mutex is a slot spun on with `compare_exchange`, a condition is a
notification counter, a barrier is arrivals and a generation -- so the
same program built for LLVM, for C, or run under CPython synchronizes the
same way.
"""

from __future__ import annotations

from ...ir import BoolType, FloatType, IntType, IRType, Operation, PtrType, VectorType

__all__ = ["lower_atomic", "lower_concurrency", "lower_cpu", "lower_simd"]

#: LLVM's names for the memory orders.
_ORDERING = {
    "relaxed": "monotonic",
    "acquire": "acquire",
    "release": "release",
    "acq_rel": "acq_rel",
    "seq_cst": "seq_cst",
}


class EmitError(Exception):
    """IR a lowering cannot express; the verifier should have caught it."""


def _align(t: IRType) -> int:
    if isinstance(t, (IntType, FloatType)):
        return max(t.width // 8, 1)
    return 8


# -- simd ------------------------------------------------------------------------


def lower_simd(emitter, op: Operation) -> None:  # type: ignore[no-untyped-def]
    ir = emitter.ir
    b = emitter.builder
    name = op.local_name
    if name == "splat":
        vector_type = op.result.type
        assert isinstance(vector_type, VectorType)
        llvm_vector = emitter.owner.llvm_type(vector_type)
        one = b.insert_element(
            ir.Constant(llvm_vector, ir.Undefined), emitter.value(op.operands[0]), _i32(ir, 0)
        )
        mask = ir.Constant(
            ir.VectorType(ir.IntType(32), vector_type.count), [0] * vector_type.count
        )
        emitter.set(op.result, b.shuffle_vector(one, ir.Constant(llvm_vector, ir.Undefined), mask))
        return
    if name == "load":
        vector_type = op.result.type
        assert isinstance(vector_type, VectorType)
        llvm_vector = emitter.owner.llvm_type(vector_type)
        pointer = b.bitcast(emitter.value(op.operands[0]), llvm_vector.as_pointer())
        emitter.set(op.result, b.load(pointer, align=_align(vector_type.element)))
        return
    if name == "store":
        vector, pointer = op.operands
        assert isinstance(vector.type, VectorType)
        llvm_vector = emitter.owner.llvm_type(vector.type)
        cast = b.bitcast(emitter.value(pointer), llvm_vector.as_pointer())
        b.store(emitter.value(vector), cast, align=_align(vector.type.element))
        return
    if name == "extract":
        vector, index = (emitter.value(v) for v in op.operands)
        emitter.set(op.result, b.extract_element(vector, index))
        return
    if name == "insert":
        vector, value, index = (emitter.value(v) for v in op.operands)
        emitter.set(op.result, b.insert_element(vector, value, index))
        return
    if name == "shuffle":
        first, second = (emitter.value(v) for v in op.operands)
        lanes = op.attributes["mask"]
        assert isinstance(lanes, tuple)
        mask = ir.Constant(ir.VectorType(ir.IntType(32), len(lanes)), [int(m) for m in lanes])
        emitter.set(op.result, b.shuffle_vector(first, second, mask))
        return
    if name.startswith("reduce_"):
        _reduce(emitter, op, name.removeprefix("reduce_"))
        return
    raise EmitError(f"{op.name} has no LLVM lowering")


def _i32(ir, value: int):  # type: ignore[no-untyped-def]
    return ir.Constant(ir.IntType(32), value)


def _reduce(emitter, op: Operation, kind: str) -> None:  # type: ignore[no-untyped-def]
    """A fold over the lanes in order, first to last.

    An integer sum is exact whatever the order, so it is the reduction
    intrinsic; a floating-point sum and every minimum and maximum keep lane
    order, which is what the reference implementation computes.
    """
    ir = emitter.ir
    b = emitter.builder
    vector = op.operands[0]
    vector_type = vector.type
    assert isinstance(vector_type, VectorType)
    element = vector_type.element
    llvm_vector = emitter.owner.llvm_type(vector_type)
    llvm_element = emitter.owner.llvm_type(element)
    value = emitter.value(vector)
    if kind == "add" and not isinstance(element, FloatType):
        suffix = f"v{vector_type.count}i{element.width if isinstance(element, IntType) else 64}"
        function = emitter.intrinsic(
            f"llvm.vector.reduce.add.{suffix}", llvm_element, [llvm_vector]
        )
        emitter.set(op.result, b.call(function, [value]))
        return
    accumulator = b.extract_element(value, _i32(ir, 0))
    floating = isinstance(element, FloatType)
    signed = element.signed if isinstance(element, IntType) else True
    for lane in range(1, vector_type.count):
        item = b.extract_element(value, _i32(ir, lane))
        if kind == "add":
            accumulator = b.fadd(accumulator, item)
            continue
        symbol = "<" if kind == "min" else ">"
        if floating:
            better = b.fcmp_ordered(symbol, item, accumulator)
        elif signed:
            better = b.icmp_signed(symbol, item, accumulator)
        else:
            better = b.icmp_unsigned(symbol, item, accumulator)
        accumulator = b.select(better, item, accumulator)
    emitter.set(op.result, accumulator)


# -- cpu -------------------------------------------------------------------------


def lower_cpu(emitter, op: Operation) -> None:  # type: ignore[no-untyped-def]
    ir = emitter.ir
    b = emitter.builder
    if op.local_name == "prefetch":
        byte_pointer = ir.IntType(8).as_pointer()
        function = emitter.intrinsic(
            "llvm.prefetch.p0i8",
            ir.VoidType(),
            [byte_pointer, ir.IntType(32), ir.IntType(32), ir.IntType(32)],
        )
        pointer = b.bitcast(emitter.value(op.operands[0]), byte_pointer)
        rw = 1 if op.attributes.get("rw", "read") == "write" else 0
        locality = int(op.attributes.get("locality", 3))  # type: ignore[call-overload]
        b.call(function, [pointer, _i32(ir, rw), _i32(ir, locality), _i32(ir, 1)])
        return
    if op.local_name == "pause":
        architecture = emitter.owner.target.architecture
        if architecture in {"x86_64", "i686", "i386"}:
            function = emitter.intrinsic("llvm.x86.sse2.pause", ir.VoidType(), [])
            b.call(function, [])
        elif architecture == "aarch64":
            asm = ir.InlineAsm(ir.FunctionType(ir.VoidType(), []), "yield", "", side_effect=True)
            b.call(asm, [])
        # Elsewhere the hint has no instruction, and a hint dropped is still correct.
        return
    raise EmitError(f"{op.name} has no LLVM lowering")


# -- atomic ------------------------------------------------------------------------


def lower_atomic(emitter, op: Operation) -> None:  # type: ignore[no-untyped-def]
    b = emitter.builder
    name = op.local_name
    order = _ORDERING[str(op.attributes.get("order", "seq_cst"))]
    if name == "load":
        pointer = op.operands[0]
        assert isinstance(pointer.type, PtrType)
        loaded = b.load_atomic(emitter.value(pointer), order, _align(pointer.type.pointee))
        emitter.set(op.result, loaded)
        return
    if name == "store":
        value, pointer = op.operands
        assert isinstance(pointer.type, PtrType)
        b.store_atomic(
            emitter.value(value), emitter.value(pointer), order, _align(pointer.type.pointee)
        )
        return
    if name == "exchange":
        pointer, value = (emitter.value(v) for v in op.operands)
        emitter.set(op.result, b.atomic_rmw("xchg", pointer, value, order))
        return
    if name.startswith("fetch_"):
        pointer, value = (emitter.value(v) for v in op.operands)
        emitter.set(op.result, b.atomic_rmw(name.removeprefix("fetch_"), pointer, value, order))
        return
    if name == "compare_exchange":
        pointer, expected, desired = (emitter.value(v) for v in op.operands)
        success = _ORDERING[str(op.attributes["success"])]
        failure = _ORDERING[str(op.attributes["failure"])]
        packed = b.cmpxchg(pointer, expected, desired, success, failure)
        emitter.set(op.results[0], b.extract_value(packed, 0))
        emitter.set(op.results[1], b.extract_value(packed, 1))
        return
    if name == "fence":
        b.fence(order)
        return
    raise EmitError(f"{op.name} has no LLVM lowering")


# -- concurrency --------------------------------------------------------------------


def lower_concurrency(emitter, op: Operation) -> None:  # type: ignore[no-untyped-def]
    if emitter.owner.target.os == "windows":
        raise EmitError(f"{op.name} has no lowering for {emitter.owner.target.triple} yet")
    name = op.local_name
    if name == "spawn":
        _spawn(emitter, op)
    elif name == "join":
        _join(emitter, op)
    elif name == "mutex_lock":
        _mutex_lock(emitter, emitter.value(op.operands[0]))
    elif name == "mutex_unlock":
        _store_release(emitter, emitter.value(op.operands[0]), 0)
    elif name == "condition_wait":
        _condition_wait(emitter, emitter.value(op.operands[0]), emitter.value(op.operands[1]))
    elif name == "condition_notify":
        emitter.builder.atomic_rmw(
            "add", emitter.value(op.operands[0]), _i64(emitter, 1), "acq_rel"
        )
    elif name == "barrier":
        _barrier(emitter, emitter.value(op.operands[0]), emitter.value(op.operands[1]))
    elif name == "thread_id":
        ir = emitter.ir
        function = emitter.extern("pthread_self", ir.IntType(64), [])
        emitter.set(op.result, emitter.builder.call(function, []))
    else:
        raise EmitError(f"{op.name} has no LLVM lowering")


def _i64(emitter, value: int):  # type: ignore[no-untyped-def]
    return emitter.ir.Constant(emitter.ir.IntType(64), value)


def _store_release(emitter, pointer, value: int) -> None:  # type: ignore[no-untyped-def]
    emitter.builder.store_atomic(_i64(emitter, value), pointer, "release", 8)


def _pause(emitter) -> None:  # type: ignore[no-untyped-def]
    ir = emitter.ir
    architecture = emitter.owner.target.architecture
    if architecture in {"x86_64", "i686", "i386"}:
        emitter.builder.call(emitter.intrinsic("llvm.x86.sse2.pause", ir.VoidType(), []), [])
    elif architecture == "aarch64":
        asm = ir.InlineAsm(ir.FunctionType(ir.VoidType(), []), "yield", "", side_effect=True)
        emitter.builder.call(asm, [])


def _mutex_lock(emitter, slot) -> None:  # type: ignore[no-untyped-def]
    """Spin until the slot goes from 0 to 1, pausing between tries."""
    b = emitter.builder
    spin = emitter.llvm.append_basic_block("mutex.spin")
    taken = emitter.llvm.append_basic_block("mutex.taken")
    b.branch(spin)
    b.position_at_end(spin)
    packed = b.cmpxchg(slot, _i64(emitter, 0), _i64(emitter, 1), "acq_rel", "monotonic")
    _pause(emitter)
    b.cbranch(b.extract_value(packed, 1), taken, spin)
    b.position_at_end(taken)


def _condition_wait(emitter, condition, mutex) -> None:  # type: ignore[no-untyped-def]
    """Release the mutex, wait for a notification after this one, take it back."""
    b = emitter.builder
    generation = b.load_atomic(condition, "acquire", 8)
    _store_release(emitter, mutex, 0)
    spin = emitter.llvm.append_basic_block("condition.spin")
    woken = emitter.llvm.append_basic_block("condition.woken")
    b.branch(spin)
    b.position_at_end(spin)
    _pause(emitter)
    current = b.load_atomic(condition, "acquire", 8)
    b.cbranch(b.icmp_signed("==", current, generation), spin, woken)
    b.position_at_end(woken)
    _mutex_lock(emitter, mutex)


def _barrier(emitter, slots, parties) -> None:  # type: ignore[no-untyped-def]
    """Arrivals in the first slot, the generation in the second: the last to
    arrive resets the count and moves the generation on, the rest wait for it."""
    ir = emitter.ir
    b = emitter.builder
    generation_slot = b.gep(slots, [_i64(emitter, 1)])
    generation = b.load_atomic(generation_slot, "acquire", 8)
    before = b.atomic_rmw("add", slots, _i64(emitter, 1), "acq_rel")
    arrived = b.add(before, _i64(emitter, 1))
    last = emitter.llvm.append_basic_block("barrier.last")
    spin = emitter.llvm.append_basic_block("barrier.spin")
    done = emitter.llvm.append_basic_block("barrier.done")
    b.cbranch(b.icmp_signed("==", arrived, parties), last, spin)
    b.position_at_end(last)
    _store_release(emitter, slots, 0)
    b.atomic_rmw("add", generation_slot, _i64(emitter, 1), "acq_rel")
    b.branch(done)
    b.position_at_end(spin)
    _pause(emitter)
    current = b.load_atomic(generation_slot, "acquire", 8)
    b.cbranch(b.icmp_signed("==", current, generation), spin, done)
    b.position_at_end(done)
    del ir


def _spawn(emitter, op: Operation) -> None:  # type: ignore[no-untyped-def]
    """`pthread_create` on a trampoline that unpacks the arguments from a
    heap context, calls the function, frees the context, and returns the
    status as the thread's result."""
    ir = emitter.ir
    b = emitter.builder
    callee_name = op.attributes["callee"].name  # type: ignore[union-attr]
    callee = emitter.owner.functions.get(callee_name)
    target = emitter.owner.module.functions.get(callee_name)
    if callee is None or target is None:
        raise EmitError(f"spawn of @{callee_name}, which was not emitted")
    atoms: list = []
    for operand, (_name, t) in zip(op.operands, target.params, strict=True):
        atoms.extend(emitter._to_atoms(t, emitter.value(operand)))
    context_type = ir.LiteralStructType([a.type for a in atoms])
    byte_pointer = ir.IntType(8).as_pointer()
    malloc = emitter.extern("malloc", byte_pointer, [ir.IntType(64)])
    size = ir.Constant(ir.IntType(64), context_type.get_abi_size(emitter.owner.data_layout))
    raw = b.call(malloc, [size])
    context = b.bitcast(raw, context_type.as_pointer())
    for index, atom in enumerate(atoms):
        b.store(atom, b.gep(context, [_i32(ir, 0), _i32(ir, index)]))
    trampoline = _trampoline(
        emitter.owner, callee_name, callee, context_type, [a.type for a in atoms]
    )
    create = emitter.extern(
        "pthread_create",
        ir.IntType(32),
        [ir.IntType(64).as_pointer(), byte_pointer, trampoline.type, byte_pointer],
    )
    handle_slot = emitter.entry_alloca(ir.IntType(64), "thread")
    status = b.call(create, [handle_slot, ir.Constant(byte_pointer, None), trampoline, raw])
    ok = b.icmp_signed("==", status, _i32(ir, 0))
    emitter.continue_if(ok, "spawn.ok")
    emitter.set(op.result, b.load(handle_slot))


def _trampoline(owner, callee_name: str, callee, context_type, atom_types):  # type: ignore[no-untyped-def]
    """`i8* f(i8* context)`, made once per spawned function."""
    ir = owner.ir
    name = f"ppy_thread_{callee_name}"
    existing = owner.llvm.globals.get(name)
    if existing is not None:
        return existing
    byte_pointer = ir.IntType(8).as_pointer()
    function = ir.Function(owner.llvm, ir.FunctionType(byte_pointer, [byte_pointer]), name=name)
    function.linkage = "private"
    block = function.append_basic_block("entry")
    b = ir.IRBuilder(block)
    context = b.bitcast(function.args[0], context_type.as_pointer())
    arguments = [
        b.load(b.gep(context, [_i32(ir, 0), _i32(ir, index)])) for index in range(len(atom_types))
    ]
    out = b.alloca(ir.IntType(64))
    status = b.call(callee, [*arguments, out])
    free = owner.llvm.globals.get("free") or ir.Function(
        owner.llvm, ir.FunctionType(ir.VoidType(), [byte_pointer]), name="free"
    )
    b.call(free, [function.args[0]])
    b.ret(b.inttoptr(b.sext(status, ir.IntType(64)), byte_pointer))
    return function


def _join(emitter, op: Operation) -> None:  # type: ignore[no-untyped-def]
    ir = emitter.ir
    b = emitter.builder
    byte_pointer = ir.IntType(8).as_pointer()
    join = emitter.extern(
        "pthread_join", ir.IntType(32), [ir.IntType(64), byte_pointer.as_pointer()]
    )
    result_slot = emitter.entry_alloca(byte_pointer, "joined")
    status = b.call(join, [emitter.value(op.operands[0]), result_slot])
    ok = b.icmp_signed("==", status, _i32(ir, 0))
    emitter.continue_if(ok, "join.ok")
    emitter.set(op.result, b.ptrtoint(b.load(result_slot), ir.IntType(64)))


def _bool_lanes(t: IRType) -> bool:
    return isinstance(t, VectorType) and isinstance(t.element, BoolType)
