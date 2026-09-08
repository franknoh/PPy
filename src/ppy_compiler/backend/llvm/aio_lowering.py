"""LLVM lowering of the async dialect: the runtime's calls, and a resume function's shape (spec 77).

A future is an `i64` handle the runtime hands out. `frame_new`, `spawn`,
`suspend`, `result`, `complete`, and `fail` are calls into `ppy_aio_*`, as
are the operations that complete later and the immediate ones; `spawn`
passes the resume function's address. A resume function takes the frame
and returns nothing: its fallback block -- reached when a callee's guard
fails -- fails the frame's future instead of returning a status.
"""

from __future__ import annotations

from ...ir import BoolType, FloatType, FutureType, Operation
from .dialect_lowerings import EmitError

__all__ = ["RUNTIME", "lower_async"]

#: The runtime's functions: (result, parameters) in LLVM spellings, i64 for a future.
RUNTIME = {
    "ppy_aio_frame_new": ("ptr", ("i64",)),
    "ppy_aio_spawn": ("i64", ("ptr", "ptr")),
    "ppy_aio_await": ("void", ("ptr", "i64")),
    "ppy_aio_start": ("void", ("i64",)),
    "ppy_aio_result": ("i64", ("ptr",)),
    "ppy_aio_complete": ("void", ("ptr", "i64")),
    "ppy_aio_fail": ("void", ("ptr", "i64")),
    "ppy_aio_sleep": ("i64", ("double",)),
    "ppy_aio_accept": ("i64", ("i64",)),
    "ppy_aio_connect": ("i64", ("ptr", "i64", "i64")),
    "ppy_aio_read": ("i64", ("i64", "ptr", "i64")),
    "ppy_aio_write": ("i64", ("i64", "ptr", "i64")),
    "ppy_aio_listen": ("i64", ("ptr", "i64", "i64", "i64")),
    "ppy_aio_port": ("i64", ("i64",)),
    "ppy_aio_close": ("void", ("i64",)),
}


def _llvm(ir, spelled: str):  # type: ignore[no-untyped-def]
    if spelled == "ptr":
        return ir.IntType(8).as_pointer()
    if spelled == "double":
        return ir.DoubleType()
    if spelled == "void":
        return ir.VoidType()
    return ir.IntType(64)


def runtime_function(emitter, name: str):  # type: ignore[no-untyped-def]
    ir = emitter.ir
    result, parameters = RUNTIME[name]
    return emitter.extern(name, _llvm(ir, result), [_llvm(ir, p) for p in parameters])


def _as_bits(emitter, value, t):  # type: ignore[no-untyped-def]
    """A scalar as the sixty-four bits a future carries."""
    ir = emitter.ir
    b = emitter.builder
    if isinstance(t, FloatType):
        wide = value if t.width == 64 else b.fpext(value, ir.DoubleType())
        return b.bitcast(wide, ir.IntType(64))
    if isinstance(t, BoolType):
        return b.zext(value, ir.IntType(64))
    width = getattr(t, "width", 64)
    if width < 64:
        return (
            b.sext(value, ir.IntType(64))
            if getattr(t, "signed", True)
            else b.zext(value, ir.IntType(64))
        )
    return value


def _from_bits(emitter, bits, t):  # type: ignore[no-untyped-def]
    ir = emitter.ir
    b = emitter.builder
    if isinstance(t, FloatType):
        wide = b.bitcast(bits, ir.DoubleType())
        return wide if t.width == 64 else b.fptrunc(wide, emitter.owner.llvm_type(t))
    if isinstance(t, BoolType):
        return b.trunc(bits, ir.IntType(1))
    if isinstance(t, FutureType):
        return bits
    width = getattr(t, "width", 64)
    return b.trunc(bits, ir.IntType(width)) if width < 64 else bits


def _pointer(emitter, value):  # type: ignore[no-untyped-def]
    ir = emitter.ir
    emitted = emitter.value(value)
    byte_pointer = ir.IntType(8).as_pointer()
    return (
        emitted if emitted.type == byte_pointer else emitter.builder.bitcast(emitted, byte_pointer)
    )


def lower_async(emitter, op: Operation) -> None:  # type: ignore[no-untyped-def]
    ir = emitter.ir
    b = emitter.builder
    name = op.local_name
    if name == "frame_new":
        slots = int(op.attributes["slots"])  # type: ignore[call-overload]
        raw = b.call(
            runtime_function(emitter, "ppy_aio_frame_new"), [ir.Constant(ir.IntType(64), slots)]
        )
        emitter.set(op.result, b.bitcast(raw, emitter.owner.llvm_type(op.result.type)))
    elif name == "spawn":
        callee = op.attributes["callee"].name  # type: ignore[union-attr]
        resume = emitter.owner.functions.get(callee)
        if resume is None:
            raise EmitError(f"spawn of @{callee}, which was not emitted")
        pointer = b.bitcast(resume, ir.IntType(8).as_pointer())
        handle = b.call(
            runtime_function(emitter, "ppy_aio_spawn"), [_pointer(emitter, op.operands[0]), pointer]
        )
        emitter.set(op.result, handle)
    elif name == "suspend":
        b.call(
            runtime_function(emitter, "ppy_aio_await"),
            [_pointer(emitter, op.operands[0]), emitter.value(op.operands[1])],
        )
    elif name == "result":
        bits = b.call(
            runtime_function(emitter, "ppy_aio_result"), [_pointer(emitter, op.operands[0])]
        )
        emitter.set(op.result, _from_bits(emitter, bits, op.result.type))
    elif name == "complete":
        if len(op.operands) > 1:
            bits = _as_bits(emitter, emitter.value(op.operands[1]), op.operands[1].type)
        else:
            bits = ir.Constant(ir.IntType(64), 0)
        b.call(
            runtime_function(emitter, "ppy_aio_complete"), [_pointer(emitter, op.operands[0]), bits]
        )
    elif name == "fail":
        code = int(op.attributes["code"])  # type: ignore[call-overload]
        b.call(
            runtime_function(emitter, "ppy_aio_fail"),
            [_pointer(emitter, op.operands[0]), ir.Constant(ir.IntType(64), code)],
        )
    elif name in {"sleep", "accept", "connect", "read", "write", "listen", "port"}:
        function = runtime_function(emitter, f"ppy_aio_{name}")
        arguments = []
        for operand, spelled in zip(op.operands, RUNTIME[f"ppy_aio_{name}"][1], strict=True):
            arguments.append(
                _pointer(emitter, operand) if spelled == "ptr" else emitter.value(operand)
            )
        emitter.set(op.result, b.call(function, arguments))
    elif name == "start":
        b.call(runtime_function(emitter, "ppy_aio_start"), [emitter.value(op.operands[0])])
    elif name == "close":
        b.call(runtime_function(emitter, "ppy_aio_close"), [emitter.value(op.operands[0])])
    else:
        raise EmitError(f"{op.name} has no LLVM lowering")
