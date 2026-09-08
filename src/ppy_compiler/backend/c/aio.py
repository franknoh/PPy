"""The async dialect as C: the runtime's calls, for a unit compiled with `ppy_aio.c` (spec 77).

A future is an `int64_t` handle; a frame an `int64_t *`. The unit declares
the runtime's functions it uses, and its header comment says which source
to compile beside it. Bits of a `double` cross a future through `memcpy`.
"""

from __future__ import annotations

from ...ir import BoolType, FloatType, Operation
from .dialects import EmitError

__all__ = ["RESUME_TYPEDEF", "declare", "emit_async"]

RESUME_TYPEDEF = "typedef void (*ppy_aio_resume_fn)(int64_t *);\n"
_PROTOTYPES = {
    "ppy_aio_frame_new": "int64_t *ppy_aio_frame_new(int64_t);",
    "ppy_aio_spawn": "int64_t ppy_aio_spawn(int64_t *, ppy_aio_resume_fn);",
    "ppy_aio_await": "void ppy_aio_await(int64_t *, int64_t);",
    "ppy_aio_start": "void ppy_aio_start(int64_t);",
    "ppy_aio_result": "int64_t ppy_aio_result(int64_t *);",
    "ppy_aio_complete": "void ppy_aio_complete(int64_t *, int64_t);",
    "ppy_aio_fail": "void ppy_aio_fail(int64_t *, int64_t);",
    "ppy_aio_sleep": "int64_t ppy_aio_sleep(double);",
    "ppy_aio_accept": "int64_t ppy_aio_accept(int64_t);",
    "ppy_aio_connect": "int64_t ppy_aio_connect(const uint8_t *, int64_t, int64_t);",
    "ppy_aio_read": "int64_t ppy_aio_read(int64_t, uint8_t *, int64_t);",
    "ppy_aio_write": "int64_t ppy_aio_write(int64_t, const uint8_t *, int64_t);",
    "ppy_aio_listen": "int64_t ppy_aio_listen(const uint8_t *, int64_t, int64_t, int64_t);",
    "ppy_aio_port": "int64_t ppy_aio_port(int64_t);",
    "ppy_aio_close": "void ppy_aio_close(int64_t);",
}
_BITS = """static inline int64_t ppy_aio_double_bits(double value) {
    int64_t bits;
    memcpy(&bits, &value, sizeof bits);
    return bits;
}
static inline double ppy_aio_bits_double(int64_t bits) {
    double value;
    memcpy(&value, &bits, sizeof value);
    return value;
}
"""


def declare(owner, name: str) -> str:  # type: ignore[no-untyped-def]
    owner.unit.aio = True
    owner.unit.prelude.setdefault("aio_resume", RESUME_TYPEDEF)
    owner.unit.externs.setdefault(name, _PROTOTYPES[name] + "\n")
    return name


def _bits_helpers(owner) -> None:  # type: ignore[no-untyped-def]
    owner.unit.headers.add("string.h")
    owner.unit.helpers.setdefault("aio_bits", _BITS)


def emit_async(fe, op: Operation) -> None:  # type: ignore[no-untyped-def]
    owner = fe.owner
    name = op.local_name
    if name == "frame_new":
        slots = int(op.attributes["slots"])  # type: ignore[call-overload]
        fe.define(op.result, f"{declare(owner, 'ppy_aio_frame_new')}({slots})")
    elif name == "spawn":
        callee = op.attributes["callee"].name  # type: ignore[union-attr]
        target = owner.module.functions.get(callee)
        if target is None:
            raise EmitError(f"spawn of @{callee}, which was not emitted")
        frame = fe.value(op.operands[0])
        symbol = owner.symbol_of(target)
        call = f"{declare(owner, 'ppy_aio_spawn')}({frame}, (ppy_aio_resume_fn){symbol})"
        fe.define(op.result, call)
    elif name == "suspend":
        frame, future = (fe.value(v) for v in op.operands)
        fe.body.append(f"    {declare(owner, 'ppy_aio_await')}({frame}, {future});")
    elif name == "result":
        bits = f"{declare(owner, 'ppy_aio_result')}({fe.value(op.operands[0])})"
        t = op.result.type
        if isinstance(t, FloatType):
            _bits_helpers(owner)
            fe.define(op.result, owner.cast(f"ppy_aio_bits_double({bits})", t))
        elif isinstance(t, BoolType):
            fe.define(op.result, f"({bits} & 1) != 0")
        else:
            fe.define(op.result, owner.cast(bits, t) if owner.c_type(t) != "int64_t" else bits)
    elif name == "complete":
        frame = fe.value(op.operands[0])
        if len(op.operands) > 1:
            value = fe.value(op.operands[1])
            t = op.operands[1].type
            if isinstance(t, FloatType):
                _bits_helpers(owner)
                bits = f"ppy_aio_double_bits({owner.cast(value, 'double')})"
            else:
                bits = owner.cast(value, "int64_t")
        else:
            bits = "0"
        fe.body.append(f"    {declare(owner, 'ppy_aio_complete')}({frame}, {bits});")
    elif name == "fail":
        code = int(op.attributes["code"])  # type: ignore[call-overload]
        fe.body.append(f"    {declare(owner, 'ppy_aio_fail')}({fe.value(op.operands[0])}, {code});")
    elif name in {"sleep", "accept", "connect", "read", "write", "listen", "port"}:
        function = declare(owner, f"ppy_aio_{name}")
        arguments = []
        for operand in op.operands:
            spelled = fe.value(operand)
            if (
                name in {"connect", "listen", "write"}
                and operand is op.operands[0 if name != "write" else 1]
            ):
                spelled = f"(const uint8_t *)({spelled})"
            elif name == "read" and operand is op.operands[1]:
                spelled = f"(uint8_t *)({spelled})"
            arguments.append(spelled)
        fe.define(op.result, f"{function}({', '.join(arguments)})")
    elif name == "start":
        fe.body.append(f"    {declare(owner, 'ppy_aio_start')}({fe.value(op.operands[0])});")
    elif name == "close":
        fe.body.append(f"    {declare(owner, 'ppy_aio_close')}({fe.value(op.operands[0])});")
    else:
        raise EmitError(f"{op.name} has no C lowering")
