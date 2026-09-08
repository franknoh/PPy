"""`sanitize`: checks the program did not ask for, and cannot fall back from (spec 79).

A build with `--sanitize` instruments the IR: `bounds` checks every buffer
index, whether or not a proof or a hoist removed the guard the frontend
wrote; `overflow` checks every wrapping or proven `int` operation;
`pointer` checks that a pointer read or written through is not null;
`alignment` checks that it is aligned for what it points at. A check that
fails is a sanitizer failure -- the function returns a status the boundary
turns into `SanitizerFailure`, never a quiet fall back to Python -- so a
guard the optimizer or the prover was wrong to remove is found, not
papered over. `lifetime` and `alias` are refused: stack lifetime is held
statically by the verifier, and aliasing has no runtime check yet.
"""

from __future__ import annotations

from ..dialects import core
from ..model import Builder, IRFunction, Operation
from ..passes import FunctionPass, PassContext
from ..types import BOOL, I64, FloatType, IntType, IRType, PtrType

__all__ = ["KINDS", "REFUSED", "Sanitize", "sanitize", "sanitizer_kinds"]

#: The kinds a build may ask for, in the order their statuses count from `STATUS_SANITIZER_BASE`.
KINDS = ("bounds", "overflow", "pointer", "alignment")
#: Candidates the directive names that have no runtime check yet, and why.
REFUSED = {
    "lifetime": "stack lifetime is held by the verifier; heap lifetime has no sanitizer yet",
    "alias": "aliasing has no runtime check yet",
}
_ARITHMETIC = {"core.add": "add", "core.sub": "sub", "core.mul": "mul"}


def sanitizer_kinds(spelled: str | list[str] | tuple[str, ...]) -> frozenset[str]:
    """The kinds `--sanitize a,b` or a configured list names; a refusal names why."""
    items = spelled.split(",") if isinstance(spelled, str) else list(spelled)
    kinds: set[str] = set()
    for item in items:
        name = item.strip()
        if not name:
            continue
        if name in REFUSED:
            raise ValueError(f"`{name}` is not a sanitizer yet: {REFUSED[name]}")
        if name not in KINDS:
            raise ValueError(f"`{name}` is not a sanitizer; the sanitizers are {', '.join(KINDS)}")
        kinds.add(name)
    return frozenset(kinds)


class Sanitize(FunctionPass):
    """Insert the checks `kinds` ask for."""

    name = "sanitize"

    def __init__(self, kinds: frozenset[str] | set[str] | tuple[str, ...]) -> None:
        self.kinds = frozenset(kinds)

    def run_on_function(self, function: IRFunction, ctx: PassContext) -> bool:
        if function.attributes.get("ppy.abi") == "resume" or function.attributes.get("gpu.kind"):
            # A coroutine's frame has no status to fail with; device code has no host to tell.
            return False
        inserted = sanitize(function, self.kinds)
        if inserted:
            ctx.remark(f"sanitizer: {inserted} check(s) inserted in @{function.name}")
        return inserted > 0


def sanitize(function: IRFunction, kinds: frozenset[str]) -> int:
    """Instrument `function`; how many checks went in."""
    inserted = 0
    for op in list(function.operations()):
        if op.parent is None:
            continue
        if "bounds" in kinds and op.name in {"core.buffer_load", "core.buffer_store"}:
            inserted += _bounds(op)
        if "overflow" in kinds and op.name in _ARITHMETIC:
            inserted += _overflow(op)
        if op.name in {"core.load", "core.store"}:
            pointer = op.operands[0] if op.name == "core.load" else op.operands[1]
            assert isinstance(pointer.type, PtrType)
            if pointer.type.address_space == "stack":
                continue
            if "pointer" in kinds:
                inserted += _pointer(op, pointer)
            if "alignment" in kinds:
                inserted += _alignment(op, pointer)
    return inserted


def _label(kind: str) -> str:
    return f"sanitize:{kind}"


def _bounds(op: Operation) -> int:
    buffer = op.operands[0] if op.name == "core.buffer_load" else op.operands[1]
    index = op.operands[1] if op.name == "core.buffer_load" else op.operands[2]
    b = Builder().before(op)
    length = core.buffer_len(b, buffer)
    position = index if index.type == I64 else core.cast(b, index, I64)
    inside = core.bitwise(
        b,
        "and",
        core.cmp(b, "ge", position, core.const(b, 0, I64)),
        core.cmp(b, "lt", position, core.cast(b, length, I64) if length.type != I64 else length),
    )
    core.guard(b, inside, "bounds", "sanitizer: index out of range", label=_label("bounds"))
    return 1


def _overflow(op: Operation) -> int:
    t = op.results[0].type
    if not isinstance(t, IntType) or op.attributes.get("overflow") not in {"wrap", "proven"}:
        return 0
    b = Builder().before(op)
    wrapped, overflowed = core.checked(b, _ARITHMETIC[op.name], op.operands[0], op.operands[1])
    ok = core.cmp(b, "eq", overflowed, core.const(b, False, BOOL))
    core.guard(b, ok, "overflow", "sanitizer: integer overflow", label=_label("overflow"))
    op.results[0].replace_all_uses_with(wrapped)
    op.erase()
    return 1


def _pointer(op: Operation, pointer) -> int:  # type: ignore[no-untyped-def]
    b = Builder().before(op)
    null = core.const(b, 0, pointer.type)
    present = core.cmp(b, "ne", pointer, null)
    core.guard(b, present, "contract", "sanitizer: null pointer", label=_label("pointer"))
    return 1


def _alignment(op: Operation, pointer) -> int:  # type: ignore[no-untyped-def]
    assert isinstance(pointer.type, PtrType)
    size = _size_of(pointer.type.pointee)
    if size <= 1:
        return 0
    b = Builder().before(op)
    address = core.cast(b, pointer, I64)
    remainder = core.mod(b, address, core.const(b, size, I64), overflow="wrap", rounding="floor")
    aligned = core.cmp(b, "eq", remainder, core.const(b, 0, I64))
    core.guard(b, aligned, "contract", "sanitizer: misaligned pointer", label=_label("alignment"))
    return 1


def _size_of(t: IRType) -> int:
    if isinstance(t, (IntType, FloatType)):
        return max(t.width // 8, 1)
    if isinstance(t, PtrType):
        return 8
    return 1
