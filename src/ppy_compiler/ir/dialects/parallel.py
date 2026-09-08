"""The parallel dialect: a loop whose iterations may run at once.

Each operation names a body function and the range it covers, and says
nothing about how the range is split: that is the lowering's choice --
serial, chunks on threads, a vectorizer, OpenMP in the C backend -- made
once per build from the configuration, and every choice gives the same
answer. `parallel.for @body(%captures...) %begin, %end` runs
`@body(captures..., begin, end)` over the range, a body that loops over
its own chunk; `parallel.reduce @body(%captures...) %begin, %end, %init
{op, reassociate} : T` folds the chunks' results with `op` from `init`,
where `@body(captures..., begin, end, acc) -> T` accumulates its chunk;
`parallel.map @body(%captures...) %begin, %end, %out` stores `@body(
captures..., i)` at `out[i]` for every index. A floating-point reduction
whose `reassociate` is false is never split, because the answer would
change; an integer one always may.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Builder, IRFunction, Operation, SymbolRef, Value
from ..types import I64, FloatType, IndexType, IntType, IRType, PtrType

if TYPE_CHECKING:
    from ..verify import Checker

__all__ = ["OPS", "ParallelDialect", "identity", "loop", "map_", "reduce"]

#: The reductions, and whether each is safe to reassociate on floats.
OPS = ("add", "mul", "min", "max")


def _callee(op: Operation, checker: Checker) -> IRFunction | None:
    callee = op.attributes.get("callee")
    if not isinstance(callee, SymbolRef):
        checker.error(op, f"{op.name} needs a `callee` symbol")
        return None
    if checker.module is None:
        return None
    target = checker.module.functions.get(callee.name)
    if target is None:
        checker.error(op, f"{op.name} of @{callee.name}, which is not a function of this module")
    return target


def _index(t: IRType) -> bool:
    return t == I64 or isinstance(t, IndexType)


def _captures(op: Operation, checker: Checker, target: IRFunction, trailing: int, own: int) -> bool:
    """The operands before the operation's `trailing` own ones match the
    body's leading parameters, and the body has `own` parameters after them."""
    given = tuple(v.type for v in op.operands[: len(op.operands) - trailing])
    expected = tuple(t for _name, t in target.params)
    if expected[: len(given)] != given or len(expected) != len(given) + own:
        checker.error(
            op,
            f"@{target.name} takes {expected}; {op.name} passes {given} and the body "
            f"takes {own} more",
        )
        return False
    return True


def _verify_for(op: Operation, checker: Checker) -> None:
    target = _callee(op, checker)
    if target is None or len(op.operands) < 2:
        if target is not None:
            checker.error(op, "parallel.for takes the captures, then begin and end")
        return
    begin, end = op.operands[-2:]
    if not (_index(begin.type) and _index(end.type)):
        checker.error(op, "a range is two i64 bounds")
    if _captures(op, checker, target, 2, 2):
        tail = tuple(t for _name, t in target.params[-2:])
        if tail != (I64, I64):
            checker.error(op, f"@{target.name} ends in (begin: i64, end: i64), not {tail}")
    if target.results:
        checker.error(
            op, f"a parallel.for body returns nothing; @{target.name} returns {target.results}"
        )


def _verify_reduce(op: Operation, checker: Checker) -> None:
    target = _callee(op, checker)
    kind = op.attributes.get("op")
    if kind not in OPS:
        checker.error(op, f"`op` is one of {', '.join(OPS)}, not {kind!r}")
    if target is None or len(op.operands) < 3:
        if target is not None:
            checker.error(op, "parallel.reduce takes the captures, then begin, end, and init")
        return
    begin, end, init = op.operands[-3:]
    if not (_index(begin.type) and _index(end.type)):
        checker.error(op, "a range is two i64 bounds")
    result = op.results[0].type
    if not isinstance(result, (IntType, IndexType, FloatType)):
        checker.error(op, f"a reduction gives a number, not {result}")
    if init.type != result:
        checker.error(op, f"init is {init.type}, the reduction {result}")
    if _captures(op, checker, target, 3, 3):
        tail = tuple(t for _name, t in target.params[-3:])
        if tail != (I64, I64, result):
            checker.error(
                op, f"@{target.name} ends in (begin: i64, end: i64, acc: {result}), not {tail}"
            )
        if tuple(target.results) != (result,):
            checker.error(op, f"@{target.name} gives {target.results}, the reduction {result}")
    if not isinstance(op.attributes.get("reassociate", True), bool):
        checker.error(op, "`reassociate` is a bool")


def _verify_map(op: Operation, checker: Checker) -> None:
    target = _callee(op, checker)
    if target is None or len(op.operands) < 3:
        if target is not None:
            checker.error(op, "parallel.map takes the captures, then begin, end, and out")
        return
    begin, end, out = op.operands[-3:]
    if not (_index(begin.type) and _index(end.type)):
        checker.error(op, "a range is two i64 bounds")
    if not isinstance(out.type, PtrType) or not out.type.mutable:
        checker.error(op, f"out is a mutable pointer, not {out.type}")
        return
    if _captures(op, checker, target, 3, 1):
        tail = tuple(t for _name, t in target.params[-1:])
        if tail != (I64,):
            checker.error(op, f"@{target.name} ends in (i: i64), not {tail}")
        if tuple(target.results) != (out.type.pointee,):
            checker.error(
                op, f"@{target.name} gives {target.results}; out holds {out.type.pointee}"
            )


class ParallelDialect(Dialect):
    name = "parallel"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        add = registry.add_op
        add(OpSpec("parallel.for", verify=_verify_for, results=0, required_attributes=("callee",)))
        add(
            OpSpec(
                "parallel.reduce",
                verify=_verify_reduce,
                results=1,
                required_attributes=("callee", "op"),
            )
        )
        add(OpSpec("parallel.map", verify=_verify_map, results=0, required_attributes=("callee",)))


def loop(
    b: Builder, callee: str, captures: tuple[Value, ...], begin: Value, end: Value
) -> Operation:
    return b.create("parallel.for", (*captures, begin, end), (), {"callee": SymbolRef(callee)})


def reduce(
    b: Builder,
    callee: str,
    captures: tuple[Value, ...],
    begin: Value,
    end: Value,
    init: Value,
    op: str,
    *,
    reassociate: bool = True,
    name: str | None = None,
) -> Value:
    if op not in OPS:
        raise ValueError(f"no reduction named {op!r}")
    return b.create(
        "parallel.reduce",
        (*captures, begin, end, init),
        (init.type,),
        {"callee": SymbolRef(callee), "op": op, "reassociate": reassociate},
        result_names=(name,),
    ).result


def map_(
    b: Builder, callee: str, captures: tuple[Value, ...], begin: Value, end: Value, out: Value
) -> Operation:
    return b.create("parallel.map", (*captures, begin, end, out), (), {"callee": SymbolRef(callee)})


def identity(op: str, t: IRType, init: Value | None = None) -> object:
    """What a chunk of the reduction starts from: the op's identity, or
    `init` itself for the idempotent ones."""
    if op == "add":
        return 0.0 if isinstance(t, FloatType) else 0
    if op == "mul":
        return 1.0 if isinstance(t, FloatType) else 1
    return init
