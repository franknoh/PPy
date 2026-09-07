"""The atomic dialect: memory shared between threads, one operation at a time.

Every operation names its memory order -- `relaxed`, `acquire`, `release`,
`acq_rel`, `seq_cst` -- the way C11 and LLVM do, so a backend lowers it to
the instruction of that order and never guesses. A load is not `release`,
a store is not `acquire`, and `compare_exchange` carries an order for each
outcome; the verifier holds those. The pointer is to the scalar the
operation reads and writes; `fetch_*` need an integer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Builder, Operation, Value
from ..types import BOOL, FloatType, IndexType, IntType, PtrType

if TYPE_CHECKING:
    from ..verify import Checker

__all__ = [
    "ORDERS",
    "RMW",
    "AtomicDialect",
    "compare_exchange",
    "exchange",
    "fence",
    "fetch",
    "load",
    "store",
]

ORDERS = ("relaxed", "acquire", "release", "acq_rel", "seq_cst")
#: The read-modify-write operations, each `%p, %x {order} : T`.
RMW = ("add", "sub", "and", "or", "xor")


def _order(op: Operation, checker: Checker, attribute: str = "order") -> str | None:
    order = op.attributes.get(attribute)
    if order not in ORDERS:
        checker.error(op, f"`{attribute}` is one of {', '.join(ORDERS)}, not {order!r}")
        return None
    return str(order)


def _pointer(op: Operation, checker: Checker, *, integer: bool = False) -> PtrType | None:
    pointer = op.operands[0].type
    if not isinstance(pointer, PtrType):
        checker.error(op, f"{op.name} takes a pointer, not {pointer}")
        return None
    element = pointer.pointee
    if integer and not isinstance(element, (IntType, IndexType)):
        checker.error(op, f"{op.name} takes a pointer to an integer, not {pointer}")
        return None
    if not isinstance(element, (IntType, IndexType, FloatType)):
        checker.error(op, f"{op.name} takes a pointer to a number, not {pointer}")
        return None
    return pointer


def _verify_load(op: Operation, checker: Checker) -> None:
    pointer = _pointer(op, checker)
    order = _order(op, checker)
    if pointer is None:
        return
    if order in {"release", "acq_rel"}:
        checker.error(op, f"a load is not `{order}`")
    if op.results[0].type != pointer.pointee:
        checker.error(op, f"loads {pointer.pointee}, not {op.results[0].type}")


def _verify_store(op: Operation, checker: Checker) -> None:
    value = op.operands[0].type
    pointer = op.operands[1].type
    order = _order(op, checker)
    if not isinstance(pointer, PtrType):
        checker.error(op, f"atomic.store writes through a pointer, not {pointer}")
        return
    if pointer.pointee != value:
        checker.error(op, f"stores {value} through {pointer}")
    if not pointer.mutable:
        checker.error(op, "writes through a const pointer")
    if order in {"acquire", "acq_rel"}:
        checker.error(op, f"a store is not `{order}`")


def _verify_rmw(op: Operation, checker: Checker) -> None:
    pointer = _pointer(op, checker, integer=op.local_name.startswith("fetch_"))
    _order(op, checker)
    if pointer is None:
        return
    if not pointer.mutable:
        checker.error(op, "writes through a const pointer")
    if op.operands[1].type != pointer.pointee:
        checker.error(op, f"{op.name} of {op.operands[1].type} through {pointer}")
    if op.results[0].type != pointer.pointee:
        checker.error(op, f"{op.name} gives the old {pointer.pointee}, not {op.results[0].type}")


def _verify_compare_exchange(op: Operation, checker: Checker) -> None:
    pointer = _pointer(op, checker)
    success = _order(op, checker, "success")
    failure = _order(op, checker, "failure")
    if pointer is None:
        return
    if not pointer.mutable:
        checker.error(op, "writes through a const pointer")
    expected, desired = (v.type for v in op.operands[1:])
    if expected != pointer.pointee or desired != pointer.pointee:
        checker.error(op, f"compare_exchange of {expected}/{desired} through {pointer}")
    if failure in {"release", "acq_rel"}:
        checker.error(op, f"the failure order is a load's, not `{failure}`")
    if (
        success is not None
        and failure is not None
        and ORDERS.index(failure) > ORDERS.index(success)
    ):
        checker.error(op, "the failure order is no stronger than the success order")
    if tuple(r.type for r in op.results) != (pointer.pointee, BOOL):
        checker.error(op, f"compare_exchange gives ({pointer.pointee}, bool)")


def _verify_fence(op: Operation, checker: Checker) -> None:
    order = _order(op, checker)
    if order == "relaxed":
        checker.error(op, "a relaxed fence orders nothing")


class AtomicDialect(Dialect):
    name = "atomic"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        add = registry.add_op
        add(
            OpSpec(
                "atomic.load",
                verify=_verify_load,
                operands=1,
                results=1,
                required_attributes=("order",),
            )
        )
        add(
            OpSpec(
                "atomic.store",
                verify=_verify_store,
                operands=2,
                results=0,
                required_attributes=("order",),
            )
        )
        add(
            OpSpec(
                "atomic.exchange",
                verify=_verify_rmw,
                operands=2,
                results=1,
                required_attributes=("order",),
            )
        )
        add(
            OpSpec(
                "atomic.compare_exchange",
                verify=_verify_compare_exchange,
                operands=3,
                results=2,
                required_attributes=("success", "failure"),
            )
        )
        for name in RMW:
            add(
                OpSpec(
                    f"atomic.fetch_{name}",
                    verify=_verify_rmw,
                    operands=2,
                    results=1,
                    required_attributes=("order",),
                )
            )
        add(
            OpSpec(
                "atomic.fence",
                verify=_verify_fence,
                operands=0,
                results=0,
                required_attributes=("order",),
            )
        )


def load(b: Builder, pointer: Value, order: str = "seq_cst", *, name: str | None = None) -> Value:
    assert isinstance(pointer.type, PtrType)
    return b.create(
        "atomic.load", (pointer,), (pointer.type.pointee,), {"order": order}, result_names=(name,)
    ).result


def store(b: Builder, value: Value, pointer: Value, order: str = "seq_cst") -> Operation:
    return b.create("atomic.store", (value, pointer), (), {"order": order})


def exchange(
    b: Builder, pointer: Value, value: Value, order: str = "seq_cst", *, name: str | None = None
) -> Value:
    assert isinstance(pointer.type, PtrType)
    return b.create(
        "atomic.exchange",
        (pointer, value),
        (pointer.type.pointee,),
        {"order": order},
        result_names=(name,),
    ).result


def fetch(
    b: Builder,
    kind: str,
    pointer: Value,
    value: Value,
    order: str = "seq_cst",
    *,
    name: str | None = None,
) -> Value:
    if kind not in RMW:
        raise ValueError(f"no atomic fetch_{kind}")
    assert isinstance(pointer.type, PtrType)
    return b.create(
        f"atomic.fetch_{kind}",
        (pointer, value),
        (pointer.type.pointee,),
        {"order": order},
        result_names=(name,),
    ).result


def compare_exchange(
    b: Builder,
    pointer: Value,
    expected: Value,
    desired: Value,
    success: str = "seq_cst",
    failure: str = "seq_cst",
    *,
    names: tuple[str | None, str | None] = (None, None),
) -> tuple[Value, Value]:
    assert isinstance(pointer.type, PtrType)
    op = b.create(
        "atomic.compare_exchange",
        (pointer, expected, desired),
        (pointer.type.pointee, BOOL),
        {"success": success, "failure": failure},
        result_names=names,
    )
    return op.results[0], op.results[1]


def fence(b: Builder, order: str = "seq_cst") -> Operation:
    return b.create("atomic.fence", (), (), {"order": order})
