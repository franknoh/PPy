"""The simd dialect: lanes made, moved, and reduced.

`vector<T, N>` is `N` scalars operated on at once, and the core dialect's
arithmetic and comparisons already take vectors; what the core cannot say
is here. `splat` makes a vector of one scalar, `load` and `store` move one
through a pointer to its element type, `extract` and `insert` reach one
lane, `shuffle` builds a vector from the lanes of two by a constant mask,
and the reductions fold a vector to a scalar -- in lane order, first to
last, so a floating-point sum is the same number on every backend and in
the reference implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Builder, Operation, Value
from ..types import BoolType, FloatType, IndexType, IntType, PtrType, VectorType

if TYPE_CHECKING:
    from ..verify import Checker

__all__ = [
    "REDUCTIONS",
    "SimdDialect",
    "extract",
    "insert",
    "load",
    "reduce",
    "shuffle",
    "splat",
    "store",
]

REDUCTIONS = ("add", "min", "max")


def _index_type(t) -> bool:  # type: ignore[no-untyped-def]
    return isinstance(t, (IndexType, IntType))


def _verify_splat(op: Operation, checker: Checker) -> None:
    result = op.results[0].type
    if not isinstance(result, VectorType):
        checker.error(op, f"splat makes a vector, not {result}")
        return
    if op.operands[0].type != result.element:
        checker.error(op, f"splat of {op.operands[0].type} cannot fill {result}")


def _verify_load(op: Operation, checker: Checker) -> None:
    pointer = op.operands[0].type
    result = op.results[0].type
    if not isinstance(pointer, PtrType):
        checker.error(op, f"simd.load reads through a pointer, not {pointer}")
        return
    if not isinstance(result, VectorType) or result.element != pointer.pointee:
        checker.error(
            op, f"simd.load through {pointer} gives vector<{pointer.pointee}, N>, not {result}"
        )


def _verify_store(op: Operation, checker: Checker) -> None:
    vector, pointer = (v.type for v in op.operands)
    if not isinstance(pointer, PtrType):
        checker.error(op, f"simd.store writes through a pointer, not {pointer}")
        return
    if not isinstance(vector, VectorType) or vector.element != pointer.pointee:
        checker.error(op, f"simd.store of {vector} through {pointer}")
    if not pointer.mutable:
        checker.error(op, "writes through a const pointer")


def _verify_extract(op: Operation, checker: Checker) -> None:
    vector, index = (v.type for v in op.operands)
    if not isinstance(vector, VectorType):
        checker.error(op, f"extract takes a vector, not {vector}")
        return
    if not _index_type(index):
        checker.error(op, f"a lane index is an integer, not {index}")
    if op.results[0].type != vector.element:
        checker.error(op, f"extract from {vector} gives {vector.element}, not {op.results[0].type}")


def _verify_insert(op: Operation, checker: Checker) -> None:
    vector, value, index = (v.type for v in op.operands)
    if not isinstance(vector, VectorType):
        checker.error(op, f"insert takes a vector, not {vector}")
        return
    if value != vector.element:
        checker.error(op, f"insert of {value} into {vector}")
    if not _index_type(index):
        checker.error(op, f"a lane index is an integer, not {index}")
    if op.results[0].type != vector:
        checker.error(op, f"insert into {vector} gives {vector}, not {op.results[0].type}")


def _verify_shuffle(op: Operation, checker: Checker) -> None:
    a, b = (v.type for v in op.operands)
    if not isinstance(a, VectorType) or a != b:
        checker.error(op, f"shuffle takes two vectors of one type, not {a} and {b}")
        return
    mask = op.attributes.get("mask")
    if not isinstance(mask, tuple) or not mask or not all(isinstance(m, int) for m in mask):
        checker.error(op, "shuffle needs a `mask` of lane numbers")
        return
    if any(m < 0 or m >= 2 * a.count for m in mask):
        checker.error(op, f"a shuffle lane is in [0, {2 * a.count}), the lanes of both vectors")
    result = op.results[0].type
    if result != VectorType(a.element, len(mask)):
        checker.error(
            op, f"shuffle by {len(mask)} lanes gives vector<{a.element}, {len(mask)}>, not {result}"
        )


def _verify_reduce(op: Operation, checker: Checker) -> None:
    vector = op.operands[0].type
    if not isinstance(vector, VectorType):
        checker.error(op, f"a reduction takes a vector, not {vector}")
        return
    if isinstance(vector.element, BoolType):
        checker.error(op, "a reduction takes numbers, not bools")
    if op.results[0].type != vector.element:
        checker.error(op, f"reducing {vector} gives {vector.element}, not {op.results[0].type}")


class SimdDialect(Dialect):
    name = "simd"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        add = registry.add_op
        add(OpSpec("simd.splat", pure=True, verify=_verify_splat, operands=1, results=1))
        add(OpSpec("simd.load", verify=_verify_load, operands=1, results=1))
        add(OpSpec("simd.store", verify=_verify_store, operands=2, results=0))
        add(OpSpec("simd.extract", pure=True, verify=_verify_extract, operands=2, results=1))
        add(OpSpec("simd.insert", pure=True, verify=_verify_insert, operands=3, results=1))
        add(
            OpSpec(
                "simd.shuffle",
                pure=True,
                verify=_verify_shuffle,
                operands=2,
                results=1,
                required_attributes=("mask",),
            )
        )
        for name in REDUCTIONS:
            add(
                OpSpec(
                    f"simd.reduce_{name}", pure=True, verify=_verify_reduce, operands=1, results=1
                )
            )


def splat(b: Builder, value: Value, count: int, *, name: str | None = None) -> Value:
    return b.create(
        "simd.splat", (value,), (VectorType(value.type, count),), result_names=(name,)
    ).result


def load(b: Builder, pointer: Value, count: int, *, name: str | None = None) -> Value:
    assert isinstance(pointer.type, PtrType)
    result = VectorType(pointer.type.pointee, count)
    return b.create("simd.load", (pointer,), (result,), result_names=(name,)).result


def store(b: Builder, vector: Value, pointer: Value) -> Operation:
    return b.create("simd.store", (vector, pointer), ())


def extract(b: Builder, vector: Value, index: Value, *, name: str | None = None) -> Value:
    assert isinstance(vector.type, VectorType)
    return b.create(
        "simd.extract", (vector, index), (vector.type.element,), result_names=(name,)
    ).result


def insert(
    b: Builder, vector: Value, value: Value, index: Value, *, name: str | None = None
) -> Value:
    return b.create(
        "simd.insert", (vector, value, index), (vector.type,), result_names=(name,)
    ).result


def shuffle(
    b: Builder, a: Value, other: Value, mask: tuple[int, ...], *, name: str | None = None
) -> Value:
    assert isinstance(a.type, VectorType)
    result = VectorType(a.type.element, len(mask))
    return b.create(
        "simd.shuffle", (a, other), (result,), {"mask": tuple(mask)}, result_names=(name,)
    ).result


def reduce(b: Builder, kind: str, vector: Value, *, name: str | None = None) -> Value:
    if kind not in REDUCTIONS:
        raise ValueError(f"no reduction named {kind!r}")
    assert isinstance(vector.type, VectorType)
    return b.create(
        f"simd.reduce_{kind}", (vector,), (vector.type.element,), result_names=(name,)
    ).result


def is_floating(t: VectorType) -> bool:
    return isinstance(t.element, FloatType)
