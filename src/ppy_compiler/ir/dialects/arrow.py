"""The arrow dialect: Arrow's own representation of an array, as the C Data
Interface hands it over (spec 52, 54).

`arrow.array<f64>` is a primitive Arrow array: a length, a null count, an
offset into its buffers, a validity bitmap that may be absent, and a values
buffer -- bit-packed for `bool`, one element per row otherwise. `arrow.import
%p` reads one from an `ArrowArray` struct a producer filled in, without a
copy; `arrow.length`, `arrow.null_count`, and `arrow.offset` read its
counts; `arrow.to_column` is the same memory as a `columnar.column<T,
nullable>`, borrowed where the offset lets it be and copied where it does
not (a bitmap sliced at a bit that is not a byte boundary). The `columnar`
dialect says what an operation means; this one says where Arrow keeps the
bytes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Builder, Operation, Value
from ..types import I64, BoolType, DialectType, FloatType, IntType, IRType, PtrType
from . import columnar

if TYPE_CHECKING:
    from ..verify import Checker

__all__ = ["ArrowDialect", "array_type", "element_of", "import_", "to_column"]

#: Byte offsets of the fields of `struct ArrowArray` (the Arrow C Data Interface).
LENGTH = 0
NULL_COUNT = 8
OFFSET = 16
N_BUFFERS = 24
N_CHILDREN = 32
BUFFERS = 40


def array_type(dtype: IRType) -> DialectType:
    return DialectType("arrow", "array", (dtype,))


def element_of(t: IRType) -> IRType | None:
    """The element type of an `arrow.array<T>`, or None for anything else."""
    if not isinstance(t, DialectType) or (t.dialect, t.name) != ("arrow", "array"):
        return None
    if verify_type(t) is not None:
        return None
    dtype = t.args[0]
    assert isinstance(dtype, IRType)
    return dtype


def verify_type(t: DialectType) -> str | None:
    if t.name != "array":
        return f"arrow defines no type {t.name!r}"
    if len(t.args) != 1 or not isinstance(t.args[0], (IntType, FloatType, BoolType)):
        return "an Arrow array is arrow.array<T> with a fixed-width scalar T"
    return None


def _array(op: Operation, checker: Checker, value: Value) -> IRType | None:
    dtype = element_of(value.type)
    if dtype is None:
        checker.error(op, f"{op.name} takes an arrow.array, not {value.type}")
    return dtype


def _verify_import(op: Operation, checker: Checker) -> None:
    pointer = op.operands[0].type
    if not isinstance(pointer, PtrType) or not (
        isinstance(pointer.pointee, IntType) and pointer.pointee.width == 8
    ):
        checker.error(op, "arrow.import reads an ArrowArray struct through a ptr<u8>")
    if element_of(op.results[0].type) is None:
        checker.error(op, f"arrow.import gives an arrow.array, not {op.results[0].type}")


def _verify_count(op: Operation, checker: Checker) -> None:
    if _array(op, checker, op.operands[0]) is None:
        return
    if op.results[0].type != I64:
        checker.error(op, f"{op.name} is an i64")


def _verify_to_column(op: Operation, checker: Checker) -> None:
    dtype = _array(op, checker, op.operands[0])
    if dtype is None:
        return
    expected = columnar.column_type(dtype, True)
    if op.results[0].type != expected:
        checker.error(op, f"arrow.to_column gives {expected}, not {op.results[0].type}")


class ArrowDialect(Dialect):
    name = "arrow"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        add = registry.add_op
        add(OpSpec("arrow.import", verify=_verify_import, operands=1, results=1))
        for name in ("length", "null_count", "offset"):
            add(OpSpec(f"arrow.{name}", pure=True, verify=_verify_count, operands=1, results=1))
        add(OpSpec("arrow.to_column", verify=_verify_to_column, operands=1, results=1))

    def verify_type(self, t: DialectType) -> str | None:
        return verify_type(t)


def import_(b: Builder, pointer: Value, dtype: IRType, name: str | None = None) -> Value:
    """The Arrow array of `dtype` an `ArrowArray` struct at `pointer` describes."""
    return b.create("arrow.import", (pointer,), (array_type(dtype),), result_names=(name,)).result


def count(b: Builder, what: str, array: Value, name: str | None = None) -> Value:
    return b.create(f"arrow.{what}", (array,), (I64,), result_names=(name,)).result


def to_column(b: Builder, array: Value, name: str | None = None) -> Value:
    dtype = element_of(array.type)
    assert dtype is not None
    return b.create(
        "arrow.to_column", (array,), (columnar.column_type(dtype, True),), result_names=(name,)
    ).result
