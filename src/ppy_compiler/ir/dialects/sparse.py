"""The sparse dialect: matrices stored by their non-zeros.

`sparse.csr<T, I, rows, cols>`, `sparse.csc<T, I, rows, cols>`, and
`sparse.coo<T, I, rows, cols>` name the format, the element type `T`, the
index type `I`, and the shape. A sparse value is made from the buffers of
its parts -- `sparse.from_parts %values, %indices, %pointers` (CSR and
CSC: the compressed pointers; COO: `%values, %rows, %cols`) -- which stay
borrowed from the program; the operations that make new non-zeros --
`add`, `convert` between formats -- own theirs. `to_dense`, `matmul` by a
dense matrix, and `reduce` over an axis give dense tensors; `transpose` of
a CSR matrix is the CSC matrix of the same parts, and the other way
round, so it costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .. import shape as shapes
from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Builder, Operation, Value
from ..types import BufferType, DialectType, FloatType, IndexType, IntType, IRType
from . import tensor as tensors

if TYPE_CHECKING:
    from ..verify import Checker

__all__ = ["FORMATS", "SparseDialect", "SparseInfo", "describe", "sparse_type"]

FORMATS = ("csr", "csc", "coo")


@dataclass(frozen=True, slots=True)
class SparseInfo:
    format: str
    dtype: IRType
    index: IRType
    rows: int
    cols: int


def sparse_type(fmt: str, dtype: IRType, index: IRType, rows: int, cols: int) -> DialectType:
    if fmt not in FORMATS:
        raise ValueError(f"no sparse format {fmt!r}")
    return DialectType("sparse", fmt, (dtype, index, rows, cols))


def describe(t: IRType) -> SparseInfo | None:
    if not isinstance(t, DialectType) or t.dialect != "sparse" or verify_sparse(t) is not None:
        return None
    dtype, index, rows, cols = t.args
    return SparseInfo(t.name, dtype, index, int(rows), int(cols))  # type: ignore[arg-type]


def verify_sparse(t: DialectType) -> str | None:
    if t.name not in FORMATS:
        return f"sparse defines no type {t.name!r}; the formats are {', '.join(FORMATS)}"
    if len(t.args) != 4:
        return f"sparse.{t.name} is sparse.{t.name}<T, I, rows, cols>"
    dtype, index, rows, cols = t.args
    if not isinstance(dtype, (IntType, FloatType, IndexType)):
        return f"a sparse matrix holds numbers, not {dtype}"
    if not isinstance(index, (IntType, IndexType)) or (
        isinstance(index, IntType) and index.width < 32
    ):
        return f"a sparse index is an integer of at least 32 bits, not {index}"
    if not isinstance(rows, int) or not isinstance(cols, int) or rows < 0 or cols < 0:
        return "a sparse matrix has a static shape"
    return None


def _sparse(op: Operation, checker: Checker, value: Value, what: str) -> SparseInfo | None:
    info = describe(value.type)
    if info is None:
        checker.error(op, f"{what} is a sparse matrix, not {value.type}")
    return info


def _verify_from_parts(op: Operation, checker: Checker) -> None:
    info = describe(op.results[0].type)
    if info is None:
        checker.error(op, "from_parts gives a sparse matrix")
        return
    values, first, second = (v.type for v in op.operands)
    if values != BufferType(info.dtype):
        checker.error(op, f"the values are a buffer<{info.dtype}>, not {values}")
    for part, name in ((first, "indices"), (second, "pointers")):
        if part != BufferType(info.index):
            checker.error(op, f"the {name} are a buffer<{info.index}>, not {part}")


def _verify_to_dense(op: Operation, checker: Checker) -> None:
    info = _sparse(op, checker, op.operands[0], "to_dense")
    if info is None:
        return
    result = tensors.describe(op.results[0].type)
    if result is None or result.dtype != info.dtype or result.shape != (info.rows, info.cols):
        checker.error(op, f"to_dense gives tensor<{info.dtype}, {info.rows}, {info.cols}>")


def _verify_matmul(op: Operation, checker: Checker) -> None:
    info = _sparse(op, checker, op.operands[0], "the sparse operand")
    dense = tensors.describe(op.operands[1].type)
    if info is None or dense is None:
        if dense is None:
            checker.error(op, "sparse.matmul takes a dense tensor on the right")
        return
    if dense.rank != 2 or dense.shape[0] != info.cols or dense.dtype != info.dtype:
        checker.error(
            op, f"sparse.matmul takes a tensor<{info.dtype}, {info.cols}, k> on the right"
        )
        return
    result = tensors.describe(op.results[0].type)
    if result is None or result.dtype != info.dtype or result.shape != (info.rows, dense.shape[1]):
        checker.error(
            op,
            f"sparse.matmul gives "
            f"tensor<{info.dtype}, {info.rows}, {shapes.spell(dense.shape[1])}>",
        )


def _verify_add(op: Operation, checker: Checker) -> None:
    a = _sparse(op, checker, op.operands[0], "the left operand")
    b = _sparse(op, checker, op.operands[1], "the right operand")
    if a is None or b is None:
        return
    if a != b or a.format == "coo":
        checker.error(op, "sparse.add takes two CSR or two CSC matrices of one type and shape")
        return
    if describe(op.results[0].type) != a:
        checker.error(op, "sparse.add gives the operands' type")


def _verify_transpose(op: Operation, checker: Checker) -> None:
    info = _sparse(op, checker, op.operands[0], "transpose")
    if info is None:
        return
    flipped = {"csr": "csc", "csc": "csr", "coo": "coo"}[info.format]
    expected = SparseInfo(flipped, info.dtype, info.index, info.cols, info.rows)
    if describe(op.results[0].type) != expected:
        checker.error(
            op, f"transpose of a {info.format} matrix is a {flipped} matrix of the transposed shape"
        )


def _verify_convert(op: Operation, checker: Checker) -> None:
    info = _sparse(op, checker, op.operands[0], "convert")
    result = describe(op.results[0].type)
    if info is None or result is None:
        if result is None:
            checker.error(op, "convert gives a sparse matrix")
        return
    if (result.dtype, result.index, result.rows, result.cols) != (
        info.dtype,
        info.index,
        info.rows,
        info.cols,
    ):
        checker.error(op, "convert changes the format and nothing else")


def _verify_reduce(op: Operation, checker: Checker) -> None:
    info = _sparse(op, checker, op.operands[0], "reduce")
    axis = op.attributes.get("axis")
    if info is None:
        return
    if axis not in {0, 1}:
        checker.error(op, "sparse.reduce sums along `axis` 0 or 1")
        return
    expected = (info.cols,) if axis == 0 else (info.rows,)
    result = tensors.describe(op.results[0].type)
    if result is None or result.dtype != info.dtype or result.shape != expected:
        checker.error(op, f"sparse.reduce gives tensor<{info.dtype}, {expected[0]}>")


class SparseDialect(Dialect):
    name = "sparse"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        add = registry.add_op
        add(OpSpec("sparse.from_parts", verify=_verify_from_parts, operands=3, results=1))
        add(OpSpec("sparse.to_dense", pure=True, verify=_verify_to_dense, operands=1, results=1))
        add(OpSpec("sparse.matmul", pure=True, verify=_verify_matmul, operands=2, results=1))
        add(OpSpec("sparse.add", pure=True, verify=_verify_add, operands=2, results=1))
        add(OpSpec("sparse.transpose", pure=True, verify=_verify_transpose, operands=1, results=1))
        add(OpSpec("sparse.convert", pure=True, verify=_verify_convert, operands=1, results=1))
        add(
            OpSpec(
                "sparse.reduce",
                pure=True,
                verify=_verify_reduce,
                operands=1,
                results=1,
                required_attributes=("axis",),
            )
        )

    def verify_type(self, t: DialectType) -> str | None:
        return verify_sparse(t)


def from_parts(
    b: Builder,
    t: DialectType,
    values: Value,
    indices: Value,
    pointers: Value,
    name: str | None = None,
) -> Value:
    return b.create(
        "sparse.from_parts", (values, indices, pointers), (t,), result_names=(name,)
    ).result


def to_dense(b: Builder, s: Value, name: str | None = None) -> Value:
    info = describe(s.type)
    assert info is not None
    return b.create(
        "sparse.to_dense",
        (s,),
        (tensors.tensor_type(info.dtype, (info.rows, info.cols)),),
        result_names=(name,),
    ).result


def matmul(b: Builder, s: Value, dense: Value, name: str | None = None) -> Value:
    info = describe(s.type)
    right = tensors.describe(dense.type)
    assert info is not None and right is not None
    return b.create(
        "sparse.matmul",
        (s, dense),
        (tensors.tensor_type(info.dtype, (info.rows, right.shape[1])),),
        result_names=(name,),
    ).result


def add(b: Builder, s: Value, t: Value, name: str | None = None) -> Value:
    return b.create("sparse.add", (s, t), (s.type,), result_names=(name,)).result


def transpose(b: Builder, s: Value, name: str | None = None) -> Value:
    info = describe(s.type)
    assert info is not None
    flipped = {"csr": "csc", "csc": "csr", "coo": "coo"}[info.format]
    return b.create(
        "sparse.transpose",
        (s,),
        (sparse_type(flipped, info.dtype, info.index, info.cols, info.rows),),
        result_names=(name,),
    ).result


def convert(b: Builder, s: Value, fmt: str, name: str | None = None) -> Value:
    info = describe(s.type)
    assert info is not None
    return b.create(
        "sparse.convert",
        (s,),
        (sparse_type(fmt, info.dtype, info.index, info.rows, info.cols),),
        result_names=(name,),
    ).result


def reduce(b: Builder, s: Value, axis: int, name: str | None = None) -> Value:
    info = describe(s.type)
    assert info is not None
    length = info.cols if axis == 0 else info.rows
    return b.create(
        "sparse.reduce",
        (s,),
        (tensors.tensor_type(info.dtype, (length,)),),
        {"axis": axis},
        result_names=(name,),
    ).result
