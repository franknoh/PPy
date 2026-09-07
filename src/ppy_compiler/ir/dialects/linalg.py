"""The linalg dialect: linear algebra on tensors, with the meaning spelled once.

`linalg.dot` of two vectors, `linalg.matmul` of two matrices,
`linalg.solve` of a square system (one or several right-hand sides),
`linalg.triangular_solve` of a triangular one, `linalg.cholesky` of a
positive-definite matrix into its lower factor, and the factorizations
`linalg.qr` (reduced: Q is m-by-k, R k-by-n), `linalg.svd` (reduced: U
m-by-k, S k, Vt k-by-n), and `linalg.eig` (values and right vectors of a
general real matrix, complex, spelled with a trailing dimension of 2 as
the fft dialect spells them). How each is computed is the lowering's
choice by what the build has: loops for the first five, LAPACK for the
factorizations, refused with the reason where there is neither.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import shape as shapes
from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Builder, Operation, Value
from ..types import FloatType
from . import tensor as tensors

if TYPE_CHECKING:
    from ..verify import Checker

__all__ = [
    "FACTORIZATIONS",
    "LinalgDialect",
    "cholesky",
    "dot",
    "eig",
    "matmul",
    "qr",
    "solve",
    "svd",
    "triangular_solve",
]

FACTORIZATIONS = ("qr", "svd", "eig")


def _matrix(
    op: Operation, checker: Checker, value: Value, what: str, *, square: bool = False
) -> tensors.TensorInfo | None:
    info = tensors.describe(value.type)
    if info is None:
        checker.error(op, f"{what} is a tensor, not {value.type}")
        return None
    if not isinstance(info.dtype, FloatType):
        checker.error(op, f"{op.name} takes floating-point tensors, not {info.dtype}")
        return None
    if info.rank != 2:
        checker.error(op, f"{what} is a matrix, not rank {info.rank}")
        return None
    if square and info.shape[0] != info.shape[1]:
        checker.error(op, f"{what} is square, not {shapes.spell_shape(info.shape)}")
        return None
    return info


def _result(op: Operation, checker: Checker, index: int, dtype, shape: shapes.Shape) -> None:  # type: ignore[no-untyped-def]
    result = tensors.describe(op.results[index].type)
    if result is None:
        checker.error(op, f"{op.name} gives tensors")
        return
    if result.dtype != dtype or result.shape != shape:
        checker.error(
            op,
            f"result {index} of {op.name} is tensor<{dtype}, {shapes.spell_shape(shape)[1:-1]}>, "
            f"written as {op.results[index].type}",
        )


def _verify_dot(op: Operation, checker: Checker) -> None:
    a, b = (tensors.describe(v.type) for v in op.operands)
    if a is None or b is None:
        checker.error(op, "dot takes two vectors")
        return
    if a.rank != 1 or b.rank != 1 or a.shape != b.shape or a.dtype != b.dtype:
        checker.error(op, "dot takes two vectors of one length and element type")
        return
    _result(op, checker, 0, a.dtype, ())


def _verify_matmul(op: Operation, checker: Checker) -> None:
    a = _matrix(op, checker, op.operands[0], "the left operand")
    b = _matrix(op, checker, op.operands[1], "the right operand")
    if a is None or b is None:
        return
    if a.dtype != b.dtype:
        checker.error(op, "matmul takes tensors of one element type")
        return
    try:
        _result(op, checker, 0, a.dtype, shapes.matmul(a.shape, b.shape))
    except shapes.ShapeError as error:
        checker.error(op, str(error))


def _verify_solve(op: Operation, checker: Checker) -> None:
    a = _matrix(op, checker, op.operands[0], "the matrix", square=True)
    rhs = tensors.describe(op.operands[1].type)
    if a is None or rhs is None:
        if rhs is None:
            checker.error(op, "the right-hand side is a tensor")
        return
    if rhs.rank not in {1, 2} or rhs.shape[0] != a.shape[0] or rhs.dtype != a.dtype:
        checker.error(op, "the right-hand side has the matrix's rows, as a vector or a matrix")
        return
    if op.local_name == "triangular_solve":
        for flag in ("lower", "unit"):
            if not isinstance(op.attributes.get(flag, False), bool):
                checker.error(op, f"`{flag}` is a bool")
    _result(op, checker, 0, a.dtype, rhs.shape)


def _verify_cholesky(op: Operation, checker: Checker) -> None:
    a = _matrix(op, checker, op.operands[0], "the matrix", square=True)
    if a is not None:
        _result(op, checker, 0, a.dtype, a.shape)


def _verify_qr(op: Operation, checker: Checker) -> None:
    a = _matrix(op, checker, op.operands[0], "the matrix")
    if a is None:
        return
    m, n = a.shape
    k = min(m, n) if isinstance(m, int) and isinstance(n, int) else m
    _result(op, checker, 0, a.dtype, (m, k))
    _result(op, checker, 1, a.dtype, (k, n))


def _verify_svd(op: Operation, checker: Checker) -> None:
    a = _matrix(op, checker, op.operands[0], "the matrix")
    if a is None:
        return
    m, n = a.shape
    k = min(m, n) if isinstance(m, int) and isinstance(n, int) else m
    _result(op, checker, 0, a.dtype, (m, k))
    _result(op, checker, 1, a.dtype, (k,))
    _result(op, checker, 2, a.dtype, (k, n))


def _verify_eig(op: Operation, checker: Checker) -> None:
    a = _matrix(op, checker, op.operands[0], "the matrix", square=True)
    if a is None:
        return
    n = a.shape[0]
    _result(op, checker, 0, a.dtype, (n, 2))
    _result(op, checker, 1, a.dtype, (n, n, 2))


class LinalgDialect(Dialect):
    name = "linalg"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        add = registry.add_op
        add(OpSpec("linalg.dot", pure=True, verify=_verify_dot, operands=2, results=1))
        add(OpSpec("linalg.matmul", pure=True, verify=_verify_matmul, operands=2, results=1))
        add(OpSpec("linalg.solve", pure=True, verify=_verify_solve, operands=2, results=1))
        add(
            OpSpec(
                "linalg.triangular_solve", pure=True, verify=_verify_solve, operands=2, results=1
            )
        )
        add(OpSpec("linalg.cholesky", pure=True, verify=_verify_cholesky, operands=1, results=1))
        add(OpSpec("linalg.qr", pure=True, verify=_verify_qr, operands=1, results=2))
        add(OpSpec("linalg.svd", pure=True, verify=_verify_svd, operands=1, results=3))
        add(OpSpec("linalg.eig", pure=True, verify=_verify_eig, operands=1, results=2))


def _info(value: Value) -> tensors.TensorInfo:
    info = tensors.describe(value.type)
    assert info is not None
    return info


def dot(b: Builder, a: Value, c: Value, name: str | None = None) -> Value:
    return b.create(
        "linalg.dot", (a, c), (tensors.tensor_type(_info(a).dtype, ()),), result_names=(name,)
    ).result


def matmul(b: Builder, a: Value, c: Value, name: str | None = None) -> Value:
    left, right = _info(a), _info(c)
    result = tensors.tensor_type(left.dtype, shapes.matmul(left.shape, right.shape))
    return b.create("linalg.matmul", (a, c), (result,), result_names=(name,)).result


def solve(b: Builder, a: Value, rhs: Value, name: str | None = None) -> Value:
    return b.create("linalg.solve", (a, rhs), (rhs.type,), result_names=(name,)).result


def triangular_solve(
    b: Builder,
    a: Value,
    rhs: Value,
    *,
    lower: bool = True,
    unit: bool = False,
    name: str | None = None,
) -> Value:
    return b.create(
        "linalg.triangular_solve",
        (a, rhs),
        (rhs.type,),
        {"lower": lower, "unit": unit},
        result_names=(name,),
    ).result


def cholesky(b: Builder, a: Value, name: str | None = None) -> Value:
    return b.create("linalg.cholesky", (a,), (a.type,), result_names=(name,)).result


def qr(b: Builder, a: Value) -> tuple[Value, Value]:
    info = _info(a)
    m, n = info.shape
    k = min(m, n)  # type: ignore[type-var]
    op = b.create(
        "linalg.qr",
        (a,),
        (tensors.tensor_type(info.dtype, (m, k)), tensors.tensor_type(info.dtype, (k, n))),
    )
    return op.results[0], op.results[1]


def svd(b: Builder, a: Value) -> tuple[Value, Value, Value]:
    info = _info(a)
    m, n = info.shape
    k = min(m, n)  # type: ignore[type-var]
    op = b.create(
        "linalg.svd",
        (a,),
        (
            tensors.tensor_type(info.dtype, (m, k)),
            tensors.tensor_type(info.dtype, (k,)),
            tensors.tensor_type(info.dtype, (k, n)),
        ),
    )
    return op.results[0], op.results[1], op.results[2]


def eig(b: Builder, a: Value) -> tuple[Value, Value]:
    info = _info(a)
    n = info.shape[0]
    op = b.create(
        "linalg.eig",
        (a,),
        (tensors.tensor_type(info.dtype, (n, 2)), tensors.tensor_type(info.dtype, (n, n, 2))),
    )
    return op.results[0], op.results[1]
