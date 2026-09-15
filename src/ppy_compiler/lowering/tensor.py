"""The shared representation of ppy.Tensor, before backend memory selection."""

from __future__ import annotations

from ..analysis.refinements import Facts
from ..backend.llvm.lowering import Unsupported
from ..ir import BF16, BOOL, F16, F32, F64, I8, I16, I32, I64, U8, U16, U32, U64, DialectType
from ..ir.dialects import tensor
from ..ir.shape import ShapeError, parse_dim

_DTYPES = {
    "bfloat16": BF16,
    "float16": F16,
    "float32": F32,
    "float64": F64,
    "int8": I8,
    "int16": I16,
    "int32": I32,
    "int64": I64,
    "uint8": U8,
    "uint16": U16,
    "uint32": U32,
    "uint64": U64,
    "bool": BOOL,
}


def common_tensor_type(facts: Facts) -> DialectType:
    """Keep proven dtype and rank/shape, without inventing missing metadata."""
    if facts.dtype is None or facts.shape is None:
        raise Unsupported("ppy.Tensor requires a known dtype and shape for canonical lowering")
    dtype = _DTYPES.get(facts.dtype)
    if dtype is None:
        raise Unsupported(f"ppy.Tensor dtype `{facts.dtype}` has no canonical IR representation")
    try:
        shape = tuple(parse_dim(dim) for dim in facts.shape)
    except ShapeError as error:
        raise Unsupported(str(error)) from error
    result = tensor.tensor_type(dtype, shape)
    reason = tensor.verify_tensor(result)
    if reason is not None:
        raise Unsupported(reason)
    return result
