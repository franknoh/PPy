"""Physical representation selection (spec 10.4)."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from . import types as T
from .refinements import Facts, width_range

__all__ = ["Repr", "Representation", "select"]


class Repr(enum.StrEnum):
    PY_OBJECT = "PyObject*"
    PY_LONG = "PyLong*"
    I1 = "i1"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    F16 = "f16"
    F32 = "f32"
    F64 = "f64"
    AGGREGATE = "LLVM aggregate"
    NATIVE_VECTOR = "native vector"
    PY_LIST = "PyListObject*"
    PY_ARRAY = "PyArrayObject*"
    NONE = "void"


@dataclass(frozen=True, slots=True)
class Representation:
    repr: Repr
    guarded: bool = False
    reason: str = ""

    def __str__(self) -> str:
        suffix = " (guarded)" if self.guarded else ""
        return f"{self.repr}{suffix}"


_INT_REPR = {8: Repr.I8, 16: Repr.I16, 32: Repr.I32, 64: Repr.I64}
_FLOAT_REPR = {16: Repr.F16, 32: Repr.F32, 64: Repr.F64}
_MACHINE_INT = width_range(64, True)


def select(t: T.Type, facts: Facts, *, escapes: bool = False) -> Representation:
    """Choose a physical representation for a semantic type plus proven facts.

    A semantic type is never conflated with its representation: a Python `int`
    may use `i64` storage while keeping arbitrary-precision semantics behind an
    overflow guard (spec 12.2).
    """
    if t == T.NONE:
        return Representation(Repr.NONE, reason="None has no payload")
    if isinstance(t, (T.AnyType, T.UnknownType)):
        return Representation(Repr.PY_OBJECT, reason="dynamic value")

    base = t.base if isinstance(t, T.Literal) else t

    if base == T.BOOL:
        return Representation(Repr.I1, reason="exact bool")
    if base == T.INT:
        if facts.width is not None:
            bits, _signed = facts.width
            return Representation(_INT_REPR[bits], reason="fixed-width marker")
        if facts.int_range is not None and _MACHINE_INT.contains(facts.int_range):
            return Representation(Repr.I64, reason="range proves machine int")
        if escapes:
            return Representation(Repr.PY_LONG, reason="escaping Python int")
        return Representation(Repr.I64, guarded=True, reason="unboxed with overflow guard")
    if base == T.FLOAT:
        bits = facts.float_bits or 64
        return Representation(_FLOAT_REPR[bits], reason="exact float")

    if isinstance(base, T.Tuple_) and not base.homogeneous and not escapes:
        return Representation(Repr.AGGREGATE, reason="non-escaping fixed tuple")
    if isinstance(base, T.Instance):
        if base.name == "list":
            if escapes:
                return Representation(Repr.PY_LIST, reason="escaping list")
            element = base.args[0] if base.args else T.ANY
            if isinstance(element, T.Instance) and element.name in {"int", "float", "bool"}:
                return Representation(Repr.NATIVE_VECTOR, reason="non-escaping homogeneous list")
            return Representation(Repr.PY_LIST, reason="heterogeneous element type")
        if base.name in {"Buffer", "memoryview", "array"}:
            element = base.args[0] if base.args else T.ANY
            if isinstance(element, T.Instance) and element.name in {"int", "float"}:
                # A buffer is pointed at, not copied, so it needs no promotion.
                return Representation(Repr.NATIVE_VECTOR, reason="borrowed contiguous buffer")
            return Representation(Repr.PY_OBJECT, reason="buffer of an unknown element type")
        if base.name in {"ndarray", "numpy.ndarray"}:
            return Representation(
                Repr.PY_ARRAY, reason="retained PyArrayObject* with an extracted data pointer"
            )
        if base.name == "torch.Tensor":
            return Representation(Repr.PY_OBJECT, reason="retained at::Tensor handle")
    return Representation(Repr.PY_OBJECT, reason="no native representation selected")
