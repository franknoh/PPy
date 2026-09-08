"""The fft dialect: discrete Fourier transforms over the last axis.

A complex tensor is a real tensor whose last dimension is 2 -- `(re, im)`
pairs -- so `fft.fft` and `fft.ifft` take `tensor<T, ..., n, 2>` and give
the same shape; `fft.rfft` takes a real `tensor<T, ..., n>` and gives
`tensor<T, ..., n // 2 + 1, 2>`; `fft.irfft` takes that and `n` and gives
the real tensor back; `fft.fftn` and `fft.ifftn` transform every axis but
the pair axis. Inverse transforms carry the `1 / n` scale NumPy's do. The
lowering is the definition -- the sum over every input for every output,
in loops, on any backend -- until a build selects an FFT library.
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

__all__ = ["FftDialect", "fft", "fftn", "ifft", "ifftn", "irfft", "rfft"]


def _complex(op: Operation, checker: Checker, value: Value, what: str) -> tensors.TensorInfo | None:
    info = tensors.describe(value.type)
    if info is None or not isinstance(info.dtype, FloatType):
        checker.error(op, f"{what} is a floating-point tensor, not {value.type}")
        return None
    if info.rank < 2 or info.shape[-1] != 2:
        checker.error(
            op, f"{what} is complex: its last dimension is 2, not {shapes.spell_shape(info.shape)}"
        )
        return None
    return info


def _same(op: Operation, checker: Checker, info: tensors.TensorInfo) -> None:
    result = tensors.describe(op.results[0].type)
    if result is None or result.dtype != info.dtype or result.shape != info.shape:
        checker.error(
            op,
            f"{op.name} keeps the shape and element type; the result is written as "
            f"{op.results[0].type}",
        )


def _verify_complex(op: Operation, checker: Checker) -> None:
    info = _complex(op, checker, op.operands[0], "the input")
    if info is not None:
        _same(op, checker, info)


def _verify_rfft(op: Operation, checker: Checker) -> None:
    info = tensors.describe(op.operands[0].type)
    if info is None or not isinstance(info.dtype, FloatType) or info.rank < 1:
        checker.error(op, "rfft takes a real floating-point tensor")
        return
    n = info.shape[-1]
    if not isinstance(n, int):
        checker.error(op, "rfft needs a static last dimension")
        return
    expected = (*info.shape[:-1], n // 2 + 1, 2)
    result = tensors.describe(op.results[0].type)
    if result is None or result.dtype != info.dtype or result.shape != expected:
        checker.error(
            op, f"rfft of {shapes.spell_shape(info.shape)} gives {shapes.spell_shape(expected)}"
        )


def _verify_irfft(op: Operation, checker: Checker) -> None:
    info = _complex(op, checker, op.operands[0], "the input")
    n = op.attributes.get("n")
    if info is None:
        return
    if not isinstance(n, int) or n < 1:
        checker.error(op, "irfft needs `n`, the length of the real output")
        return
    if info.shape[-2] != n // 2 + 1:
        checker.error(
            op, f"irfft with n = {n} takes {n // 2 + 1} frequencies, not {info.shape[-2]}"
        )
    expected = (*info.shape[:-2], n)
    result = tensors.describe(op.results[0].type)
    if result is None or result.dtype != info.dtype or result.shape != expected:
        checker.error(op, f"irfft gives {shapes.spell_shape(expected)}")


class FftDialect(Dialect):
    name = "fft"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        for name in ("fft", "ifft", "fftn", "ifftn"):
            registry.add_op(
                OpSpec(f"fft.{name}", pure=True, verify=_verify_complex, operands=1, results=1)
            )
        registry.add_op(OpSpec("fft.rfft", pure=True, verify=_verify_rfft, operands=1, results=1))
        registry.add_op(
            OpSpec(
                "fft.irfft",
                pure=True,
                verify=_verify_irfft,
                operands=1,
                results=1,
                required_attributes=("n",),
            )
        )


def _unary(b: Builder, name: str, x: Value, result_type, attributes=None, hint=None):  # type: ignore[no-untyped-def]
    return b.create(
        f"fft.{name}", (x,), (result_type,), attributes or {}, result_names=(hint,)
    ).result


def fft(b: Builder, x: Value, name: str | None = None) -> Value:
    return _unary(b, "fft", x, x.type, hint=name)


def ifft(b: Builder, x: Value, name: str | None = None) -> Value:
    return _unary(b, "ifft", x, x.type, hint=name)


def fftn(b: Builder, x: Value, name: str | None = None) -> Value:
    return _unary(b, "fftn", x, x.type, hint=name)


def ifftn(b: Builder, x: Value, name: str | None = None) -> Value:
    return _unary(b, "ifftn", x, x.type, hint=name)


def rfft(b: Builder, x: Value, name: str | None = None) -> Value:
    info = tensors.describe(x.type)
    assert info is not None
    n = info.shape[-1]
    assert isinstance(n, int)
    return _unary(
        b, "rfft", x, tensors.tensor_type(info.dtype, (*info.shape[:-1], n // 2 + 1, 2)), hint=name
    )


def irfft(b: Builder, x: Value, n: int, name: str | None = None) -> Value:
    info = tensors.describe(x.type)
    assert info is not None
    return _unary(
        b, "irfft", x, tensors.tensor_type(info.dtype, (*info.shape[:-2], n)), {"n": n}, hint=name
    )
