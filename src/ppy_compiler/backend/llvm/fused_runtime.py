"""Guarded dispatch for fused NumPy kernels (spec 19.3, 19.5, 19.10, 19.11)."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Callable

from .fusion import FusedLoop

__all__ = ["FusedBinding", "bind_fused"]

_SAFE_ERROR_STATES = frozenset({"ignore"})


@dataclass(slots=True)
class FusedBinding:
    loop: FusedLoop
    wrapper: Callable[..., object]
    fallback: Callable[..., object]
    calls: int = 0
    fallbacks: int = 0
    reason: str = ""


def bind_fused(loop: FusedLoop, address: int, fallback: Callable[..., object]) -> FusedBinding:
    """Wrap one fused kernel in the guards its fast-path domain requires."""
    try:
        import numpy
    except ImportError:
        binding = FusedBinding(loop, fallback, fallback, reason="numpy is not importable")
        return binding

    double = ctypes.c_double
    pointer = ctypes.POINTER(double)
    array_count = len(loop.arrays)
    scalar_count = len(loop.scalars)

    if loop.returns_scalar:
        prototype = ctypes.CFUNCTYPE(
            double, *([pointer] * array_count), *([double] * scalar_count), ctypes.c_int64
        )
    else:
        prototype = ctypes.CFUNCTYPE(
            None, pointer, *([pointer] * array_count), *([double] * scalar_count), ctypes.c_int64
        )
    native = prototype(address)

    float64 = numpy.dtype("float64")
    binding = FusedBinding(loop, lambda *a: None, fallback)

    def wrapper(*args: object) -> object:
        if len(args) != array_count + scalar_count:
            return fallback(*args)
        arrays = args[:array_count]
        scalars = args[array_count:]

        shape = None
        for value in arrays:
            # An exact ndarray only: a subclass may override dispatch entirely.
            if type(value) is not numpy.ndarray:
                binding.fallbacks += 1
                return fallback(*args)
            if value.dtype != float64 or not value.dtype.isnative:
                binding.fallbacks += 1
                return fallback(*args)
            if not value.flags["C_CONTIGUOUS"]:
                binding.fallbacks += 1
                return fallback(*args)
            if shape is None:
                shape = value.shape
            elif value.shape != shape:
                # Array-to-array broadcasting is left to NumPy in v1.
                binding.fallbacks += 1
                return fallback(*args)
        if shape is None:
            binding.fallbacks += 1
            return fallback(*args)

        for value in scalars:
            if type(value) not in (int, float):
                binding.fallbacks += 1
                return fallback(*args)

        pointers = [ctypes.cast(a.ctypes.data, pointer) for a in arrays]
        widened = [float(s) for s in scalars]
        length = int(arrays[0].size)
        strict = not _errors_ignored(numpy)

        if loop.returns_scalar:
            result = native(*pointers, *widened, length)
            if strict and not _finite_result(numpy, result, arrays):
                binding.fallbacks += 1
                return fallback(*args)
            binding.calls += 1
            return float(result)

        out = numpy.empty(shape, dtype=float64)
        native(ctypes.cast(out.ctypes.data, pointer), *pointers, *widened, length)
        if strict and not _finite_result(numpy, out, arrays):
            binding.fallbacks += 1
            return fallback(*args)
        binding.calls += 1
        return out

    wrapper.__name__ = loop.symbol
    wrapper.__ppy_fused__ = loop  # type: ignore[attr-defined]
    binding.wrapper = wrapper
    return binding


def _errors_ignored(numpy) -> bool:  # type: ignore[no-untyped-def]
    """Is NumPy's floating-point error state one the fused loop can honor?

    A generated loop raises no NumPy warning of its own, so unless every error
    category is ignored the result must be checked before it is trusted
    (spec 19.11).
    """
    state = numpy.geterr()
    return all(value in _SAFE_ERROR_STATES for value in state.values())


def _finite_result(numpy, result, arrays) -> bool:  # type: ignore[no-untyped-def]
    """Confirm the kernel raised no floating-point condition NumPy would report.

    Divide-by-zero, overflow, and invalid operations all surface as a non-finite
    value. If the inputs were finite and the output is too, none occurred;
    otherwise the Python path re-runs and reports exactly what NumPy would.
    """
    if not numpy.isfinite(result).all():
        return False
    return all(numpy.isfinite(a).all() for a in arrays)
