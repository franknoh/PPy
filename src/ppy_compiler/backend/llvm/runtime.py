"""Python-ABI trampolines for natively lowered functions (spec 16.4, 25.3)."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Callable

from .lowering import STATUS_OK, NativeSignature

__all__ = ["NativeBinding", "bind"]

_I64_LOW = -(1 << 63)
_I64_HIGH = (1 << 63) - 1

_CTYPES = {
    "i64": ctypes.c_int64,
    "double": ctypes.c_double,
    "i8": ctypes.c_int8,
}


@dataclass(slots=True)
class NativeBinding:
    """A guarded native entry point with a Python fallback."""

    signature: NativeSignature
    wrapper: Callable[..., object]
    fallback: Callable[..., object]
    calls: int = 0
    fallbacks: int = 0


def bind(signature: NativeSignature, address: int, fallback: Callable[..., object]) -> NativeBinding:
    """Build the Python-callable wrapper for one native function.

    `ctypes.CFUNCTYPE` releases the GIL around the foreign call, which is what
    a native region touching no Python objects is allowed to do (spec 16.6).
    """
    argument_types = [_CTYPES[p] for p in signature.params]
    result_type = _CTYPES[signature.ret]
    prototype = ctypes.CFUNCTYPE(ctypes.c_int32, *argument_types, ctypes.POINTER(result_type))
    native = prototype(address)

    guards = [_guard_for(p) for p in signature.params]
    converters = [_converter_for(p) for p in signature.params]
    finalize = _result_for(signature.ret)
    binding = NativeBinding(signature=signature, wrapper=lambda *a: None, fallback=fallback)

    def wrapper(*args: object) -> object:
        if len(args) != len(guards):
            return fallback(*args)
        for value, guard in zip(args, guards):
            if not guard(value):
                binding.fallbacks += 1
                return fallback(*args)
        slot = result_type()
        status = native(*[convert(v) for convert, v in zip(converters, args)], ctypes.byref(slot))
        if status != STATUS_OK:
            binding.fallbacks += 1
            return fallback(*args)
        binding.calls += 1
        return finalize(slot.value)

    wrapper.__name__ = signature.qualname.rpartition(".")[2]
    wrapper.__qualname__ = signature.qualname
    wrapper.__doc__ = getattr(fallback, "__doc__", None)
    wrapper.__ppy_native__ = signature  # type: ignore[attr-defined]
    wrapper.__ppy_fallback__ = fallback  # type: ignore[attr-defined]
    binding.wrapper = wrapper
    return binding


def _guard_for(abi: str) -> Callable[[object], bool]:
    if abi == "i64":
        def guard_int(value: object) -> bool:
            return type(value) is int and _I64_LOW <= value <= _I64_HIGH or type(value) is bool
        return guard_int
    if abi == "double":
        def guard_float(value: object) -> bool:
            return type(value) is float or (type(value) is int and _I64_LOW <= value <= _I64_HIGH)
        return guard_float
    def guard_bool(value: object) -> bool:
        return type(value) is bool
    return guard_bool


def _converter_for(abi: str) -> Callable[[object], object]:
    if abi == "double":
        return lambda value: float(value)  # type: ignore[arg-type]
    if abi == "i8":
        return lambda value: int(bool(value))
    return lambda value: int(value)  # type: ignore[arg-type]


def _result_for(abi: str) -> Callable[[object], object]:
    if abi == "double":
        return lambda value: float(value)  # type: ignore[arg-type]
    if abi == "i8":
        return lambda value: bool(value)
    return lambda value: int(value)  # type: ignore[arg-type]
