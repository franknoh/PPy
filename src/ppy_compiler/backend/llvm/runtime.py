"""Python-ABI trampolines for natively lowered functions (spec 16.4, 25.3)."""

from __future__ import annotations

import array
import ctypes
from dataclasses import dataclass, field
from typing import Callable

from .lowering import STATUS_OK, NativeParam, NativeSignature
from .specialize import SpecializationKey, SpecializationPolicy, Specializer, key_for

__all__ = ["NativeBinding", "bind"]

_I64_LOW = -(1 << 63)
_I64_HIGH = (1 << 63) - 1

_CTYPES = {
    "i64": ctypes.c_int64,
    "double": ctypes.c_double,
    "i8": ctypes.c_int8,
}

_ELEMENT_CTYPES = {"int": ctypes.c_int64, "float": ctypes.c_double}

#: `array` type codes matching the native element types. Unlike a ctypes slice
#: assignment, `array.array` rejects an out-of-range value instead of
#: truncating it, which is what keeps Python integer semantics intact.
_ELEMENT_CODES = {"int": "q", "float": "d"}

#: Buffer-protocol formats a borrowed buffer accepts per element type. `l` is
#: a signed long, which is the same width as `q` where this matters.
_ELEMENT_FORMATS = {"int": ("q", "l"), "float": ("d",)}


class GuardFailed(Exception):
    """A runtime guard rejected an argument, so the Python path must run."""


@dataclass(slots=True)
class NativeBinding:
    """A guarded native entry point with a Python fallback."""

    signature: NativeSignature
    wrapper: Callable[..., object]
    fallback: Callable[..., object]
    calls: int = 0
    fallbacks: int = 0
    specialized_calls: int = 0
    #: (matcher, entry) pairs, checked in order on every call.
    selectors: list = field(default_factory=list)
    key_counts: dict = field(default_factory=dict)
    observations: int = 0
    observing: bool = False

    @property
    def specialization_count(self) -> int:
        return len(self.selectors)


def bind(
    signature: NativeSignature,
    address: int,
    fallback: Callable[..., object],
    *,
    specializer: Specializer | None = None,
    policy: SpecializationPolicy | None = None,
    info: object | None = None,
) -> NativeBinding:
    """Build the Python-callable wrapper for one native function.

    `ctypes.CFUNCTYPE` releases the GIL around the foreign call, which is what
    a native region touching no Python objects is allowed to do (spec 16.6).

    When the function asked for it, repeated argument shapes are compiled into
    guarded specializations and selected here (spec 16.9).
    """
    argument_types: list[type] = []
    for parameter in signature.parameters:
        if parameter.is_buffer:
            argument_types.append(ctypes.POINTER(_ELEMENT_CTYPES[parameter.element]))
            argument_types.append(ctypes.c_int64)
        else:
            argument_types.extend(_CTYPES[atom] for atom in parameter.abi)

    result_types = [_CTYPES[atom] for atom in signature.returns]
    prototype = ctypes.CFUNCTYPE(
        ctypes.c_int32, *argument_types, *[ctypes.POINTER(t) for t in result_types]
    )
    native = prototype(address)

    expanders = [_expander_for(p) for p in signature.parameters]
    finalizers = [_result_for(atom) for atom in signature.returns]
    returns_tuple = signature.returns_tuple
    observing = bool(
        specializer is not None
        and policy is not None
        and policy.enabled
        and policy.maximum > 0
        and info is not None
    )
    binding = NativeBinding(
        signature=signature, wrapper=lambda *a: None, fallback=fallback, observing=observing
    )

    def wrapper(*args: object) -> object:
        if len(args) != len(expanders):
            return fallback(*args)
        atoms: list[object] = []
        # `borrowed` keeps each unboxed buffer alive for the duration of the call.
        borrowed: list[object] = []
        try:
            for expand, value in zip(expanders, args):
                expand(value, atoms, borrowed)
        except GuardFailed:
            binding.fallbacks += 1
            return fallback(*args)
        entry = None
        for matches, candidate in binding.selectors:
            if matches(args):
                entry = candidate
                break
        else:
            if binding.observing:
                entry = _observe(binding, signature, args, policy, specializer, info, prototype)

        slots = [result_type() for result_type in result_types]
        target = entry or native
        status = target(*atoms, *[ctypes.byref(slot) for slot in slots])
        if status != STATUS_OK:
            binding.fallbacks += 1
            return fallback(*args)
        binding.calls += 1
        binding.specialized_calls += int(entry is not None)
        if returns_tuple:
            return tuple(finish(slot.value) for finish, slot in zip(finalizers, slots))
        return finalizers[0](slots[0].value)

    wrapper.__name__ = signature.qualname.rpartition(".")[2]
    wrapper.__qualname__ = signature.qualname
    wrapper.__doc__ = getattr(fallback, "__doc__", None)
    wrapper.__ppy_native__ = signature  # type: ignore[attr-defined]
    wrapper.__ppy_fallback__ = fallback  # type: ignore[attr-defined]
    binding.wrapper = wrapper
    return binding


def _expander_for(parameter: NativeParam) -> Callable[[object, list, list], None]:
    """Build the guard-and-convert step for one source-level parameter."""
    if parameter.is_borrowed:
        element_type = _ELEMENT_CTYPES[parameter.element]
        pointer_type = ctypes.POINTER(element_type)
        formats = _ELEMENT_FORMATS[parameter.element]

        def expand_view(value: object, atoms: list, borrowed: list) -> None:
            """Point at the caller's memory: no copy, no conversion."""
            try:
                view = memoryview(value)
            except TypeError as exc:
                raise GuardFailed from exc
            if (
                view.ndim != 1
                or not view.c_contiguous
                or view.format not in formats
                or view.itemsize != ctypes.sizeof(element_type)
                or view.readonly
            ):
                raise GuardFailed
            # `borrowed` keeps the view alive for the duration of the call, so
            # the memory cannot move or be freed underneath the native code.
            borrowed.append(view)
            length = view.shape[0]
            if length:
                holder = (element_type * length).from_buffer(view)
                borrowed.append(holder)
                atoms.append(ctypes.cast(holder, pointer_type))
            else:
                atoms.append(ctypes.cast(0, pointer_type))
            atoms.append(length)

        return expand_view

    if parameter.is_buffer:
        code = _ELEMENT_CODES[parameter.element]
        pointer_type = ctypes.POINTER(_ELEMENT_CTYPES[parameter.element])

        def expand_buffer(value: object, atoms: list, borrowed: list) -> None:
            if type(value) is not list:
                raise GuardFailed
            try:
                buffer = array.array(code, value)  # type: ignore[arg-type]
            except (TypeError, OverflowError, ValueError) as exc:
                raise GuardFailed from exc
            borrowed.append(buffer)
            address, length = buffer.buffer_info()
            atoms.append(ctypes.cast(address, pointer_type))
            atoms.append(length)

        return expand_buffer

    if parameter.is_tuple:
        element_guards = [_scalar_guard(atom) for atom in parameter.abi]

        def expand_tuple(value: object, atoms: list, borrowed: list) -> None:
            if type(value) is not tuple or len(value) != len(element_guards):
                raise GuardFailed
            for item, convert in zip(value, element_guards):
                atoms.append(convert(item))

        return expand_tuple

    abi = parameter.abi[0]
    if abi == "i64":

        def expand_int(value: object, atoms: list, borrowed: list) -> None:
            if type(value) is bool:
                atoms.append(int(value))
                return
            if type(value) is not int or not _I64_LOW <= value <= _I64_HIGH:
                raise GuardFailed
            atoms.append(value)

        return expand_int

    if abi == "double":

        def expand_float(value: object, atoms: list, borrowed: list) -> None:
            if type(value) is float:
                atoms.append(value)
                return
            if type(value) is int and _I64_LOW <= value <= _I64_HIGH:
                atoms.append(float(value))
                return
            raise GuardFailed

        return expand_float

    def expand_bool(value: object, atoms: list, borrowed: list) -> None:
        if type(value) is not bool:
            raise GuardFailed
        atoms.append(int(value))

    return expand_bool


def _result_for(abi: str) -> Callable[[object], object]:
    if abi == "double":
        return lambda value: float(value)  # type: ignore[arg-type]
    if abi == "i8":
        return lambda value: bool(value)
    return lambda value: int(value)  # type: ignore[arg-type]


def _scalar_guard(abi: str) -> Callable[[object], object]:
    """Guard and convert one scalar, raising `GuardFailed` when it does not fit."""
    if abi == "i64":

        def as_int(value: object) -> object:
            if type(value) is bool:
                return int(value)
            if type(value) is not int or not _I64_LOW <= value <= _I64_HIGH:
                raise GuardFailed
            return value

        return as_int
    if abi == "double":

        def as_float(value: object) -> object:
            if type(value) is float:
                return value
            if type(value) is int and _I64_LOW <= value <= _I64_HIGH:
                return float(value)
            raise GuardFailed

        return as_float

    def as_bool(value: object) -> object:
        if type(value) is not bool:
            raise GuardFailed
        return int(value)

    return as_bool


def _observe(
    binding: NativeBinding,
    signature: NativeSignature,
    args: tuple[object, ...],
    policy: SpecializationPolicy,
    specializer: Specializer,
    info: object,
    prototype,  # type: ignore[no-untyped-def]
):
    """Watch the arguments, and compile a specialization once one repeats.

    This runs only while learning. Once a specialization exists its matcher
    handles the call, and once the budget is spent the watching stops, so a
    function whose arguments never settle pays nothing to have asked.
    """
    binding.observations += 1
    if binding.observations > policy.budget:
        binding.observing = False
        return None

    key = key_for(signature, args, policy)
    if not key:
        binding.observing = False
        return None

    seen = binding.key_counts.get(key, 0) + 1
    binding.key_counts[key] = seen
    if seen < policy.threshold:
        return None

    specialization = specializer.specialize(info, key)  # type: ignore[arg-type]
    if specialization is None or not specialization.ok:
        # Refused once: never retry, and stop watching if nothing can work.
        binding.key_counts[key] = -(1 << 30)
        return None

    entry = prototype(specialization.address)
    binding.selectors.append((key.matcher(), entry))
    if binding.specialization_count >= policy.maximum:
        binding.observing = False
    return entry
