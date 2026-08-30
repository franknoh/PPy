"""Python-ABI trampolines for natively lowered functions (spec 16.4, 25.3).

Runtime-only: a built artifact binds through this module with no compiler
installed. JIT specialization is an optional hook the compiler passes in,
and its machinery is imported only when it is actually used.
"""

from __future__ import annotations

import array
import ctypes
from collections.abc import Callable
from dataclasses import dataclass, field

from .abi import STATUS_OK, NativeParam, NativeSignature

__all__ = ["NativeBinding", "adopt", "bind", "observation_wanted", "value_class_types"]

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


def observation_wanted(specializer: object, policy: object, info: object) -> bool:
    """Whether this function still wants Python watching for specialization."""
    return bool(
        specializer is not None
        and policy is not None
        and policy.enabled  # type: ignore[attr-defined]
        and policy.maximum > 0  # type: ignore[attr-defined]
        and info is not None
    )


def value_class_types(signature: NativeSignature, fallback: Callable[..., object]) -> tuple | None:
    """The runtime classes a generated wrapper guards value parameters on.

    Resolved from the defining module, so a class the wrapper cannot see means
    no fast entry rather than a wrong one.
    """
    namespace = getattr(fallback, "__globals__", None)
    found = []
    for parameter in signature.parameters:
        if not parameter.is_object:
            continue
        if namespace is None:
            return None
        cls = namespace.get(parameter.class_name.rpartition(".")[2])
        if not isinstance(cls, type):
            return None
        found.append(cls)
    return tuple(found)


def adopt(
    signature: NativeSignature,
    entry: Callable[..., object],
    fallback: Callable[..., object],
    *,
    owner: object | None = None,
) -> NativeBinding:
    """Adopt a generated wrapper that already holds its Python fallback in C.

    Nothing stands between the caller and the C entry point, so per-call
    statistics are not collected on this path.
    """
    return NativeBinding(
        signature=signature, wrapper=entry, fallback=fallback, fast_entry=entry, owner=owner
    )


@dataclass(slots=True)
class NativeBinding:
    """A guarded native entry point with a Python fallback."""

    signature: NativeSignature
    wrapper: Callable[..., object]
    fallback: Callable[..., object]
    calls: int = 0
    fallbacks: int = 0
    specialized_calls: int = 0
    #: The generated CPython-ABI entry point, when one was compiled.
    fast_entry: object | None = None
    #: Whatever owns the compiled code, kept so it cannot be freed while a
    #: wrapper still points at it.
    owner: object | None = None
    #: (matcher, entry) pairs the ctypes boundary checks on every call.
    selectors: list = field(default_factory=list)
    #: Specializations handed to the generated wrapper, which selects them
    #: itself, so this side only needs to know how many exist.
    registered: int = 0
    key_counts: dict = field(default_factory=dict)
    observations: int = 0
    observing: bool = False

    @property
    def specialization_count(self) -> int:
        return len(self.selectors) + self.registered


def bind(
    signature: NativeSignature,
    address: int,
    fallback: Callable[..., object],
    *,
    specializer: object | None = None,
    policy: object | None = None,
    info: object | None = None,
    fast_entry: Callable[..., object] | None = None,
    owner: object | None = None,
    register: Callable[[int, tuple], bool] | None = None,
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

    namespace = getattr(fallback, "__globals__", None)
    expanders = [
        _expander_for(p, (lambda: namespace) if namespace is not None else None)
        for p in signature.parameters
    ]
    finalizers = [_result_for(atom) for atom in signature.returns]
    returns_tuple = signature.returns_tuple
    # Without a way to register one, a specialization could not be reached.
    observing = observation_wanted(specializer, policy, info) and (
        fast_entry is None or register is not None
    )
    binding = NativeBinding(
        signature=signature,
        wrapper=lambda *a: None,
        fallback=fallback,
        observing=observing,
        fast_entry=fast_entry,
        owner=owner,
    )

    if fast_entry is not None:
        # The generated wrapper does the parsing, the guards, the specialization
        # choice, the call, and the boxing in C; `NotImplemented` is its signal
        # that a guard failed. While the function is still learning which
        # argument shapes repeat, Python watches alongside.
        def fast_wrapper(*args: object) -> object:
            if binding.observing:
                _watch(binding, signature, args, policy, specializer, info, register)
            result = fast_entry(*args)
            if result is NotImplemented:
                binding.fallbacks += 1
                return fallback(*args)
            binding.calls += 1
            return result

        fast_wrapper.__name__ = signature.qualname.rpartition(".")[2]
        fast_wrapper.__qualname__ = signature.qualname
        fast_wrapper.__doc__ = getattr(fallback, "__doc__", None)
        fast_wrapper.__ppy_native__ = signature  # type: ignore[attr-defined]
        fast_wrapper.__ppy_fallback__ = fallback  # type: ignore[attr-defined]
        binding.wrapper = fast_wrapper
        return binding

    def wrapper(*args: object) -> object:
        if len(args) != len(expanders):
            return fallback(*args)
        atoms: list[object] = []
        # `borrowed` keeps each unboxed buffer alive for the duration of the call.
        borrowed: list[object] = []
        try:
            for expand, value in zip(expanders, args, strict=False):
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
            return tuple(
                finish(slot.value) for finish, slot in zip(finalizers, slots, strict=False)
            )
        return finalizers[0](slots[0].value)

    wrapper.__name__ = signature.qualname.rpartition(".")[2]
    wrapper.__qualname__ = signature.qualname
    wrapper.__doc__ = getattr(fallback, "__doc__", None)
    wrapper.__ppy_native__ = signature  # type: ignore[attr-defined]
    wrapper.__ppy_fallback__ = fallback  # type: ignore[attr-defined]
    binding.wrapper = wrapper
    return binding


def _expander_for(
    parameter: NativeParam, namespace: Callable[[], dict] | None = None
) -> Callable[[object, list, list], None]:
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

    if parameter.is_object:
        field_guards = [(attr, _scalar_guard(_abi_of(scalar))) for attr, scalar in parameter.fields]
        short_name = parameter.class_name.rpartition(".")[2]
        resolved: list[type | None] = [None]

        def expand_object(value: object, atoms: list, borrowed: list) -> None:
            """Read a value class's fields, guarded on its exact class.

            The class is looked up lazily in the defining module, so the order
            of definitions in the source does not matter.
            """
            expected = resolved[0]
            if expected is None:
                expected = namespace().get(short_name) if namespace is not None else None
                if not isinstance(expected, type):
                    raise GuardFailed
                resolved[0] = expected
            # A subclass may override attribute access, so only the exact class
            # is flattened; anything else runs the Python body (spec 25.4).
            if type(value) is not expected:
                raise GuardFailed
            for attr, convert in field_guards:
                try:
                    atoms.append(convert(getattr(value, attr)))
                except AttributeError as exc:
                    raise GuardFailed from exc

        return expand_object

    if parameter.is_tuple:
        element_guards = [_scalar_guard(atom) for atom in parameter.abi]

        def expand_tuple(value: object, atoms: list, borrowed: list) -> None:
            if type(value) is not tuple or len(value) != len(element_guards):
                raise GuardFailed
            for item, convert in zip(value, element_guards, strict=False):
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
        return float  # type: ignore[arg-type]
    if abi == "i8":
        return bool
    return int  # type: ignore[arg-type]


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
    policy,  # type: ignore[no-untyped-def]
    specializer,
    info: object,
    prototype,  # type: ignore[no-untyped-def]
):
    """Watch the arguments, and compile a specialization once one repeats.

    This runs only while learning. Once a specialization exists its matcher
    handles the call, and once the budget is spent the watching stops, so a
    function whose arguments never settle pays nothing to have asked.
    """
    # Specialization is the JIT compiler's business; a prebuilt artifact
    # never passes a specializer, so the compiler import never happens there.
    from ppy_compiler.backend.llvm.specialize import key_for

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


def _abi_of(scalar: str) -> str:
    return {"int": "i64", "float": "double", "bool": "i8"}[scalar]


def _watch(
    binding: NativeBinding,
    signature: NativeSignature,
    args: tuple[object, ...],
    policy,  # type: ignore[no-untyped-def]
    specializer,
    info: object,
    register: Callable[[int, tuple], bool] | None,
) -> None:
    """Watch argument shapes, and register a specialization once one repeats.

    Selecting the specialization is the generated wrapper's job; this only
    decides when one is worth compiling, and stops once it knows.
    """
    # Specialization is the JIT compiler's business; a prebuilt artifact
    # never passes a specializer, so the compiler import never happens there.
    from ppy_compiler.backend.llvm.specialize import key_for

    binding.observations += 1
    if binding.observations > policy.budget:
        binding.observing = False
        return

    key = key_for(signature, args, policy)
    if not key:
        binding.observing = False
        return

    seen = binding.key_counts.get(key, 0) + 1
    binding.key_counts[key] = seen
    if seen < policy.threshold:
        return

    specialization = specializer.specialize(info, key)  # type: ignore[arg-type]
    if specialization is None or not specialization.ok:
        binding.key_counts[key] = -(1 << 30)
        return
    if register is None or not register(specialization.address, key.pins()):
        binding.observing = False
        return

    binding.registered += 1
    if binding.specialization_count >= policy.maximum:
        binding.observing = False
