"""Python-ABI trampolines for natively lowered functions (spec 16.4, 25.3).

Runtime-only: a built artifact binds through this module with no compiler
installed. JIT specialization is an optional hook the compiler passes in,
and its machinery is imported only when it is actually used.
"""

from __future__ import annotations

import array
import contextlib
import ctypes
import sys
from collections.abc import Callable
from typing import Any

from . import _cpu
from ._record import field
from ._record import record as dataclass
from .abi import (
    SANITIZERS,
    STATUS_OK,
    STATUS_RAISED,
    STATUS_SANITIZER_BASE,
    TEXT,
    VARIADIC,
    NativeParam,
    NativeSignature,
)

__all__ = [
    "NativeBinding",
    "adopt",
    "as_method",
    "bind",
    "bind_globals",
    "keyed",
    "observation_wanted",
    "python_of",
    "value_class_types",
]

_I64_LOW = -(1 << 63)
_I64_HIGH = (1 << 63) - 1

_CTYPES = {
    "i64": ctypes.c_int64,
    "double": ctypes.c_double,
    "i8": ctypes.c_int8,
    # A collection's handle, which the boundary makes and reads.
    "i8*": ctypes.c_void_p,
}

_ELEMENT_CTYPES = {
    "int": ctypes.c_int64,
    "float": ctypes.c_double,
    "i8": ctypes.c_int8,
    "u8": ctypes.c_uint8,
}

#: `array` type codes matching the native element types. Unlike a ctypes slice
#: assignment, `array.array` rejects an out-of-range value instead of
#: truncating it, which is what keeps Python integer semantics intact.
_ELEMENT_CODES = {"int": "q", "float": "d", "i8": "b", "u8": "B"}

#: Buffer-protocol formats a borrowed buffer accepts per element type. `l` is
#: a signed long, which is the same width as `q` where this matters.
#: Buffer-protocol formats each element accepts. Signedness is part of the
#: element, not a detail: reading `array("B", [255])` as `i8` would answer
#: -1 where Python answers 255, so the wrong one is refused and the Python
#: body runs instead.
_ELEMENT_FORMATS = {
    "int": ("q", "l"),
    "float": ("d",),
    "i8": ("b",),
    "u8": ("B",),
}


class SanitizerFailure(RuntimeError):
    """A `--sanitize` check failed in native code; the message names the kind and the function."""


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


def value_class_types(
    signature: NativeSignature, fallback: Callable[..., object], *, globals_read: bool = False
) -> tuple | None:
    """The runtime classes a generated wrapper guards value parameters on.

    Resolved from the defining module, so a class the wrapper cannot see means
    no fast entry rather than a wrong one. A function passed module globals
    has one only where the caller reads them for it (`bind_globals`).
    """
    if signature.reads_globals and not globals_read:
        # The Python-level binding reads the globals; a C wrapper would not.
        return None
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


#: The C entry points that stand in a module's namespace themselves. No Python
#: frame is on their call path, so there is nothing to hang `__ppy_native__` on;
#: they are known by identity and kept here, as their modules keep them.
_ADOPTED: dict[int, tuple[object, NativeSignature, object]] = {}


def remember(
    entry: Callable[..., object], signature: NativeSignature, fallback: object = None
) -> None:
    """Record that `entry`, a generated C entry point, runs `signature` natively,
    standing for the Python function `fallback`."""
    _ADOPTED[id(entry)] = (entry, signature, fallback)


def python_of(function: object) -> object:
    """The Python function a native entry point stands for, whose source a tool
    like `ppy.grad` reads; `function` itself where it is one."""
    found = getattr(function, "__ppy_fallback__", None)
    if found is not None:
        return found
    known = _ADOPTED.get(id(function))
    if known is not None and known[0] is function and known[2] is not None:
        return known[2]
    return function


def as_method(entry: Callable[..., object], fallback: object, key: str) -> Callable[..., object]:
    """A generated C entry point that stands in a class body for a plain method,
    made an instance method: a builtin function does not bind, so `obj.f(x)`
    would not pass `obj`. A static method's entry is left as it is."""
    import types  # pylint: disable=import-outside-toplevel

    if "." not in key or not isinstance(fallback, types.FunctionType):
        return entry
    made = _INSTANCE_METHOD(entry)
    known = _ADOPTED.get(id(entry))
    if known is not None and known[0] is entry:
        remember(made, known[1], known[2])
    return made  # type: ignore[no-any-return]


_INSTANCE_METHOD = ctypes.pythonapi.PyInstanceMethod_New
_INSTANCE_METHOD.restype = ctypes.py_object
_INSTANCE_METHOD.argtypes = (ctypes.py_object,)


def signature_of(function: object) -> NativeSignature | None:
    """The native signature calling `function` runs under here, or None when the
    call is Python's: the wrapper's attribute, or the adopted C entry point."""
    signature = getattr(function, "__ppy_native__", None)
    if signature is not None:
        return signature
    known = _ADOPTED.get(id(function))
    if known is not None and known[0] is function:
        return known[1]
    # A method bound to its instance: the entry point is its function.
    inner = getattr(function, "__func__", None)
    if inner is not None and inner is not function:
        return signature_of(inner)
    return None


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
    remember(entry, signature, fallback)
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
    globals_read: bool = False,
) -> NativeBinding:
    """Build the Python-callable wrapper for one native function.

    A function that draws random numbers draws from `random`'s own generator:
    its state is saved before the call and put back before a fallback, so
    the Python rerun draws what the native call drew.
    """
    if not signature.draws:
        return _bind(
            signature,
            address,
            fallback,
            specializer=specializer,
            policy=policy,
            info=info,
            fast_entry=fast_entry,
            owner=owner,
            register=register,
            globals_read=globals_read,
        )
    generator = _shared_generator(owner)
    if generator is None:
        # Native draws would not be Python's: the Python definition runs.
        return NativeBinding(
            signature=signature, wrapper=fallback, fallback=fallback, fast_entry=None, owner=owner
        )
    binding = _bind(
        signature,
        address,
        generator.restoring(fallback),
        specializer=specializer,
        policy=policy,
        info=info,
        fast_entry=fast_entry,
        owner=owner,
        register=register,
        globals_read=globals_read,
    )
    if binding.wrapper is not binding.fallback:
        saved = generator.saving(binding.wrapper)
        _dress(saved, signature, fallback)
        saved.__ppy_native__ = signature  # type: ignore[attr-defined]
        saved.__ppy_fallback__ = fallback  # type: ignore[attr-defined]
        binding.wrapper = saved
    binding.fallback = fallback
    return binding


#: `RandomObject`'s state after the object header: `int index`, then 624 words.
_STATE_BYTES = 4 + 624 * 4


class _SharedGenerator:
    """`random._inst`'s Mersenne Twister, which native code draws from in place."""

    def __init__(self, address: int, reseeded: Callable[[], int]) -> None:
        import threading  # pylint: disable=import-outside-toplevel

        self.address = address
        self.reseeded = reseeded
        self.saved = threading.local()

    def _stack(self) -> list[bytes]:
        stack = getattr(self.saved, "stack", None)
        if stack is None:
            stack = self.saved.stack = []
        return stack

    def saving(self, wrapper: Callable[..., object]) -> Callable[..., object]:
        def saved(*args: object, **keywords: object) -> object:
            stack = self._stack()
            stack.append(ctypes.string_at(self.address, _STATE_BYTES))
            try:
                return wrapper(*args, **keywords)
            finally:
                stack.pop()
                if self.reseeded():
                    # `random.seed` forgets a pending `gauss` value too.
                    import random  # pylint: disable=import-outside-toplevel

                    random._inst.gauss_next = None  # type: ignore[attr-defined]

        return saved

    def restoring(self, fallback: Callable[..., object]) -> Callable[..., object]:
        def restored(*args: object, **keywords: object) -> object:
            stack = self._stack()
            if stack:
                ctypes.memmove(self.address, stack[-1], _STATE_BYTES)
                self.reseeded()
            return fallback(*args, **keywords)

        restored.__wrapped__ = fallback  # type: ignore[attr-defined]
        return restored


_generators: dict[int, _SharedGenerator | None] = {}


def attach_random(wrappers: object, owner: object = None) -> bool:
    """Hand a generated wrapper module `random._inst`'s state and the runtime's
    `ppy_random_reseeded`, for the wrappers of functions that draw to save and
    put back the state in C (`wrapper._DRAWS`); whether it took them. Asked
    once per module."""
    hand = getattr(wrappers, "ppy_random", None)
    if hand is None:
        return False
    attached = getattr(wrappers, "__ppy_random__", None)
    if attached is not None:
        return bool(attached)
    taken = False
    generator = _shared_generator(owner)
    if generator is not None:
        import random  # pylint: disable=import-outside-toplevel

        try:
            reseeded = ctypes.cast(generator.reseeded, ctypes.c_void_p).value or 0
            taken = bool(hand(generator.address, reseeded, random._inst))  # type: ignore[attr-defined]
        except (AttributeError, TypeError, ValueError, OverflowError):
            taken = False
    with contextlib.suppress(AttributeError, TypeError):
        wrappers.__ppy_random__ = taken  # type: ignore[attr-defined]
    return taken


def _shared_generator(owner: object) -> _SharedGenerator | None:
    """The runtime whose native code draws, bound to `random._inst`'s state:
    the library's own where it carries the runtime, else the one `ppy run`
    compiles. None where CPython's layout is not the one expected."""
    library = (
        owner if isinstance(owner, ctypes.CDLL) and hasattr(owner, "ppy_random_bind") else None
    )
    if library is None:
        from . import collection_boundary  # pylint: disable=import-outside-toplevel

        library = collection_boundary.runtime(None)
        if library is None:
            return None
    key = id(library)
    if key in _generators:
        return _generators[key]
    found = None
    address = _random_state_address()
    if address is not None:
        library.ppy_random_bind.argtypes = (ctypes.c_int64,)
        library.ppy_random_bind.restype = None
        library.ppy_random_reseeded.argtypes = ()
        library.ppy_random_reseeded.restype = ctypes.c_int64
        library.ppy_random_bind(address)
        found = _SharedGenerator(address, library.ppy_random_reseeded)
    _generators[key] = found
    return found


def _random_state_address() -> int | None:
    """Where `random._inst` keeps its index and state words, checked against
    `getstate()` so a build laid out otherwise never has its memory written."""
    import random  # pylint: disable=import-outside-toplevel

    if sys.implementation.name != "cpython":
        return None
    inst = random._inst  # type: ignore[attr-defined]
    _version, words, _gauss = inst.getstate()
    address = id(inst) + object.__basicsize__
    index = ctypes.c_int32.from_address(address).value
    state = (ctypes.c_uint32 * 624).from_address(address + 4)
    if index != words[-1] or tuple(state) != tuple(words[:-1]):
        return None
    return address


def _bind(
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
    globals_read: bool = False,
) -> NativeBinding:
    """`bind`, below the layer that saves and restores `random`'s state.

    `ctypes.CFUNCTYPE` releases the GIL around the foreign call, which is what
    a native region touching no Python objects is allowed to do (spec 16.6).

    When the function asked for it, repeated argument shapes are compiled into
    guarded specializations and selected here (spec 16.9).
    """
    missing = set(signature.cpu_features) - set(_cpu.features())
    if missing:
        # Compiled for a machine with more than this one has: the Python
        # definition is the function here, and says nothing about it.
        return NativeBinding(
            signature=signature, wrapper=fallback, fallback=fallback, fast_entry=None, owner=owner
        )
    if signature.reads_globals and not globals_read:
        bound = _bind_globals(signature, address, fallback, owner)
        assert bound is not None
        return bound
    effects = None
    if signature.effects:
        from .effects import effects_for, register_namespace

        effects = effects_for(owner)
        if effects is None:
            # No runtime to print through: the function is its Python definition.
            return NativeBinding(
                signature=signature,
                wrapper=fallback,
                fallback=fallback,
                fast_entry=None,
                owner=owner,
            )
        register_namespace(signature.qualname, _namespace(fallback))
        # The generated wrapper knows nothing of held output.
        fast_entry = None
    argument_types: list[type] = []
    for parameter in signature.parameters:
        if parameter.is_buffer:
            argument_types.append(ctypes.POINTER(_ELEMENT_CTYPES[parameter.element]))
            argument_types.append(ctypes.c_int64)
        elif parameter.is_text:
            argument_types.extend((ctypes.c_char_p, ctypes.c_int64))
        else:
            argument_types.extend(_CTYPES[atom] for atom in parameter.abi)

    # A string result is two out slots: the address of a UTF-8 copy the
    # native code made, and its length.
    text_result = signature.returns == (TEXT,)
    result_types = (
        [ctypes.c_void_p, ctypes.c_int64]
        if text_result
        else [_CTYPES[atom] for atom in signature.returns]
    )
    prototype = ctypes.CFUNCTYPE(
        ctypes.c_int32, *argument_types, *[ctypes.POINTER(t) for t in result_types]
    )
    native = prototype(address)

    # A `str` result comes back as text, like any string: only a collection
    # needs the crossing.
    text_only = text_result and signature.returned == "str"
    if signature.crosses_collections and not (
        text_only and not any(p.is_handle for p in signature.parameters)
    ):
        return _bind_collections(signature, native, result_types, fallback, owner, effects)

    namespace = _namespace(fallback)
    expanders = [
        _expander_for(p, (lambda: namespace) if namespace is not None else None)
        for p in signature.parameters
    ]
    finalizers = [_result_for(atom) for atom in signature.returns if atom != TEXT]
    if signature.future:
        from .aio import NativeFuture, runtime_for

        runtime = runtime_for(owner)
        if runtime is None:
            # No async runtime here: the coroutine is its Python definition.
            return NativeBinding(
                signature=signature,
                wrapper=fallback,
                fallback=fallback,
                fast_entry=None,
                owner=owner,
            )
        kind = signature.future
        finalizers = [lambda bits: NativeFuture(bits, kind, runtime, signature.qualname)]
        fast_entry = None
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

    arity = len(signature.parameters)
    if fast_entry is not None:
        # The generated wrapper does the parsing, the guards, the specialization
        # choice, the call, and the boxing in C; `NotImplemented` is its signal
        # that a guard failed. While the function is still learning which
        # argument shapes repeat, Python watches alongside.
        def fast_wrapper(*args: object, **keywords: object) -> object:
            if keywords or len(args) != arity:
                # Python binds keywords and defaults; the native code takes
                # its arguments in order.
                return _keyword_call(fallback, fast_wrapper, arity, args, keywords)
            if binding.observing:
                _watch(binding, signature, args, policy, specializer, info, register)
            result = fast_entry(*args)
            if result is NotImplemented:
                binding.fallbacks += 1
                return fallback(*args)
            binding.calls += 1
            return result

        _dress(fast_wrapper, signature, fallback)
        fast_wrapper.__ppy_native__ = signature  # type: ignore[attr-defined]
        fast_wrapper.__ppy_fallback__ = fallback  # type: ignore[attr-defined]
        binding.wrapper = fast_wrapper
        return binding

    def wrapper(*args: object, **keywords: object) -> object:
        if keywords or len(args) != len(expanders):
            return _keyword_call(fallback, wrapper, len(expanders), args, keywords)
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
        if effects is not None:
            outer = effects.enter()
            status = target(*atoms, *[ctypes.byref(slot) for slot in slots])
            settled = _settled(effects, status, outer, signature, owner, target)
            if isinstance(settled, BaseException):
                raise settled
            if not settled:
                binding.fallbacks += 1
                return fallback(*args)
            binding.calls += 1
            try:
                return _answer(slots)
            finally:
                effects.commit()
        status = target(*atoms, *[ctypes.byref(slot) for slot in slots])
        if status != STATUS_OK:
            if status == STATUS_RAISED:
                _let_go_of_raised(owner, target)
            if status >= STATUS_SANITIZER_BASE:
                kind = SANITIZERS[min(status - STATUS_SANITIZER_BASE, len(SANITIZERS) - 1)]
                raise SanitizerFailure(
                    f"sanitizer: a {kind} check failed in `{signature.qualname}`"
                )
            binding.fallbacks += 1
            return fallback(*args)
        binding.calls += 1
        binding.specialized_calls += int(entry is not None)
        return _answer(slots)

    nothing = signature.returns_none

    def _answer(slots: list) -> object:  # type: ignore[type-arg]
        if nothing:
            return None
        if text_result:
            return _text_result(slots[0].value, slots[1].value)
        if returns_tuple:
            return tuple(
                finish(slot.value) for finish, slot in zip(finalizers, slots, strict=False)
            )
        return finalizers[0](slots[0].value)

    _dress(wrapper, signature, fallback)
    wrapper.__ppy_native__ = signature  # type: ignore[attr-defined]
    wrapper.__ppy_fallback__ = fallback  # type: ignore[attr-defined]
    binding.wrapper = wrapper
    return binding


#: Each Python function's signature, as `_keyword_call` binds by it; None
#: where the native entry cannot take what it binds.
_SIGNATURES: dict[int, tuple[object, Any]] = {}


def _binder(fallback: Callable[..., object], count: int) -> Any:
    """The `inspect.Signature` a call to `fallback` binds by, where its
    parameters are the native entry's `count` ones in order: no `*args`, no
    `**kwargs`. None otherwise."""
    found = _SIGNATURES.get(id(fallback))
    if found is not None and found[0] is fallback:
        return found[1]
    import inspect  # pylint: disable=import-outside-toplevel

    try:
        signature = inspect.signature(fallback)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        kinds = [p.kind for p in signature.parameters.values()]
        variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        if len(kinds) != count or any(kind in variadic for kind in kinds):
            signature = None
    _SIGNATURES[id(fallback)] = (fallback, signature)
    return signature


def _keyword_call(
    fallback: Callable[..., object],
    entry: Callable[..., object],
    count: int,
    args: tuple,
    keywords: dict[str, object],
) -> object:
    """A call Python spelled with keywords, or with defaults left out: bound
    as Python binds it, then made in order through the native entry. A call
    that does not bind is the Python function's, which raises CPython's
    `TypeError` for it."""
    signature = _binder(fallback, count)
    if signature is None:
        return fallback(*args, **keywords)
    try:
        bound = signature.bind(*args, **keywords)
    except TypeError:
        return fallback(*args, **keywords)
    bound.apply_defaults()
    return entry(*bound.arguments.values())


def keyed(
    fallback: Callable[..., object], entry: Callable[..., object], count: int
) -> Callable[..., object]:
    """What a generated C entry point calls for a call spelled with keywords or
    with defaults left out: `_keyword_call`, which binds it as Python does
    and calls `entry` in order."""

    def call(*args: object, **keywords: object) -> object:
        return _keyword_call(fallback, entry, count, args, keywords)

    return call


def _namespace(function: object) -> dict | None:
    """The globals a Python function reads: its module's, or, for the one
    `_bind_globals` makes, those of the function it stands for. The wrapper
    that restores `random`'s state before a fallback is looked through."""
    found = getattr(function, "__ppy_globals__", None)
    if found is not None:
        return found  # type: ignore[no-any-return]
    wrapped = getattr(function, "__wrapped__", None)
    if wrapped is not None:
        return _namespace(wrapped)
    return getattr(function, "__globals__", None)


def bind_globals(  # type: ignore[no-untyped-def]
    signature: NativeSignature,
    address: int,
    fallback: Callable[..., object],
    owner,
    entry_for: Callable[[Callable[..., object]], Callable[..., object] | None],
) -> NativeBinding | None:
    """`_bind_globals` over a generated C entry point: `entry_for(spelled)` binds
    the entry, which takes the globals after Python's arguments, with
    `spelled` as its fallback. None where it binds none."""
    return _bind_globals(signature, address, fallback, owner, entry_for)


def _bind_globals(  # type: ignore[no-untyped-def]
    signature: NativeSignature,
    address: int,
    fallback: Callable[..., object],
    owner,
    entry_for: Callable[[Callable[..., object]], Callable[..., object] | None] | None = None,
) -> NativeBinding | None:
    """A function passed the settled module globals it reads (`NativeParam.
    source`). Python's caller spells the other arguments; the wrapper reads
    each global from its module at the call and passes it after them, where
    the native entry takes it. A global the module no longer holds, or holds
    as something the function does not take, runs the Python body.
    """
    count = sum(1 for p in signature.parameters if not p.source)
    own = signature.qualname.rpartition(".")[0]
    namespace = _namespace(fallback)
    places: list[tuple[str, str]] = []
    for parameter in signature.parameters[count:]:
        module, _, name = parameter.source.rpartition(":")
        places.append((module, name))
    # `*args`: the positions after the named ones, which the native entry
    # takes as one list, before any global.
    variadic = any(p.source == VARIADIC for p in signature.parameters)

    def spelled(*args: object, **keywords: object) -> object:
        # The Python function takes the arguments Python spelled; the globals
        # the native entry takes after them are left off.
        if variadic:
            return fallback(*args[:count], *args[count], **keywords)  # type: ignore[misc]
        return fallback(*args[:count], **keywords)

    spelled.__ppy_globals__ = namespace  # type: ignore[attr-defined]
    if entry_for is not None:
        entry = entry_for(spelled)
        if entry is None:
            return None
        remember(entry, signature)
        inner = NativeBinding(
            signature=signature, wrapper=entry, fallback=spelled, fast_entry=entry, owner=owner
        )
    else:
        inner = _bind(signature, address, spelled, owner=owner, globals_read=True)
    if inner.wrapper is inner.fallback:
        # No native entry here after all: the Python function is the function.
        inner.wrapper = inner.fallback = fallback
        return inner
    native = inner.wrapper

    def read(module: str, name: str) -> object:
        if module.endswith(".<locals>"):
            # A variable of the function the Python function was defined in:
            # what its cell holds now. An empty cell raises `ValueError`, and
            # the Python body raises CPython's `NameError` for it.
            function = fallback
            while getattr(function, "__wrapped__", None) is not None:
                function = function.__wrapped__  # type: ignore[attr-defined]
            code = getattr(function, "__code__", None)
            cells = getattr(function, "__closure__", None) or ()
            if code is None or name not in code.co_freevars:
                raise KeyError(name)
            return cells[code.co_freevars.index(name)].cell_contents
        if module == own and namespace is not None:
            return namespace[name]
        return sys.modules[module].__dict__[name]

    def wrapper(*args: object, **keywords: object) -> object:
        if variadic and (keywords or len(args) < count):
            return fallback(*args, **keywords)
        if not variadic and (keywords or len(args) != count):
            return _keyword_call(fallback, wrapper, count, args, keywords)
        try:
            values = [
                list(args[count:]) if module == "" and name == VARIADIC else read(module, name)
                for module, name in places
            ]
        except (KeyError, ValueError):
            inner.fallbacks += 1
            return fallback(*args)
        return native(*args[:count], *values)

    _dress(wrapper, signature, fallback)
    wrapper.__ppy_native__ = signature  # type: ignore[attr-defined]
    wrapper.__ppy_fallback__ = fallback  # type: ignore[attr-defined]
    inner.wrapper = wrapper
    inner.fallback = fallback
    return inner


def _settled(effects, status, outer, signature, owner, target, let_go=None) -> bool | BaseException:  # type: ignore[no-untyped-def]
    """Whether a call with effects answered: what it printed is written out once
    its result is read (`Effects.commit`); where it fell back, dropped. What it
    raised after an effect it cannot take back comes back, for the caller to raise.
    `let_go` sweeps what a raising call left; a crossing passes one that waits
    until it has let go of its own handles."""
    if status >= STATUS_SANITIZER_BASE:
        effects.abandon(outer)
        kind = SANITIZERS[min(status - STATUS_SANITIZER_BASE, len(SANITIZERS) - 1)]
        return SanitizerFailure(f"sanitizer: a {kind} check failed in `{signature.qualname}`")
    return effects.leave(  # type: ignore[no-any-return]
        status,
        outer,
        signature.qualname,
        let_go if let_go is not None else lambda: _let_go_of_raised(owner, target),
    )


def _dress(wrapper, signature, fallback) -> None:  # type: ignore[no-untyped-def]
    """Give a wrapper the function's own name, docstring, and module, so
    `help`, `doctest`, and `__name__` see what the program wrote."""
    name = signature.qualname.rpartition(".")[2]
    wrapper.__name__ = getattr(fallback, "__name__", name)
    wrapper.__qualname__ = getattr(fallback, "__qualname__", signature.qualname)
    wrapper.__doc__ = getattr(fallback, "__doc__", None)
    wrapper.__module__ = getattr(fallback, "__module__", wrapper.__module__)


def _text_result(address: int | None, length: int) -> str:
    """A string the native code returned: its UTF-8 copy read, then freed."""
    try:
        return ctypes.string_at(address or 0, length).decode("utf-8") if length else ""
    finally:
        _LIBC.free(ctypes.c_void_p(address))


_LIBC = ctypes.CDLL(None)
_LIBC.free.argtypes = (ctypes.c_void_p,)
_LIBC.free.restype = None


def _bind_collections(  # type: ignore[no-untyped-def]
    signature: NativeSignature, native, result_types: list, fallback, owner, effects=None
) -> NativeBinding:
    """The boundary of a function a collection crosses: each argument copied into
    native memory, the result copied out, and a written argument copied back.

    Anything that does not match its declared type, or a runtime that cannot be
    loaded, runs the Python body, as a failed guard does.
    """
    from . import collection_boundary as crossing  # pylint: disable=import-outside-toplevel

    unbound = NativeBinding(
        signature=signature, wrapper=fallback, fallback=fallback, fast_entry=None, owner=owner
    )
    library = owner if isinstance(owner, ctypes.CDLL) else None
    rt = crossing.runtime(library)
    if rt is None:
        return unbound
    described = {c.qualname: c for c in signature.classes}
    classes = crossing.Classes(signature.classes, _class_finder(fallback)) if described else None
    specs = [
        crossing.parse(p.element + ("?" if p.nullable else ""), described) if p.is_handle else None
        for p in signature.parameters
    ]
    parameters = signature.parameters
    if any(p.is_handle and spec is None for p, spec in zip(parameters, specs, strict=True)):
        return unbound
    nothing = signature.returned == crossing.RETURNS_NOTHING
    returned = (
        crossing.parse(signature.returned, described)
        if signature.returned and not nothing
        else None
    )
    if (
        signature.returned
        and not nothing
        and (returned is None or returned.kind == "random.Random")
    ):
        return unbound
    # A value class among the arguments is found in the function's module, as
    # `_bind` finds it: without the namespace, every call would run the
    # Python body.
    namespace = _namespace(fallback)
    find = (lambda: namespace) if namespace is not None else None
    expanders = [None if p.is_handle else _expander_for(p, find) for p in signature.parameters]
    written = [p.is_handle and p.written for p in signature.parameters]
    binding = NativeBinding(
        signature=signature, wrapper=lambda *a: None, fallback=fallback, owner=owner
    )

    def wrapper(*args: object, **keywords: object) -> object:
        if keywords or len(args) != len(expanders):
            return _keyword_call(fallback, wrapper, len(expanders), args, keywords)
        boundary = crossing.Boundary(rt, classes)
        raised: list[bool] = []
        try:
            if effects is not None:
                answered, answer = _cross_with_effects(boundary, args, raised)
            else:
                answered, answer = _cross(boundary, args, raised)
        finally:
            # Every handle the crossing made is let go of before any Python
            # runs: a fallback that calls native code again must not find
            # them on the thread's list, which a failed call's sweep frees.
            boundary.close()
            if raised:
                # And only then is what the raising call left swept: the sweep
                # frees every handle on the list, the crossing's own included.
                _let_go_of_raised(owner, native)
        if answered:
            binding.calls += 1
            return answer
        # A generator an argument lent is put back as it was before Python reruns.
        boundary.restore()
        binding.fallbacks += 1
        return fallback(*args)

    def _cross(boundary, args: tuple, raised: list) -> tuple[bool, object]:  # type: ignore[no-untyped-def,type-arg]
        """The native call and its conversions: whether it answered, and what."""
        atoms: list[object] = []
        borrowed: list[object] = []
        try:
            for expand, spec, value in zip(expanders, specs, args, strict=True):
                if spec is not None:
                    atoms.append(boundary.argument(value, spec))
                else:
                    assert expand is not None
                    expand(value, atoms, borrowed)
        except (GuardFailed, crossing.Refused):
            return False, None
        slots = [result_type() for result_type in result_types]
        status = native(*atoms, *[ctypes.byref(slot) for slot in slots])
        if status != STATUS_OK:
            if status == STATUS_RAISED:
                raised.append(True)
            if status >= STATUS_SANITIZER_BASE:
                kind = SANITIZERS[min(status - STATUS_SANITIZER_BASE, len(SANITIZERS) - 1)]
                raise SanitizerFailure(
                    f"sanitizer: a {kind} check failed in `{signature.qualname}`"
                )
            return False, None
        return True, _read(boundary, args, slots)

    def _cross_with_effects(boundary, args: tuple, raised: list) -> tuple[bool, object]:  # type: ignore[no-untyped-def,type-arg]
        """As `_cross`, for a function with effects (`ppy_runtime/effects.py`)."""
        atoms: list[object] = []
        borrowed: list[object] = []
        try:
            for expand, spec, value in zip(expanders, specs, args, strict=True):
                if spec is not None:
                    atoms.append(boundary.argument(value, spec))
                else:
                    assert expand is not None
                    expand(value, atoms, borrowed)
        except (GuardFailed, crossing.Refused):
            return False, None
        slots = [result_type() for result_type in result_types]
        outer = effects.enter()
        status = native(*atoms, *[ctypes.byref(slot) for slot in slots])
        settled = _settled(
            effects, status, outer, signature, owner, native, lambda: raised.append(True)
        )
        if isinstance(settled, BaseException):
            raise settled
        if not settled:
            return False, None
        try:
            return True, _read(boundary, args, slots)
        finally:
            effects.commit()

    def _read(boundary, args: tuple, slots: list) -> object:  # type: ignore[no-untyped-def,type-arg]
        """What an answered call wrote back and gave back."""
        boundary.sync(
            [
                (value, spec)
                for value, spec, wrote in zip(args, specs, written, strict=True)
                if wrote and spec is not None
            ]
        )
        if returned is not None:
            return boundary.result(slots[0].value, returned)
        if nothing or not slots:
            return None
        if signature.returns == (TEXT,):
            return _text_result(slots[0].value, slots[1].value)
        if len(slots) > 1:
            return tuple(
                _result_for(atom)(slot.value)
                for atom, slot in zip(signature.returns, slots, strict=True)
            )
        return _result_for(signature.returns[0])(slots[0].value)

    _dress(wrapper, signature, fallback)
    wrapper.__ppy_native__ = signature  # type: ignore[attr-defined]
    wrapper.__ppy_fallback__ = fallback  # type: ignore[attr-defined]
    binding.wrapper = wrapper
    return binding


def _class_finder(fallback: Callable[..., object]) -> Callable[[object], object]:
    """How the boundary finds a described class: in the function's own module's
    namespace, where the program defines it, else in its module."""
    namespace = _namespace(fallback)

    def find(described):  # type: ignore[no-untyped-def]
        if namespace is not None:
            found = namespace.get(described.name)
            if isinstance(found, type) and found.__qualname__ == described.name:
                return found
        module = sys.modules.get(described.module)
        return getattr(module, described.name, None) if module is not None else None

    return find


def _expander_for(
    parameter: NativeParam, namespace: Callable[[], dict] | None = None
) -> Callable[[object, list, list], None]:
    """Build the guard-and-convert step for one source-level parameter."""
    if parameter.is_text:

        def expand_text(value: object, atoms: list, borrowed: list) -> None:
            """A `str` as its UTF-8 bytes; one with a lone surrogate has none."""
            if type(value) is not str:
                raise GuardFailed
            try:
                data = value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise GuardFailed from exc
            borrowed.append(data)
            atoms.append(data)
            atoms.append(len(data))

        return expand_text

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

        exact_floats = parameter.exact and parameter.element in {"float", "f64", "f32"}

        def expand_buffer(value: object, atoms: list, borrowed: list) -> None:
            if type(value) is not list:
                raise GuardFailed
            if exact_floats and not all(type(item) is float for item in value):
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
        field_guards = [
            (attr, _scalar_guard(_abi_of(scalar), parameter.exact))
            for attr, scalar in parameter.fields
        ]
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
        element_guards = [_scalar_guard(atom, parameter.exact) for atom in parameter.abi]

        def expand_tuple(value: object, atoms: list, borrowed: list) -> None:
            if type(value) is not tuple or len(value) != len(element_guards):
                raise GuardFailed
            for item, convert in zip(value, element_guards, strict=False):
                atoms.append(convert(item))

        return expand_tuple

    abi = parameter.abi[0]
    if abi == "i64":

        def expand_int(value: object, atoms: list, borrowed: list) -> None:
            # A `bool` stays a `bool` in Python where native code would make it
            # 1 or 0; the compiled wrapper refuses it too.
            if type(value) is not int or not _I64_LOW <= value <= _I64_HIGH:
                raise GuardFailed
            atoms.append(value)

        return expand_int

    if abi == "double":
        exact = parameter.exact

        def expand_float(value: object, atoms: list, borrowed: list) -> None:
            if type(value) is float:
                atoms.append(value)
                return
            if not exact and type(value) is int and -_EXACT_INT <= value <= _EXACT_INT:
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


#: The largest int a double holds exactly, and every int below it.
_EXACT_INT = 1 << 53


def _scalar_guard(abi: str, exact: bool = False) -> Callable[[object], object]:
    """Guard and convert one scalar, raising `GuardFailed` when it does not fit;
    an `exact` float takes only a `float`."""
    if abi == "i64":

        def as_int(value: object) -> object:
            if type(value) is not int or not _I64_LOW <= value <= _I64_HIGH:
                raise GuardFailed
            return value

        return as_int
    if abi == "double":

        def as_float(value: object) -> object:
            if type(value) is float:
                return value
            if not exact and type(value) is int and -_EXACT_INT <= value <= _EXACT_INT:
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
    policy: Any,
    specializer: Any,
    info: object,
    prototype: Any,
) -> Any:
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
    policy: Any,
    specializer: Any,
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


def _let_go_of_raised(owner: object, native: object) -> None:
    """A native call ended on an exception nothing native caught: Python runs the
    call again and raises it. What the native call made, the exception with
    it, is garbage now; the collections runtime in the native code's own
    library frees all of it at once."""
    sweep = getattr(owner, "ppy_coll_sweep", None) if isinstance(owner, ctypes.CDLL) else None
    if sweep is None:
        sweep = _sweep_beside(native)
    if sweep is not None:
        sweep()


def _sweep_beside(native: object):  # type: ignore[no-untyped-def]
    """`ppy_coll_sweep` of the library `native` was loaded from, if any."""

    class _Found(ctypes.Structure):
        _fields_ = [
            ("dli_fname", ctypes.c_char_p),
            ("dli_fbase", ctypes.c_void_p),
            ("dli_sname", ctypes.c_char_p),
            ("dli_saddr", ctypes.c_void_p),
        ]

    try:
        libc = ctypes.CDLL(None)
        found = _Found()
        address = ctypes.cast(native, ctypes.c_void_p)  # type: ignore[arg-type]
        if not libc.dladdr(address, ctypes.byref(found)) or not found.dli_fname:
            return None
        return getattr(ctypes.CDLL(found.dli_fname.decode()), "ppy_coll_sweep", None)
    except (OSError, AttributeError, TypeError):
        return None
