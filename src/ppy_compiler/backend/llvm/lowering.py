"""The native ABI and the eligibility rules of the LLVM backend (spec 16).

The lowering itself is the IR road -- `ppy_compiler.lowering` makes the
canonical IR, `backend/llvm/from_ir` makes LLVM from it -- and what is
here is what both agree on before any IR exists: which functions are
native at all, what their native signatures are, and whether the Python
boundary is worth crossing for them.

Only effect-free functions are lowered. That restriction is what makes the
guard-and-fall-back model sound: when a native fast path bails out, the
original Python implementation can be re-executed with identical observable
behavior (spec 16.8).

Floating-point ordering is strict by default: no reassociation, no
contraction, no reciprocal substitution. `@ppy.fastmath` is what permits those,
and nothing else does -- optimization level alone never will (spec 3.4, 12.5).

A function may take scalars, fixed-size tuples, value classes whose fields
are all scalars, and homogeneous `list[int]` / `list[float]` arguments. A fixed
tuple flattens into scalar SSA
values rather than an allocated object, and a list is unboxed into a borrowed
native buffer at the Python boundary -- transparent because a lowered function
may not mutate it (spec 13.2, 13.3, 13.5).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ppy_runtime.abi import STATUS_FALLBACK, STATUS_OK, NativeParam, NativeSignature

from ...analysis import types as T
from ...analysis.checker import FunctionAnalysis
from ...analysis.effects import Effect
from ...analysis.symbols import FunctionInfo

__all__ = [
    "STATUS_FALLBACK",
    "STATUS_OK",
    "LoweredFunction",
    "NativeParam",
    "NativeSignature",
    "Unsupported",
    "eligible",
]

#: The native entry point returns a status; a non-zero status means the caller
#: must re-run the Python implementation (spec 16.9).

_SCALARS = {"int", "float", "bool", "i8", "u8"}

#: Element types a native buffer parameter may carry. `i8`/`u8` are one byte
#: each, which is what text and packed data want; everything else is a word.
_BUFFER_ELEMENTS = {"int", "float", "i8", "u8"}

#: Buffer elements narrower than a machine word, and whether they are signed.
_NARROW = {"i8": True, "u8": False}


def _read_as(element: str) -> str:
    """What a read of that element hands out: a byte comes out an `int`."""
    return "int" if element in _NARROW else element


#: A tuple wider than this stays boxed rather than expanding the ABI.
_MAX_TUPLE_WIDTH = 8

#: The transformations `@ppy.fastmath` permits, and nothing permits otherwise.
FASTMATH_FLAGS = ("fast",)

#: A class with more fields than this stays boxed rather than expanding the ABI.
_MAX_CLASS_WIDTH = 8

_MATH_INTRINSICS = {
    "sqrt": "llvm.sqrt.f64",
    "sin": "llvm.sin.f64",
    "cos": "llvm.cos.f64",
    "exp": "llvm.exp.f64",
    "log": "llvm.log.f64",
    "log2": "llvm.log2.f64",
    "log10": "llvm.log10.f64",
    "fabs": "llvm.fabs.f64",
    "floor": "llvm.floor.f64",
    "ceil": "llvm.ceil.f64",
    "pow": "llvm.pow.f64",
    "trunc": "llvm.trunc.f64",
}

#: How the obligation language spells each guarded operator.
_OPERATOR_SPELLING = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}


def _declared_bounds(interval) -> tuple[int | None, int | None]:  # type: ignore[no-untyped-def]
    """The bounds of a range as the machine can check them: a bound past the
    word is no bound."""
    if interval is None:
        return None, None
    low = interval.low if interval.low is not None and interval.low >= -(1 << 63) else None
    high = interval.high if interval.high is not None and interval.high < (1 << 63) else None
    return low, high


class Unsupported(Exception):
    """Raised when a construct has no native lowering."""


@dataclass(slots=True)
class LoweredFunction:
    info: FunctionInfo
    signature: NativeSignature
    reason: str = ""
    #: Whether the Python/native boundary is worth crossing for this
    #: function. Native callers use the direct symbol either way.
    exposed: bool = True
    exposure_reason: str = ""


@dataclass(slots=True)
class LoweringResult:
    ir: str
    #: The module's own IR as text (`.ppyir`), for the linker.
    ppyir: str = ""
    functions: dict[str, LoweredFunction] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)
    #: Per function, the arithmetic whose overflow guard a proof left out.
    proved: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Shared libraries the C bindings need, by name.
    libraries: tuple[str, ...] = ()
    #: Public C symbols: name -> the qualname of the function behind it.
    exports: dict[str, str] = field(default_factory=dict)
    #: What the lowering and the passes said about the code, as remarks.
    remarks: tuple[str, ...] = ()


def _scalar_name(t: T.Type) -> str | None:
    base = T.strip_literal(t)
    if isinstance(base, T.Instance) and base.name in _SCALARS:
        return base.name
    return None


def _buffer_element(t: T.Type) -> tuple[str, str] | None:
    """The kind and element scalar of a buffer parameter, if it is one."""
    base = T.strip_literal(t)
    if not isinstance(base, T.Instance) or len(base.args) != 1:
        return None
    if base.name not in {"list", "Sequence", "Buffer", "memoryview", "array"}:
        return None
    element = _scalar_name(base.args[0])
    if element not in _BUFFER_ELEMENTS:
        return None
    # A `Sequence` promises only reading, which is what a copied-in buffer
    # does; anything the caller passes is unpacked the same way.
    kinds = {"list": "list", "Sequence": "sequence"}
    return kinds.get(base.name, "view"), element


def _tuple_elements(t: T.Type) -> tuple[str, ...] | None:
    """The scalar element kinds of a fixed-size tuple, if it has any."""
    base = T.strip_literal(t)
    if not isinstance(base, T.Tuple_) or base.homogeneous:
        return None
    if not base.items or len(base.items) > _MAX_TUPLE_WIDTH:
        return None
    kinds = [_scalar_name(item) for item in base.items]
    if any(kind is None for kind in kinds):
        return None
    return tuple(kind for kind in kinds if kind is not None)


#: Layouts of the value classes this module may flatten, by qualified name.
ClassLayouts = dict


def _class_fields(
    t: T.Type, layouts: ClassLayouts
) -> tuple[str, tuple[tuple[str, str], ...]] | None:
    """The flattened field layout of a value-class parameter, if it has one."""
    base = T.strip_literal(t)
    if not isinstance(base, T.Instance):
        return None
    fields = layouts.get(base.name)
    if not fields or len(fields) > _MAX_CLASS_WIDTH:
        return None
    return base.name, tuple(fields)


def _pointer_element(t: T.Type) -> tuple[str, str] | None:
    """(kind, element) of a `ppy.native.ptr[T]` / `const_ptr[T]`, if `t` is one."""
    base = T.strip_literal(t)
    if not isinstance(base, T.Instance) or len(base.args) != 1:
        return None
    if base.name not in {"ppy.native.ptr", "ppy.native.const_ptr"}:
        return None
    element = _scalar_name(base.args[0])
    if element is None:
        return None
    return ("ptr" if base.name == "ppy.native.ptr" else "const_ptr"), element


def _native_param(name: str, t: T.Type, layouts: ClassLayouts | None = None) -> NativeParam | None:
    scalar = _scalar_name(t)
    if scalar is not None:
        return NativeParam(name, scalar)
    pointer = _pointer_element(t)
    if pointer is not None:
        return NativeParam(name, pointer[0], pointer[1])
    buffer = _buffer_element(t)
    if buffer is not None:
        kind, element = buffer
        return NativeParam(name, kind, element)
    elements = _tuple_elements(t)
    if elements is not None:
        return NativeParam(name, "tuple", elements=elements)
    described = _class_fields(t, layouts or {})
    if described is not None:
        class_name, fields = described
        return NativeParam(name, "object", fields=fields, class_name=class_name)
    return None


def _return_atoms(t: T.Type) -> tuple[str, ...] | None:
    scalar = _scalar_name(t)
    if scalar is not None:
        return (scalar,)
    return _tuple_elements(t)


def eligible(
    info: FunctionInfo,
    analysis: FunctionAnalysis,
    layouts: ClassLayouts | None = None,
    *,
    allow_io: bool = False,
    allow_launch: bool = False,
    allow_async: bool = False,
) -> tuple[bool, str]:
    """Can this function be lowered to a native scalar entry point?

    `allow_io` is the standalone build's dispensation: printing goes through
    native shims there, so the IO effect alone is not disqualifying.
    `allow_launch` is the GPU source backends': a kernel launch is written
    as one, where the CPU backends have no launch runtime yet.
    """
    if info.is_generator:
        return False, "generators use the boxed runtime"
    if info.is_async and not allow_async:
        return False, "a coroutine runs natively only where the async runtime does"
    written = analysis.mutated_params | analysis.delegated_writes
    for name in sorted(written):
        declared = next((p.type for p in info.params if p.name == name), None)
        described = _buffer_element(declared) if declared is not None else None
        # Writing through a borrowed buffer is visible to the caller, which is
        # what borrowing means. Anything else would lose the write.
        if described is None or described[0] != "view":
            return False, f"mutates `{name}`, which is not a borrowed buffer"
    if analysis.foreign_writes:
        return False, "writes through a target the compiler cannot identify"

    violations = set(analysis.effects.violations())
    # Native memory is what native code is for: a read or a write through a
    # pointer lands where the program aimed it and needs no interpreter.
    violations.discard(Effect.READ_MEMORY)
    violations.discard(Effect.WRITE_MEMORY)
    # Atomics, synchronization, and threads are native code's own business:
    # `ppy.atomic` and `ppy.concurrent` lower to instructions and pthreads.
    violations.discard(Effect.ATOMIC)
    violations.discard(Effect.SYNC)
    violations.discard(Effect.THREAD)
    if allow_io:
        violations.discard(Effect.IO)
    if allow_launch:
        violations.discard(Effect.GPU_LAUNCH)
    if allow_async:
        # A sleep and a socket are the async runtime's own business.
        violations.discard(Effect.TIME)
        violations.discard(Effect.NETWORK)
    if written or analysis.writes_only_allocations:
        # Those writes land in memory the caller lent us, and nowhere else --
        # whether this function performed them or a callee it handed the
        # buffer to did. A write to something the function allocated itself
        # is the same story with a shorter lifetime, and handing that memory
        # on does not change it: passing a buffer to a native callee makes
        # the write visible to that callee, not to CPython.
        violations.discard(Effect.WRITE_OBJECT)
    if violations:
        listed = ", ".join(sorted(str(e) for e in violations))
        return False, f"has effects that must run on CPython: {listed}"
    for param in info.params:
        if param.kind in {"var_positional", "var_keyword"}:
            return False, "variadic parameters have no native ABI"
        if _native_param(param.name, param.type, layouts) is None:
            return False, f"parameter `{param.name}` is `{param.type}`, which has no native ABI"
    if _return_atoms(info.ret) is None and not _returns_none(info.ret):
        return False, f"returns `{info.ret}`, which has no native ABI"
    return True, ""


def _returns_none(t: T.Type) -> bool:
    return t == T.NONE


#: Directives that are an explicit request for the native boundary.
_EXPOSURE_DIRECTIVES = ("native", "jit", "specialize", "parallel")

#: Below this much straight-line work, the ~0.2 us Python/native crossing
#: costs more than the native body saves over CPython.
_EXPOSURE_WORK = 16


def can_lower_native(
    info: FunctionInfo, analysis: FunctionAnalysis, layouts: ClassLayouts | None = None
) -> tuple[bool, str]:
    """Can PPY generate correct native code for this function?"""
    return eligible(info, analysis, layouts)


def should_lower_native(info: FunctionInfo, analysis: FunctionAnalysis) -> tuple[bool, str]:
    """Is native execution through the Python boundary expected to be faster?

    Eligibility and profitability are different questions: a two-instruction
    add lowers perfectly and still loses to CPython once the boundary's
    guards and conversions are paid. Native-to-native calls never cross the
    boundary, so a helper this keeps off it is still called directly by any
    native caller.
    """
    del analysis
    if info.is_async:
        # The future is the boundary value; a coroutine is called to be run.
        return True, "a coroutine's future crosses the boundary"
    for api in ("cuda", "hip"):
        if any(info.directive(f"{api}.{kind}") is not None for kind in ("kernel", "device")):
            # Device code has no CPU form to bind; a launch runs it, and under
            # CPython its own definition is the reference.
            return False, "device code runs where it is launched"
    if _returns_none(info.ret):
        # The boundary hands back a value; a function with none to hand
        # back is native code's to call -- a thread's body, a helper.
        return False, "returns nothing, which has no Python boundary"
    for param in info.params:
        native = _native_param(param.name, param.type)
        if native is not None and native.is_pointer:
            # A machine address has no Python object to come from, whatever
            # the directives ask: the function is native code's to call.
            return False, "takes a native pointer, which has no Python boundary"
    for name in _EXPOSURE_DIRECTIVES:
        if info.directive(name) is not None:
            return True, f"@ppy.{name} asks for the boundary"
    for param in info.params:
        native = _native_param(param.name, param.type)
        if native is not None and native.is_buffer:
            # Buffer work scales with the data; the crossing is flat.
            return True, "takes a buffer"
    for child in ast.walk(info.node):
        if isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
            return True, "contains a loop"
    work = sum(
        isinstance(child, (ast.BinOp, ast.Compare, ast.BoolOp, ast.Call, ast.Subscript))
        for child in ast.walk(info.node)
    )
    if work >= _EXPOSURE_WORK:
        return True, f"straight-line work ({work} operations)"
    return False, "the boundary crossing costs more than the body saves"


#: What a standalone build can allocate for itself, and the element it holds.
#: Both are eight bytes wide, which is what the native ABI passes.
_ALLOCATIONS = {
    "ppy.buffer[int]": ("int", False),
    "ppy.buffer[float]": ("float", False),
    "ppy.buffer[ppy.i8]": ("i8", False),
    "ppy.buffer[ppy.u8]": ("u8", False),
    "ppy.input[Buffer[int]]": ("int", True),
}


def _default_triple() -> str:
    try:
        import llvmlite.binding as llvm

        return llvm.get_process_triple()
    except Exception:  # noqa: BLE001 - triple is informational for IR dumps
        return ""


def _signature(
    info: FunctionInfo,
    layouts: ClassLayouts | None = None,
    analysis: FunctionAnalysis | None = None,
) -> NativeSignature:
    parameters = tuple(
        _native_param(p.name, p.type, layouts) or NativeParam(p.name, "int") for p in info.params
    )
    atoms = _return_atoms(info.ret) or ("int",)
    returns = tuple(_abi_name(atom) for atom in atoms)
    future = ""
    if info.is_async:
        # A coroutine hands back the runtime's future, an i64 handle, and
        # the boundary reads the kind it carries from `future`.
        future = "none" if _returns_none(info.ret) else atoms[0]
        returns = ("i64",)
    return NativeSignature(
        qualname=info.qualname,
        symbol="ppy_" + info.qualname.replace(".", "_"),
        parameters=parameters,
        returns=returns,
        releases_gil=_releases_gil(analysis) if analysis is not None else False,
        cpu_features=_cpu_features(info),
        future=future,
    )


def _cpu_features(info: FunctionInfo) -> tuple[str, ...]:
    directive = info.directive("cpu.target")
    if directive is None:
        return ()
    return tuple(str(f) for f in directive.options.get("features", ()))  # type: ignore[union-attr]


#: Effects that mean the body can reach the interpreter while it runs, so the
#: GIL has to be held for the whole call.
_NEEDS_GIL = (Effect.PYTHON_CALLBACK, Effect.EXTERNAL_UNKNOWN, Effect.IO)


def _releases_gil(analysis: FunctionAnalysis) -> bool:
    """May the boundary drop the GIL around this call? (spec 16.6)

    Arguments are unpacked into machine values before the call and the result
    is built after it, so the only question is whether the body itself can
    touch a Python object. A borrowed buffer does not count: the caller holds
    the reference and the boundary pins the memory for the whole call, which is
    the same guarantee NumPy relies on.
    """
    return not any(effect in analysis.effects for effect in _NEEDS_GIL)


def _abi_name(scalar: str) -> str:
    return {"int": "i64", "float": "double", "bool": "i8", "i8": "i8", "u8": "i8"}[scalar]
