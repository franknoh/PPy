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
import builtins
import dataclasses
from dataclasses import dataclass, field

from ppy_runtime._record import replace
from ppy_runtime.abi import (
    STATUS_FALLBACK,
    STATUS_OK,
    CrossingClass,
    NativeParam,
    NativeSignature,
)
from ppy_runtime.collection_boundary import RETURNS_NOTHING
from ppy_runtime.collection_boundary import parse as crossing_spec

from ...analysis import types as T
from ...analysis.checker import FunctionAnalysis
from ...analysis.closures import callable_spelled, is_plain_callable
from ...analysis.collections import spelled as collection_spelled
from ...analysis.effects import Effect
from ...analysis.settled import implicit_parameter_name
from ...analysis.symbols import FunctionInfo, ParamInfo

__all__ = [
    "STATUS_FALLBACK",
    "STATUS_OK",
    "LoweredFunction",
    "NativeParam",
    "NativeSignature",
    "Unsupported",
    "called_back_only",
    "eligible",
    "written_params",
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
    #: The entry Python calls, when it is a thunk around `signature`'s symbol
    #: (strings cross as UTF-8 bytes there, as handles here).
    boundary: NativeSignature | None = None
    #: Why Python's calls run the Python body whatever the cost says: it copies
    #: an argument Python could change at a barrier (`_check_effects`).
    withheld: str = ""

    @property
    def python(self) -> NativeSignature:
        """The signature the Python boundary binds."""
        return self.boundary or self.signature


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
    #: Per function with effects, the rule they run under (`lowering/effects.py`).
    effects: dict[str, str] = field(default_factory=dict)


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
    if element is None or element not in _BUFFER_ELEMENTS:
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


#: The `ppy` collections, which cross between native functions as handles.
_COLLECTIONS = frozenset(
    {
        "ppy.Vec",
        "ppy.Deque",
        "ppy.Heap",
        "ppy.MaxHeap",
        "ppy.LinkedList",
        "ppy.HashMap",
        "ppy.HashSet",
        "ppy.TreeMap",
        "ppy.TreeSet",
    }
)


#: Python's own containers, which native code holds as the runtime's collections.
_BUILTIN_CONTAINERS = frozenset({"list", "dict", "set"})

#: The standard library's containers native code holds (`lowering/library.py`).
_LIBRARY_CONTAINERS = frozenset(
    {
        "collections.deque",
        "collections.defaultdict",
        "collections.OrderedDict",
        "collections.Counter",
        "random.Random",
    }
)


def written_params(analysis: FunctionAnalysis | None) -> frozenset[str]:
    """The parameters held by handle rather than lent as buffers: every one, when
    the function writes through any parameter, and none otherwise.

    A list of numbers only read is lent as a buffer, a copy of its words. A
    function that writes through a parameter may be handed the same list
    twice (`f(xs, xs)`), and a copy would not see the write: then each list
    goes by handle, and the boundary keeps one object one handle."""
    if analysis is None or not writes(analysis):
        return frozenset()
    return frozenset(p.name for p in analysis.info.params) | {
        implicit_parameter_name(analysis, held) for held in analysis.implicit_globals
    }


def writes(analysis: FunctionAnalysis) -> set[str]:
    """The parameters the function writes through, itself or by handing them to
    a callee that does, and the settled globals passed to it that it or a
    callee writes, by the names it takes them as."""
    return (
        analysis.mutated_params
        | analysis.delegated_writes
        | {
            implicit_parameter_name(analysis, held)
            for held in analysis.implicit_globals
            if held.written
        }
    )


def _holds_strings(info: FunctionInfo) -> bool:
    """Whether a parameter or the result is a container with strings in it."""

    def inside(t: T.Type) -> bool:
        base = T.strip_literal(t)
        if isinstance(base, T.Union_):
            return any(inside(member) for member in base.members)
        if not isinstance(base, T.Instance):
            return False
        return base.name == "str" or any(inside(a) for a in base.args)

    def container(t: T.Type) -> bool:
        base = T.strip_literal(t)
        return isinstance(base, T.Instance) and any(inside(a) for a in base.args)

    return container(info.ret) or any(container(p.type) for p in info.params)


def _nested_loop(function: ast.AST) -> bool:
    """Whether a loop runs inside another loop: work per element of more than
    one step, which pays for copying the element across."""
    loops = (ast.For, ast.While, ast.AsyncFor)
    comprehensions = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    for child in ast.walk(function):
        if isinstance(child, loops):
            inner = (n for statement in child.body for n in ast.walk(statement))
            if any(isinstance(n, (*loops, ast.comprehension)) for n in inner):
                return True
        elif isinstance(child, comprehensions) and len(child.generators) > 1:
            return True
    return False


def _collection_param(
    name: str, t: T.Type, layouts: ClassLayouts | None = None, written: bool = True
) -> NativeParam | None:
    """A collection or an object parameter: a handle native callers pass.

    A collection is a `ppy.Vec[int]` or another; an object is an instance of a
    project class `layouts` marks as one (an empty layout), `Node` or `Node |
    None`. Its element is the whole type written out (`ppy.Vec[ppy.Vec[int]]`),
    which is what an argument is matched against.
    """
    base = T.strip_literal(t)
    if base == T.STR:
        return NativeParam(name, "handle", "str", class_name="str")
    if isinstance(base, T.Callable_):
        # A function value: a closure's handle. None crosses from Python.
        if not is_plain_callable(base):
            return None
        return NativeParam(name, "handle", callable_spelled(base), class_name="callable")
    if isinstance(base, T.Instance) and base.name in {"Iterator", "Generator"} and base.args:
        # A generator's frame: native code's own, never crossing to Python.
        element = T.strip_literal(base.args[0])
        return NativeParam(name, "handle", f"generator[{element}]", class_name="generator")
    if isinstance(base, T.Instance) and base.name in _LIBRARY_CONTAINERS:
        # `deque`, `defaultdict`, `Counter`, `OrderedDict`, `random.Random`:
        # native code's own, passed by handle, never crossing to Python.
        if base.name != "random.Random" and not _held_natively(base, layouts):
            return None
        from ...lowering.collections import kind_of  # pylint: disable=import-outside-toplevel

        records = {name: (tuple(fields), False) for name, fields in (layouts or {}).items()}
        kind = kind_of(base, records)
        if kind is None:
            return None
        return NativeParam(name, "handle", kind.spelled, class_name=base.name)
    if isinstance(base, T.Instance) and base.name in _BUILTIN_CONTAINERS and base.args:
        if not written and _buffer_element(base) is not None:
            # A list of numbers the function only reads is lent as a buffer.
            return None
        if not _held_natively(base, layouts):
            return None
        return NativeParam(name, "handle", str(base), class_name=base.name)
    nullable = False
    if isinstance(base, T.Union_):
        members = [m for m in base.members if m != T.NONE]
        if len(members) != 1 or len(members) == len(base.members):
            return None
        base = T.strip_literal(members[0])
        if not isinstance(base, T.Instance) or base.name in _COLLECTIONS:
            return None
        nullable = True
    if not isinstance(base, T.Instance):
        return None
    if base.name in _COLLECTIONS and base.args:
        return NativeParam(name, "handle", collection_spelled(base), class_name=base.name)
    if layouts is not None and layouts.get(base.name) == ():
        return NativeParam(
            name, "handle", collection_spelled(base), class_name=base.name, nullable=nullable
        )
    return None


def _held_natively(base: T.Instance, layouts: ClassLayouts | None) -> bool:
    """Whether native code holds this `list`, `dict`, or `set`: what it holds has
    a native form, and a key is one native code hashes as CPython does."""
    from ...lowering.collections import kind_of  # pylint: disable=import-outside-toplevel

    records = {name: (tuple(fields), False) for name, fields in (layouts or {}).items()}
    return kind_of(base, records) is not None


def _native_param(
    name: str, t: T.Type, layouts: ClassLayouts | None = None, written: bool = False
) -> NativeParam | None:
    collection = _collection_param(name, t, layouts, written)
    if collection is not None:
        return collection
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


def _return_atoms(t: T.Type, layouts: ClassLayouts | None = None) -> tuple[str, ...] | None:
    if _collection_param("", t, layouts) is not None:
        return ("handle",)
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
    allow_python: bool = False,
    allow_globals: bool = False,
) -> tuple[bool, str]:
    """Can this function be lowered to a native scalar entry point?

    `allow_io` is the standalone build's dispensation: printing goes through
    native shims there, so the IO effect alone is not disqualifying.
    `allow_launch` is the GPU source backends': a kernel launch is written
    as one, where the CPU backends have no launch runtime yet. `allow_python`
    is `ppy run`'s: a call native code cannot make it makes through Python
    (`lowering/effects.py`), so a call of unknown effect is the lowering's to
    refuse, one call at a time. `allow_globals` is `ppy run`'s too: the
    settled globals the function reads are parameters of `info` (see
    `with_implicit_globals`), which Python's boundary reads from the module
    at the call.
    """
    if analysis.python_only:
        return False, analysis.python_only[0]
    if info.is_generator and info.is_async:
        return False, "an async generator runs on CPython"
    if info.is_async and not allow_async:
        return False, "a coroutine runs natively only where the async runtime does"
    written = writes(analysis)
    passes_globals = allow_globals and analysis.globals_native
    for name in sorted(written):
        declared = next((p.type for p in info.params if p.name == name), None)
        if declared is None and not passes_globals:
            # A global written where it is not passed: reading it is the blocker.
            continue
        described = _buffer_element(declared) if declared is not None else None
        handle = _collection_param(name, declared, layouts) if declared is not None else None
        # Writing through a borrowed buffer is visible to the caller, which is
        # what borrowing means, and so is writing through a collection's
        # handle. Anything else would lose the write.
        if handle is None and (described is None or described[0] != "view"):
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
    # A draw is a write to the generator's state, which native code shares
    # with Python's `random` (the boundary saves and restores it).
    violations.discard(Effect.RANDOM)
    if passes_globals:
        # Every global it reads is one no one rebinds, passed in as a parameter.
        violations.discard(Effect.READ_GLOBAL)
    if allow_io:
        violations.discard(Effect.IO)
    if allow_python:
        violations.discard(Effect.EXTERNAL_UNKNOWN)
        violations.discard(Effect.PYTHON_CALLBACK)
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
        if _native_param(param.name, param.type, layouts, param.name in written) is None:
            return False, f"parameter `{param.name}` is `{param.type}`, which has no native ABI"
    if _return_atoms(info.ret, layouts) is None and not _returns_none(info.ret):
        return False, f"returns `{info.ret}`, which has no native ABI"
    return True, ""


def with_implicit_globals(info: FunctionInfo, analysis: FunctionAnalysis) -> FunctionInfo:
    """`info` with the settled globals native code passes it (see
    `analysis/settled.py`) as parameters after its own."""
    if not analysis.implicit_globals:
        return info
    added = [
        ParamInfo(
            implicit_parameter_name(analysis, held),
            held.type,
            annotated=True,
            global_of=held.key,
        )
        for held in analysis.implicit_globals
    ]
    return dataclasses.replace(info, params=[*info.params, *added])


def called_back_only(info: FunctionInfo) -> bool:
    """`def __eq__(self, other: object)` of a class: `other` has no native ABI, and
    what calls it natively is a collection comparing two of its own keys, which
    lowers the method again with `other` an instance of the class."""
    return (
        info.name == "__eq__"
        and bool(info.owner)
        and len(info.params) == 2
        and T.strip_literal(info.params[1].type) == T.OBJECT
    )


#: What counts as an operation of straight-line work.
_WORK_NODES = (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.Call, ast.Subscript, ast.IfExp)


def _crossing_costs_more(
    info: FunctionInfo,
    layouts: ClassLayouts | None,
    written: frozenset[str],
    classes: tuple[CrossingClass, ...] = (),
    filled: frozenset[str] = frozenset(),
) -> str | None:
    """Why copying the function's containers across the boundary would cost more
    than running it natively saves, or None when it pays.

    The boundary copies a container that crosses whole, in and back, on every
    call. That is work proportional to its size, so the body has to do work
    proportional to it too: a loop that walks it, or works on it element by
    element. How much work depends on what an element costs to copy, which
    `_copy_cost` puts in operations of a Python loop's body: a list of
    numbers next to nothing, a dict's entry or an inner list a few, an object
    more. A container of strings costs more still, a native string made for
    every element, about what one pass of a Python loop spends on it, so it
    pays only when the body makes more than one pass (a loop in a loop).
    """
    crossing = [
        (param.name, native)
        for param in info.params
        if (native := _native_param(param.name, param.type, layouts, param.name in written))
        is not None
        and native.is_handle
        and native.element != "str"
        and _crosses(native, classes)
    ]
    names = [name for name, _native in crossing]
    if crossing and not _works_through(info.node, names, info.name):
        return "copying the collections in costs more than the body does with them"
    for name, native in crossing:
        need = _copy_cost(native, classes, name in filled)
        if need and _loop_work(info.node, name, info.name) < need:
            return "copying the collections in costs more than the body does with them"
    if _holds_strings(info) and not _nested_loop(info.node):
        return "copying its strings across costs what one pass over them saves"
    return None


def _copy_cost(parameter: NativeParam, classes: tuple[CrossingClass, ...], written: bool) -> int:
    """What copying one element of a crossing parameter costs, in operations of
    a CPython loop's body (each about what `s += x` costs), in and back.
    Measured on the generated wrapper (`crossing.c`): a number in a list costs
    a few nanoseconds; a dict's or a set's entry is hashed and put; a list in
    a list, and an object, is a handle made, filled, and let go of."""
    described = {c.qualname: c for c in classes}
    spec = crossing_spec(parameter.element, described)
    if spec is None:
        return 0
    if spec.kind == "object" or (spec.value is not None and spec.value.kind == "object"):
        return 8 if written else 6
    if spec.key is not None:
        return 6 if written else 3
    if spec.value is not None and (spec.value.collection or spec.value.kind == "str"):
        return 3 if written else 2
    return 1 if written else 0


#: Work in a loop's body, as `_loop_work` counts it.
_LOOP_WORK = (*_WORK_NODES, ast.AugAssign)


def _loop_work(function: ast.AST, name: str, own: str) -> int:
    """The work a loop does per pass over `name`: the operations in the body of
    the heaviest loop that touches it (names it, or follows a field), less a
    lookup into `name` itself, which costs native code what it costs Python.
    A function that walks by calling itself counts its body."""
    local = _local_names(function)
    best = 0
    walked = False
    for loop in ast.walk(function):
        if isinstance(loop, (ast.For, ast.AsyncFor, ast.While)):
            body: list = loop.body
        elif isinstance(loop, ast.comprehension):
            body = loop.ifs
        else:
            continue
        names = {n.id for n in ast.walk(loop) if isinstance(n, ast.Name)}
        follows = isinstance(loop, ast.While) and _follows_a_field(loop)
        if name not in names and not follows:
            continue
        walked = True
        best = max(best, sum(_work_of(statement, name, local) for statement in body))
    if not walked and own:
        # Recursion over linked objects, a builtin over the whole, or a global
        # only a callee reads: the function's own work is what it costs.
        return sum(_work_of(statement, name, local) for statement in getattr(function, "body", []))
    return best


#: A call of one of the program's functions counts as this many operations:
#: it is a body of its own, which may loop.
_CALL_WORK = 6


def _local_names(function: ast.AST) -> frozenset[str]:
    """The names the function binds: its parameters and what it assigns."""
    found = {
        n.id for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    arguments = getattr(function, "args", None)
    if arguments is not None:
        found.update(
            a.arg for a in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
        )
    return frozenset(found)


def _program_call(call: ast.Call, local: frozenset[str]) -> bool:
    """A call of one of the program's functions: `f(x)`, or `module.f(x)`."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id not in _BUILTIN_NAMES
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id not in local
        and func.value.id not in _BUILTIN_NAMES
    )


def _work_of(node: ast.AST, name: str, local: frozenset[str] = frozenset()) -> int:
    count = 0
    for child in ast.walk(node):
        if not isinstance(child, _LOOP_WORK):
            continue
        if (
            isinstance(child, ast.Subscript)
            and isinstance(child.value, ast.Name)
            and child.value.id == name
        ):
            continue
        if isinstance(child, ast.Call) and _program_call(child, local):
            count += _CALL_WORK
            continue
        count += 1
    return count


_BUILTIN_NAMES = frozenset(dir(builtins))


#: Builtins that go over a whole container given to them.
_WHOLE_BUILTINS = frozenset(
    {"sum", "sorted", "min", "max", "any", "all", "list", "set", "dict", "tuple", "reversed",
     "enumerate", "zip", "map", "filter"}
)  # fmt: skip

#: Methods that go over a whole container, its own or the one they are given.
_WHOLE_METHODS = frozenset(
    {"sort", "copy", "count", "index", "remove", "reverse", "extend", "update", "union",
     "intersection", "difference", "symmetric_difference", "issubset", "issuperset",
     "isdisjoint", "join", "values", "items", "keys", "to_sorted", "between"}
)  # fmt: skip


def _works_through(function: ast.AST, names: list[str], own: str = "") -> bool:
    """Whether the function does work that grows with one of `names`, which pays
    for copying it across the boundary: a loop or a comprehension walks it, a
    loop's body calls a method on it or writes an element of it, an operator
    takes it whole (`s & t`), or a builtin or a method goes over all of it
    (`sum(v)`, `v.sort()`, `", ".join(v)`). Objects linked to one another are
    walked by a loop that follows a field (`node = node.next`) or by the
    function calling itself on one (`height(node.left)`)."""
    wanted = set(names)

    def whole(node: ast.expr) -> bool:
        """The container itself (or a field of it), not one of its elements."""
        while isinstance(node, ast.Attribute):
            node = node.value
        return isinstance(node, ast.Name) and node.id in wanted

    def rooted(node: ast.expr) -> bool:
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        return isinstance(node, ast.Name) and node.id in wanted

    for node in ast.walk(function):
        # Work over the whole container without a loop of the function's own.
        if isinstance(node, ast.comprehension) and rooted(node.iter):
            return True
        if (
            own
            and isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == own
            and any(isinstance(a, ast.Attribute) and rooted(a) for a in node.args)
        ):
            return True
        if isinstance(node, ast.While) and _follows_a_field(node):
            return True
        if isinstance(node, ast.BinOp) and (whole(node.left) or whole(node.right)):
            return True  # `s & t`, `v + w`
        if isinstance(node, ast.Call):
            called = node.func
            if (
                isinstance(called, ast.Name)
                and called.id in _WHOLE_BUILTINS
                and any(whole(argument) for argument in node.args)
            ):
                return True
            if (
                isinstance(called, ast.Attribute)
                and called.attr in _WHOLE_METHODS
                and (whole(called.value) or any(whole(a) for a in node.args))
            ):
                return True
    for loop in ast.walk(function):
        if not isinstance(loop, (ast.For, ast.While, ast.AsyncFor)):
            continue
        header = loop.iter if isinstance(loop, (ast.For, ast.AsyncFor)) else loop.test
        if any(isinstance(n, ast.Name) and n.id in wanted for n in ast.walk(header)):
            return True
        for statement in loop.body:
            for child in ast.walk(statement):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and rooted(child.func.value)
                ):
                    return True
                if isinstance(child, (ast.Assign, ast.AugAssign)):
                    targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                    if any(isinstance(t, ast.Subscript) and rooted(t) for t in targets):
                        return True
    return False


def _follows_a_field(loop: ast.While) -> bool:
    """A loop that steps along linked objects: `node = node.next` in its body."""
    for child in ast.walk(loop):
        if (
            isinstance(child, ast.Assign)
            and len(child.targets) == 1
            and isinstance(child.targets[0], ast.Name)
            and isinstance(child.value, ast.Attribute)
        ):
            root = child.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id == child.targets[0].id:
                return True
    return False


def _returns_none(t: T.Type) -> bool:
    return t == T.NONE


#: Directives that are an explicit request for the native boundary.
_EXPOSURE_DIRECTIVES = ("native", "jit", "specialize", "parallel")


def can_lower_native(
    info: FunctionInfo, analysis: FunctionAnalysis, layouts: ClassLayouts | None = None
) -> tuple[bool, str]:
    """Can PPY generate correct native code for this function?"""
    return eligible(info, analysis, layouts)


def should_lower_native(
    info: FunctionInfo,
    analysis: FunctionAnalysis,
    layouts: ClassLayouts | None = None,
    classes: tuple[CrossingClass, ...] = (),
) -> tuple[bool, str]:
    """Is native execution through the Python boundary expected to be faster?

    Eligibility and profitability are different questions: a two-instruction
    add lowers perfectly and still loses to CPython once the boundary's
    guards and conversions are paid. Native-to-native calls never cross the
    boundary, so a helper this keeps off it is still called directly by any
    native caller.
    """
    written = written_params(analysis)
    if info.is_async:
        # The future is the boundary value; a coroutine is called to be run.
        return True, "a coroutine's future crosses the boundary"
    for api in ("cuda", "hip"):
        if any(info.directive(f"{api}.{kind}") is not None for kind in ("kernel", "device")):
            # Device code has no CPU form to bind; a launch runs it, and under
            # CPython its own definition is the reference.
            return False, "device code runs where it is launched"
    filled = writes(analysis)
    fills = any(
        _crosses(native, classes) and native is not None and native.name in filled
        for native in (
            _native_param(p.name, p.type, layouts, p.name in written) for p in info.params
        )
    )
    if _returns_none(info.ret) and not fills and not (info.params and _loops(info.node)):
        # A function with no value to hand back is native code's to call -- a
        # thread's body, a helper, a check that raises -- unless what it does
        # is fill a collection the caller passed, or loop over what it is
        # given. A `main()` that takes nothing runs once, and Python runs it:
        # what it calls goes native on its own terms.
        return False, "returns nothing, which has no Python boundary"
    returned = _collection_param("", info.ret, layouts)
    if returned is not None and returned.element != "str" and not _crosses(returned, classes):
        return False, "returns an object, which native callers receive by handle"
    for param in info.params:
        native = _native_param(param.name, param.type, layouts, param.name in written)
        if native is not None and native.is_pointer:
            # A machine address has no Python object to come from, whatever
            # the directives ask: the function is native code's to call.
            return False, "takes a native pointer, which has no Python boundary"
        crosses = native is not None and (native.element == "str" or _crosses(native, classes))
        if native is not None and native.is_handle and not crosses:
            return False, "takes an object, which native callers pass by handle"
    for name in _EXPOSURE_DIRECTIVES:
        if info.directive(name) is not None:
            return True, f"@ppy.{name} asks for the boundary"
    refused = _crossing_costs_more(info, layouts, written, classes, frozenset(filled))
    if refused is not None:
        return False, refused
    for param in info.params:
        native = _native_param(param.name, param.type, layouts, param.name in written)
        if native is not None and native.is_buffer:
            # Buffer work scales with the data; the crossing is flat.
            return True, "takes a buffer"
    for child in ast.walk(info.node):
        if isinstance(child, (ast.For, ast.While, ast.AsyncFor, ast.comprehension)):
            return True, "contains a loop"
    work = sum(isinstance(child, _WORK_NODES) for child in ast.walk(info.node))
    if _calls_itself(info):
        # Its depth is the argument's to decide, and native code has no
        # recursion limit to raise `RecursionError` at: CPython's frames do.
        return False, "calls itself without a loop; CPython's recursion limit stays in force"
    if work >= _crossing_cost(info, analysis, layouts, written):
        return True, f"straight-line work ({work} operations)"
    return False, "the boundary crossing costs more than the body saves"


#: The crossing's cost in operations of a CPython body, measured with
#: `examples/bench_boundary.py`: the generated wrapper's call costs what a
#: Python call does, so a body of two operations already gains. A string is
#: made natively on the way in and decoded on the way out, about four
#: operations each; a value class's field is an attribute read. Module
#: globals the function reads are read by a Python frame in front of the
#: wrapper. A function that draws random numbers is bound by the Python-level
#: binding, which saves `random`'s state through `ctypes`. Output a native
#: call holds is written out through Python when the call ends, which costs
#: more than CPython's `print` for one line and much less for many: only a
#: loop that prints pays for it.
_CROSSING_BASE = 2
_CROSSING_TEXT = 4
_CROSSING_GLOBALS = 6
_CROSSING_SLOW = 16
_CROSSING_HELD = 40


def _crossing_cost(
    info: FunctionInfo,
    analysis: FunctionAnalysis,
    layouts: ClassLayouts | None,
    written: frozenset[str],
) -> int:
    """How much straight-line work pays for a call through the boundary."""
    if Effect.RANDOM in analysis.effects:
        return _CROSSING_SLOW
    if any(effect in analysis.effects for effect in _NEEDS_GIL):
        return _CROSSING_HELD
    cost = _CROSSING_BASE + (_CROSSING_GLOBALS if analysis.implicit_globals else 0)
    for param in info.params:
        native = _native_param(param.name, param.type, layouts, param.name in written)
        if native is None:
            continue
        if native.is_handle and native.element == "str":
            cost += _CROSSING_TEXT
        elif native.is_object:
            cost += len(native.fields)
    returned = _collection_param("", info.ret, layouts)
    if returned is not None and returned.element == "str":
        cost += _CROSSING_TEXT
    return cost


#: What a standalone build can allocate for itself, and the element it holds.
#: Both are eight bytes wide, which is what the native ABI passes.
_ALLOCATIONS = {
    "ppy.buffer[int]": ("int", False),
    "ppy.buffer[float]": ("float", False),
    "ppy.buffer[ppy.i8]": ("i8", False),
    "ppy.buffer[ppy.u8]": ("u8", False),
    "ppy.scan[Buffer[int]]": ("int", True),
    "ppy.scan[Buffer[float]]": ("float", True),
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
    written = writes(analysis) if analysis is not None else set()
    # Held by handle as the function's IR takes them: each one, when it writes
    # through any (see `written_params`).
    held = written_params(analysis)
    parameters = tuple(
        _sourced(
            _written(
                _native_param(p.name, p.type, layouts, p.name in held)
                or NativeParam(p.name, "int"),
                written,
            ),
            p.global_of,
        )
        for p in info.params
    )
    returned = _collection_param("", info.ret, layouts)
    atoms = _return_atoms(info.ret, layouts) or ("int",)
    returns = tuple(_abi_name(atom) for atom in atoms)
    future = ""
    if info.is_async:
        # A coroutine hands back the runtime's future, an i64 handle, and
        # the boundary reads the kind it carries from `future`.
        future = "none" if _returns_none(info.ret) else atoms[0]
        returns = ("i64",)
    return NativeSignature(
        qualname=info.qualname,
        symbol="ppy_" + info.qualname.replace(".<locals>.", "_locals_").replace(".", "_"),
        parameters=parameters,
        returns=returns,
        releases_gil=_releases_gil(analysis) and _runs_long(info)
        if analysis is not None and not _cached(info)
        else False,
        cpu_features=_cpu_features(info),
        future=future,
        returned=_returned(info, returned),
        draws=analysis is not None and Effect.RANDOM in analysis.effects,
    )


def _returned(info: FunctionInfo, returned: NativeParam | None) -> str:
    """What the boundary builds from the result: a collection's type, or `None`
    for a function that returns nothing."""
    if returned is not None:
        return returned.element
    if _returns_none(info.ret) and not info.is_async:
        return RETURNS_NOTHING
    return ""


def _written(parameter: NativeParam, written: set[str]) -> NativeParam:
    """A collection parameter the function writes through, marked for the boundary."""
    if parameter.is_handle and parameter.name in written:
        return replace(parameter, written=True)
    return parameter


def _sourced(parameter: NativeParam, source: str) -> NativeParam:
    """A settled global passed as a parameter, marked for the boundary to read."""
    return replace(parameter, source=source) if source else parameter


def _crosses(parameter: NativeParam | None, classes: tuple[CrossingClass, ...] = ()) -> bool:
    """Whether a handle has a Python form at the boundary: a collection of numbers,
    strings, tuples of numbers, and collections of those, or an object of a
    class among `classes`."""
    if parameter is None:
        return False
    described = {c.qualname: c for c in classes}
    return crossing_spec(parameter.element, described) is not None


def _cpu_features(info: FunctionInfo) -> tuple[str, ...]:
    directive = info.directive("cpu.target")
    if directive is None:
        return ()
    return tuple(str(f) for f in directive.options.get("features", ()))  # type: ignore[union-attr]


#: Effects that mean the body can reach the interpreter while it runs, so the
#: GIL has to be held for the whole call.
# A draw moves `random._inst`'s state, which Python code may be reading.
_NEEDS_GIL = (Effect.PYTHON_CALLBACK, Effect.EXTERNAL_UNKNOWN, Effect.IO, Effect.RANDOM)


def _cached(info: FunctionInfo) -> bool:
    """A cached function writes its table, which Python's callers share, and
    one taking a `random.Random` draws from a Python object's memory: the GIL
    is held for both."""
    if any(name in {"functools.cache", "functools.lru_cache"} for name in info.decorators):
        return True
    return any(
        isinstance(base := T.strip_literal(p.type), T.Instance) and base.name == "random.Random"
        for p in info.params
    )


def _releases_gil(analysis: FunctionAnalysis) -> bool:
    """May the boundary drop the GIL around this call? (spec 16.6)

    Arguments are unpacked into machine values before the call and the result
    is built after it, so the only question is whether the body itself can
    touch a Python object. A borrowed buffer does not count: the caller holds
    the reference and the boundary pins the memory for the whole call, which is
    the same guarantee NumPy relies on.
    """
    return not any(effect in analysis.effects for effect in _NEEDS_GIL)


def _calls_itself(info: FunctionInfo) -> bool:
    """Whether the body calls the function by its own name (or `self.name`)."""
    for child in ast.walk(info.node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name) and func.id == info.name:
            return True
        if isinstance(func, ast.Attribute) and func.attr == info.name and info.owner:
            return True
    return False


def _loops(node: ast.AST) -> bool:
    return any(
        isinstance(child, (ast.For, ast.While, ast.AsyncFor, ast.comprehension))
        for child in ast.walk(node)
    )


def _runs_long(info: FunctionInfo) -> bool:
    """Whether a call may run long enough to be worth dropping the GIL for: a
    loop, or a call that may be one (recursion, a callee's loop). Dropping it
    and taking it back costs about what a two-operation body does, so a
    short straight-line body keeps it, as CPython's own builtins do."""
    for child in ast.walk(info.node):
        if isinstance(child, (ast.For, ast.While, ast.AsyncFor, ast.comprehension)):
            return True
        if isinstance(child, ast.Call) and not _brief_call(child):
            return True
    return False


#: Builtins whose call is a step or two natively; `min` and `max` of two or
#: more values, not of a collection.
_BRIEF_BUILTINS = frozenset(
    {"abs", "len", "int", "float", "bool", "round", "divmod", "pow", "ord", "chr"}
)


def _brief_call(call: ast.Call) -> bool:
    if not isinstance(call.func, ast.Name):
        return False
    if call.func.id in {"min", "max"}:
        return len(call.args) >= 2
    return call.func.id in _BRIEF_BUILTINS


def _abi_name(scalar: str) -> str:
    names = {"int": "i64", "float": "double", "bool": "i8", "i8": "i8", "u8": "i8"}
    return names.get(scalar, "i8*")
