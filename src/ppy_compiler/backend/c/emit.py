"""Canonical IR -> C11, or C++17.

The C backend reads the IR and nothing else, and writes one translation
unit: a typedef per aggregate type, the overflow helpers the unit uses,
the runtime shims it calls, a prototype per function, the functions in the
internal ABI the runtime binds -- atoms in, result slots out, an `int32_t`
status back -- and, for every `ppy.export`, a public symbol with a C
signature that aborts where Python would have taken the fallback. Blocks
are labels, branches are `goto`s, and a branch's arguments are assigned to
the target block's parameters all at once, so a swap through two arguments
stays a swap.

The C++ form is the same emitter with `Language.CPP` set: it makes its
structural choices -- `extern "C"` around the public symbols, `static_cast`
rather than a C cast, `T{...}` rather than a compound literal, `<cstdint>`
rather than `<stdint.h>` -- as it emits, never by transforming C text
afterwards.

Correct, deterministic, portable, compilable: the four things the output
is for. Pretty is not one of them.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

from ppy_runtime.abi import SANITIZERS, STATUS_FALLBACK, STATUS_OK, STATUS_SANITIZER_BASE

from ...ir import (
    BoolType,
    BufferType,
    FloatType,
    FutureType,
    IndexType,
    IntType,
    IRFunction,
    IRModule,
    IRType,
    Operation,
    PtrType,
    StructType,
    Successor,
    TupleType,
    Value,
    VectorType,
)
from ...ir.dialects.gpu import kind_of
from ...target import TargetInfo, host_target
from .aio import declare as declare_async
from .aio import emit_async
from .dialects import EmitError as _DialectEmitError
from .dialects import (
    emit_atomic,
    emit_concurrency,
    emit_cpu,
    emit_parallel,
    emit_simd,
    emit_special,
    touches_vector,
    vector_core,
    vector_name,
)
from .gpu import emit_gpu
from .prof import emit_prof
from .runtime import SHIMS, definition, program_main

__all__ = ["EmitError", "HeaderOnlyError", "Language", "emit_module"]


class EmitError(Exception):
    """IR the C backend cannot lower; the verifier should have caught it."""


class HeaderOnlyError(EmitError):
    """A header-only unit was asked to carry state a process must own."""


class Language(enum.StrEnum):
    C = "c"
    CPP = "cpp"
    CUDA = "cuda"
    HIP = "hip"


#: The C standard headers a C++ unit includes as `<cX>`; any other keeps its name.
_C_STANDARD_HEADERS = frozenset(
    {"stdint.h", "stdlib.h", "math.h", "stdbool.h", "string.h", "stdio.h", "limits.h", "float.h"}
)
_DIALECT_NAMES = {
    Language.C: "C11",
    Language.CPP: "C++17",
    Language.CUDA: "CUDA C++",
    Language.HIP: "HIP C++",
}


_CMP = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_MATH_INTRINSICS = {
    f"math.{name}": name
    for name in (
        "sqrt",
        "sin",
        "cos",
        "exp",
        "log",
        "log2",
        "log10",
        "fabs",
        "floor",
        "ceil",
        "pow",
        "trunc",
    )
}

#: One overflow helper per (operation, width, signedness), made on demand.
_HELPER = """static inline int ppy_ovf_{op}_{name}({t} a, {t} b, {t} *out) {{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_{op}_overflow(a, b, out);
#else
{portable}
#endif
}}
"""
_PORTABLE_64 = {
    ("add", True): """    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) {
        return 1;
    }
    *out = a + b;
    return 0;""",
    ("sub", True): """    if ((b < 0 && a > INT64_MAX + b) || (b > 0 && a < INT64_MIN + b)) {
        return 1;
    }
    *out = a - b;
    return 0;""",
    ("mul", True): """    if (a > 0) {
        if (b > 0) { if (a > INT64_MAX / b) return 1; }
        else if (b < INT64_MIN / a) return 1;
    } else if (b > 0) {
        if (a < INT64_MIN / b) return 1;
    } else if (a != 0 && b < INT64_MAX / a) return 1;
    *out = a * b;
    return 0;""",
    ("add", False): """    *out = a + b;
    return *out < a;""",
    ("sub", False): """    *out = a - b;
    return a < b;""",
    ("mul", False): """    if (a != 0 && b > UINT64_MAX / a) return 1;
    *out = a * b;
    return 0;""",
}
_PORTABLE_NARROW = """    {wide} r = ({wide})a {symbol} ({wide})b;
    if (r < {minimum} || r > {maximum}) return 1;
    *out = ({t})r;
    return 0;"""

_UNREACHABLE = """#if defined(__GNUC__) || defined(__clang__)
#define PPY_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
#define PPY_UNREACHABLE() __assume(0)
#else
#define PPY_UNREACHABLE() abort()
#endif
"""


@dataclass(slots=True)
class _Buffer:
    data: str
    length: str


#: C library functions the standard headers declare: a prototype of our own
#: would conflict with theirs, so arguments and result are cast instead.
_LIBC: dict[str, tuple[str, tuple[str, ...], str]] = {
    "malloc": ("void *", ("size_t",), "stdlib.h"),
    "calloc": ("void *", ("size_t", "size_t"), "stdlib.h"),
    "realloc": ("void *", ("void *", "size_t"), "stdlib.h"),
    "free": ("void", ("void *",), "stdlib.h"),
    "memcpy": ("void *", ("void *", "const void *", "size_t"), "string.h"),
    "memmove": ("void *", ("void *", "const void *", "size_t"), "string.h"),
    "memset": ("void *", ("void *", "int", "size_t"), "string.h"),
}


@dataclass(slots=True)
class _Unit:
    """Everything a translation unit accumulates, in emission order."""

    aggregates: dict[str, str] = field(default_factory=dict)
    helpers: dict[str, str] = field(default_factory=dict)
    shims: dict[str, None] = field(default_factory=dict)
    externs: dict[str, str] = field(default_factory=dict)
    strings: dict[str, str] = field(default_factory=dict)
    prototypes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    headers: set[str] = field(default_factory=set)
    #: Macro blocks a dialect needs (`PPY_PAUSE`, ...), keyed by what they are.
    prelude: dict[str, str] = field(default_factory=dict)
    #: The unit spawns threads: it links pthreads, and each spawned function
    #: gets a trampoline, emitted once the prototypes are known.
    pthread: bool = False
    trampolines: dict[str, str] = field(default_factory=dict)
    #: The unit has OpenMP regions: it compiles with `-fopenmp`.
    openmp: bool = False
    #: The unit awaits: it is compiled beside the async runtime's source.
    aio: bool = False


def emit_module(
    module: IRModule,
    language: Language = Language.C,
    *,
    header_only: bool = False,
    entry: str | None = None,
    target: TargetInfo | None = None,
) -> str:
    """The translation unit (or, header-only, the header) for `module`.

    `entry` names the symbol of a standalone program's `main`; the unit then
    ends in a C `main` that calls it and fails the process where a guard
    fails, so the text is a whole program. `target` is the machine the C
    is for, where a dialect's lowering depends on it.
    """
    try:
        return _ModuleEmitter(module, language, header_only, entry, target or host_target()).run()
    except _DialectEmitError as error:
        raise EmitError(str(error)) from error


class _ModuleEmitter:
    def __init__(
        self,
        module: IRModule,
        language: Language,
        header_only: bool,
        entry: str | None,
        target: TargetInfo,
    ) -> None:
        self.module = module
        self.language = language
        self.cpp = language is not Language.C
        #: CUDA and HIP: device code is written, and a launch is a triple chevron.
        self.gpu = language in {Language.CUDA, Language.HIP}
        self.header_only = header_only
        self.entry = entry
        self.target = target
        self.unit = _Unit()
        self.type_names: dict[IRType, str] = {}

    # -- spellings the two languages make differently -----------------------

    def cast(self, expression: str, t: IRType | str) -> str:
        spelled = t if isinstance(t, str) else self.c_type(t)
        if self.cpp and not spelled.endswith("*"):
            return f"static_cast<{spelled}>({expression})"
        return f"(({spelled})({expression}))"

    def literal_of(self, aggregate: str, parts: list[str]) -> str:
        inner = ", ".join(parts)
        return f"{aggregate}{{{inner}}}" if self.cpp else f"(({aggregate}){{{inner}}})"

    def std(self, name: str) -> str:
        return f"std::{name}" if self.cpp else name

    @property
    def storage(self) -> str:
        return "static inline " if self.header_only else ""

    # -- types ---------------------------------------------------------------

    def c_type(self, t: IRType) -> str:
        if isinstance(t, BoolType):
            return "bool"
        if isinstance(t, IntType):
            return f"{'' if t.signed else 'u'}int{t.width}_t"
        if isinstance(t, IndexType):
            return "int64_t"
        if isinstance(t, FloatType):
            if t.width == 16:
                raise EmitError("a 16-bit float has no portable C type")
            return "float" if t.width == 32 else "double"
        if isinstance(t, PtrType):
            const = "" if t.mutable else "const "
            return f"{const}{self.c_type(t.pointee)} *"
        if isinstance(t, (TupleType, StructType)):
            return self.aggregate(t)
        if isinstance(t, VectorType):
            return vector_name(self, t)
        if isinstance(t, FutureType):
            return "int64_t"
        raise EmitError(f"{t} has no C representation")

    def declare(self, spelled: str, name: str) -> str:
        return _declare(spelled, name)

    def symbol_of(self, function: IRFunction) -> str:
        return _ident(str(function.attributes.get("ppy.symbol", function.name)))

    def aggregate(self, t: TupleType | StructType) -> str:
        """A typedef'd struct per aggregate type, in order of first use."""
        known = self.type_names.get(t)
        if known is not None:
            return known
        name = f"ppy_agg{len(self.type_names)}"
        self.type_names[t] = name
        body = "".join(f"    {_declare(self.c_type(item), field)};\n" for field, item in _fields(t))
        self.unit.aggregates[name] = f"typedef struct {name} {{\n{body}}} {name};\n"
        return name

    def atoms(self, t: IRType) -> list[str]:
        """The C types a value of `t` crosses the boundary as."""
        if isinstance(t, BoolType):
            return ["int8_t"]
        if isinstance(t, BufferType):
            return [f"{self.c_type(t.element)} *", "int64_t"]
        if isinstance(t, (TupleType, StructType)):
            return [atom for _name, item in _fields(t) for atom in self.atoms(item)]
        return [self.c_type(t)]

    def helper(self, op: str, t: IRType) -> str:
        """The overflow-checking helper for `op` over `t`, defined once."""
        width = t.width if isinstance(t, IntType) else 64
        signed = t.signed if isinstance(t, IntType) else True
        spelled = self.c_type(t)
        name = f"{'i' if signed else 'u'}{width}"
        key = f"ppy_ovf_{op}_{name}"
        if key not in self.unit.helpers:
            if width == 64:
                portable = _PORTABLE_64[(op, signed)]
            else:
                portable = _PORTABLE_NARROW.format(
                    wide="int64_t" if signed else "uint64_t",
                    symbol={"add": "+", "sub": "-", "mul": "*"}[op],
                    minimum=f"INT{width}_MIN" if signed else "0",
                    maximum=f"{'' if signed else 'U'}INT{width}_MAX",
                    t=spelled,
                )
            self.unit.helpers[key] = _HELPER.format(op=op, name=name, t=spelled, portable=portable)
        return key

    def shim(self, name: str) -> None:
        """A runtime shim the unit calls: carried in, with what it needs."""
        described = SHIMS[name]
        if self.header_only and described.stateful:
            raise HeaderOnlyError(
                f"`{name}` keeps state the process owns; a header-only unit cannot carry it"
            )
        for needed in described.needs:
            self.shim(needed)
        self.unit.headers.update(described.headers)
        self.unit.shims.setdefault(name)

    # -- the unit --------------------------------------------------------------

    def run(self) -> str:
        for name, item in self.module.globals.items():
            if isinstance(item.type, BufferType) and isinstance(item.value, str):
                data = item.value.encode("utf-8")
                literal = ", ".join(str(b) for b in data) or "0"
                self.unit.strings[name] = (
                    f"static uint8_t {_ident(name)}[{max(len(data), 1)}] = {{{literal}}};\n"
                )
        defined = [f for f in self.module.functions.values() if not f.is_declaration]
        for function in self.module.functions.values():
            if function.is_declaration and function.attributes.get("ppy.external"):
                # Another module's function: its prototype, never static.
                self.unit.prototypes.append(self.prototype(function)[len(self.storage) :] + ";")
        device = [f for f in defined if kind_of(f) != "host"]
        # Device code is the GPU backends': CUDA and HIP write it first, C and C++ leave it.
        defined = [f for f in defined if kind_of(f) == "host"]
        if self.gpu:
            defined = device + defined
        for function in defined:
            self.unit.prototypes.append(self.prototype(function) + ";")
        for function in defined:
            self.unit.functions.append(_FunctionEmitter(self, function).run())
        for function in defined:
            if "ppy.export" in function.attributes:
                self.unit.exports.append(self.export(function))
        if self.entry is not None:
            if self.header_only:
                raise HeaderOnlyError("a program's `main` is a definition a header cannot carry")
            self.unit.headers.add("stdio.h")
            self.unit.exports.append(program_main(_ident(self.entry), self.std("fputs")))
        return self.assemble()

    def prototype(self, function: IRFunction) -> str:
        if kind_of(function) != "host":
            return self.device_prototype(function)
        if function.attributes.get("ppy.abi") == "resume":
            return f"static void {self.symbol_of(function)}(int64_t *a0)"
        parameters: list[str] = []
        position = 0
        for _name, t in function.params:
            for atom in self.atoms(t):
                parameters.append(_declare(atom, f"a{position}"))
                position += 1
        outs = [atom for t in function.results for atom in self.atoms(t)] or ["int64_t"]
        parameters.extend(_declare(atom, f"*out{index}") for index, atom in enumerate(outs))
        symbol = _ident(str(function.attributes.get("ppy.symbol", function.name)))
        return f"{self.storage}int32_t {symbol}({', '.join(parameters)})"

    def device_prototype(self, function: IRFunction) -> str:
        """A kernel is `__global__` and returns nothing; a device function returns its value."""
        parameters: list[str] = []
        position = 0
        for _name, t in function.params:
            for atom in self.atoms(t):
                parameters.append(_declare(atom, f"a{position}"))
                position += 1
        spelled = ", ".join(parameters) or "void"
        if kind_of(function) == "kernel":
            return f"__global__ void {self.symbol_of(function)}({spelled})"
        atoms = [atom for t in function.results for atom in self.atoms(t)]
        if len(atoms) > 1:
            raise EmitError(f"@{function.name}: a device function returns one scalar")
        result = atoms[0] if atoms else "void"
        return f"static __device__ {result} {self.symbol_of(function)}({spelled})"

    def export(self, function: IRFunction) -> str:
        """The public symbol: the internal function behind a C signature.

        A C caller has no Python to fall back to, so a failed guard aborts,
        never answers wrong.
        """
        name = str(function.attributes["ppy.export"])
        symbol = _ident(str(function.attributes.get("ppy.symbol", function.name)))
        parameters: list[str] = []
        arguments: list[str] = []
        for pname, t in function.params:
            atoms = self.atoms(t)
            for index, atom in enumerate(atoms):
                if isinstance(t, BufferType):
                    spelled = _ident(pname) if index == 0 else f"{_ident(pname)}_len"
                elif len(atoms) > 1:
                    spelled = f"{_ident(pname)}_{index}"
                else:
                    spelled = _ident(pname)
                parameters.append(_declare(atom, spelled))
                arguments.append(spelled)
        result = function.results[0] if function.results else None
        result_atom = self.atoms(result)[0] if result is not None else None
        lines = [
            f"{self.storage}{result_atom or 'void'} {name}({', '.join(parameters) or 'void'}) {{",
            f"    {result_atom or 'int64_t'} result;",
            f"    if ({symbol}({', '.join([*arguments, '&result'])}) != {STATUS_OK}) {{",
            f"        {self.std('abort')}();",
            "    }",
            "    return result;" if result is not None else "    (void)result;",
            "}",
        ]
        return "\n".join(lines) + "\n"

    def assemble(self) -> str:
        cpp = self.cpp
        lines = [f"/* {self.module.name}: generated by ppy, {_DIALECT_NAMES[self.language]} */"]
        libraries = list(self.module.attributes.get("ppy.libraries", ()))  # type: ignore[arg-type]
        libraries = [name for name in libraries if name != "ppy_aio"]
        if self.unit.pthread and "pthread" not in libraries:
            libraries.append("pthread")
        if libraries:
            flags = " ".join(f"-l{name}" for name in libraries)
            lines.append(f"/* link with: {flags} */")
        if self.unit.openmp:
            lines.append("/* compile with: -fopenmp */")
        if self.unit.aio:
            lines.append("/* compile with: the async runtime, ppy_runtime/aio/ppy_aio.c */")
        guard = "PPY_" + _ident(self.module.name).upper() + ("_HPP" if cpp else "_H")
        if self.header_only:
            lines += [f"#ifndef {guard}", f"#define {guard}"]
        headers = {"stdint.h", "stdlib.h", "math.h"} | self.unit.headers
        if not cpp:
            headers.add("stdbool.h")
        if self.language is Language.CUDA:
            headers.add("cuda_runtime.h")
        elif self.language is Language.HIP:
            headers.add("hip/hip_runtime.h")
        for header in sorted(headers):
            if cpp and header in _C_STANDARD_HEADERS:
                lines.append(f"#include <c{header[:-2]}>")
            else:
                lines.append(f"#include <{header}>")
        lines.append("")
        if any("PPY_UNREACHABLE" in f for f in self.unit.functions):
            lines.append(_UNREACHABLE)
        lines.extend(self.unit.prelude.values())
        lines.extend(self.unit.aggregates.values())
        lines.extend(self.unit.helpers.values())
        if self.unit.shims:
            lines.extend(
                definition(name, "static inline " if self.header_only else "static ")
                for name in self.unit.shims
            )
        if self.unit.externs:
            if cpp:
                lines.append('extern "C" {')
            lines.extend(self.unit.externs.values())
            if cpp:
                lines.append("}")
            lines.append("")
        lines.extend(self.unit.strings.values())
        if cpp:
            # The internal ABI is a C ABI: the runtime binds these symbols
            # by name, so C++ must not mangle them.
            lines.append('extern "C" {\n')
        lines.extend(self.unit.prototypes)
        lines.append("")
        lines.extend(self.unit.trampolines.values())
        lines.extend(self.unit.functions)
        lines.extend(self.unit.exports)
        if cpp:
            lines.append('} /* extern "C" */')
        if self.header_only:
            lines.append(f"#endif /* {guard} */")
        return "\n".join(lines).rstrip("\n") + "\n"


class _FunctionEmitter:
    """One IR function as one C function."""

    def __init__(self, owner: _ModuleEmitter, function: IRFunction) -> None:
        self.owner = owner
        self.function = function
        self.scalars: dict[int, str] = {}
        self.buffers: dict[int, _Buffer] = {}
        self.declarations: list[str] = []
        self.body: list[str] = []
        self.counter = 0
        self.labels: dict[int, str] = {}
        #: A kernel or a device function: it returns its value, and nothing falls back.
        self.device = kind_of(function) != "host"
        #: A coroutine's resume function: the frame in, nothing out.
        self.resume = function.attributes.get("ppy.abi") == "resume"

    # -- values ------------------------------------------------------------------

    def fresh(self, hint: str | None) -> str:
        self.counter += 1
        return f"v{self.counter}_{_ident(hint or 'v')}"

    def value(self, v: Value) -> str:
        found = self.scalars.get(id(v))
        if found is None:
            raise EmitError(f"value %{v.name or '?'} was never emitted")
        return found

    def buffer(self, v: Value) -> _Buffer:
        found = self.buffers.get(id(v))
        if found is None:
            raise EmitError(f"buffer %{v.name or '?'} was never emitted")
        return found

    def declare(self, v: Value) -> None:
        """A local (two for a buffer) for `v`, declared at the top."""
        name = self.fresh(v.name)
        if isinstance(v.type, BufferType):
            self.declarations.append(f"    {self.owner.c_type(v.type.element)} *{name}_data;")
            self.declarations.append(f"    int64_t {name}_len;")
            self.buffers[id(v)] = _Buffer(f"{name}_data", f"{name}_len")
            return
        self.declarations.append(f"    {_declare(self.owner.c_type(v.type), name)};")
        self.scalars[id(v)] = name

    def define(self, v: Value, expression: str) -> str:
        self.declare(v)
        name = self.value(v)
        self.body.append(f"    {name} = {expression};")
        if not v.uses:
            # Computed for its effect alone; say so, or a C compiler warns.
            self.body.append(f"    (void){name};")
        return name

    def define_buffer(self, v: Value, data: str, length: str) -> None:
        self.declare(v)
        b = self.buffer(v)
        self.body.append(f"    {b.data} = {data};")
        self.body.append(f"    {b.length} = {length};")
        # The length is read only where a `core.buffer_len` or a call
        # takes the buffer whole; a C compiler warns about the rest.
        if not v.uses:
            self.body.append(f"    (void){b.data};")
        self.body.append(f"    (void){b.length};")

    def reads(self, v: Value) -> list[str]:
        """The expressions that read `v`: one, or a buffer's two."""
        if isinstance(v.type, BufferType):
            b = self.buffer(v)
            return [b.data, b.length]
        return [self.value(v)]

    def flatten(self, v: Value) -> list[str]:
        """The atoms `v` crosses a boundary as."""
        t = v.type
        if isinstance(t, BufferType):
            return self.reads(v)
        if isinstance(t, BoolType):
            return [self.owner.cast(self.value(v), "int8_t")]
        if isinstance(t, (TupleType, StructType)):
            return self._flatten_fields(self.value(v), t)
        return [self.value(v)]

    def _flatten_fields(self, expression: str, t: TupleType | StructType) -> list[str]:
        atoms: list[str] = []
        for field_name, item in _fields(t):
            member = f"{expression}.{field_name}"
            if isinstance(item, BoolType):
                atoms.append(self.owner.cast(member, "int8_t"))
            elif isinstance(item, (TupleType, StructType)):
                atoms.extend(self._flatten_fields(member, item))
            else:
                atoms.append(member)
        return atoms

    def rebuild(self, t: IRType, atoms: list[str], position: int) -> tuple[str, int]:
        """A value of `t` from boundary atoms; (expression, next position)."""
        if isinstance(t, BoolType):
            return f"{atoms[position]} != 0", position + 1
        if isinstance(t, (TupleType, StructType)):
            parts = []
            for _name, item in _fields(t):
                part, position = self.rebuild(item, atoms, position)
                parts.append(part)
            return self.owner.literal_of(self.owner.aggregate(t), parts), position
        return atoms[position], position + 1

    def fail_unless(self, condition: str, label: str) -> None:
        self.body.append(f"    if (!({condition})) goto fallback; /* {label} */")

    # -- the function ------------------------------------------------------------

    def run(self) -> str:
        function = self.function
        entry = function.entry
        assert entry is not None
        for index, block in enumerate(function.body.blocks):
            self.labels[id(block)] = f"L{index}_{_ident(block.name)}"
        position = 0
        for argument in entry.arguments:
            position = self.bind_parameter(argument, position)
        for block in function.body.blocks[1:]:
            for argument in block.arguments:
                self.declare(argument)
        for index, block in enumerate(function.body.blocks):
            if index:
                self.body.append(f"{self.labels[id(block)]}:;")
            for op in block.operations:
                self.op(op)
        if any("goto fallback;" in line for line in self.body):
            self.body.append("fallback:")
            if self.resume:
                declare_async(self.owner, "ppy_aio_fail")
                self.body.append("    ppy_aio_fail(a0, 0);")
                self.body.append("    return;")
            else:
                self.body.append(f"    return {STATUS_FALLBACK};")
        head = self.owner.prototype(function)
        parts = [head + " {", *self.declarations, *self.body, "}", ""]
        return "\n".join(parts) + "\n"

    def bind_parameter(self, argument: Value, position: int) -> int:
        t = argument.type
        if isinstance(t, BufferType):
            self.buffers[id(argument)] = _Buffer(f"a{position}", f"a{position + 1}")
            return position + 2
        if isinstance(t, (BoolType, TupleType, StructType)):
            atoms = [f"a{position + i}" for i in range(len(self.owner.atoms(t)))]
            expression, _ = self.rebuild(t, atoms, 0)
            self.define(argument, expression)
            return position + len(atoms)
        self.scalars[id(argument)] = f"a{position}"
        return position + 1

    # -- operations ----------------------------------------------------------------

    def op(self, op: Operation) -> None:
        c = self.owner.c_type
        name = op.local_name
        if op.dialect == "math":
            self.math(op, name, [self.value(v) for v in op.operands])
            return
        emitter = _DIALECTS.get(op.dialect)
        if emitter is not None:
            emitter(self, op)
            return
        if op.dialect != "core":
            raise EmitError(
                f"{op.name}: the C backend has no lowering for the {op.dialect} dialect"
            )
        if touches_vector(op) and name in _VECTOR_CORE:
            vector_core(self, op, name)
            return
        match name:
            case "const":
                self.define(op.result, self.literal(op.result.type, op.attributes["value"]))
            case "add" | "sub" | "mul":
                self.arith(op, name)
            case "div" | "mod":
                self.divmod(op, name)
            case "neg":
                self.neg(op)
            case "and" | "or" | "xor":
                symbol = {"and": "&", "or": "|", "xor": "^"}[name]
                a, b = (self.value(v) for v in op.operands)
                self.define(op.result, f"{a} {symbol} {b}")
            case "shl" | "shr":
                self.shift(op, name)
            case "cmp":
                self.cmp(op)
            case "select":
                cond, a, b = (self.value(v) for v in op.operands)
                self.define(op.result, f"{cond} ? {a} : {b}")
            case "cast":
                self.cast(op)
            case "br":
                self.branch(op.successors[0], "    ")
            case "cond_br":
                self.body.append(f"    if ({self.value(op.operands[0])}) {{")
                self.branch(op.successors[0], "        ")
                self.body.append("    } else {")
                self.branch(op.successors[1], "        ")
                self.body.append("    }")
            case "ret":
                if self.device or self.resume:
                    atoms = [atom for value in op.operands for atom in self.flatten(value)]
                    self.body.append(f"    return {atoms[0]};" if atoms else "    return;")
                    return
                position = 0
                for value in op.operands:
                    for atom in self.flatten(value):
                        self.body.append(f"    *out{position} = {atom};")
                        position += 1
                if not op.operands:
                    self.body.append("    *out0 = 0;")
                self.body.append(f"    return {STATUS_OK};")
            case "unreachable":
                self.body.append("    PPY_UNREACHABLE();")
            case "alloca":
                pointer = op.result.type
                assert isinstance(pointer, PtrType)
                slot = self.fresh(op.result.name or "slot")
                count = int(op.attributes.get("count", 1))  # type: ignore[call-overload]
                slots = max(count, 1)
                self.declarations.append(f"    {_declare(c(pointer.pointee), slot)}[{slots}];")
                self.scalars[id(op.result)] = slot
            case "load":
                self.define(op.result, f"*{self.value(op.operands[0])}")
            case "store":
                value, pointer = (self.value(v) for v in op.operands)
                self.body.append(f"    *{pointer} = {value};")
            case "ptr_offset":
                pointer, offset = (self.value(v) for v in op.operands)
                self.define(op.result, f"{pointer} + {offset}")
            case "buffer_data":
                self.define(op.result, self.buffer(op.operands[0]).data)
            case "buffer_len":
                self.define(op.result, self.buffer(op.operands[0]).length)
            case "buffer_load":
                b = self.buffer(op.operands[0])
                self.define(op.result, f"{b.data}[{self.value(op.operands[1])}]")
            case "buffer_store":
                b = self.buffer(op.operands[1])
                self.body.append(
                    f"    {b.data}[{self.value(op.operands[2])}] = {self.value(op.operands[0])};"
                )
            case "tuple_make" | "struct_make":
                t = op.result.type
                assert isinstance(t, (TupleType, StructType))
                parts = [self.value(v) for v in op.operands]
                self.define(op.result, self.owner.literal_of(self.owner.aggregate(t), parts))
            case "tuple_extract":
                index = int(op.attributes["index"])  # type: ignore[call-overload]
                self.define(op.result, f"{self.value(op.operands[0])}.f{index}")
            case "struct_extract":
                field_name = _ident(str(op.attributes["field"]))
                self.define(op.result, f"{self.value(op.operands[0])}.{field_name}")
            case "call":
                self.call(op)
            case "call_extern":
                self.call_extern(op)
            case "call_intrinsic":
                self.call_intrinsic(op)
            case "guard":
                label = str(op.attributes.get("label") or f"{op.attributes['kind']}.ok")
                if label.startswith("sanitize:"):
                    # A sanitizer's check returns its status; nothing falls back.
                    status = STATUS_SANITIZER_BASE + SANITIZERS.index(label.partition(":")[2])
                    self.body.append(
                        f"    if (!({self.value(op.operands[0])})) return {status}; /* {label} */"
                    )
                else:
                    self.fail_unless(self.value(op.operands[0]), label)
            case _:
                raise EmitError(f"{op.name} has no C lowering")

    def literal(self, t: IRType, value: object) -> str:
        if isinstance(t, BoolType):
            return "true" if value else "false"
        if isinstance(t, FloatType):
            number = float(value)  # type: ignore[arg-type]
            if math.isnan(number):
                return "NAN"
            if number in (float("inf"), float("-inf")):
                return "INFINITY" if number > 0 else "-INFINITY"
            text = repr(number)
            return text if ("." in text or "e" in text) else f"{text}.0"
        if isinstance(t, PtrType):
            return f"(({self.owner.c_type(t)})0)"
        number = int(value)  # type: ignore[call-overload]
        if isinstance(t, IntType) and not t.signed:
            return f"UINT64_C({number})"
        if number == -(1 << 63):
            return "INT64_MIN"
        return f"INT64_C({number})"

    def math(self, op: Operation, name: str, arguments: list[str]) -> None:
        t = op.results[0].type
        if isinstance(t, VectorType) and isinstance(t.element, FloatType):
            self.vector_math(op, name, arguments, t)
            return
        if not isinstance(t, FloatType) or t.width == 16:
            raise EmitError(f"math.{name} over {t} has no C lowering")
        function = "fabs" if name == "abs" else name
        if not self.owner.cpp and t.width == 32:
            function += "f"
        self.define(op.results[0], f"{self.owner.std(function)}({', '.join(arguments)})")

    def vector_math(self, op: Operation, name: str, arguments: list[str], t: VectorType) -> None:
        function = "fabs" if name == "abs" else name
        if not self.owner.cpp and t.element.width == 32:  # type: ignore[union-attr]
            function += "f"
        vector = vector_name(self.owner, t)
        key = f"{vector}_{name}"
        parameters = ", ".join(f"{vector} a{i}" for i in range(len(arguments)))
        call = (
            self.owner.std(function)
            + "("
            + ", ".join(f"a{i}.lanes[i]" for i in range(len(arguments)))
            + ")"
        )
        self.owner.unit.helpers.setdefault(
            key,
            f"static inline {vector} {key}({parameters}) {{\n    {vector} r;\n"
            f"    for (int i = 0; i < {t.count}; i++) {{\n        r.lanes[i] = {call};\n    }}\n"
            "    return r;\n}\n",
        )
        self.define(op.results[0], f"{key}({', '.join(arguments)})")

    def branch(self, successor: Successor, indent: str) -> None:
        """Assign the target's parameters -- all at once -- and jump."""
        block = successor.block
        if successor.arguments:
            staged: list[tuple[str, str, str]] = []
            for argument, value in zip(block.arguments, successor.arguments, strict=True):
                for target, expression in zip(self.reads(argument), self.reads(value), strict=True):
                    temporary = self.fresh("pass")
                    spelled = self.slot_type(argument, target)
                    self.declarations.append(f"    {_declare(spelled, temporary)};")
                    staged.append((target, temporary, expression))
            for _target, temporary, expression in staged:
                self.body.append(f"{indent}{temporary} = {expression};")
            for target, temporary, _expression in staged:
                self.body.append(f"{indent}{target} = {temporary};")
        self.body.append(f"{indent}goto {self.labels[id(block)]};")

    def slot_type(self, v: Value, slot: str) -> str:
        if isinstance(v.type, BufferType):
            return "int64_t" if slot.endswith("_len") else f"{self.owner.c_type(v.type.element)} *"
        return self.owner.c_type(v.type)

    def arith(self, op: Operation, name: str) -> None:
        t = op.result.type
        a, b = (self.value(v) for v in op.operands)
        symbol = {"add": "+", "sub": "-", "mul": "*"}[name]
        if isinstance(t, FloatType):
            self.define(op.result, f"{a} {symbol} {b}")
            return
        overflow = op.attributes.get("overflow", "python")
        if overflow in {"wrap", "proven"}:
            self.define(op.result, self.wrapped(t, a, b, symbol))
            return
        result = self.define(op.result, "0")
        helper = self.owner.helper(name, t)
        self.fail_unless(f"!{helper}({a}, {b}, &{result})", "arith.ok")

    def wrapped(self, t: IRType, a: str, b: str, symbol: str) -> str:
        """Two's-complement arithmetic through the unsigned type: signed
        overflow is undefined in C, unsigned wraps."""
        unsigned = _unsigned(t)
        cast = self.owner.cast
        return cast(f"{cast(a, unsigned)} {symbol} {cast(b, unsigned)}", t)

    def neg(self, op: Operation) -> None:
        t = op.result.type
        x = self.value(op.operands[0])
        if isinstance(t, FloatType):
            self.define(op.result, f"-{x}")
            return
        if op.attributes.get("overflow", "python") == "wrap":
            self.define(op.result, self.wrapped(t, "0", x, "-"))
            return
        result = self.define(op.result, "0")
        helper = self.owner.helper("sub", t)
        self.fail_unless(f"!{helper}(0, {x}, &{result})", "arith.ok")

    def divmod(self, op: Operation, name: str) -> None:
        t = op.result.type
        a, b = (self.value(v) for v in op.operands)
        if isinstance(t, FloatType):
            if name == "mod":
                raise EmitError("float remainder has no C lowering with Python semantics")
            self.define(op.result, f"{a} / {b}")
            return
        signed = t.signed if isinstance(t, IntType) else True
        symbol = "/" if name == "div" else "%"
        if not signed:
            self.define(op.result, f"{a} {symbol} {b}")
            return
        width = t.width if isinstance(t, IntType) else 64
        if op.attributes.get("overflow", "python") != "wrap":
            self.fail_unless(f"!({a} == INT{width}_MIN && {b} == -1)", "div.ok")
        if op.attributes.get("rounding", "floor") == "trunc":
            self.define(op.result, f"{a} {symbol} {b}")
            return
        c = self.owner.c_type(t)
        remainder = self.fresh("r")
        self.declarations.append(f"    {c} {remainder};")
        self.body.append(f"    {remainder} = {a} % {b};")
        adjust = f"({remainder} != 0 && (({a} < 0) != ({b} < 0)))"
        if name == "div":
            quotient = self.fresh("q")
            self.declarations.append(f"    {c} {quotient};")
            self.body.append(f"    {quotient} = {a} / {b};")
            self.define(op.result, f"{adjust} ? {quotient} - 1 : {quotient}")
        else:
            self.define(op.result, f"{adjust} ? {remainder} + {b} : {remainder}")

    def shift(self, op: Operation, name: str) -> None:
        t = op.result.type
        a, b = (self.value(v) for v in op.operands)
        if name == "shr":
            self.define(op.result, f"{a} >> {b}")
            return
        shifted = self.define(op.result, self.wrapped(t, a, b, "<<"))
        if op.attributes.get("overflow", "wrap") != "wrap":
            self.fail_unless(f"({shifted} >> {b}) == {a}", "shl.ok")

    def cmp(self, op: Operation) -> None:
        a, b = (self.value(v) for v in op.operands)
        predicate = str(op.attributes["predicate"])
        if predicate == "ne" and isinstance(op.operands[0].type, FloatType):
            # Ordered, like the LLVM road: NaN compares neither less nor greater.
            self.define(op.result, f"{a} < {b} || {a} > {b}")
            return
        self.define(op.result, f"{a} {_CMP[predicate]} {b}")

    def cast(self, op: Operation) -> None:
        value = self.value(op.operands[0])
        source = op.operands[0].type
        target = op.result.type
        if isinstance(source, PtrType) and isinstance(target, PtrType):
            self.define(op.result, f"(({self.owner.c_type(target)})({value}))")
            return
        if isinstance(source, PtrType):
            self.define(op.result, f"((int64_t)(intptr_t)({value}))")
            return
        if isinstance(target, BoolType):
            zero = "0.0" if isinstance(source, FloatType) else "0"
            self.define(op.result, f"{value} != {zero}")
            return
        if source == target:
            self.define(op.result, value)
            return
        self.define(op.result, self.owner.cast(value, target))

    def call(self, op: Operation) -> None:
        callee_name = op.attributes["callee"].name  # type: ignore[union-attr]
        target = self.owner.module.functions.get(callee_name)
        if target is None:
            raise EmitError(f"call to @{callee_name}, which was not emitted")
        symbol = _ident(str(target.attributes.get("ppy.symbol", target.name)))
        arguments: list[str] = []
        for operand in op.operands:
            arguments.extend(self.flatten(operand))
        if self.device:
            direct = f"{symbol}({', '.join(arguments)})"
            if op.results:
                self.define(op.results[0], direct)
            else:
                self.body.append(f"    {direct};")
            return
        slots: list[list[str]] = []
        for t in target.results or ():
            names = []
            for atom in self.owner.atoms(t):
                slot = self.fresh("res")
                self.declarations.append(f"    {_declare(atom, slot)};")
                names.append(slot)
            slots.append(names)
            arguments.extend(f"&{n}" for n in names)
        if not target.results:
            slot = self.fresh("res")
            self.declarations.append(f"    int64_t {slot};")
            arguments.append(f"&{slot}")
        results = list(op.results)
        call = f"{symbol}({', '.join(arguments)})"
        if op.attributes.get("capture_status"):
            self.define(results.pop(), self.owner.cast(call, "int64_t"))
        else:
            self.fail_unless(f"{call} == {STATUS_OK}", "call.ok")
        for result, t, names in zip(results, target.results, slots, strict=True):
            if isinstance(t, BufferType):
                self.define_buffer(result, names[0], names[1])
            else:
                self.define(result, self.rebuild(t, names, 0)[0])

    def call_extern(self, op: Operation) -> None:
        symbol = str(op.attributes["callee"])
        shim = SHIMS.get(symbol)
        libc = _LIBC.get(symbol)
        if libc is not None:
            result_type, parameter_types, header = libc
            self.owner.unit.headers.add(header)
            arguments = [
                self.owner.cast(self.extern_atom(operand), expected)
                for operand, expected in zip(op.operands, parameter_types, strict=True)
            ]
            call = f"{symbol}({', '.join(arguments)})"
            if op.results:
                wanted = self.extern_type(op.results[0])
                if wanted != result_type:
                    call = self.owner.cast(call, wanted)
                self.define(op.results[0], call)
            else:
                self.body.append(f"    {call};")
            return
        if shim is not None:
            self.owner.shim(symbol)
            parameter_types = [p.rsplit(" ", 1)[0] for p in shim.parameters]
            arguments = []
            for operand, expected in zip(op.operands, parameter_types, strict=True):
                atom = self.extern_atom(operand)
                if expected != self.extern_type(operand):
                    atom = self.owner.cast(atom, expected)
                arguments.append(atom)
        else:
            parameter_types = [self.extern_type(v) for v in op.operands]
            arguments = [self.extern_atom(v) for v in op.operands]
            result = "void" if not op.results else self.extern_type(op.results[0])
            self.owner.unit.externs.setdefault(
                symbol, f"{result} {symbol}({', '.join(parameter_types) or 'void'});\n"
            )
        call = f"{symbol}({', '.join(arguments)})"
        if not op.results:
            self.body.append(f"    {call};")
            return
        if isinstance(op.results[0].type, BoolType):
            call = f"{call} != 0"
        self.define(op.results[0], call)

    def extern_type(self, value: Value) -> str:
        t = value.type
        if isinstance(t, BoolType):
            return "int8_t"
        if isinstance(t, BufferType):
            return f"{self.owner.c_type(t.element)} *"
        return self.owner.c_type(t)

    def extern_atom(self, value: Value) -> str:
        """How a value crosses into C: a bool is a byte, a buffer its pointer."""
        if isinstance(value.type, BoolType):
            return self.owner.cast(self.value(value), "int8_t")
        if isinstance(value.type, BufferType):
            return self.buffer(value).data
        return self.value(value)

    def call_intrinsic(self, op: Operation) -> None:
        name = str(op.attributes["intrinsic"])
        if name in _MATH_INTRINSICS:
            self.math(op, _MATH_INTRINSICS[name], [self.value(v) for v in op.operands])
            return
        if name == "ppy.buffer_from_parts":
            data, length = (self.value(v) for v in op.operands)
            self.define_buffer(op.results[0], data, length)
            return
        if name == "ppy.string_data":
            self.define(op.results[0], _ident(str(op.attributes["symbol"])))
            return
        if name in {"llvm.smin.i64", "llvm.smax.i64"}:
            a, b = (self.value(v) for v in op.operands)
            symbol = "<" if name == "llvm.smin.i64" else ">"
            self.define(op.results[0], f"{a} {symbol} {b} ? {a} : {b}")
            return
        if name.startswith("ppy.checked_"):
            operation = name.removeprefix("ppy.checked_")
            a, b = (self.value(v) for v in op.operands)
            result = self.define(op.results[0], "0")
            helper = self.owner.helper(operation, op.operands[0].type)
            self.define(op.results[1], f"{helper}({a}, {b}, &{result}) != 0")
            return
        raise EmitError(f"intrinsic {name!r} has no C lowering")


#: The dialects beyond core and math, each emitted by its own function.
_DIALECTS = {
    "simd": emit_simd,
    "cpu": emit_cpu,
    "atomic": emit_atomic,
    "concurrency": emit_concurrency,
    "parallel": emit_parallel,
    "special": emit_special,
    "gpu": emit_gpu,
    "async": emit_async,
    "prof": emit_prof,
}
#: Core operations that take vectors, emitted as lane helpers.
_VECTOR_CORE = frozenset(
    {"add", "sub", "mul", "div", "neg", "and", "or", "xor", "cmp", "select", "cast"}
)


def _fields(t: TupleType | StructType) -> list[tuple[str, IRType]]:
    if isinstance(t, TupleType):
        return [(f"f{i}", item) for i, item in enumerate(t.items)]
    return list(t.fields)


def _declare(spelled: str, name: str) -> str:
    """`int64_t x`, but `double *p`: the star hugs the name."""
    return f"{spelled}{name}" if spelled.endswith("*") else f"{spelled} {name}"


def _ident(name: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    return out if out and not out[0].isdigit() else f"_{out}"


def _unsigned(t: IRType) -> str:
    width = t.width if isinstance(t, IntType) else 64
    return f"uint{width}_t"
