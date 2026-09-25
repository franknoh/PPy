"""Canonical IR -> C11, or C++17.

The C backend reads the IR and nothing else, and writes one translation
unit: a typedef per aggregate type, the overflow helpers the unit uses,
the runtime shims it calls, a prototype per function, the functions in the
internal ABI the runtime binds -- atoms in, result slots out, an `int32_t`
status back -- and, for every `ppy.export`, a public symbol with a C
signature that aborts where Python would have taken the fallback.

The control flow is written the way a person would write it. The emitter
reads the dominator tree and the loops of the function's graph and writes
`while`, `if`/`else`, `break`, `continue`, and `return`; a block reached
from one place is written where it is reached, a block several branches
meet at is written once after them, and a branch's arguments are assigned
to the target block's parameters all at once, so a swap through two
arguments stays a swap. A value used once, in the block that computed it,
is written where it is used; a stack slot that is only ever loaded and
stored is a plain variable; a parameter keeps the name the Python had.
A graph the reconstruction cannot express -- one that Python control flow
does not produce -- is written with labels and `goto`s instead, for that
function only, so the output is always correct.

The C++ form is the same emitter with `Language.CPP` set: it makes its
structural choices -- `extern "C"` around the public symbols, `static_cast`
rather than a C cast, `T{...}` rather than a compound literal, `<cstdint>`
rather than `<stdint.h>` -- as it emits, never by transforming C text
afterwards.

Correct, deterministic, portable, compilable, and readable: the five
things the output is for.
"""

from __future__ import annotations

import enum
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import groupby

from ppy_runtime.abi import SANITIZERS, STATUS_FALLBACK, STATUS_OK, STATUS_SANITIZER_BASE

from ...ir import (
    Block,
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
from ...ir.dialects.core import RAISED_VALUES
from ...ir.dialects.gpu import kind_of
from ...ir.raising import overflow_text, raised_text
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
from .presentation import PrintFormat, string_literal, string_symbol
from .prof import emit_prof
from .ranges import truncating_divisions
from .runtime import SHIMS, definition, program_main
from .source import definition_order

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
    {
        "stdint.h",
        "stdlib.h",
        "math.h",
        "stdbool.h",
        "string.h",
        "stdio.h",
        "limits.h",
        "float.h",
        "inttypes.h",
    }
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


#: Names a local may not take: the languages' keywords, the C functions this
#: unit may call, and the prefixes its own helpers use.
_KEYWORDS = frozenset(
    """alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t
    char16_t char32_t class compl concept const consteval constexpr constinit const_cast continue
    co_await co_return co_yield decltype default delete do double dynamic_cast else enum explicit
    export extern false float for friend goto if inline int long mutable namespace new noexcept not
    not_eq nullptr operator or or_eq private protected public register reinterpret_cast requires
    restrict return short signed sizeof static static_assert static_cast struct switch template
    this thread_local throw true try typedef typeid typename union unsigned using virtual void
    volatile wchar_t while xor xor_eq _Bool _Complex _Imaginary _Alignas _Alignof _Atomic _Generic
    _Noreturn _Static_assert _Thread_local main abort exit free malloc calloc realloc memcpy
    memmove memset printf scanf fflush stdout stderr stdin std fputs puts sqrt sin cos tan exp exp2
    log log2 log10 floor ceil trunc fabs
    pow fmod hypot round erf erfc tgamma lgamma isnan isinf NAN INFINITY NULL
    int8_t int16_t int32_t int64_t uint8_t uint16_t uint32_t uint64_t intptr_t uintptr_t size_t
    INT_MIN INT_MAX UINT_MAX CHAR_BIT
    INT64_MIN INT64_MAX UINT64_MAX INT64_C UINT64_C PRId64 PRIu64 SCNd64 SCNu64
    remove rename tmpfile tmpnam fopen freopen fclose fread fwrite fgetc fputc getc putc getchar
    putchar gets fgets sprintf snprintf vsnprintf fprintf sscanf fscanf perror feof ferror rewind
    fseek ftell clearerr setbuf setvbuf ungetc EOF FILE BUFSIZ atoi atol atoll atof strtol strtoul
    strtoll strtoull strtod strtof qsort bsearch rand srand abs labs llabs div ldiv system getenv
    atexit memcmp memchr strlen strcpy strncpy strcat strncat strcmp strncmp strchr strrchr strstr
    strtok strerror time clock signal raise assert errno fmin fmax asin acos atan atan2 sinh cosh
    tanh asinh acosh atanh cbrt modf frexp ldexp isfinite signbit copysign nan""".split()  # noqa: SIM905
)
_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
_NUMBER = re.compile(r"^(\d[\w.+-]*|\.\d+)$")
_SIMPLE_JUMP = re.compile(r"^(break;|continue;|return\b.*;)$")


def _cname(name: str) -> str:
    """A C identifier for a Python name: the name itself unless C has taken it."""
    out = _ident(name)
    if out in _KEYWORDS or out.startswith(("ppy_", "__")):
        out += "_"
    return out


def _closes_at_end(text: str, start: int) -> bool:
    depth = 0
    for index in range(start, len(text)):
        c = text[index]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return index == len(text) - 1
    return False


def _is_atomic(text: str) -> bool:
    """Whether `text` may stand as an operand with no parentheses around it.

    Identifiers, numbers, calls, indexes, member reads, a parenthesized group,
    and a C cast of an atom are atoms; anything with an operator in the open
    is not.
    """
    if not text:
        return False
    if _IDENTIFIER.match(text) or _NUMBER.match(text):
        return True
    if text.startswith("static_cast<"):
        close = text.find(">(")
        return close > 0 and _closes_at_end(text, close + 1)
    if text[0] == "(":
        if _closes_at_end(text, 0):
            return True
        # `(type)atom` -- a cast whose operand is itself an atom.
        depth, end = 0, -1
        for index, c in enumerate(text):
            depth += (c == "(") - (c == ")")
            if depth == 0:
                end = index
                break
        inner = text[1:end]
        return re.fullmatch(r"[A-Za-z_][\w *]*", inner) is not None and _is_atomic(text[end + 1 :])
    match = re.match(r"[A-Za-z_]\w*", text)
    if match is None:
        return False
    index = match.end()
    while index < len(text):
        c = text[index]
        if c in "([":
            depth = 0
            while index < len(text):
                depth += text[index] in "(["
                depth -= text[index] in ")]"
                index += 1
                if depth == 0:
                    break
            if depth:
                return False
        elif (
            c == "."
            and index + 1 < len(text)
            and (text[index + 1].isalpha() or text[index + 1] == "_")
        ):
            member = re.match(r"\w+", text[index + 1 :])
            index += 1 + (member.end() if member else 0)
        elif text.startswith("->", index):
            member = re.match(r"\w+", text[index + 2 :])
            index += 2 + (member.end() if member else 0)
        else:
            return False
    return True


def _negate(text: str) -> str:
    """`!text`: an equality flipped, a negation undone, `!` otherwise."""
    if text.startswith("!") and _is_atomic(text[1:]):
        rest = text[1:]
        if rest.startswith("(") and _closes_at_end(rest, 0):
            return rest[1:-1]
        return rest
    flipped = _flip_equality(text)
    if flipped is not None:
        return flipped
    return f"!{text}" if _is_atomic(text) else f"!({text})"


def _flip_equality(text: str) -> str | None:
    """`a != b` for `a == b` (and back) when both sides are atoms; exact for any type."""
    for symbol in (" == ", " != "):
        head, found, tail = text.partition(symbol)
        if found and _is_atomic(head) and _is_atomic(tail):
            other = " != " if symbol == " == " else " == "
            return f"{head}{other}{tail}"
    return None


class _Body(list):
    """The statements of a function, indented for the depth they are written at.

    A dialect appends a statement with the four spaces of a function's top
    level; the emitter's current depth replaces them, and deeper lines of a
    multi-line statement keep their relative indentation.
    """

    def __init__(self, depth: int = 1) -> None:
        super().__init__()
        self.depth = depth

    def append(self, text: str) -> None:
        prefix = "    " * self.depth
        for line in text.split("\n"):
            if not line:
                super().append("")
            else:
                super().append(prefix + line.removeprefix("    "))


class _Unstructured(Exception):
    """A graph the structured writer cannot express; the labels writer can."""


@dataclass(slots=True)
class _Loop:
    header: Block
    latches: list = field(default_factory=list)
    body: set = field(default_factory=set)
    exits: list = field(default_factory=list)
    #: Where the loop's `while` line went: which list of lines, and which line.
    writer: object = None
    start: int = 0

    @property
    def exit(self):  # type: ignore[no-untyped-def]
        return self.exits[0] if len(self.exits) == 1 else None


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
    readable: bool = False,
) -> str:
    """The translation unit (or, header-only, the header) for `module`.

    `entry` names the symbol of a standalone program's `main`; the unit then
    ends in a C `main` that calls it and fails the process where a guard
    fails, so the text is a whole program. `target` is the machine the C
    is for, where a dialect's lowering depends on it.
    """
    try:
        return _ModuleEmitter(
            module, language, header_only, entry, target or host_target(), readable
        ).run()
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
        readable: bool = False,
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
        #: The shims whose needs are being carried in, against a cycle.
        self._carrying: set[str] = set()
        self.readable = readable and entry is not None
        #: A standalone program: a failed guard prints what CPython would
        #: raise and exits, since there is no Python to fall back to.
        self.standalone = bool(module.attributes.get("ppy.standalone")) and entry is not None
        #: Whether a failed call leaves handles for the collections runtime to free.
        self.collections = "ppy_collections" in tuple(
            module.attributes.get("ppy.libraries", ())  # type: ignore[arg-type]
        )
        self.int32 = self.readable and module.attributes.get("ppy.source_int_width") == 32
        if self.int32:
            self.unit.headers.add("limits.h")
            assertion = "static_assert" if self.cpp else "_Static_assert"
            self.unit.prelude["int_width"] = (
                f"{assertion}(INT_MAX == 2147483647 && INT_MIN == (-2147483647 - 1), "
                '"this program requires 32-bit int");\n'
            )
        operations = [op for function in module.functions.values() for op in function.operations()]
        self.native_input = not any(
            op.name == "core.call_extern"
            and str(op.attributes.get("callee", "")).startswith(
                ("ppy_rt_scan_", "ppy_rt_input_", "ppy_rt_read_")
            )
            and op.attributes.get("callee") not in {"ppy_rt_input_int", "ppy_rt_scan_int"}
            for op in operations
        )
        self.names: dict[str, str] = {}
        self.namespaces: dict[str, str] = {}
        self.direct: set[str] = set()
        self.reserved: set[str] = set()
        if self.readable:
            self._source_names()
            # Status capture and runtime callbacks have an ABI consumer other
            # than ordinary source calls, so their signatures must stay intact.
            for op in operations:
                if op.name != "core.call" or op.attributes.get("capture_status"):
                    callee = getattr(op.attributes.get("callee"), "name", None)
                    self.direct.discard(callee)
            self.reserved = set(self.names.values()) | {
                scope.split("::")[0] for scope in self.namespaces.values() if scope
            }
        self.type_names: dict[IRType, str] = {}

    def _source_names(self) -> None:
        """Assign source names once, reserving library and global identifiers."""
        functions = list(self.module.functions.values())
        parts = {
            f.name: str(f.attributes.get("ppy.qualname", f.name)).rpartition(".") for f in functions
        }
        used = set(_LIBC) | {"main", "printf", "scanf", "fflush", "stdout", "std"}
        namespace_names: dict[str, str] = {}
        prefixes: dict[tuple[str, ...], str] = {(): ""}
        scoped_names: dict[str, set[str]] = {"": used}
        for module in sorted({p[0] for p in parts.values()}):
            segments = module.split(".") if module else ["program"]
            for index, segment in enumerate(segments):
                prefix = tuple(segments[: index + 1])
                if prefix not in prefixes:
                    parent = prefixes[prefix[:-1]]
                    name = _unique(_cname(segment), scoped_names.setdefault(parent, set()))
                    prefixes[prefix] = f"{parent}::{name}" if parent else name
            namespace_names[module] = prefixes[tuple(segments)]
        used = set(_LIBC) | {"main", "printf", "scanf", "fflush", "stdout"}
        for function in sorted(functions, key=lambda f: f.name):
            module, _, short = parts[function.name]
            name = _cname(short)
            if str(function.attributes.get("ppy.symbol")) == self.entry:
                name = "main"
                scope = ""
            elif self.cpp:
                scope = namespace_names[module]
                name = _unique(name, scoped_names.setdefault(scope, set()))
            else:
                scope = ""
                if sum(_cname(p[2]) == name for p in parts.values()) > 1 or name in used:
                    name = _cname(module.replace(".", "_") + "_" + short)
                name = _unique(name, used)
            self.names[function.name] = name
            self.namespaces[function.name] = scope
            if (
                not function.is_declaration
                and kind_of(function) == "host"
                and function.attributes.get("ppy.abi") != "resume"
                and len(function.results) <= 1
                and all(
                    isinstance(t, (IntType, IndexType, BoolType, FloatType, PtrType))
                    for t in function.results
                )
            ):
                self.direct.add(function.name)

    def scoped(self, function: IRFunction, text: str) -> str:
        scope = self.namespaces.get(function.name)
        return f"namespace {scope} {{\n{text}\n}} // namespace {scope}\n" if scope else text

    def is_main(self, function: IRFunction) -> bool:
        return self.readable and str(function.attributes.get("ppy.symbol")) == self.entry

    # -- spellings the two languages make differently -----------------------

    def cast(self, expression: str, t: IRType | str) -> str:
        spelled = t if isinstance(t, str) else self.c_type(t)
        if self.cpp and not spelled.endswith("*"):
            if _IDENTIFIER.match(spelled):
                return f"{spelled}({expression})"
            return f"static_cast<{spelled}>({expression})"
        if _is_atomic(expression):
            return f"({spelled}){expression}"
        return f"({spelled})({expression})"

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
            if not self.cpp:
                self.unit.headers.add("stdbool.h")
            return "bool"
        if isinstance(t, IntType):
            if self.int32 and t.width == 32:
                return "int" if t.signed else "unsigned int"
            self.unit.headers.add("stdint.h")
            name = f"{'' if t.signed else 'u'}int{t.width}_t"
            return self.std(name) if self.readable else name
        if isinstance(t, IndexType):
            self.unit.headers.add("stdint.h")
            return self.std("int64_t") if self.readable else "int64_t"
        if isinstance(t, FloatType):
            if t.width == 16:
                raise EmitError("a 16-bit float has no portable C type")
            return "float" if t.width == 32 else "double"
        if isinstance(t, PtrType):
            const = "" if t.mutable else "const "
            pointee = self.c_type(t.pointee)
            return f"{const}{pointee}{'' if pointee.endswith('*') else ' '}*"
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
        if self.readable:
            name = self.names[function.name]
            scope = self.namespaces[function.name]
            return f"{scope}::{name}" if scope else name
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
            return [self.c_type(t)] if self.readable else ["int8_t"]
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
        if name in self.unit.shims or name in self._carrying:
            # Carried, or being carried: the runtime's functions may call each
            # other in a cycle (a collection lets go of the strings it keys by).
            return
        self._carrying.add(name)
        for needed in described.needs:
            self.shim(needed)
        self._carrying.discard(name)
        self.unit.headers.update(described.headers)
        self.unit.shims.setdefault(name)

    # -- the unit --------------------------------------------------------------

    def run(self) -> str:
        for name, item in self.module.globals.items():
            if isinstance(item.type, BufferType) and isinstance(item.value, str):
                if self.readable:
                    continue
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
        prototypes = {function.name for function in defined}
        if self.readable:
            defined, prototypes = definition_order(defined)
        for function in defined:
            if not self.is_main(function) and function.name in prototypes:
                self.unit.prototypes.append(self.scoped(function, self.prototype(function) + ";"))
        bodies: list[tuple[IRFunction, str]] = []
        for function in defined:
            try:
                text = _FunctionEmitter(self, function).run()
            except _Unstructured:
                text = _FunctionEmitter(self, function, structured=False).run()
            bodies.append((function, text))
        for _, group in groupby(bodies, key=lambda item: self.namespaces.get(item[0].name)):
            members = list(group)
            self.unit.functions.append(
                self.scoped(members[0][0], "\n".join(text for _, text in members))
            )
        for function in defined:
            if "ppy.export" in function.attributes and not self.readable:
                self.unit.exports.append(self.export(function))
        if self.entry is not None:
            if self.header_only:
                raise HeaderOnlyError("a program's `main` is a definition a header cannot carry")
            if not self.readable:
                self.unit.headers.add("stdio.h")
                if self.collections:
                    self.shim("ppy_coll_collect")
                self.unit.exports.append(
                    program_main(_ident(self.entry), self.std("fputs"), collect=self.collections)
                )
        return self.assemble()

    def parameter_atoms(self, function: IRFunction) -> list[tuple[str, str]]:
        """The C parameters a function's parameters cross the boundary as: (type, name).

        A parameter keeps the name the Python gave it; a buffer is its data
        and `<name>_len`; a value class or tuple is `<name>_0`, `<name>_1`,
        ...; a name C has taken gets a trailing underscore.
        """
        used: set[str] = set(self.reserved)
        atoms: list[tuple[str, str]] = []
        for index, (pname, t) in enumerate(function.params):
            base = _cname(pname) if pname and not pname.isdigit() else f"a{index}"
            kinds = self.atoms(t)
            if isinstance(t, BufferType):
                names = [base, f"{base}_len"]
            elif len(kinds) > 1:
                names = [f"{base}_{i}" for i in range(len(kinds))]
            else:
                names = [base]
            for kind, name in zip(kinds, names, strict=True):
                atoms.append((kind, _unique(name, used)))
        return atoms

    def result_atoms(self, function: IRFunction, used: set[str]) -> list[tuple[str, str]]:
        """The result slots: `out`, or `out0`, `out1`, ... for several atoms."""
        outs = [atom for t in function.results for atom in self.atoms(t)] or ["int64_t"]
        names = ["out"] if len(outs) == 1 else [f"out{i}" for i in range(len(outs))]
        return [(atom, _unique(name, used)) for atom, name in zip(outs, names, strict=True)]

    def prototype(self, function: IRFunction) -> str:
        if self.is_main(function):
            return "int main()" if self.cpp else "int main(void)"
        if function.name in self.direct:
            parameters = [_declare(kind, name) for kind, name in self.parameter_atoms(function)]
            result = self.c_type(function.results[0]) if function.results else "void"
            storage = "" if self.cpp else "static "
            spelled = ", ".join(parameters) or ("" if self.cpp else "void")
            return f"{storage}{result} {self.names[function.name]}({spelled})"
        if kind_of(function) != "host":
            return self.device_prototype(function)
        if function.attributes.get("ppy.abi") == "resume":
            symbol = self.names[function.name] if self.readable else self.symbol_of(function)
            return f"static void {symbol}(int64_t *frame)"
        atoms = self.parameter_atoms(function)
        parameters = [_declare(kind, name) for kind, name in atoms]
        results = self.result_atoms(function, {name for _kind, name in atoms})
        parameters.extend(_declare(kind, f"*{name}") for kind, name in results)
        symbol = self.names[function.name] if self.readable else self.symbol_of(function)
        return f"{self.storage}int32_t {symbol}({', '.join(parameters)})"

    def device_prototype(self, function: IRFunction) -> str:
        """A kernel is `__global__` and returns nothing; a device function returns its value."""
        parameters = [_declare(kind, name) for kind, name in self.parameter_atoms(function)]
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
        # The async runtime is its own file to compile with; the collections
        # runtime is carried in the unit as shims.
        libraries = [name for name in libraries if name not in {"ppy_aio", "ppy_collections"}]
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
        headers = (
            set(self.unit.headers)
            if self.readable
            else {"stdint.h", "stdlib.h", "math.h"} | self.unit.headers
        )
        if not cpp and not self.readable:
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
        # A device kernel may call a vector helper: in a CUDA or HIP unit every
        # helper is compiled for both sides.
        lines.extend(
            helper.replace("static inline ", "static inline __host__ __device__ ", 1)
            if self.gpu
            else helper
            for helper in self.unit.helpers.values()
        )
        if self.unit.shims:
            storage = "static inline " if self.header_only else "static "
            # A shim in a cycle calls one defined after it: that one is declared first.
            order = {name: position for position, name in enumerate(self.unit.shims)}
            later = {
                needed
                for name in self.unit.shims
                for needed in SHIMS[name].needs
                if order.get(needed, -1) > order[name]
            }
            lines.extend(
                SHIMS[name].prototype(name, storage) + ";"
                for name in self.unit.shims
                if name in later
            )
            lines.extend(definition(name, storage) for name in self.unit.shims)
        if self.unit.externs:
            if cpp:
                lines.append('extern "C" {')
            lines.extend(self.unit.externs.values())
            if cpp:
                lines.append("}")
            lines.append("")
        lines.extend(self.unit.strings.values())
        if cpp and not self.readable:
            # The internal ABI is a C ABI: the runtime binds these symbols
            # by name, so C++ must not mangle them.
            lines.append('extern "C" {\n')
        lines.extend(self.unit.prototypes)
        lines.append("")
        lines.extend(self.unit.trampolines.values())
        lines.extend(self.unit.functions)
        lines.extend(self.unit.exports)
        if cpp and not self.readable:
            lines.append('} /* extern "C" */')
        if self.header_only:
            lines.append(f"#endif /* {guard} */")
        return "\n".join(lines).rstrip("\n") + "\n"


#: C's precedence, as the level an expression binds at: higher binds tighter.
_ATOM = 100
_UNARY = 14
_TERNARY = 3
_INFIX = {
    "*": 13,
    "/": 13,
    "%": 13,
    "+": 12,
    "-": 12,
    "<<": 11,
    ">>": 11,
    "<": 10,
    "<=": 10,
    ">": 10,
    ">=": 10,
    "==": 9,
    "!=": 9,
    "&": 8,
    "^": 7,
    "|": 6,
    "&&": 5,
    "||": 4,
}
_FLIPPED = {"==": "!=", "!=": "==", "<": ">=", ">=": "<", ">": "<=", "<=": ">"}
_LITERAL = re.compile(
    r"^-?(\d[\w.+-]*|\.\d+)$|^-?(INFINITY|NAN)$|^INT64_MIN$|^U?INT64_C\(-?\d+\)$|^(true|false)$"
)

#: An expression as text, with the level its outermost operator binds at.
_Expr = tuple[str, int]


def _wrap(text: str, parenthesize: bool) -> str:
    return f"({text})" if parenthesize else text


def _infix(symbol: str, left: _Expr, right: _Expr) -> _Expr:
    """`left symbol right`, parenthesized where C's grammar or a compiler's
    `-Wparentheses` would want it."""
    own = _INFIX[symbol]
    (lt, ll), (rt, rl) = left, right
    if own >= 12:
        lp, rp = ll < own, rl <= own
    elif own == 11:
        # Arithmetic inside a shift reads as a mistake.
        lp, rp = ll <= 12, rl <= 12
    elif own >= 9:
        # Comparisons never chain.
        lp, rp = ll <= 10, rl <= 10
    elif own >= 6:
        lp, rp = ll < _UNARY, rl < _UNARY
    elif own == 5:
        lp, rp = ll < 5, rl <= 5
    else:
        # `&&` inside `||` is spelled out.
        lp, rp = ll < 4 or ll == 5, rl <= 5
    return f"{_wrap(lt, lp)} {symbol} {_wrap(rt, rp)}", own


def _prefix(symbol: str, operand: _Expr) -> _Expr:
    text, level = operand
    parenthesize = level < _UNARY or (symbol in "-+" and text.startswith(symbol))
    return f"{symbol}{_wrap(text, parenthesize)}", _UNARY


def _ternary(condition: _Expr, a: _Expr, b: _Expr) -> _Expr:
    (ct, cl), (at, al), (bt, bl) = condition, a, b
    chosen = f"{_wrap(at, al <= _TERNARY)} : {_wrap(bt, bl <= _TERNARY)}"
    return f"{_wrap(ct, cl <= 5)} ? {chosen}", _TERNARY


def _member(base: _Expr, name: str) -> _Expr:
    text, level = base
    return f"{_wrap(text, level < _ATOM)}.{name}", _ATOM


class _FunctionEmitter:
    """One IR function as one C function.

    Structured by default: the graph's dominator tree and loops become
    `while`, `if`/`else`, `break`, `continue`, and `return`. With
    `structured=False` -- the fallback for a graph the reconstruction
    cannot express -- blocks are labels and branches are `goto`s.
    """

    def __init__(self, owner: _ModuleEmitter, function: IRFunction, structured: bool = True):
        self.owner = owner
        self.function = function
        self.structured = structured
        #: What reads each value: a variable's name, or the expression itself
        #: when the value is written where it is used.
        self.scalars: dict[int, str] = {}
        #: The level a written-in-place expression binds at.
        self.levels: dict[int, int] = {}
        #: The memory an expression reads: slot names, or "*" for any.
        self.reads_of: dict[int, frozenset[str]] = {}
        #: A comparison written in place, spelled flipped, for its negation.
        self.flipped: dict[int, str] = {}
        #: An infix expression written in place: (symbol, left, right), for
        #: `x op= y` and `p[i]`.
        self.infixes: dict[int, tuple[str, _Expr, _Expr]] = {}
        #: Wrapping arithmetic written in place, before its cast back to signed.
        self.unsigned_form: dict[int, _Expr] = {}
        self.buffers: dict[int, _Buffer] = {}
        #: A stack slot only ever loaded and stored: a plain variable.
        self.slots: dict[int, str] = {}
        #: Slots stored once: nothing changes them after, so a load is the name.
        self.immutable: set[str] = set()
        #: Stores made for us, by whoever wrote the slot itself.
        self.skipped: set[int] = set()
        #: Declarations at the top of the function: slots, block parameters,
        #: and anything a `goto` may otherwise jump across.
        self.declarations: list[str] = []
        #: First stores that can introduce a variable in their lexical scope.
        self.slot_declarations: dict[int, str] = {}
        self.body = _Body()
        self.print_format: PrintFormat | None = None
        self.used: set[str] = set(owner.reserved)
        self.counter = 0
        self.labels: dict[int, str] = {}
        #: A kernel or a device function: it returns its value, and nothing falls back.
        self.device = kind_of(function) != "host"
        #: A coroutine's resume function: the frame in, nothing out.
        self.resume = function.attributes.get("ppy.abi") == "resume"
        self.direct = function.name in owner.direct
        self.truncating = truncating_divisions(function) if owner.readable else set()
        self.parameters = owner.parameter_atoms(function)
        self.used.update(name for _kind, name in self.parameters)
        self.results = (
            owner.result_atoms(function, self.used) if not (self.device or self.direct) else []
        )
        if self.resume:
            self.used.add("frame")
        # The graph.
        self.blocks = list(function.body.blocks)
        self.preds: dict[int, list] = {}
        self.idom: dict[int, object] = {}
        self.children: dict[int, list] = {}
        self.loops: dict[int, _Loop] = {}
        self.loop_of: dict[int, _Loop | None] = {}
        self.succ_block: dict[int, object] = {}
        self.loop_stack: list[_Loop] = []
        self.emitted: set[int] = set()
        #: Pure operations nothing reads, transitively: never written.
        self.dead: set[int] = set()
        for block in reversed(self.blocks):
            for op in reversed(block.operations):
                if op.results and _pure(op) and all(self._dead_use(v) for v in op.results):
                    self.dead.add(id(op))

    def _dead_use(self, v: Value) -> bool:
        return all(isinstance(u, Operation) and id(u) in self.dead for u, _i in v.uses)

    # -- names ----------------------------------------------------------------------

    def fresh(self, hint: str | None) -> str:
        """A local's name: the IR's, made unique; `t1`, `t2`, ... for an unnamed value."""
        if hint and not hint.isdigit():
            return _unique(_cname(hint), self.used)
        while True:
            self.counter += 1
            candidate = f"t{self.counter}"
            if candidate not in self.used:
                self.used.add(candidate)
                return candidate

    # -- values ------------------------------------------------------------------

    def bare(self, v: Value) -> str:
        """What reads `v` where no operator follows: the right of `=`, an argument."""
        found = self.scalars.get(id(v))
        if found is None:
            raise EmitError(f"value %{v.name or '?'} was never emitted")
        return found

    def expr(self, v: Value) -> _Expr:
        text = self.bare(v)
        level = self.levels.get(id(v))
        if level is None:
            level = _ATOM if _is_atomic(text) else 0
        return text, level

    def value(self, v: Value) -> str:
        """What reads `v` inside a larger expression: parenthesized when it needs to be."""
        text, level = self.expr(v)
        return text if level == _ATOM or _is_atomic(text) else f"({text})"

    def buffer(self, v: Value) -> _Buffer:
        found = self.buffers.get(id(v))
        if found is None:
            raise EmitError(f"buffer %{v.name or '?'} was never emitted")
        return found

    def line(self, text: str) -> None:
        self.body.append("    " + text)

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

    def variable(self, v: Value) -> str:
        """A variable for `v` with no initializer yet -- for a helper to write into."""
        name = self.fresh(v.name)
        spelled = _declare(self.owner.c_type(v.type), name)
        if self.structured and not self._crosses_scope(v):
            self.line(f"{spelled};")
        else:
            self.declarations.append(f"    {spelled};")
        self.scalars[id(v)] = name
        return name

    def define(self, v: Value, expression: str) -> str:
        """A variable holding `expression`; its name."""
        slot = self.sink(v)
        if slot is not None:
            self.line(f"{slot} = {expression};")
            return slot
        name = self.fresh(v.name)
        spelled = _declare(self.owner.c_type(v.type), name)
        if self.structured and not self._crosses_scope(v):
            self.line(f"{spelled} = {expression};")
        else:
            self.declarations.append(f"    {spelled};")
            self.line(f"{name} = {expression};")
        if not v.uses:
            # Computed for its effect alone; say so, or a C compiler warns.
            self.line(f"(void){name};")
        self.scalars[id(v)] = name
        return name

    def effect(self, v: Value, expression: str) -> str:
        """`expression` has an effect -- a call: written where `v` is read when
        that is a store or a return right after it, held in a variable otherwise."""
        if self.structured and self._consumed_next(v):
            self.scalars[id(v)] = expression
            self.levels[id(v)] = _ATOM if _is_atomic(expression) else 0
            return expression
        return self.define(v, expression)

    def fold(self, v: Value, expression: _Expr, *, reads: frozenset[str] = frozenset()) -> str:
        """A pure value: written where it is used when that reads the same,
        held in a variable otherwise, dropped when nothing reads it."""
        text, level = expression
        if not v.uses:
            return text
        reads = reads - self.immutable
        if self._foldable(v, text, reads):
            self.scalars[id(v)] = text
            self.levels[id(v)] = level
            self.reads_of[id(v)] = reads
            return text
        return self.define(v, text)

    def infix(self, v: Value, symbol: str, a: _Expr, b: _Expr, reads: frozenset[str]) -> None:
        """`a symbol b` as `v`, remembered by its parts for a `symbol=` store."""
        text, level = _infix(symbol, a, b)
        if self.fold(v, (text, level), reads=reads) == text:
            self.infixes[id(v)] = (symbol, a, b)

    def operands(self, op: Operation) -> tuple[list[_Expr], frozenset[str]]:
        """The operands as expressions, and the memory they read between them."""
        return [self.expr(v) for v in op.operands], self.reading(op.operands)

    def reading(self, values: Sequence[Value]) -> frozenset[str]:
        reads: frozenset[str] = frozenset()
        for v in values:
            reads |= self.reads_of.get(id(v), frozenset())
        return reads

    def temporary(self, hint: str, spelled: str, expression: str) -> str:
        """A helper variable of the emitter's own, initialized where it is made."""
        return self.local(spelled, hint, expression)

    def local(self, spelled: str, hint: str, expression: str | None = None) -> str:
        """A variable of the emitter's or a dialect's own: declared where it is
        made, or at the top where a `goto` may otherwise cross it; its name."""
        name = self.fresh(hint)
        declaration = _declare(spelled, name)
        if self.structured:
            self.line(
                f"{declaration} = {expression};" if expression is not None else f"{declaration};"
            )
        else:
            self.declarations.append(f"    {declaration};")
            if expression is not None:
                self.line(f"{name} = {expression};")
        return name

    def _foldable(self, v: Value, text: str, reads: frozenset[str]) -> bool:
        cheap = _IDENTIFIER.match(text) is not None or _LITERAL.match(text) is not None
        if cheap and not reads:
            # A name or a literal reads the same anywhere.
            return True
        producer = v.owner
        if not isinstance(producer, Operation) or producer.parent is None:
            return False
        block = producer.parent
        users: list[Operation] = []
        for user, _index in v.uses:
            op = user if isinstance(user, Operation) else self.succ_block[id(user)].terminator
            if op is None or op.parent is not block:
                return False
            users.append(op)
        if not cheap and len(users) != 1:
            return False
        operations = block.operations
        last = max(operations.index(u) for u in users)
        between = operations[operations.index(producer) + 1 : last]
        return not any(self._clobbers(op, reads) for op in between)

    def _clobbers(self, op: Operation, reads: frozenset[str]) -> bool:
        """Whether `op` may change what an expression reading `reads` reads."""
        if not reads:
            return False
        if op.name == "core.store":
            slot = self.slots.get(id(op.operands[1]))
            if slot is not None:
                return slot in reads
            return "*" in reads
        if _pure(op):
            return False
        # A call, a buffer store, a dialect's operation: anything but a plain slot.
        return "*" in reads

    def _crosses_scope(self, v: Value) -> bool:
        """Whether a use of `v` lies outside the loop its definition is written in."""
        producer = v.owner
        home_block = producer.parent if isinstance(producer, Operation) else producer
        home = self.loop_of.get(id(home_block))
        for user, _index in v.uses:
            block = user.parent if isinstance(user, Operation) else self.succ_block.get(id(user))
            if block is None:
                return True
            loop = self.loop_of.get(id(block))
            while loop is not None and loop is not home:
                loop = self._enclosing(loop)
            if loop is not home:
                return True
        return False

    def _enclosing(self, loop: _Loop) -> _Loop | None:
        outer = self.idom.get(id(loop.header))
        return self.loop_of.get(id(outer)) if outer is not None else None

    def sink(self, v: Value) -> str | None:
        """The plain slot `v` goes straight into, when a store right after its
        definition is all that reads it: whoever makes `v` writes the slot."""
        if len(v.uses) != 1 or not isinstance(v.owner, Operation):
            return None
        user, index = v.uses[0]
        while (
            isinstance(user, Operation)
            and user.name == "core.cast"
            and len(user.result.uses) == 1
            and self.owner.c_type(user.operands[0].type) == self.owner.c_type(user.result.type)
        ):
            # A cast between types C spells the same: the store is on its other side.
            user, index = user.result.uses[0]
        if not isinstance(user, Operation) or user.name != "core.store" or index != 0:
            return None
        block = v.owner.parent
        if block is None or user.parent is not block:
            return None
        slot = self.slots.get(id(user.operands[1]))
        if slot is None:
            return None
        operations = block.operations
        between = operations[operations.index(v.owner) + 1 : operations.index(user)]
        if not all(_pure(op) for op in between):
            return None
        # Writing `v` into the slot where it is made moves the store up past
        # these operations, so none of them may read the slot's old value.
        target = user.operands[1]
        if any(op.name == "core.load" and op.operands[0] is target for op in between):
            return None
        self.skipped.add(id(user))
        self.scalars[id(v)] = slot
        declaration = self.slot_declarations.pop(id(user), None)
        if declaration is not None:
            self.line(f"{declaration};")
        return slot

    def _consumed_next(self, v: Value) -> bool:
        """Whether the one thing reading `v` is a store or a return right after it:
        an impure expression can then be written there, nothing running between."""
        if len(v.uses) != 1 or not isinstance(v.owner, Operation):
            return False
        user, index = v.uses[0]
        if not isinstance(user, Operation) or user.parent is not v.owner.parent:
            return False
        if user.name not in {"core.store", "core.buffer_store", "core.ret"} or index != 0:
            return False
        if user.name == "core.ret" and isinstance(v.type, (TupleType, StructType, BoolType)):
            return False
        block = user.parent
        if block is None:
            return False
        operations = block.operations
        return operations.index(user) == operations.index(v.owner) + 1

    def define_buffer(self, v: Value, data: str, length: str) -> None:
        self.declare(v)
        b = self.buffer(v)
        self.line(f"{b.data} = {data};")
        self.line(f"{b.length} = {length};")
        # The length is read only where a `core.buffer_len` or a call
        # takes the buffer whole; a C compiler warns about the rest.
        if not v.uses:
            self.line(f"(void){b.data};")
        self.line(f"(void){b.length};")

    def reads(self, v: Value) -> list[str]:
        """The expressions that read `v`: one, or a buffer's two."""
        if isinstance(v.type, BufferType):
            b = self.buffer(v)
            return [b.data, b.length]
        return [self.bare(v)]

    def flatten(self, v: Value) -> list[str]:
        """The atoms `v` crosses a boundary as."""
        t = v.type
        if isinstance(t, BufferType):
            return self.reads(v)
        if isinstance(t, BoolType):
            return [self.cast_text(self.expr(v), "int8_t")[0]]
        if isinstance(t, (TupleType, StructType)):
            producer = v.owner
            if (
                id(v) in self.levels
                and isinstance(producer, Operation)
                and producer.name in _MAKERS
            ):
                # Written in place: its parts cross one by one, with no aggregate between.
                return [atom for part in producer.operands for atom in self.flatten(part)]
            return self._flatten_fields(self.expr(v), t)
        return [self.bare(v)]

    def _flatten_fields(self, expression: _Expr, t: TupleType | StructType) -> list[str]:
        atoms: list[str] = []
        for field_name, item in _fields(t):
            member = _member(expression, field_name)
            if isinstance(item, BoolType):
                atoms.append(self.cast_text(member, "int8_t")[0])
            elif isinstance(item, (TupleType, StructType)):
                atoms.extend(self._flatten_fields(member, item))
            else:
                atoms.append(member[0])
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

    def cast_text(self, operand: _Expr, t: IRType | str) -> _Expr:
        spelled = t if isinstance(t, str) else self.owner.c_type(t)
        text, level = operand
        if self.owner.cpp and not spelled.endswith("*"):
            if _IDENTIFIER.match(spelled):
                # C++'s functional cast: `int64_t(x)`.
                return f"{spelled}({text})", _ATOM
            return f"static_cast<{spelled}>({text})", _ATOM
        return f"({spelled}){_wrap(text, level < _UNARY)}", _UNARY

    def deref(self, pointer: Value) -> _Expr:
        """What `pointer` points at: `p[i]` for an offset written in place, `*p` otherwise."""
        parts = self.infixes.get(id(pointer))
        if parts is not None and parts[0] == "+":
            _symbol, base, index = parts
            return f"{_wrap(base[0], base[1] < _ATOM)}[{index[0]}]", _ATOM
        return _prefix("*", self.expr(pointer))

    def negated(self, v: Value) -> str:
        """`!v`: a comparison flipped, an `and`/`or` of them turned around, `!` otherwise."""
        exact = self._negation(v)
        if exact is not None:
            return exact[0]
        text, level = self.expr(v)
        return _negate(text) if level == _ATOM or _is_atomic(text) else f"!({text})"

    def _negation(self, v: Value) -> _Expr | None:
        flipped = self.flipped.get(id(v))
        if flipped is not None:
            return flipped, _INFIX["=="]
        producer = v.owner
        if (
            id(v) not in self.levels
            or not isinstance(producer, Operation)
            or producer.name not in {"core.and", "core.or"}
            or not isinstance(v.type, BoolType)
        ):
            return None
        parts = [self._negation(operand) for operand in producer.operands]
        if any(part is None for part in parts):
            return None
        a, b = parts
        assert a is not None and b is not None
        return _infix("||" if producer.name == "core.and" else "&&", a, b)

    def fail_unless(
        self,
        condition: str,
        label: str,
        *,
        failed: str | None = None,
        raises: str = "",
        values: tuple[str, ...] = (),
    ) -> None:
        """Leave the function the way a failed guard does, unless `condition`
        holds; `failed` spells its negation where the caller knows a better one.

        In a standalone program, `raises` is what the failure says instead.
        """
        raised = self.raised(raises, values)
        if not self.structured:
            leave = raised or "goto fallback;"
            self.body.append(f"    if (!({condition})) {leave} /* {label} */")
            return
        self.line(f"if ({failed or _negate(condition)}) {raised or self.failure()} /* {label} */")

    def guard(self, v: Value, label: str, raises: str = "", values: tuple[Value, ...] = ()) -> None:
        condition = self.bare(v)
        if condition == "true":
            return
        shown = tuple(self.bare(value) for value in values)
        if condition == "false" and self.structured:
            self.line(f"{self.raised(raises, shown) or self.failure()} /* {label} */")
            return
        self.fail_unless(condition, label, failed=self.negated(v), raises=raises, values=shown)

    def raised(self, raises: str, values: tuple[str, ...]) -> str:
        """In a standalone program, the statement that says what CPython would
        raise and exits; elsewhere nothing, and the failure falls back."""
        if not raises or not self.owner.standalone or self.resume:
            return ""
        self.owner.shim("ppy_rt_raise")
        padded = [f"(int64_t)({value})" for value in values]
        padded += ["0"] * (RAISED_VALUES - len(padded))
        text = string_literal(raises.encode("utf-8"))
        return f"ppy_rt_raise({text}, {len(values)}, {', '.join(padded)});"

    def failure(self) -> str:
        """The statement a failed guard runs: the fallback status, or a failed frame."""
        if self.resume:
            declare_async(self.owner, "ppy_aio_fail")
            return "{ ppy_aio_fail(frame, 0); return; }"
        if self.direct:
            if self.owner.is_main(self.function):
                return "return 1;"
            self.owner.unit.headers.add("stdlib.h")
            return f"{self.owner.std('exit')}(1);"
        if self.owner.collections and not self.owner.standalone:
            # What this call made is garbage now; the runtime frees it.
            self.owner.shim("ppy_coll_failed")
            return f"{{ ppy_coll_failed(); return {STATUS_FALLBACK}; }}"
        return f"return {STATUS_FALLBACK};"

    # -- the function ------------------------------------------------------------

    def run(self) -> str:
        function = self.function
        entry = function.entry
        assert entry is not None
        self._analyze()
        position = 0
        for argument in entry.arguments:
            position = self.bind_parameter(argument, position)
        self._name_slots()
        for block in self.blocks[1:]:
            for argument in block.arguments:
                self.declare(argument)
        if self.structured:
            self._region(entry, None)
            self._hoist()
            if self.body and self.body[-1] == "    return;":
                self.body.pop()
        else:
            self._labels()
        head = self.owner.prototype(function)
        parts = [head + " {", *self.declarations]
        if self.declarations and self.body:
            parts.append("")
        parts.extend(self.body)
        parts.append("}")
        return "\n".join(parts) + "\n"

    def bind_parameter(self, argument: Value, position: int) -> int:
        t = argument.type
        if isinstance(t, BufferType):
            data, length = self.parameters[position][1], self.parameters[position + 1][1]
            self.buffers[id(argument)] = _Buffer(data, length)
            return position + 2
        if isinstance(t, BoolType):
            name = self.parameters[position][1]
            self.scalars[id(argument)] = f"{name} != 0"
            self.levels[id(argument)] = _INFIX["!="]
            self.flipped[id(argument)] = f"{name} == 0"
            return position + 1
        if isinstance(t, (TupleType, StructType)):
            count = len(self.owner.atoms(t))
            atoms = [self.parameters[position + i][1] for i in range(count)]
            expression, _ = self.rebuild(t, atoms, 0)
            if argument.uses:
                self.define(argument, expression)
            return position + count
        self.scalars[id(argument)] = self.parameters[position][1]
        return position + 1

    # -- the graph -----------------------------------------------------------------

    def _analyze(self) -> None:
        """Predecessors, dominators, and loops of the function's graph."""
        blocks = self.blocks
        self.preds = {id(b): [] for b in blocks}
        for block in blocks:
            terminator = block.terminator
            if terminator is None:
                continue
            for successor in terminator.successors:
                self.preds[id(successor.block)].append(block)
                self.succ_block[id(successor)] = block
        order: list = []
        seen: set[int] = {id(blocks[0])}
        stack = [(blocks[0], iter(blocks[0].successors))]
        while stack:
            block, successors = stack[-1]
            for successor in successors:
                if id(successor) not in seen:
                    seen.add(id(successor))
                    stack.append((successor, iter(successor.successors)))
                    break
            else:
                stack.pop()
                order.append(block)
        rpo = order[::-1]
        position = {id(b): i for i, b in enumerate(rpo)}
        idom: dict[int, object] = {id(rpo[0]): rpo[0]}

        def intersect(a, b):  # type: ignore[no-untyped-def]
            while a is not b:
                while position[id(a)] > position[id(b)]:
                    a = idom[id(a)]
                while position[id(b)] > position[id(a)]:
                    b = idom[id(b)]
            return a

        changed = True
        while changed:
            changed = False
            for block in rpo[1:]:
                candidate = None
                for pred in self.preds[id(block)]:
                    if id(pred) not in idom:
                        continue
                    candidate = pred if candidate is None else intersect(pred, candidate)
                if candidate is not None and idom.get(id(block)) is not candidate:
                    idom[id(block)] = candidate
                    changed = True
        idom[id(rpo[0])] = None
        self.idom = idom
        self.children = {id(b): [] for b in blocks}
        for block in rpo[1:]:
            parent = idom.get(id(block))
            if parent is not None:
                self.children[id(parent)].append(block)
        for block in rpo:
            for successor in block.successors:
                if self._dominates(successor, block):
                    self.loops.setdefault(id(successor), _Loop(successor)).latches.append(block)
        for loop in self.loops.values():
            loop.body = {id(loop.header)}
            work = list(loop.latches)
            while work:
                block = work.pop()
                if id(block) in loop.body:
                    continue
                loop.body.add(id(block))
                work.extend(self.preds[id(block)])
            for member in blocks:
                if id(member) not in loop.body:
                    continue
                for successor in member.successors:
                    if id(successor) not in loop.body and successor not in loop.exits:
                        loop.exits.append(successor)
        self.loop_of = {id(b): None for b in blocks}
        for loop in sorted(self.loops.values(), key=lambda found: -len(found.body)):
            for member in blocks:
                if id(member) in loop.body:
                    self.loop_of[id(member)] = loop

    def _dominates(self, a, b) -> bool:  # type: ignore[no-untyped-def]
        while b is not None:
            if b is a:
                return True
            b = self.idom.get(id(b))
        return False

    def _entering(self, block) -> list:  # type: ignore[no-untyped-def]
        """The predecessors of `block` that are not back edges into it."""
        loop = self.loops.get(id(block))
        latches = set(map(id, loop.latches)) if loop is not None else set()
        return [p for p in self.preds[id(block)] if id(p) not in latches]

    def _name_slots(self) -> None:
        """A stack slot that is only loaded and stored is a variable named
        after it -- or the parameter itself, when the slot is that parameter's home."""
        entry = self.blocks[0]
        for block in self.blocks:
            for op in block.operations:
                if op.name != "core.alloca" or op.attributes.get("count", 1) != 1:
                    continue
                pointer = op.result
                if not pointer.uses:
                    continue
                stores: list[Operation] = []
                plain = True
                for user, index in pointer.uses:
                    if not isinstance(user, Operation):
                        plain = False
                    elif user.name == "core.store" and index == 1:
                        stores.append(user)
                    elif not (user.name == "core.load" and index == 0):
                        plain = False
                if not plain:
                    continue
                t = pointer.type
                assert isinstance(t, PtrType)
                name = self._parameter_slot(stores, entry)
                if name is None:
                    hint = re.sub(r"[._]addr$", "", pointer.name or "") or "slot"
                    name = self.fresh(hint)
                    declaration = _declare(self.owner.c_type(t.pointee), name)
                    first = self._slot_start(pointer, stores)
                    if first is not None:
                        self.slot_declarations[id(first)] = declaration
                    else:
                        self.declarations.append(f"    {declaration};")
                self.slots[id(pointer)] = name
                if len(stores) <= 1:
                    self.immutable.add(name)

    def _slot_start(self, pointer: Value, stores: list[Operation]) -> Operation | None:
        """A store dominating every access, in a scope containing every access.

        A loop-local declaration must be initialized before each iteration's
        reads; a value carried around a back edge remains outside that loop.
        """
        if not self.structured or not self.owner.readable:
            return None
        users = [user for user, _ in pointer.uses if isinstance(user, Operation)]
        for store in stores:
            block = store.parent
            if block is None:
                continue
            if any(
                not self._dominates(block, user.parent)
                or (
                    user.parent is block
                    and block.operations.index(user) < block.operations.index(store)
                )
                for user in users
            ):
                continue
            loop = self.loop_of.get(id(block))
            if loop is not None and any(id(user.parent) not in loop.body for user in users):
                continue
            # Entry and the unconditional start of a while body have stable
            # lexical scopes. Elsewhere, keep only block-local declarations.
            body_start = (
                loop is not None
                and block is not loop.header
                and block in loop.header.successors
                and self._entering(block) == [loop.header]
            )
            if block is self.blocks[0] or body_start or all(user.parent is block for user in users):
                return store
        return None

    def _parameter_slot(self, stores: list[Operation], entry) -> str | None:  # type: ignore[no-untyped-def]
        """The parameter a slot is the home of: stored into it on entry and
        read nowhere else, so the C parameter is the variable."""
        if not stores:
            return None
        first = stores[0]
        stored = first.operands[0]
        if first.parent is not entry or stored.owner is not entry or len(stored.uses) != 1:
            return None
        name = self.scalars.get(id(stored))
        if name is None or _IDENTIFIER.match(name) is None or id(stored) in self.levels:
            return None
        self.skipped.add(id(first))
        return name

    # -- structured control flow -----------------------------------------------------

    def _region(self, block, follow) -> None:  # type: ignore[no-untyped-def]
        """Write `block` and what it owns, until control reaches `follow` or leaves."""
        if id(block) in self.emitted:
            raise _Unstructured(f"^{block.name} is reached twice")
        self.emitted.add(id(block))
        loop = self.loops.get(id(block))
        if loop is None:
            self._block(block, follow)
            return
        if len(loop.exits) > 1:
            raise _Unstructured(f"the loop at ^{block.name} leaves to several places")
        exit_ = loop.exit
        if exit_ is not None and exit_ is not follow and not self._dominates(block, exit_):
            raise _Unstructured(f"the loop at ^{block.name} leaves to ^{exit_.name}")
        loop.writer = self.body
        loop.start = len(self.body)
        self.line("while (1) {")
        self.body.depth += 1
        self.loop_stack.append(loop)
        self._block(block, None)
        self.loop_stack.pop()
        self.body.depth -= 1
        inner = "    " * (self.body.depth + 1)
        if len(self.body) > loop.start + 1 and self.body[-1] == inner + "continue;":
            self.body.pop()
        self.line("}")
        if exit_ is not None and exit_ is not follow:
            self._region(exit_, follow)

    def _block(self, block, follow) -> None:  # type: ignore[no-untyped-def]
        terminator = block.terminator
        if terminator is None:
            raise _Unstructured(f"^{block.name} has no terminator")
        for op in block.operations[:-1]:
            if id(op) not in self.dead:
                self.op(op)
        self.flush_print()
        if terminator.name == "core.br":
            self._jump(block, terminator.successors[0], follow)
        elif terminator.name == "core.cond_br":
            self._conditional(block, terminator, follow)
        else:
            self.op(terminator)

    def _jump(self, block, successor: Successor, follow, *, inline: bool = True) -> None:  # type: ignore[no-untyped-def]
        """Control leaves `block` for `successor`: inline, fall through, continue, or break."""
        target = successor.block
        loop = self.loop_stack[-1] if self.loop_stack else None
        if loop is not None and target is loop.header:
            self._moves(block, successor)
            self.line("continue;")
            return
        if loop is not None and target is loop.exit:
            self._moves(block, successor)
            self.line("break;")
            return
        if target is follow:
            self._moves(block, successor)
            return
        if (
            inline
            and self.idom.get(id(target)) is block
            and self._entering(target) == [block]
            and id(target) not in self.emitted
        ):
            self._moves(block, successor)
            self._region(target, follow)
            return
        raise _Unstructured(f"^{block.name} jumps to ^{target.name}")

    def _conditional(self, block, terminator: Operation, follow) -> None:  # type: ignore[no-untyped-def]
        chooser = terminator.operands[0]
        condition = self.bare(chooser)
        then, otherwise = terminator.successors
        protected: set[int] = set()
        for loop in self.loop_stack:
            protected.add(id(loop.header))
            if loop.exit is not None:
                protected.add(id(loop.exit))
        merge = None
        for child in self.children[id(block)]:
            if child is then.block or child is otherwise.block or id(child) in protected:
                continue
            if merge is not None:
                raise _Unstructured(f"^{block.name} has two join points")
            merge = child
        for successor in (then, otherwise):
            target = successor.block
            if target is follow or id(target) in protected:
                continue
            if len(self._entering(target)) > 1:
                if merge is not None and merge is not target:
                    raise _Unstructured(f"^{block.name} has two join points")
                merge = target
        inner = merge if merge is not None else follow

        def kind(successor: Successor) -> str:
            target = successor.block
            if target is merge or target is inner or id(target) in protected:
                return "jump"
            if self.idom.get(id(target)) is block and self._entering(target) == [block]:
                return "own"
            raise _Unstructured(f"^{block.name} branches to ^{target.name}, reached elsewhere")

        kinds = kind(then), kind(otherwise)
        depth = self.body.depth

        def jump(successor: Successor, at: int) -> list[str]:
            return self._capture(lambda: self._jump(block, successor, inner, inline=False), at)

        def own(successor: Successor, at: int) -> list[str]:
            def emit() -> None:
                self._moves(block, successor)
                self._region(successor.block, inner)

            return self._capture(emit, at)

        # The loop whose header this block is, when its `while (1)` line is the
        # last thing written: the test can then be the loop's own.
        header: _Loop | None = None
        if self.loop_stack:
            loop = self.loop_stack[-1]
            if (
                block is loop.header
                and self.body is loop.writer
                and len(self.body) == loop.start + 1
            ):
                header = loop
        if kinds == ("jump", "jump"):
            then_lines, else_lines = jump(then, depth + 1), jump(otherwise, depth + 1)
            self._if_else(condition, self.negated(chooser), then_lines, else_lines)
        elif kinds == ("own", "jump"):
            else_lines = jump(otherwise, depth + 1)
            if header is not None and _only_break(else_lines):
                self.body[header.start] = "    " * (depth - 1) + f"while ({condition}) {{"
                self.body.extend(own(then, depth))
            elif _leaves(else_lines):
                self._if(self.negated(chooser), else_lines)
                self.body.extend(own(then, depth))
            else:
                self._if_else(condition, self.negated(chooser), own(then, depth + 1), else_lines)
        elif kinds == ("jump", "own"):
            then_lines = jump(then, depth + 1)
            if header is not None and _only_break(then_lines):
                self.body[header.start] = (
                    "    " * (depth - 1) + f"while ({self.negated(chooser)}) {{"
                )
                self.body.extend(own(otherwise, depth))
            elif _leaves(then_lines):
                self._if(condition, then_lines)
                self.body.extend(own(otherwise, depth))
            else:
                self._if_else(
                    condition, self.negated(chooser), then_lines, own(otherwise, depth + 1)
                )
        else:
            self._if_else(
                condition, self.negated(chooser), own(then, depth + 1), own(otherwise, depth + 1)
            )
        if merge is not None:
            self._region(merge, follow)

    def _capture(self, emit, depth: int) -> list[str]:  # type: ignore[no-untyped-def]
        """The lines `emit` writes, at `depth`, without writing them yet."""
        saved = self.body
        self.body = _Body(depth)
        try:
            emit()
            return list(self.body)
        finally:
            self.body = saved

    def _if(self, condition: str, lines: list[str]) -> None:
        if not lines:
            return
        if len(lines) == 1 and _SIMPLE_JUMP.match(lines[0].strip()):
            self.line(f"if ({condition}) {lines[0].strip()}")
            return
        self.line(f"if ({condition}) {{")
        self.body.extend(lines)
        self.line("}")

    def _if_else(
        self, condition: str, negated: str, then_lines: list[str], else_lines: list[str]
    ) -> None:
        if not then_lines:
            self._if(negated, else_lines)
            return
        if not else_lines:
            self._if(condition, then_lines)
            return
        if len(then_lines) == 1 and _leaves(then_lines):
            # `if (c) return;` needs no `else`.
            self._if(condition, then_lines)
            self.body.extend(line.removeprefix("    ") for line in else_lines)
            return
        self.line(f"if ({condition}) {{")
        self.body.extend(then_lines)
        if _one_if(else_lines):
            # `else if`: the branch's own `if` joins this one's `else`.
            self.line(f"}} else {else_lines[0].strip()}")
            self.body.extend(line.removeprefix("    ") for line in else_lines[1:])
            return
        self.line("} else {")
        self.body.extend(else_lines)
        self.line("}")

    def _moves(self, block, successor: Successor) -> None:  # type: ignore[no-untyped-def]
        """Assign the target's parameters -- all at once -- for one edge."""
        target = successor.block
        pairs: list[tuple[str, str, str]] = []
        for argument, value in zip(target.arguments, successor.arguments, strict=True):
            for destination, source in zip(self.reads(argument), self.reads(value), strict=True):
                if destination != source:
                    pairs.append((destination, source, self.slot_type(argument, destination)))
        if not pairs:
            return
        destinations = [d for d, _s, _t in pairs]
        overlap = any(
            re.search(rf"\b{re.escape(d)}\b", source)
            for d in destinations
            for _d, source, _t in pairs
        )
        if not overlap:
            for destination, source, _t in pairs:
                self.line(f"{destination} = {source};")
            return
        staged = [(d, self.temporary("pass", t, source)) for d, source, t in pairs]
        for destination, held in staged:
            self.line(f"{destination} = {held};")

    def _hoist(self) -> None:
        """A top declaration whose first mention is a whole assignment, and
        whose every mention lies in the block of that assignment, becomes
        the assignment's declaration."""
        kept: list[str] = []
        body = self.body
        for declaration in self.declarations:
            match = re.fullmatch(r"    ((?:[\w:]+ )+\**)(\w+);", declaration)
            if match is None:
                kept.append(declaration)
                continue
            declared, name = str(match.group(1)), str(match.group(2))
            spelled = declared if declared.endswith("*") else declared.rstrip()
            pattern = re.compile(rf"\b{re.escape(name)}\b")
            mentions = [i for i, line in enumerate(body) if pattern.search(line)]
            if not mentions:
                kept.append(declaration)
                continue
            first = mentions[0]
            indent = _indent(body[first])
            prefix = " " * indent + f"{name} = "
            rest = body[first][len(prefix) :]
            if not body[first].startswith(prefix) or not rest.endswith(";") or pattern.search(rest):
                kept.append(declaration)
                continue
            # The block the assignment stands in: the lines that open and close it.
            opener = next(
                (i for i in range(first - 1, -1, -1) if 0 <= _indent(body[i]) < indent), -1
            )
            closer = next(
                (i for i in range(first + 1, len(body)) if 0 <= _indent(body[i]) < indent),
                len(body),
            )
            if any(not opener < i < closer for i in mentions):
                kept.append(declaration)
                continue
            body[first] = " " * indent + f"{_declare(spelled, name)} = {rest}"
        self.declarations = kept

    # -- labels: the fallback --------------------------------------------------------

    def _labels(self) -> None:
        for index, block in enumerate(self.blocks):
            self.labels[id(block)] = f"L{index}_{_ident(block.name)}"
        for index, block in enumerate(self.blocks):
            if index:
                self.body.append(f"{self.labels[id(block)]}:;")
            for op in block.operations:
                if id(op) not in self.dead:
                    self.op(op)
            self.flush_print()
        if any("goto fallback;" in line for line in self.body):
            self.body.append("fallback:")
            if self.resume:
                declare_async(self.owner, "ppy_aio_fail")
                self.body.append("    ppy_aio_fail(frame, 0);")
                self.body.append("    return;")
            else:
                self.body.append(f"    {self.failure()}")

    def branch(self, successor: Successor, indent: str) -> None:
        """Labels mode: assign the target's parameters, then jump."""
        block = successor.block
        if successor.arguments:
            staged: list[tuple[str, str, str]] = []
            for argument, value in zip(block.arguments, successor.arguments, strict=True):
                for target, expression in zip(self.reads(argument), self.reads(value), strict=True):
                    held = self.fresh("pass")
                    spelled = self.slot_type(argument, target)
                    self.declarations.append(f"    {_declare(spelled, held)};")
                    staged.append((target, held, expression))
            for _target, held, expression in staged:
                self.body.append(f"{indent}{held} = {expression};")
            for target, held, _expression in staged:
                self.body.append(f"{indent}{target} = {held};")
        self.body.append(f"{indent}goto {self.labels[id(block)]};")

    def slot_type(self, v: Value, slot: str) -> str:
        if isinstance(v.type, BufferType):
            return "int64_t" if slot.endswith("_len") else f"{self.owner.c_type(v.type.element)} *"
        return self.owner.c_type(v.type)

    # -- operations ----------------------------------------------------------------

    def op(self, op: Operation) -> None:
        if self.owner.readable:
            if self.print_operation(op):
                return
            setup = op.name == "core.const" or (
                op.name == "core.call_intrinsic"
                and op.attributes.get("intrinsic") == "ppy.string_data"
            )
            if not setup:
                self.flush_print()
        c = self.owner.c_type
        name = op.local_name
        if op.dialect == "math":
            self.math(op, name, [self.bare(v) for v in op.operands])
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
                self.fold(op.result, self.literal(op.result.type, op.attributes["value"]))
            case "add" | "sub" | "mul":
                self.arith(op, name)
            case "div" | "mod":
                self.divmod(op, name)
            case "neg":
                self.neg(op)
            case "and" | "or" | "xor":
                symbol = {"and": "&", "or": "|", "xor": "^"}[name]
                (a, b), reads = self.operands(op)
                self.infix(op.result, symbol, a, b, reads)
            case "shl" | "shr":
                self.shift(op, name)
            case "cmp":
                self.cmp(op)
            case "select":
                (cond, a, b), reads = self.operands(op)
                self.fold(op.result, _ternary(cond, a, b), reads=reads)
            case "cast":
                self.cast(op)
            case "br":
                self.branch(op.successors[0], "    ")
            case "cond_br":
                self.body.append(f"    if ({self.bare(op.operands[0])}) {{")
                self.branch(op.successors[0], "        ")
                self.body.append("    } else {")
                self.branch(op.successors[1], "        ")
                self.body.append("    }")
            case "ret":
                if self.direct:
                    if self.owner.is_main(self.function):
                        for value in op.operands:
                            self.line(f"(void)({self.bare(value)});")
                        self.line("return 0;")
                    else:
                        self.line(
                            f"return {self.bare(op.operands[0])};" if op.operands else "return;"
                        )
                    return
                if self.device or self.resume:
                    atoms = [atom for value in op.operands for atom in self.flatten(value)]
                    self.line(f"return {atoms[0]};" if atoms else "return;")
                    return
                position = 0
                for value in op.operands:
                    for atom in self.flatten(value):
                        self.line(f"*{self.results[position][1]} = {atom};")
                        position += 1
                if not op.operands:
                    self.line(f"*{self.results[0][1]} = 0;")
                self.line(f"return {STATUS_OK};")
            case "unreachable":
                self.line("PPY_UNREACHABLE();")
            case "alloca":
                if id(op.result) in self.slots:
                    return
                pointer = op.result.type
                assert isinstance(pointer, PtrType)
                slot = self.fresh(op.result.name or "slot")
                count = int(op.attributes.get("count", 1))  # type: ignore[call-overload]
                slots = max(count, 1)
                self.declarations.append(f"    {_declare(c(pointer.pointee), slot)}[{slots}];")
                self.scalars[id(op.result)] = slot
            case "load":
                slot = self.slots.get(id(op.operands[0]))
                if slot is not None:
                    self.fold(op.result, (slot, _ATOM), reads=frozenset({slot}))
                else:
                    reads = self.reading(op.operands) | {"*"}
                    self.fold(op.result, self.deref(op.operands[0]), reads=reads)
            case "store":
                if id(op) in self.skipped:
                    return
                value, pointer = op.operands
                slot = self.slots.get(id(pointer))
                if slot is None:
                    self.line(f"{self.deref(pointer)[0]} = {self.bare(value)};")
                    return
                declaration = self.slot_declarations.pop(id(op), None)
                if declaration is not None:
                    self.line(f"{declaration} = {self.bare(value)};")
                    return
                parts = self.infixes.get(id(value))
                if parts is not None and parts[1][0] == slot:
                    symbol, _left, right = parts
                    self.line(f"{slot} {symbol}= {right[0]};")
                else:
                    self.line(f"{slot} = {self.bare(value)};")
            case "ptr_offset":
                (pointer, offset), reads = self.operands(op)
                self.infix(op.result, "+", pointer, offset, reads)
            case "buffer_data":
                self.fold(op.result, (self.buffer(op.operands[0]).data, _ATOM))
            case "buffer_len":
                self.fold(op.result, (self.buffer(op.operands[0]).length, _ATOM))
            case "buffer_load":
                b = self.buffer(op.operands[0])
                index = op.operands[1]
                reads = self.reading([index]) | {"*"}
                self.fold(op.result, (f"{b.data}[{self.bare(index)}]", _ATOM), reads=reads)
            case "buffer_store":
                b = self.buffer(op.operands[1])
                self.line(f"{b.data}[{self.bare(op.operands[2])}] = {self.bare(op.operands[0])};")
            case "tuple_make" | "struct_make":
                t = op.result.type
                assert isinstance(t, (TupleType, StructType))
                parts, reads = self.operands(op)
                literal = self.owner.literal_of(self.owner.aggregate(t), [p for p, _l in parts])
                self.fold(op.result, (literal, _ATOM), reads=reads)
            case "tuple_extract":
                index = int(op.attributes["index"])  # type: ignore[call-overload]
                (whole,), reads = self.operands(op)
                self.fold(op.result, _member(whole, f"f{index}"), reads=reads)
            case "struct_extract":
                field_name = _ident(str(op.attributes["field"]))
                (whole,), reads = self.operands(op)
                self.fold(op.result, _member(whole, field_name), reads=reads)
            case "call":
                self.call(op)
            case "call_extern":
                self.call_extern(op)
            case "call_intrinsic":
                self.call_intrinsic(op)
            case "guard":
                label = str(op.attributes.get("label") or f"{op.attributes['kind']}.ok")
                if label.startswith("sanitize:") and not self.direct:
                    # A sanitizer's check returns its status; nothing falls back.
                    status = STATUS_SANITIZER_BASE + SANITIZERS.index(label.partition(":")[2])
                    if self.structured:
                        failed = self.negated(op.operands[0])
                        self.line(f"if ({failed}) return {status}; /* {label} */")
                    else:
                        condition = self.bare(op.operands[0])
                        self.body.append(f"    if (!({condition})) return {status}; /* {label} */")
                else:
                    self.guard(op.operands[0], label, raised_text(op), tuple(op.operands[1:]))
            case _:
                raise EmitError(f"{op.name} has no C lowering")

    def literal(self, t: IRType, value: object) -> _Expr:
        if isinstance(t, BoolType):
            if not self.owner.cpp:
                self.owner.unit.headers.add("stdbool.h")
            return ("true" if value else "false"), _ATOM
        if isinstance(t, FloatType):
            number = float(value)  # type: ignore[arg-type]
            if math.isnan(number):
                self.owner.unit.headers.add("math.h")
                return "NAN", _ATOM
            if number in (float("inf"), float("-inf")):
                self.owner.unit.headers.add("math.h")
                return ("INFINITY", _ATOM) if number > 0 else ("-INFINITY", _UNARY)
            text = repr(number)
            text = text if ("." in text or "e" in text) else f"{text}.0"
            return text, (_UNARY if text.startswith("-") else _ATOM)
        if isinstance(t, PtrType):
            if self.owner.cpp:
                return "nullptr", _ATOM
            return f"(({self.owner.c_type(t)})0)", _ATOM
        number = int(value)  # type: ignore[call-overload]
        if self.owner.int32 and isinstance(t, IntType) and t.width == 32:
            if not t.signed:
                return f"{number}u", _ATOM
            if number == -(1 << 31):
                return "(-2147483647 - 1)", _ATOM
            return str(number), (_UNARY if number < 0 else _ATOM)
        if isinstance(t, IntType) and not t.signed:
            return (str(number) if number < (1 << 31) else f"UINT64_C({number})"), _ATOM
        if number == -(1 << 63):
            return "INT64_MIN", _ATOM
        if -(1 << 31) <= number < (1 << 31):
            # An `int` literal widens to the operation's type on its own.
            return str(number), (_UNARY if number < 0 else _ATOM)
        return f"INT64_C({number})", _ATOM

    def constant(self, v: Value) -> int | None:
        """The integer `v` is, when it is a constant."""
        producer = v.owner
        if not isinstance(producer, Operation) or producer.name != "core.const":
            return None
        if not isinstance(v.type, (IntType, IndexType)):
            return None
        return int(producer.attributes["value"])  # type: ignore[call-overload]

    def math(self, op: Operation, name: str, arguments: list[str]) -> None:
        self.owner.unit.headers.add("math.h")
        t = op.results[0].type
        if isinstance(t, VectorType) and isinstance(t.element, FloatType):
            self.vector_math(op, name, arguments, t)
            return
        if not isinstance(t, FloatType) or t.width == 16:
            raise EmitError(f"math.{name} over {t} has no C lowering")
        function = "fabs" if name == "abs" else name
        if not self.owner.cpp and t.width == 32:
            function += "f"
        call = f"{self.owner.std(function)}({', '.join(arguments)})"
        self.fold(op.results[0], (call, _ATOM), reads=self.reading(op.operands))

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

    def arith(self, op: Operation, name: str) -> None:
        t = op.result.type
        (a, b), reads = self.operands(op)
        symbol = {"add": "+", "sub": "-", "mul": "*"}[name]
        if isinstance(t, FloatType):
            self.infix(op.result, symbol, a, b, reads)
            return
        overflow = op.attributes.get("overflow", "python")
        if overflow == "native" and not any(self.machine_typed(v) for v in op.operands):
            # Small literals and conditional expressions may have C's `int`
            # type even when their canonical type is i64. Widen before the
            # operation, not after an intermediate has already overflowed.
            a = self.cast_text(a, t)
        if overflow in {"proven", "native"}:
            # Shown never to overflow: the plain operation is defined.
            self.infix(op.result, symbol, a, b, reads)
            return
        if overflow == "wrap":
            self.wrapped(op.result, t, op.operands[0], op.operands[1], symbol)
            return
        result = self.sink(op.result) or self.variable(op.result)
        helper = self.owner.helper(name, t)
        self.fail_unless(
            f"!{helper}({a[0]}, {b[0]}, &{result})",
            "arith.ok",
            raises=overflow_text(t.width if isinstance(t, IntType) else 64),
        )

    def machine_typed(self, value: Value) -> bool:
        """Whether the emitted expression already evaluates at its IR width."""
        if not isinstance(value.type, (IntType, IndexType)):
            return False
        if self.owner.int32 and value.type == IntType(32):
            return True  # Every signed integer expression is at least C's int.
        if _IDENTIFIER.fullmatch(self.bare(value)):
            return True
        producer = value.owner
        if not isinstance(producer, Operation):
            return False
        if producer.name in {
            "core.add",
            "core.sub",
            "core.mul",
            "core.neg",
            "core.div",
            "core.mod",
        }:
            # These emitters promote before arithmetic, including literal-only inputs.
            return producer.attributes.get("overflow") in {"native", "wrap"} or any(
                self.machine_typed(v) for v in producer.operands
            )
        if producer.name == "core.select":
            return any(self.machine_typed(v) for v in producer.operands[1:])
        if producer.name == "core.cast":
            source = producer.operands[0]
            return self.owner.c_type(source.type) != self.owner.c_type(
                value.type
            ) or self.machine_typed(source)
        if producer.name == "core.call":
            return True
        if producer.name == "core.const":
            number = self.constant(value)
            return number is not None and not -(1 << 31) < number < (1 << 31)
        return False

    def wrapped(self, v: Value, t: IRType, a: Value | _Expr, b: Value | _Expr, symbol: str) -> None:
        """`v` as two's-complement arithmetic through the unsigned type: signed
        overflow is undefined in C, unsigned wraps. An operand that is such
        arithmetic itself stays unsigned in between."""
        unsigned = self.owner.c_type(IntType(t.width if isinstance(t, IntType) else 64, False))
        wide = _infix(symbol, self._unsigned(a, unsigned), self._unsigned(b, unsigned))
        text, level = self.cast_text(wide, t)
        reads = self.reading([x for x in (a, b) if isinstance(x, Value)])
        if self.fold(v, (text, level), reads=reads) == text:
            self.unsigned_form[id(v)] = wide

    def _unsigned(self, x: Value | _Expr, unsigned: str) -> _Expr:
        if isinstance(x, Value):
            form = self.unsigned_form.get(id(x))
            if form is not None:
                return form
            x = self.expr(x)
        if re.fullmatch(r"\d+", x[0]):
            # An unsigned literal: `1u` widens to the operation's type.
            return f"{x[0]}u", _ATOM
        return self.cast_text(x, unsigned)

    def neg(self, op: Operation) -> None:
        t = op.result.type
        (x,), reads = self.operands(op)
        if op.attributes.get("overflow") == "native" and not self.machine_typed(op.operands[0]):
            x = self.cast_text(x, t)
        if isinstance(t, FloatType) or op.attributes.get("overflow") == "native":
            self.fold(op.result, _prefix("-", x), reads=reads)
            return
        if op.attributes.get("overflow", "python") == "wrap":
            self.wrapped(op.result, t, ("0", _ATOM), op.operands[0], "-")
            return
        result = self.sink(op.result) or self.variable(op.result)
        helper = self.owner.helper("sub", t)
        self.fail_unless(
            f"!{helper}(0, {x[0]}, &{result})",
            "arith.ok",
            raises=overflow_text(t.width if isinstance(t, IntType) else 64),
        )

    def divmod(self, op: Operation, name: str) -> None:
        t = op.result.type
        (a, b), reads = self.operands(op)
        if op.attributes.get("overflow") == "native" and not any(
            self.machine_typed(v) for v in op.operands
        ):
            a = self.cast_text(a, t)
        if isinstance(t, FloatType):
            if name == "mod":
                raise EmitError("float remainder has no C lowering with Python semantics")
            self.fold(op.result, _infix("/", a, b), reads=reads)
            return
        signed = t.signed if isinstance(t, IntType) else True
        symbol = "/" if name == "div" else "%"
        if not signed:
            self.fold(op.result, _infix(symbol, a, b), reads=reads)
            return
        width = t.width if isinstance(t, IntType) else 64
        divisor = self.constant(op.operands[1])
        zero, minus_one = ("0", _ATOM), ("-1", _UNARY)
        if op.attributes.get("overflow", "python") not in {"wrap", "native"} and divisor is None:
            minimum = (f"INT{width}_MIN", _ATOM)
            held = _infix("&&", _infix("==", a, minimum), _infix("==", b, minus_one))
            failed = _infix("||", _infix("!=", a, minimum), _infix("!=", b, minus_one))
            self.fail_unless(failed[0], "div.ok", failed=held[0], raises=overflow_text(width))
        if op.attributes.get("rounding", "floor") == "trunc" or id(op) in self.truncating:
            self.fold(op.result, _infix(symbol, a, b), reads=reads)
            return
        if divisor is not None and 0 < divisor < (1 << 31) and _IDENTIFIER.match(a[0]):
            # Python's floor division by a positive constant, in one expression.
            remainder = _infix("%", a, b)
            if name == "mod":
                if 2 * divisor - 1 <= (1 << (width - 1)) - 1:
                    self.fold(op.result, _infix("%", _infix("+", remainder, b), b), reads=reads)
                    return
                # Adding the divisor to a positive remainder can overflow at
                # 32 bits. Python's correction is needed only when negative.
                r = (self.temporary("r", self.owner.c_type(t), remainder[0]), _ATOM)
                corrected = _ternary(_infix("<", r, zero), _infix("+", r, b), r)
                self.fold(op.result, corrected, reads=reads)
            else:
                negative = _infix("<", remainder, zero)
                self.fold(op.result, _infix("-", _infix("/", a, b), negative), reads=reads)
            return
        c = self.owner.c_type(t)
        remainder = (self.temporary("r", c, _infix("%", a, b)[0]), _ATOM)
        differ = _infix("!=", _infix("<", a, zero), _infix("<", b, zero))
        adjust = _infix("&&", _infix("!=", remainder, zero), differ)
        if name == "div":
            quotient = (self.temporary("q", c, _infix("/", a, b)[0]), _ATOM)
            floored = _ternary(adjust, _infix("-", quotient, ("1", _ATOM)), quotient)
            self.fold(op.result, floored, reads=reads)
        else:
            self.fold(
                op.result, _ternary(adjust, _infix("+", remainder, b), remainder), reads=reads
            )

    def shift(self, op: Operation, name: str) -> None:
        t = op.result.type
        (a, b), reads = self.operands(op)
        if name == "shr":
            self.infix(op.result, ">>", a, b, reads)
            return
        unsigned = self.owner.c_type(IntType(t.width if isinstance(t, IntType) else 64, False))
        wide = _infix("<<", self._unsigned(op.operands[0], unsigned), self.cast_text(b, unsigned))
        if op.attributes.get("overflow", "wrap") in {"wrap", "native"}:
            text, level = self.cast_text(wide, t)
            if self.fold(op.result, (text, level), reads=reads) == text:
                self.unsigned_form[id(op.result)] = wide
            return
        shifted = (self.define(op.result, self.cast_text(wide, t)[0]), _ATOM)
        back = _infix(">>", shifted, b)
        self.fail_unless(
            _infix("==", back, a)[0],
            "shl.ok",
            failed=_infix("!=", back, a)[0],
            raises=overflow_text(t.width if isinstance(t, IntType) else 64),
        )

    def cmp(self, op: Operation) -> None:
        (a, b), reads = self.operands(op)
        predicate = str(op.attributes["predicate"])
        floating = isinstance(op.operands[0].type, FloatType)
        if predicate == "ne" and floating:
            # Ordered, like the LLVM road: NaN compares neither less nor greater.
            self.fold(op.result, _infix("||", _infix("<", a, b), _infix(">", a, b)), reads=reads)
            return
        symbol = _CMP[predicate]
        if a[0] == b[0] and (not floating or symbol in {"<", ">"}):
            # A value against itself: settled here, or a C compiler warns about it.
            self.fold(op.result, ("true" if symbol in {"==", "<=", ">="} else "false", _ATOM))
            return
        text, level = _infix(symbol, a, b)
        written = self.fold(op.result, (text, level), reads=reads) == text
        if written and (not floating or symbol in {"==", "!="}):
            # NaN makes `!(a < b)` and `a >= b` differ; equality flips exactly.
            self.flipped[id(op.result)] = _infix(_FLIPPED[symbol], a, b)[0]

    def cast(self, op: Operation) -> None:
        (x,), reads = self.operands(op)
        source = op.operands[0].type
        target = op.result.type
        if isinstance(source, PtrType) and isinstance(target, PtrType):
            self.fold(op.result, self.cast_text(x, target), reads=reads)
            return
        if isinstance(source, PtrType):
            wide = self.cast_text(self.cast_text(x, "intptr_t"), "int64_t")
            self.fold(op.result, wide, reads=reads)
            return
        if isinstance(target, BoolType):
            zero = "0.0" if isinstance(source, FloatType) else "0"
            self.fold(op.result, _infix("!=", x, (zero, _ATOM)), reads=reads)
            return
        if source == target or self.owner.c_type(source) == self.owner.c_type(target):
            self.fold(op.result, x, reads=reads)
            return
        self.fold(op.result, self.cast_text(x, target), reads=reads)

    def call(self, op: Operation) -> None:
        callee_name = op.attributes["callee"].name  # type: ignore[union-attr]
        target = self.owner.module.functions.get(callee_name)
        if target is None:
            raise EmitError(f"call to @{callee_name}, which was not emitted")
        symbol = self.owner.symbol_of(target)
        arguments: list[str] = []
        for operand in op.operands:
            if target.name in self.owner.direct and isinstance(operand.type, BoolType):
                arguments.append(self.bare(operand))
            else:
                arguments.extend(self.flatten(operand))
        if self.device or target.name in self.owner.direct:
            direct = f"{symbol}({', '.join(arguments)})"
            if op.results and op.results[0].uses:
                self.effect(op.results[0], direct)
            else:
                self.line(f"{direct};")
            return
        results = list(op.results)
        captured = op.attributes.get("capture_status")
        status_result = results.pop() if captured else None
        slots: list[list[str]] = []
        for result, t in zip(results, target.results or (), strict=True):
            names = []
            atoms = self.owner.atoms(t)
            if atoms == [self.owner.c_type(t)] and result.uses:
                # The callee writes the result's own variable -- or its slot.
                names.append(self.sink(result) or self.variable(result))
            else:
                for atom in atoms:
                    slot = self.fresh(result.name or "res")
                    self._declare_slot(atom, slot)
                    names.append(slot)
            slots.append(names)
            arguments.extend(f"&{n}" for n in names)
        if not target.results:
            slot = self.fresh("res")
            self._declare_slot("int64_t", slot)
            arguments.append(f"&{slot}")
        call = f"{symbol}({', '.join(arguments)})"
        if status_result is not None:
            self.define(status_result, self.cast_text((call, _ATOM), "int64_t")[0])
        else:
            self.fail_unless(f"{call} == {STATUS_OK}", "call.ok", failed=f"{call} != {STATUS_OK}")
        for result, t, names in zip(results, target.results, slots, strict=True):
            if isinstance(t, BufferType):
                self.define_buffer(result, names[0], names[1])
            elif id(result) in self.scalars:
                continue
            elif result.uses:
                self.define(result, self.rebuild(t, names, 0)[0])

    def _declare_slot(self, spelled: str, name: str) -> None:
        if self.structured:
            self.line(f"{_declare(spelled, name)};")
        else:
            self.declarations.append(f"    {_declare(spelled, name)};")

    def call_extern(self, op: Operation) -> None:
        symbol = str(op.attributes["callee"])
        if (
            self.owner.readable
            and self.owner.native_input
            and symbol in {"ppy_rt_input_int", "ppy_rt_scan_int"}
        ):
            # Unsafe source assumes valid, in-range input for the host stdio parser.
            self.owner.unit.headers.add("stdio.h")
            if not self.owner.int32:
                self.owner.unit.headers.add("inttypes.h")
            result = self.sink(op.result) or self.variable(op.result)
            scan = self.owner.std("scanf")
            spec = '"%d"' if self.owner.int32 else '"%" SCNd64'
            self.line(f"{scan}({spec}, &{result});")
            return
        shim = SHIMS.get(symbol)
        libc = _LIBC.get(symbol)
        if libc is not None:
            result_type, parameter_types, header = libc
            self.owner.unit.headers.add(header)
            arguments = [
                self.cast_text(self.extern_atom(operand), expected)[0]
                for operand, expected in zip(op.operands, parameter_types, strict=True)
            ]
            call = f"{symbol}({', '.join(arguments)})"
            if op.results and op.results[0].uses:
                wanted = self.extern_type(op.results[0])
                if wanted != result_type:
                    call = self.cast_text((call, _ATOM), wanted)[0]
                self.effect(op.results[0], call)
            else:
                self.line(f"{call};")
            return
        if shim is not None:
            self.owner.shim(symbol)
            parameter_types = [re.sub(r"[A-Za-z_]\w*$", "", p).strip() for p in shim.parameters]
            arguments = []
            for operand, expected in zip(op.operands, parameter_types, strict=True):
                atom = self.extern_atom(operand)
                if expected != self.extern_type(operand):
                    atom = self.cast_text(atom, expected)
                arguments.append(atom[0])
        else:
            parameter_types = [self.extern_type(v) for v in op.operands]
            arguments = [self.extern_atom(v)[0] for v in op.operands]
            result = "void" if not op.results else self.extern_type(op.results[0])
            self.owner.unit.externs.setdefault(
                symbol, f"{result} {symbol}({', '.join(parameter_types) or 'void'});\n"
            )
        call = f"{symbol}({', '.join(arguments)})"
        if not op.results or not op.results[0].uses:
            self.line(f"{call};")
            return
        if isinstance(op.results[0].type, BoolType):
            call = f"{call} != 0"
        self.effect(op.results[0], call)

    def extern_type(self, value: Value) -> str:
        t = value.type
        if isinstance(t, BoolType):
            return "int8_t"
        if isinstance(t, BufferType):
            return f"{self.owner.c_type(t.element)} *"
        return self.owner.c_type(t)

    def extern_atom(self, value: Value) -> _Expr:
        """How a value crosses into C: a bool is a byte, a buffer its pointer."""
        if isinstance(value.type, BoolType):
            return self.cast_text(self.expr(value), "int8_t")
        if isinstance(value.type, BufferType):
            return self.buffer(value).data, _ATOM
        return self.expr(value)

    def call_intrinsic(self, op: Operation) -> None:
        name = str(op.attributes["intrinsic"])
        if name in _MATH_INTRINSICS:
            self.math(op, _MATH_INTRINSICS[name], [self.bare(v) for v in op.operands])
            return
        if name == "ppy.buffer_from_parts":
            data, length = (self.bare(v) for v in op.operands)
            self.define_buffer(op.results[0], data, length)
            return
        if name == "ppy.string_data":
            if self.owner.readable:
                symbol = str(op.attributes["symbol"])
                data = str(self.owner.module.globals[symbol].value).encode("utf-8")
                # Printing consumes literal bytes directly; other consumers keep
                # their addressable storage and the original pointer contract.
                if all(
                    isinstance(u, Operation)
                    and u.name == "core.call_extern"
                    and u.attributes.get("callee") == "ppy_rt_print_str"
                    for u, _ in op.result.uses
                ):
                    self.scalars[id(op.result)] = string_literal(data)
                    return
                self.owner.unit.strings[symbol] = (
                    f"static {self.owner.c_type(IntType(8, False))} {_ident(symbol)}[] = "
                    "{" + (", ".join(str(b) for b in data) or "0") + "};\n"
                )
            self.fold(op.results[0], (_ident(str(op.attributes["symbol"])), _ATOM))
            return
        if name in {"llvm.smin.i64", "llvm.smax.i64"}:
            (a, b), reads = self.operands(op)
            symbol = "<" if name == "llvm.smin.i64" else ">"
            self.fold(op.results[0], _ternary(_infix(symbol, a, b), a, b), reads=reads)
            return
        if name.startswith("ppy.checked_"):
            operation = name.removeprefix("ppy.checked_")
            a, b = (self.bare(v) for v in op.operands)
            result = self.sink(op.results[0]) or self.variable(op.results[0])
            helper = self.owner.helper(operation, op.operands[0].type)
            self.define(op.results[1], f"{helper}({a}, {b}, &{result}) != 0")
            return
        raise EmitError(f"intrinsic {name!r} has no C lowering")

    def flush_print(self) -> None:
        if self.print_format is not None:
            self.line(f"{self.owner.std('printf')}({self.print_format.finish()});")
            self.print_format = None

    def print_operation(self, op: Operation) -> bool:
        if op.name != "core.call_extern":
            return False
        symbol = str(op.attributes["callee"])
        if symbol == "ppy_rt_flush_stdout":
            self.flush_print()
            self.owner.unit.headers.add("stdio.h")
            self.line(f"{self.owner.std('fflush')}(stdout);")
            return True
        if symbol not in {
            "ppy_rt_print_str",
            "ppy_rt_print_i64",
            "ppy_rt_print_u64",
            "ppy_rt_print_bool",
            "ppy_rt_print_nl",
            "ppy_rt_print_sep",
        }:
            return False
        data = None
        if symbol == "ppy_rt_print_str":
            name = string_symbol(op.operands[0])
            length = self.constant(op.operands[1])
            if name is None or length is None or length < 0:
                return False
            data = str(self.owner.module.globals[name].value).encode("utf-8")[:length]
        self.owner.unit.headers.add("stdio.h")
        if self.print_format is None:
            self.print_format = PrintFormat()
        if data is not None:
            self.print_format.literal(data)
        elif symbol in {"ppy_rt_print_sep", "ppy_rt_print_nl"}:
            self.print_format.literal(b" " if symbol.endswith("sep") else b"\n")
        elif symbol == "ppy_rt_print_bool":
            expression = _ternary(self.expr(op.operands[0]), ('"True"', _ATOM), ('"False"', _ATOM))
            self.print_format.value("%s", expression[0])
        else:
            unsigned = symbol == "ppy_rt_print_u64"
            operand = op.operands[0]
            argument = self.bare(operand)
            width = 32 if self.owner.int32 else 64
            exact_type = operand.type == IntType(width, not unsigned) or (
                isinstance(operand.type, IndexType) and not unsigned
            )
            if not exact_type or not self.machine_typed(operand):
                argument = self.owner.cast(
                    argument, self.owner.c_type(IntType(width, not unsigned))
                )
            if self.owner.int32:
                self.print_format.value("%u" if unsigned else "%d", argument)
            else:
                self.owner.unit.headers.add("inttypes.h")
                self.print_format.value("%", argument, "PRIu64" if unsigned else "PRId64")
        if symbol == "ppy_rt_print_nl" or (data is not None and data.endswith(b"\n")):
            self.flush_print()
        return True


#: Operations with no effect on memory: a value read before one of these can
#: be written after it without changing what it reads.
_PURE = frozenset(
    {
        "core.const",
        "core.add",
        "core.sub",
        "core.mul",
        "core.div",
        "core.mod",
        "core.neg",
        "core.and",
        "core.or",
        "core.xor",
        "core.shl",
        "core.shr",
        "core.cmp",
        "core.select",
        "core.cast",
        "core.load",
        "core.alloca",
        "core.guard",
        "core.buffer_data",
        "core.buffer_len",
        "core.buffer_load",
        "core.tuple_make",
        "core.tuple_extract",
        "core.struct_make",
        "core.struct_extract",
        "core.ptr_offset",
    }
)


_MAKERS = frozenset({"core.tuple_make", "core.struct_make"})


def _pure(op: Operation) -> bool:
    return op.name in _PURE or op.dialect == "math"


def _indent(line: str) -> int:
    """A line's depth in spaces; -1 for an empty line, which opens and closes nothing."""
    return len(line) - len(line.lstrip(" ")) if line.strip() else -1


def _leaves(lines: list[str]) -> bool:
    """Whether control never runs past `lines`: they end in a jump out."""
    return bool(lines) and _SIMPLE_JUMP.match(lines[-1].strip()) is not None


def _only_break(lines: list[str]) -> bool:
    return [line.strip() for line in lines] == ["break;"]


def _one_if(lines: list[str]) -> bool:
    """Whether `lines` are exactly one braced `if` statement, for an `else if`."""
    if len(lines) < 2 or not lines[0].lstrip().startswith("if (") or not lines[0].endswith("{"):
        return False
    if lines[-1].strip() != "}":
        return False
    depth = 0
    for index, line in enumerate(lines):
        depth += line.count("{") - line.count("}")
        if depth == 0 and index != len(lines) - 1:
            return False
    return depth == 0


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


def _unique(name: str, used: set[str]) -> str:
    """`name`, or `name_2`, `name_3`, ... when it is taken; recorded as used."""
    candidate = name
    n = 1
    while candidate in used:
        n += 1
        candidate = f"{name}_{n}"
    used.add(candidate)
    return candidate
