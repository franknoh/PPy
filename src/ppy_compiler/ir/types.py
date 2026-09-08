"""The types a value in the IR can have, and their text.

Every value has exactly one type. The core types are the ones every backend
must know -- machine scalars, pointers, buffers, vectors, tuples, structs,
and futures -- and a dialect adds its own through `DialectType`, which the
core parses generically as `name<args>` and the dialect verifies.

A type is an immutable value that compares by content and prints to one
spelling, so the same program always prints to the same bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

__all__ = [
    "BOOL",
    "F16",
    "F32",
    "F64",
    "I8",
    "I16",
    "I32",
    "I64",
    "INDEX",
    "U8",
    "U16",
    "U32",
    "U64",
    "VOID",
    "BoolType",
    "BufferType",
    "DialectType",
    "FloatType",
    "FutureType",
    "IRType",
    "IndexType",
    "IntType",
    "PtrType",
    "StructType",
    "TupleType",
    "TypeError_",
    "VectorType",
    "VoidType",
    "is_arithmetic",
    "is_integer",
    "is_scalar",
    "parse_type",
    "scalar_of",
]


class TypeError_(ValueError):
    """A type that cannot be spelled or parsed."""


@dataclass(frozen=True, slots=True)
class IRType:
    """Base of every IR type. Subclasses are frozen dataclasses."""

    def __str__(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class VoidType(IRType):
    def __str__(self) -> str:
        return "void"


@dataclass(frozen=True, slots=True)
class BoolType(IRType):
    def __str__(self) -> str:
        return "bool"


@dataclass(frozen=True, slots=True)
class IntType(IRType):
    width: int
    signed: bool = True

    WIDTHS: ClassVar[tuple[int, ...]] = (8, 16, 32, 64)

    def __post_init__(self) -> None:
        if self.width not in self.WIDTHS:
            raise TypeError_(f"no integer type of width {self.width}")

    def __str__(self) -> str:
        return f"{'i' if self.signed else 'u'}{self.width}"

    @property
    def min(self) -> int:
        return -(1 << (self.width - 1)) if self.signed else 0

    @property
    def max(self) -> int:
        return (1 << (self.width - 1)) - 1 if self.signed else (1 << self.width) - 1

    def fits(self, value: int) -> bool:
        return self.min <= value <= self.max


@dataclass(frozen=True, slots=True)
class FloatType(IRType):
    width: int

    WIDTHS: ClassVar[tuple[int, ...]] = (16, 32, 64)

    def __post_init__(self) -> None:
        if self.width not in self.WIDTHS:
            raise TypeError_(f"no float type of width {self.width}")

    def __str__(self) -> str:
        return f"f{self.width}"


@dataclass(frozen=True, slots=True)
class IndexType(IRType):
    """A size or offset: as wide as a pointer on the target."""

    def __str__(self) -> str:
        return "index"


@dataclass(frozen=True, slots=True)
class PtrType(IRType):
    """`ptr<T, space, mut|const>`: a pointer to `T` in an address space.

    `stack` memory comes from `core.alloca` and may not escape the function;
    `generic` is anything the host addresses. A dialect adds spaces of its
    own (`global`, `shared`, ...).
    """

    pointee: IRType
    address_space: str = "generic"
    mutable: bool = True

    def __str__(self) -> str:
        parts = [str(self.pointee)]
        if self.address_space != "generic" or not self.mutable:
            parts.append(self.address_space)
        if not self.mutable:
            parts.append("const")
        return f"ptr<{', '.join(parts)}>"


@dataclass(frozen=True, slots=True)
class BufferType(IRType):
    """`buffer<T>`: contiguous elements with a length the program can read."""

    element: IRType

    def __str__(self) -> str:
        return f"buffer<{self.element}>"


@dataclass(frozen=True, slots=True)
class VectorType(IRType):
    """`vector<T, N>`: `N` scalars operated on at once."""

    element: IRType
    count: int

    def __post_init__(self) -> None:
        if self.count < 1:
            raise TypeError_(f"a vector needs at least one lane, not {self.count}")
        if not is_scalar(self.element):
            raise TypeError_(f"a vector holds scalars, not {self.element}")

    def __str__(self) -> str:
        return f"vector<{self.element}, {self.count}>"


@dataclass(frozen=True, slots=True)
class TupleType(IRType):
    items: tuple[IRType, ...]

    def __str__(self) -> str:
        return f"tuple<{', '.join(str(item) for item in self.items)}>"


@dataclass(frozen=True, slots=True)
class StructType(IRType):
    """`struct<Name, f: T, ...>`: named fields with a nominal identity."""

    name: str
    fields: tuple[tuple[str, IRType], ...]

    def __str__(self) -> str:
        spelled = ", ".join(f"{name}: {t}" for name, t in self.fields)
        return f"struct<{self.name}{', ' + spelled if spelled else ''}>"

    def field_type(self, name: str) -> IRType | None:
        for field_name, t in self.fields:
            if field_name == name:
                return t
        return None

    def field_index(self, name: str) -> int | None:
        for index, (field_name, _t) in enumerate(self.fields):
            if field_name == name:
                return index
        return None


@dataclass(frozen=True, slots=True)
class FutureType(IRType):
    inner: IRType

    def __str__(self) -> str:
        return f"future<{self.inner}>"


@dataclass(frozen=True, slots=True)
class DialectType(IRType):
    """`dialect.name<args>`: a type a dialect owns.

    The core keeps the arguments as parsed -- types, integers, or bare
    words -- and the dialect's verifier gives them meaning.
    """

    dialect: str
    name: str
    args: tuple[IRType | int | str, ...] = ()

    def __str__(self) -> str:
        head = f"{self.dialect}.{self.name}"
        if not self.args:
            return head
        return f"{head}<{', '.join(str(a) for a in self.args)}>"


VOID = VoidType()
BOOL = BoolType()
I8 = IntType(8, True)
I16 = IntType(16, True)
I32 = IntType(32, True)
I64 = IntType(64, True)
U8 = IntType(8, False)
U16 = IntType(16, False)
U32 = IntType(32, False)
U64 = IntType(64, False)
F16 = FloatType(16)
F32 = FloatType(32)
F64 = FloatType(64)
INDEX = IndexType()

_NAMED: dict[str, IRType] = {
    "void": VOID,
    "bool": BOOL,
    "i8": I8,
    "i16": I16,
    "i32": I32,
    "i64": I64,
    "u8": U8,
    "u16": U16,
    "u32": U32,
    "u64": U64,
    "f16": F16,
    "f32": F32,
    "f64": F64,
    "index": INDEX,
}


def is_integer(t: IRType) -> bool:
    return isinstance(t, (IntType, IndexType))


def is_arithmetic(t: IRType) -> bool:
    """Something `core.add` and friends operate on: a number, or a vector of them."""
    if isinstance(t, VectorType):
        return is_arithmetic(t.element)
    return isinstance(t, (IntType, FloatType, IndexType))


def is_scalar(t: IRType) -> bool:
    return isinstance(t, (BoolType, IntType, FloatType, IndexType))


def scalar_of(t: IRType) -> IRType:
    """The element of a vector, or the type itself."""
    return t.element if isinstance(t, VectorType) else t


# -- text -------------------------------------------------------------------


def parse_type(text: str) -> IRType:
    """The type `text` spells; raises `TypeError_` on anything else."""
    parser = _TypeParser(text)
    result = parser.type_()
    parser.skip_space()
    if parser.pos != len(parser.text):
        raise TypeError_(f"trailing text after type in {text!r}")
    return result


class _TypeParser:
    """A small recursive-descent parser over the type grammar.

    The IR parser hands its own tokens here for the same grammar, so
    `type_` is the one place that knows how a type is spelled.
    """

    def __init__(self, text: str, pos: int = 0) -> None:
        self.text = text
        self.pos = pos

    def skip_space(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t":
            self.pos += 1

    def peek(self) -> str:
        self.skip_space()
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def expect(self, char: str) -> None:
        if self.peek() != char:
            raise TypeError_(f"expected {char!r} at {self.pos} in {self.text!r}")
        self.pos += 1

    def word(self) -> str:
        self.skip_space()
        start = self.pos
        while self.pos < len(self.text) and (
            self.text[self.pos].isalnum() or self.text[self.pos] in "_."
        ):
            self.pos += 1
        if start == self.pos:
            raise TypeError_(f"expected a type at {self.pos} in {self.text!r}")
        return self.text[start : self.pos]

    def type_(self) -> IRType:
        name = self.word()
        if self.peek() != "<":
            if name in _NAMED:
                return _NAMED[name]
            if "." in name:
                dialect, _, local = name.partition(".")
                return DialectType(dialect, local)
            raise TypeError_(f"unknown type {name!r}")
        self.expect("<")
        if name == "ptr":
            pointee = self.type_()
            space = "generic"
            mutable = True
            while self.peek() == ",":
                self.pos += 1
                option = self.word()
                if option == "const":
                    mutable = False
                elif option == "mut":
                    mutable = True
                else:
                    space = option
            self.expect(">")
            return PtrType(pointee, space, mutable)
        if name == "buffer":
            element = self.type_()
            self.expect(">")
            return BufferType(element)
        if name == "future":
            inner = self.type_()
            self.expect(">")
            return FutureType(inner)
        if name == "vector":
            element = self.type_()
            self.expect(",")
            count = int(self.word())
            self.expect(">")
            return VectorType(element, count)
        if name == "tuple":
            items: list[IRType] = []
            if self.peek() != ">":
                items.append(self.type_())
                while self.peek() == ",":
                    self.pos += 1
                    items.append(self.type_())
            self.expect(">")
            return TupleType(tuple(items))
        if name == "struct":
            struct_name = self.word()
            fields: list[tuple[str, IRType]] = []
            while self.peek() == ",":
                self.pos += 1
                field_name = self.word()
                self.expect(":")
                fields.append((field_name, self.type_()))
            self.expect(">")
            return StructType(struct_name, tuple(fields))
        if "." in name:
            dialect, _, local = name.partition(".")
            args: list[IRType | int | str] = []
            if self.peek() != ">":
                args.append(self.argument())
                while self.peek() == ",":
                    self.pos += 1
                    args.append(self.argument())
            self.expect(">")
            return DialectType(dialect, local, tuple(args))
        raise TypeError_(f"{name!r} takes no type arguments")

    def argument(self) -> IRType | int | str:
        """One dialect-type argument: a type, an integer, a bare word, or an
        expression in parentheses, kept as its text for the dialect to read."""
        self.skip_space()
        start = self.pos
        if self.peek() == "(":
            depth = 0
            while self.pos < len(self.text):
                char = self.text[self.pos]
                self.pos += 1
                depth += char == "("
                depth -= char == ")"
                if depth == 0:
                    return self.text[start : self.pos]
            raise TypeError_(f"unbalanced parentheses at {start} in {self.text!r}")
        if self.peek() in "-0123456789":
            self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
            return int(self.text[start : self.pos])
        word = self.word()
        if self.peek() == "<" or word in _NAMED or "." in word:
            self.pos = start
            return self.type_()
        return word
