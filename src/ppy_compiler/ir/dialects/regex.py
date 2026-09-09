"""The regex dialect: a regular expression matched natively over bytes.

`regex.search %buf, %pos, %endpos {pattern = "...", flags = 0}` is what
`PATTERN.search(buf, pos, endpos)` means for a pattern compiled from a bytes
literal; `regex.match` and `regex.fullmatch` are the anchored forms. The
operation yields whether a match was found and then the span of every group
-- start and end of group 0, of group 1, ... -- with -1 for a group that did
not take part. The `lower-regex` pass compiles each pattern into a matcher
function of core operations, so every backend runs it as it runs anything
else, and no backend needs a regex library.

The pattern is read by CPython's own parser, so its syntax and its meaning
are `re`'s: ordered alternation, greedy and lazy repetition, the group's
last iteration, `$` before a trailing newline, `\\b` at ASCII word edges.
What a backtracking matcher over bytes cannot reproduce exactly --
backreferences, lookaround, atomic groups and possessive repeats, locale
and Unicode categories -- is refused, and the function stays on Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..dialect import Dialect, DialectRegistry, OpSpec
from ..model import Attribute, Builder, Operation, Value
from ..types import BOOL, I64, U8, BufferType, IntType, IRType

if TYPE_CHECKING:
    from ..verify import Checker

try:
    import re as _re
    from re import _constants as _C
    from re import _parser as _P
except ImportError:  # pragma: no cover - CPython always has these
    _re = _C = _P = None  # type: ignore[assignment]

__all__ = [
    "MODES",
    "Alt",
    "Anchor",
    "Chars",
    "Compiled",
    "Group",
    "Node",
    "RegexDialect",
    "Repeat",
    "Seq",
    "Unsupported",
    "analyse",
    "create",
    "pattern_of",
    "supported_flags",
]

#: The anchored forms and the free one, each its own operation name.
MODES = ("search", "match", "fullmatch")

#: The flags a native matcher honours; `re.ASCII` is what bytes mean anyway,
#: and `re.VERBOSE` only shapes the parse.
_HONOURED = 0
if _re is not None:
    _HONOURED = _re.IGNORECASE | _re.MULTILINE | _re.DOTALL | _re.ASCII | _re.VERBOSE


class Unsupported(ValueError):
    """A pattern the native matcher cannot reproduce exactly."""


@dataclass(frozen=True, slots=True)
class Chars:
    """One byte from a set: four 64-bit words, bit `c` for byte `c`."""

    words: tuple[int, int, int, int]

    def has(self, byte: int) -> bool:
        return bool(self.words[byte >> 6] >> (byte & 63) & 1)

    @property
    def only(self) -> int | None:
        """The one byte of a singleton set, or None."""
        members = [c for c in range(256) if self.has(c)]
        return members[0] if len(members) == 1 else None


@dataclass(frozen=True, slots=True)
class Anchor:
    """A position test that consumes nothing."""

    #: start, line_start, string_start, end, line_end, string_end,
    #: boundary, not_boundary
    kind: str


@dataclass(frozen=True, slots=True)
class Group:
    #: The capture index, or None for `(?:...)`.
    index: int | None
    body: Node


@dataclass(frozen=True, slots=True)
class Seq:
    items: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class Alt:
    branches: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class Repeat:
    body: Node
    minimum: int
    #: None for unbounded.
    maximum: int | None
    greedy: bool


Node = Chars | Anchor | Group | Seq | Alt | Repeat


@dataclass(frozen=True, slots=True)
class Compiled:
    """A pattern the native matcher can take, analysed."""

    pattern: bytes
    flags: int
    tree: Node
    #: Capture groups, group 0 not counted.
    groups: int
    names: dict[str, int]

    @property
    def spans(self) -> int:
        """How many integers a match carries: two per group, group 0 included."""
        return 2 * (self.groups + 1)


def supported_flags() -> int:
    return _HONOURED


def _set(members: set[int]) -> Chars:
    words = [0, 0, 0, 0]
    for c in members:
        words[c >> 6] |= 1 << (c & 63)
    return Chars((words[0], words[1], words[2], words[3]))


_WORD = {c for c in range(256) if chr(c).isalnum() and c < 128} | {ord("_")}
_DIGIT = set(range(ord("0"), ord("9") + 1))
_SPACE = {ord(c) for c in " \t\n\r\f\v"}
_LETTERS = {c for c in range(128) if chr(c).isalpha()}


def _folded(members: set[int]) -> set[int]:
    """The set with each ASCII letter in both cases: `re.IGNORECASE` over bytes."""
    out = set(members)
    for c in members:
        if c in _LETTERS:
            out.add(ord(chr(c).swapcase()))
    return out


def analyse(pattern: bytes, flags: int = 0) -> Compiled:
    """The pattern as the matcher's tree; `Unsupported` says what stops it."""
    if _P is None:
        raise Unsupported("this Python has no regex parser to read the pattern with")
    if not isinstance(pattern, bytes):
        raise Unsupported("a native pattern is a bytes literal, matched over bytes")
    try:
        parsed = _P.parse(pattern, flags)
    except _re.error as error:
        raise Unsupported(f"the pattern does not parse: {error}") from error
    effective = parsed.state.flags
    if effective & ~_HONOURED:
        unknown = _re.RegexFlag(effective & ~_HONOURED)
        raise Unsupported(f"flag {unknown} has no native matcher")
    tree = _Walker(effective).sequence(parsed)
    return Compiled(pattern, flags, tree, parsed.state.groups - 1, dict(parsed.state.groupdict))


class _Walker:
    def __init__(self, flags: int) -> None:
        self.flags = flags

    def sequence(self, items) -> Node:  # type: ignore[no-untyped-def]
        nodes = [self.node(op, argument) for op, argument in items]
        return nodes[0] if len(nodes) == 1 else Seq(tuple(nodes))

    def node(self, op, argument) -> Node:  # type: ignore[no-untyped-def]
        ignore = bool(self.flags & _re.IGNORECASE)
        if op is _C.LITERAL:
            members = {argument}
            return _set(_folded(members) if ignore else members)
        if op is _C.NOT_LITERAL:
            members = {argument}
            members = _folded(members) if ignore else members
            return _set(set(range(256)) - members)
        if op is _C.ANY:
            return _set(set(range(256)) if self.flags & _re.DOTALL else set(range(256)) - {10})
        if op is _C.IN:
            return self.charset(argument, ignore)
        if op is _C.AT:
            return self.anchor(argument)
        if op is _C.SUBPATTERN:
            group, add, remove, body = argument
            saved = self.flags
            self.flags = (self.flags | add) & ~remove
            if self.flags & ~_HONOURED:
                raise Unsupported("an inline flag with no native matcher")
            try:
                inner = self.sequence(body)
            finally:
                self.flags = saved
            return Group(group, inner)
        if op is _C.BRANCH:
            _none, alternatives = argument
            return Alt(tuple(self.sequence(alternative) for alternative in alternatives))
        if op is _C.MAX_REPEAT or op is _C.MIN_REPEAT:
            minimum, maximum, body = argument
            bound = None if maximum is _C.MAXREPEAT else int(maximum)
            return Repeat(self.sequence(body), int(minimum), bound, op is _C.MAX_REPEAT)
        if op is _C.POSSESSIVE_REPEAT:
            raise Unsupported("a possessive repeat (`*+`, `++`, `?+`) has no native matcher")
        if op is _C.ATOMIC_GROUP:
            raise Unsupported("an atomic group `(?>...)` has no native matcher")
        if op is _C.GROUPREF or op is _C.GROUPREF_EXISTS:
            raise Unsupported("a backreference has no native matcher")
        if op is _C.ASSERT or op is _C.ASSERT_NOT:
            raise Unsupported("lookahead and lookbehind have no native matcher")
        raise Unsupported(f"`{op.name.lower()}` has no native matcher")

    def charset(self, items, ignore: bool) -> Chars:  # type: ignore[no-untyped-def]
        members: set[int] = set()
        negate = False
        for op, argument in items:
            if op is _C.NEGATE:
                negate = True
            elif op is _C.LITERAL:
                members.add(argument)
            elif op is _C.RANGE:
                low, high = argument
                members.update(range(low, high + 1))
            elif op is _C.CATEGORY:
                members |= self.category(argument)
            else:
                raise Unsupported(f"`{op.name.lower()}` in a character set has no native matcher")
        if ignore:
            members = _folded(members)
        return _set(set(range(256)) - members if negate else members)

    @staticmethod
    def category(which) -> set[int]:  # type: ignore[no-untyped-def]
        table = {
            _C.CATEGORY_DIGIT: _DIGIT,
            _C.CATEGORY_NOT_DIGIT: set(range(256)) - _DIGIT,
            _C.CATEGORY_SPACE: _SPACE,
            _C.CATEGORY_NOT_SPACE: set(range(256)) - _SPACE,
            _C.CATEGORY_WORD: _WORD,
            _C.CATEGORY_NOT_WORD: set(range(256)) - _WORD,
        }
        found = table.get(which)
        if found is None:
            raise Unsupported(f"`{which.name.lower()}` has no native matcher over bytes")
        return found

    def anchor(self, which) -> Anchor:  # type: ignore[no-untyped-def]
        multiline = bool(self.flags & _re.MULTILINE)
        table = {
            _C.AT_BEGINNING: "line_start" if multiline else "start",
            _C.AT_BEGINNING_LINE: "line_start",
            _C.AT_BEGINNING_STRING: "string_start",
            _C.AT_END: "line_end" if multiline else "end",
            _C.AT_END_LINE: "line_end",
            _C.AT_END_STRING: "string_end",
            _C.AT_BOUNDARY: "boundary",
            _C.AT_NON_BOUNDARY: "not_boundary",
        }
        found = table.get(which)
        if found is None:
            raise Unsupported(f"`{which.name.lower()}` has no native matcher over bytes")
        return Anchor(found)


# -- the operations --------------------------------------------------------------


def pattern_of(op: Operation) -> tuple[bytes, int]:
    """The pattern and flags an operation carries."""
    pattern = str(op.attributes["pattern"]).encode("latin-1")
    return pattern, int(op.attributes.get("flags", 0))  # type: ignore[call-overload]


def _verify(op: Operation, checker: Checker) -> None:
    if len(op.operands) != 3:
        checker.error(op, "takes the buffer, the start, and the end")
        return
    buffer, pos, endpos = (v.type for v in op.operands)
    if buffer != BufferType(U8):
        checker.error(op, f"matches over buffer<u8>, not {buffer}")
    if pos != I64 or endpos != I64:
        checker.error(op, "the start and the end are i64")
    try:
        compiled = analyse(*pattern_of(op))
    except Unsupported as error:
        checker.error(op, str(error))
        return
    expected: list[IRType] = [BOOL, *([I64] * compiled.spans)]
    if [r.type for r in op.results] != expected:
        checker.error(op, f"yields (bool, i64 x {compiled.spans}) for this pattern")


class RegexDialect(Dialect):
    name = "regex"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        for mode in MODES:
            registry.add_op(
                OpSpec(
                    f"regex.{mode}",
                    pure=True,
                    verify=_verify,
                    operands=3,
                    required_attributes=("pattern",),
                )
            )


def create(
    b: Builder,
    mode: str,
    buffer: Value,
    pos: Value,
    endpos: Value,
    compiled: Compiled,
    name: str | None = None,
) -> Operation:
    """`regex.<mode>` over `buffer[pos:endpos]`: found, then every group's span."""
    if mode not in MODES:
        raise ValueError(f"a regex operation is one of {MODES}, not {mode!r}")
    results: list[IRType] = [BOOL, *([I64] * compiled.spans)]
    names: list[str | None] = [name or "found"]
    for group in range(compiled.groups + 1):
        names.extend((f"start{group}", f"end{group}"))
    attributes: dict[str, Attribute] = {
        "pattern": compiled.pattern.decode("latin-1"),
        "flags": int(compiled.flags),
    }
    return b.create(f"regex.{mode}", (buffer, pos, endpos), results, attributes, result_names=names)


def _unused(_t: IntType) -> None:
    return None
