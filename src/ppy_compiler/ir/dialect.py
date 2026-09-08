"""Dialects: the units the IR is extended in.

A dialect owns a namespace of operation names and types, and says what each
operation requires -- how many operands and results, which attributes, what
the types must be -- through an `OpSpec` the verifier consults. The core
dialect is one of them; every other is registered the same way, by the
compiler or by a plugin, and a module records which dialects it uses at
which version so that a reader without one refuses it instead of guessing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .types import DialectType, IRType

if TYPE_CHECKING:
    from .model import IRFunction, Operation
    from .verify import Checker

__all__ = ["Dialect", "DialectRegistry", "OpSpec", "registry"]


@dataclass(frozen=True, slots=True)
class OpSpec:
    """What one operation name requires, and what it may be treated as."""

    name: str
    #: Ends a block: has successors, nothing follows it.
    terminator: bool = False
    #: No side effects: unused results make it dead, equal ones fold together.
    pure: bool = False
    commutative: bool = False
    #: An attribute printed as a suffix of the name (`core.cmp.lt`), with the
    #: values it may take.
    variant_attribute: str | None = None
    variants: tuple[str, ...] = ()
    #: An attribute printed inline after the name (`core.const 0`).
    inline_attribute: str | None = None
    #: The dialect's own checks, given the operation and where to report.
    verify: Callable[[Operation, Checker], None] | None = None
    #: The number of operands, or None for any.
    operands: int | None = None
    results: int | None = None
    #: Attributes every instance must carry.
    required_attributes: tuple[str, ...] = ()
    #: Successor count, or None for any.
    successors: int | None = None
    regions: int = 0


class Dialect:
    """A namespace of types and operations; subclasses register theirs."""

    name: str = ""
    version: int = 1

    def register_types(self, registry: DialectRegistry) -> None:
        """Types this dialect owns are verified by `verify_type`; nothing to do by default."""

    def register_operations(self, registry: DialectRegistry) -> None:
        """Add this dialect's `OpSpec`s: `registry.add_op(spec)`."""

    def register_patterns(self, registry: object) -> None:
        """Canonicalization patterns (`ir.pattern`); nothing by default."""

    def register_lowerings(self, registry: object) -> None:
        """Lowerings to other dialects or to a backend; nothing by default."""

    def verify_type(self, t: DialectType) -> str | None:
        """Why `t` is not a type this dialect defines, or None when it is."""
        return f"{self.name} defines no type {t.name!r}"

    def address_spaces(self) -> frozenset[str]:
        """Address spaces this dialect gives pointers."""
        return frozenset()

    def verify_function(self, function: IRFunction, checker: Checker) -> None:
        """Rules over a whole function -- its attributes, what its body may hold.

        The verifier asks every registered dialect after a function's own
        checks; the default has nothing to say.
        """


class DialectRegistry:
    """The dialects a compiler knows, and every operation they define."""

    def __init__(self) -> None:
        self.dialects: dict[str, Dialect] = {}
        self._ops: dict[str, OpSpec] = {}
        #: Patterns contributed outside any dialect: a plugin's rewrites.
        self.patterns: list[object] = []

    def register(self, dialect: Dialect) -> None:
        if not dialect.name:
            raise ValueError("a dialect needs a name")
        existing = self.dialects.get(dialect.name)
        if existing is not None and type(existing) is not type(dialect):
            raise ValueError(
                f"dialect {dialect.name!r} is already registered by "
                f"{type(existing).__module__}.{type(existing).__qualname__}"
            )
        self.dialects[dialect.name] = dialect
        dialect.register_types(self)
        dialect.register_operations(self)

    def add_op(self, spec: OpSpec) -> None:
        dialect = spec.name.partition(".")[0]
        if dialect not in self.dialects:
            raise ValueError(f"{spec.name} belongs to unregistered dialect {dialect!r}")
        self._ops[spec.name] = spec

    def add_pattern(self, pattern: object) -> None:
        """A rewrite pattern that belongs to no dialect of its own."""
        self.patterns.append(pattern)

    def dialect(self, name: str) -> Dialect | None:
        return self.dialects.get(name)

    def op_spec(self, name: str) -> OpSpec | None:
        return self._ops.get(name)

    def ops_of(self, dialect: str) -> list[OpSpec]:
        return [
            spec for name, spec in sorted(self._ops.items()) if spec.name.startswith(dialect + ".")
        ]

    def address_spaces(self) -> frozenset[str]:
        spaces: set[str] = set()
        for dialect in self.dialects.values():
            spaces |= dialect.address_spaces()
        return frozenset(spaces)

    def verify_type(self, t: IRType) -> str | None:
        """Why a type is not one any registered dialect defines, or None."""
        if not isinstance(t, DialectType):
            return None
        dialect = self.dialects.get(t.dialect)
        if dialect is None:
            return f"unknown dialect {t.dialect!r} in type {t}"
        return dialect.verify_type(t)


_registry: DialectRegistry | None = None


def registry() -> DialectRegistry:
    """The process-wide registry, with the builtin dialects registered."""
    global _registry  # noqa: PLW0603 - one registry per process, made on first use
    if _registry is None:
        _registry = DialectRegistry()
        from .dialects import builtin_dialects

        for dialect in builtin_dialects():
            _registry.register(dialect)
    return _registry
