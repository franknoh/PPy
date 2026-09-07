"""The IR's data model: modules, functions, blocks, operations, values.

Typed SSA with an explicit control-flow graph. Every value is defined once
-- as a block argument or as the result of one operation -- and carries one
type; every block ends in one terminator; branches pass arguments to the
blocks they reach. The structure is plain objects with use lists, so a
rewrite can replace a value everywhere it is read and a verifier can walk
what it wants.

Nothing here knows what an operation means. An operation is a name in a
dialect, operands, results, attributes, successors, and nested regions; the
dialect's `OpSpec` says what is allowed, and the verifier asks it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from .types import IRType

__all__ = [
    "Attribute",
    "Block",
    "BlockArgument",
    "Builder",
    "Global",
    "IRFunction",
    "IRModule",
    "OpResult",
    "Operation",
    "Region",
    "SourceLocation",
    "Successor",
    "Symbol",
    "SymbolRef",
    "Value",
]


@dataclass(frozen=True, slots=True)
class SourceLocation:
    file: str
    line: int
    column: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"


@dataclass(frozen=True, slots=True)
class SymbolRef:
    """A reference to a symbol by name: the `@callee` of a call."""

    name: str

    def __str__(self) -> str:
        return f"@{self.name}"


@dataclass(frozen=True, slots=True)
class Symbol:
    """A name in the module's symbol table and who may see it."""

    name: str
    visibility: str = "public"  # public | private | extern


#: What an attribute may hold. Nested tuples spell lists; a dict spells a
#: nested attribute table.
Attribute = int | float | bool | str | IRType | SymbolRef | tuple | dict


class Value:
    """Something an operation reads: a block argument or an operation result."""

    __slots__ = ("name", "type", "uses")

    def __init__(self, type_: IRType, name: str | None = None) -> None:
        self.type = type_
        self.name = name
        #: (operation, operand index) for every read of this value, plus
        #: (successor, index) for every branch argument.
        self.uses: list[tuple[object, int]] = []

    @property
    def owner(self) -> Operation | Block:
        raise NotImplementedError

    def replace_all_uses_with(self, other: Value) -> None:
        """Every read of this value reads `other` from now on."""
        if other is self:
            return
        for user, index in list(self.uses):
            if isinstance(user, Operation):
                user.set_operand(index, other)
            else:
                assert isinstance(user, Successor)
                user.set_argument(index, other)

    def __repr__(self) -> str:
        return f"%{self.name or '?'}: {self.type}"


class BlockArgument(Value):
    __slots__ = ("block", "index")

    def __init__(self, block: Block, index: int, type_: IRType, name: str | None = None) -> None:
        super().__init__(type_, name)
        self.block = block
        self.index = index

    @property
    def owner(self) -> Block:
        return self.block


class OpResult(Value):
    __slots__ = ("index", "op")

    def __init__(self, op: Operation, index: int, type_: IRType, name: str | None = None) -> None:
        super().__init__(type_, name)
        self.op = op
        self.index = index

    @property
    def owner(self) -> Operation:
        return self.op


class Successor:
    """A branch target with the arguments it receives."""

    __slots__ = ("arguments", "block")

    def __init__(self, block: Block, arguments: Sequence[Value] = ()) -> None:
        self.block = block
        self.arguments: list[Value] = []
        for value in arguments:
            self.arguments.append(value)
            value.uses.append((self, len(self.arguments) - 1))

    def set_argument(self, index: int, value: Value) -> None:
        old = self.arguments[index]
        old.uses.remove((self, index))
        self.arguments[index] = value
        value.uses.append((self, index))

    def drop_uses(self) -> None:
        for index, value in enumerate(self.arguments):
            value.uses.remove((self, index))


class Region:
    """A list of blocks; the first is the entry."""

    __slots__ = ("blocks", "parent")

    def __init__(self, parent: Operation | IRFunction | None = None) -> None:
        self.blocks: list[Block] = []
        self.parent = parent

    @property
    def entry(self) -> Block | None:
        return self.blocks[0] if self.blocks else None

    def add_block(self, name: str, arguments: Iterable[tuple[str | None, IRType]] = ()) -> Block:
        block = Block(name, self)
        for argument_name, type_ in arguments:
            block.add_argument(type_, argument_name)
        self.blocks.append(block)
        return block

    def __iter__(self) -> Iterator[Block]:
        return iter(self.blocks)


class Block:
    __slots__ = ("arguments", "name", "operations", "region")

    def __init__(self, name: str, region: Region | None = None) -> None:
        self.name = name
        self.region = region
        self.arguments: list[BlockArgument] = []
        self.operations: list[Operation] = []

    def add_argument(self, type_: IRType, name: str | None = None) -> BlockArgument:
        argument = BlockArgument(self, len(self.arguments), type_, name)
        self.arguments.append(argument)
        return argument

    @property
    def terminator(self) -> Operation | None:
        if self.operations and self.operations[-1].spec_is_terminator():
            return self.operations[-1]
        return None

    def append(self, op: Operation) -> None:
        op.parent = self
        self.operations.append(op)

    def insert(self, index: int, op: Operation) -> None:
        op.parent = self
        self.operations.insert(index, op)

    def remove(self, op: Operation) -> None:
        self.operations.remove(op)
        op.parent = None

    @property
    def successors(self) -> list[Block]:
        terminator = self.terminator
        return [s.block for s in terminator.successors] if terminator is not None else []

    def __iter__(self) -> Iterator[Operation]:
        return iter(self.operations)

    def __repr__(self) -> str:
        return f"^{self.name}"


class Operation:
    """One operation: `results = dialect.name operands successors attrs regions`."""

    __slots__ = (
        "attributes",
        "location",
        "name",
        "operands",
        "parent",
        "regions",
        "results",
        "successors",
    )

    def __init__(
        self,
        name: str,
        operands: Sequence[Value] = (),
        result_types: Sequence[IRType] = (),
        attributes: dict[str, Attribute] | None = None,
        successors: Sequence[Successor] = (),
        location: SourceLocation | None = None,
        result_names: Sequence[str | None] = (),
    ) -> None:
        if "." not in name:
            raise ValueError(f"an operation name is `dialect.name`, not {name!r}")
        self.name = name
        self.operands: list[Value] = []
        for value in operands:
            self.operands.append(value)
            value.uses.append((self, len(self.operands) - 1))
        self.results: list[OpResult] = [
            OpResult(self, index, type_, result_names[index] if index < len(result_names) else None)
            for index, type_ in enumerate(result_types)
        ]
        self.attributes: dict[str, Attribute] = dict(attributes or {})
        self.successors: list[Successor] = list(successors)
        self.regions: list[Region] = []
        self.location = location
        self.parent: Block | None = None

    @property
    def dialect(self) -> str:
        return self.name.partition(".")[0]

    @property
    def local_name(self) -> str:
        return self.name.partition(".")[2]

    @property
    def result(self) -> OpResult:
        if len(self.results) != 1:
            raise ValueError(f"{self.name} has {len(self.results)} results, not one")
        return self.results[0]

    def spec_is_terminator(self) -> bool:
        from .dialect import registry

        spec = registry().op_spec(self.name)
        return spec is not None and spec.terminator

    def set_operand(self, index: int, value: Value) -> None:
        old = self.operands[index]
        old.uses.remove((self, index))
        self.operands[index] = value
        value.uses.append((self, index))

    def add_region(self) -> Region:
        region = Region(self)
        self.regions.append(region)
        return region

    def erase(self) -> None:
        """Remove this operation from its block and drop every use it holds."""
        for index, value in enumerate(self.operands):
            value.uses.remove((self, index))
        self.operands = []
        for successor in self.successors:
            successor.drop_uses()
        self.successors = []
        if self.parent is not None:
            self.parent.remove(self)

    def __repr__(self) -> str:
        return f"<{self.name}>"


@dataclass(slots=True)
class Global:
    """A module-level constant or variable."""

    symbol: Symbol
    type: IRType
    value: Attribute | None = None
    constant: bool = True
    location: SourceLocation | None = None


class IRFunction:
    __slots__ = (
        "attributes",
        "body",
        "location",
        "param_attributes",
        "params",
        "results",
        "symbol",
    )

    def __init__(
        self,
        symbol: Symbol,
        params: Sequence[tuple[str, IRType]],
        results: Sequence[IRType],
        attributes: dict[str, Attribute] | None = None,
        location: SourceLocation | None = None,
    ) -> None:
        self.symbol = symbol
        self.params: tuple[tuple[str, IRType], ...] = tuple(params)
        self.param_attributes: list[dict[str, Attribute]] = [{} for _ in self.params]
        self.results: tuple[IRType, ...] = tuple(results)
        self.attributes: dict[str, Attribute] = dict(attributes or {})
        self.body = Region(self)
        self.location = location

    @property
    def name(self) -> str:
        return self.symbol.name

    @property
    def is_declaration(self) -> bool:
        return not self.body.blocks

    @property
    def entry(self) -> Block | None:
        return self.body.entry

    def add_entry_block(self) -> Block:
        """The entry block, with one argument per parameter, named after it."""
        return self.body.add_block("entry", self.params)

    def blocks(self) -> Iterator[Block]:
        return iter(self.body.blocks)

    def operations(self) -> Iterator[Operation]:
        """Every operation in the function, nested regions included, in order."""
        pending: list[Region] = [self.body]
        while pending:
            region = pending.pop(0)
            for block in region.blocks:
                for op in block.operations:
                    yield op
                    pending.extend(op.regions)

    def __repr__(self) -> str:
        return f"<func @{self.name}>"


class IRModule:
    __slots__ = ("attributes", "dialects", "functions", "globals", "name")

    def __init__(self, name: str, dialects: dict[str, int] | None = None) -> None:
        self.name = name
        #: Every dialect the module uses, with the version it was written against.
        self.dialects: dict[str, int] = dict(dialects or {"core": 1})
        self.functions: dict[str, IRFunction] = {}
        self.globals: dict[str, Global] = {}
        self.attributes: dict[str, Attribute] = {}

    def add_function(
        self,
        name: str,
        params: Sequence[tuple[str, IRType]],
        results: Sequence[IRType],
        *,
        visibility: str = "public",
        attributes: dict[str, Attribute] | None = None,
        location: SourceLocation | None = None,
    ) -> IRFunction:
        if name in self.functions or name in self.globals:
            raise ValueError(f"@{name} is already defined in module {self.name}")
        function = IRFunction(Symbol(name, visibility), params, results, attributes, location)
        self.functions[name] = function
        return function

    def add_global(
        self,
        name: str,
        type_: IRType,
        value: Attribute | None = None,
        *,
        constant: bool = True,
        visibility: str = "public",
    ) -> Global:
        if name in self.functions or name in self.globals:
            raise ValueError(f"@{name} is already defined in module {self.name}")
        item = Global(Symbol(name, visibility), type_, value, constant)
        self.globals[name] = item
        return item

    def symbol(self, name: str) -> IRFunction | Global | None:
        return self.functions.get(name) or self.globals.get(name)

    def require(self, dialect: str, version: int) -> None:
        """Record that the module uses `dialect` at `version`."""
        self.dialects[dialect] = max(self.dialects.get(dialect, 0), version)


class Builder:
    """Appends operations at an insertion point.

    The dialect modules put their typed constructors on top of `create`:
    `core.add(builder, x, y, overflow="python")`.
    """

    __slots__ = ("anchor", "block", "location")

    def __init__(self, block: Block | None = None, location: SourceLocation | None = None) -> None:
        self.block = block
        #: Every created operation goes before this one; None means at the end.
        self.anchor: Operation | None = None
        self.location = location

    def at_end(self, block: Block) -> Builder:
        self.block = block
        self.anchor = None
        return self

    def before(self, op: Operation) -> Builder:
        assert op.parent is not None
        self.block = op.parent
        self.anchor = op
        return self

    def create(
        self,
        name: str,
        operands: Sequence[Value] = (),
        result_types: Sequence[IRType] = (),
        attributes: dict[str, Attribute] | None = None,
        successors: Sequence[Successor] = (),
        result_names: Sequence[str | None] = (),
        location: SourceLocation | None = None,
    ) -> Operation:
        op = Operation(
            name,
            operands,
            result_types,
            attributes,
            successors,
            location or self.location,
            result_names,
        )
        if self.block is None:
            raise ValueError("the builder has no insertion point")
        if self.anchor is None:
            self.block.append(op)
        else:
            self.block.insert(self.block.operations.index(self.anchor), op)
        return op
