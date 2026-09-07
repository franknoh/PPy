"""The verifier: what every module must satisfy before anything reads it.

Structure first -- every block ends in one terminator, branches reach blocks
of the same region with the arguments those blocks take, every value is
defined before it is used along every path, every symbol a call names
exists -- then the dialects' own rules, operation by operation. A failure
is a list of errors with the operation each sits on, never an exception
out of the middle of a walk; `verify_or_raise` turns the list into one
exception for callers that want that.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dialect import DialectRegistry
from .dialect import registry as default_registry
from .model import Block, IRFunction, IRModule, Operation, Region, Value
from .types import PtrType, VectorType, VoidType

__all__ = ["Checker", "VerificationError", "VerifyError", "verify", "verify_or_raise"]


@dataclass(frozen=True, slots=True)
class VerifyError:
    message: str
    function: str = ""
    block: str = ""
    op: str = ""
    location: str = ""

    def __str__(self) -> str:
        where = ""
        if self.function:
            where = f"@{self.function}"
            if self.block:
                where += f" ^{self.block}"
            if self.op:
                where += f" {self.op}"
        text = f"{where}: {self.message}" if where else self.message
        return f"{text} ({self.location})" if self.location else text


class VerificationError(Exception):
    """A module the verifier refused, with every error it found."""

    def __init__(self, errors: list[VerifyError]) -> None:
        self.errors = errors
        super().__init__("\n".join(str(e) for e in errors))


class Checker:
    """Collects errors with their position; dialect rules call `error`."""

    def __init__(self, registry: DialectRegistry, module: IRModule | None) -> None:
        self.registry = registry
        self.module = module
        self.function: IRFunction | None = None
        self.block: Block | None = None
        self.errors: list[VerifyError] = []

    def error(self, op: Operation | None, message: str) -> None:
        self.errors.append(
            VerifyError(
                message,
                self.function.name if self.function else "",
                self.block.name if self.block else "",
                op.name if op is not None else "",
                str(op.location) if op is not None and op.location is not None else "",
            )
        )


def verify(module: IRModule, registry: DialectRegistry | None = None) -> list[VerifyError]:
    """Every error in `module`; an empty list means it is well-formed."""
    checker = Checker(registry or default_registry(), module)
    _verify_dialects(module, checker)
    seen: set[str] = set()
    for name, function in module.functions.items():
        if name in seen:
            checker.errors.append(VerifyError(f"@{name} is defined twice"))
        seen.add(name)
        _verify_function(function, checker)
    for name, item in module.globals.items():
        if name in seen:
            checker.errors.append(VerifyError(f"@{name} is defined twice"))
        seen.add(name)
        reason = checker.registry.verify_type(item.type)
        if reason is not None:
            checker.errors.append(VerifyError(f"global @{name}: {reason}"))
    return checker.errors


def verify_or_raise(module: IRModule, registry: DialectRegistry | None = None) -> None:
    errors = verify(module, registry)
    if errors:
        raise VerificationError(errors)


def _verify_dialects(module: IRModule, checker: Checker) -> None:
    for name, version in module.dialects.items():
        dialect = checker.registry.dialect(name)
        if dialect is None:
            checker.errors.append(VerifyError(f"unknown dialect {name!r}"))
        elif version > dialect.version:
            checker.errors.append(
                VerifyError(
                    f"dialect {name!r} version {version} is newer than the "
                    f"registered version {dialect.version}"
                )
            )


def _verify_function(function: IRFunction, checker: Checker) -> None:
    checker.function = function
    checker.block = None
    for _name, t in function.params:
        reason = checker.registry.verify_type(t)
        if reason is not None:
            checker.error(None, reason)
        if isinstance(t, VoidType):
            checker.error(None, "a parameter cannot be void")
    for t in function.results:
        reason = checker.registry.verify_type(t)
        if reason is not None:
            checker.error(None, reason)
    if function.is_declaration:
        checker.function = None
        return
    entry = function.entry
    assert entry is not None
    entry_types = tuple(a.type for a in entry.arguments)
    param_types = tuple(t for _name, t in function.params)
    if entry_types != param_types:
        checker.error(
            None,
            f"entry block takes ({', '.join(map(str, entry_types))}), "
            f"parameters are ({', '.join(map(str, param_types))})",
        )
    _verify_region(function.body, checker, defined_outside=set())
    _verify_names(function, checker)
    checker.function = None


def _verify_region(region: Region, checker: Checker, defined_outside: set[int]) -> None:
    """Structure and dominance for one region; nested regions recurse."""
    if not region.blocks:
        return
    names: dict[str, Block] = {}
    for block in region.blocks:
        if block.name in names:
            checker.block = block
            checker.error(None, f"block ^{block.name} is defined twice")
        names[block.name] = block
    for block in region.blocks:
        checker.block = block
        _verify_block_structure(block, region, checker)
    dominators = _dominators(region)
    # What each block may read: everything defined outside, its dominators'
    # definitions, its own arguments, and then its operations in order.
    definitions: dict[Block, set[int]] = {}
    for block in region.blocks:
        available = set(defined_outside)
        for dominator in dominators.get(block, {block}):
            if dominator is not block:
                available |= definitions.get(dominator, set()) or _defined_in(dominator)
        available |= {id(a) for a in block.arguments}
        checker.block = block
        for op in block.operations:
            for value in op.operands:
                if id(value) not in available:
                    checker.error(op, f"{_spell(value)} is used before it is defined")
            for successor in op.successors:
                for value in successor.arguments:
                    if id(value) not in available:
                        checker.error(op, f"{_spell(value)} is used before it is defined")
            for nested in op.regions:
                _verify_region(nested, checker, available)
                checker.block = block
            for result in op.results:
                available.add(id(result))
            _verify_op(op, checker)
        definitions[block] = _defined_in(block)


def _defined_in(block: Block) -> set[int]:
    found = {id(a) for a in block.arguments}
    for op in block.operations:
        found.update(id(r) for r in op.results)
    return found


def _verify_block_structure(block: Block, region: Region, checker: Checker) -> None:
    if not block.operations:
        checker.error(None, f"block ^{block.name} is empty: it needs a terminator")
        return
    for op in block.operations[:-1]:
        if op.spec_is_terminator():
            checker.error(op, "a terminator in the middle of a block")
    last = block.operations[-1]
    if not last.spec_is_terminator():
        checker.error(last, f"block ^{block.name} does not end in a terminator")
    for op in block.operations:
        for successor in op.successors:
            if successor.block.region is not region:
                checker.error(op, f"branch to ^{successor.block.name} leaves the region")


def _verify_op(op: Operation, checker: Checker) -> None:
    spec = checker.registry.op_spec(op.name)
    if spec is None:
        dialect = op.dialect
        if checker.registry.dialect(dialect) is None:
            checker.error(op, f"unknown dialect {dialect!r}")
        else:
            checker.error(op, f"dialect {dialect!r} defines no operation {op.local_name!r}")
        return
    module = checker.module
    if module is not None and op.dialect not in module.dialects:
        checker.error(op, f"the module does not declare dialect {op.dialect!r}")
    if spec.operands is not None and len(op.operands) != spec.operands:
        checker.error(op, f"takes {spec.operands} operand(s), given {len(op.operands)}")
        return
    if spec.results is not None and len(op.results) != spec.results:
        checker.error(op, f"yields {spec.results} result(s), given {len(op.results)}")
        return
    if spec.successors is not None and len(op.successors) != spec.successors:
        checker.error(op, f"has {spec.successors} successor(s), given {len(op.successors)}")
        return
    if not spec.terminator and op.successors:
        checker.error(op, "only a terminator has successors")
        return
    if len(op.regions) != spec.regions:
        checker.error(op, f"has {spec.regions} region(s), given {len(op.regions)}")
        return
    for attribute in spec.required_attributes:
        if attribute not in op.attributes:
            checker.error(op, f"needs attribute {attribute!r}")
            return
    if spec.variant_attribute is not None:
        variant = op.attributes.get(spec.variant_attribute)
        if variant not in spec.variants:
            checker.error(op, f"{spec.variant_attribute} {variant!r} is not one of {spec.variants}")
            return
    for value in (*op.operands, *(r for r in op.results)):
        reason = checker.registry.verify_type(value.type)
        if reason is not None:
            checker.error(op, reason)
            return
        if isinstance(value.type, VoidType):
            checker.error(op, "a value cannot be void")
            return
        if isinstance(value.type, PtrType):
            space = value.type.address_space
            if space not in checker.registry.address_spaces():
                checker.error(op, f"unknown address space {space!r}")
                return
        if isinstance(value.type, VectorType):
            reason = checker.registry.verify_type(value.type.element)
            if reason is not None:
                checker.error(op, reason)
                return
    if spec.verify is not None:
        spec.verify(op, checker)


def _verify_names(function: IRFunction, checker: Checker) -> None:
    """A value name, when given, names one value in its function."""
    seen: dict[str, Value] = {}
    for block in function.body.blocks:
        for value in block.arguments:
            _note_name(value, seen, checker, None)
    for op in function.operations():
        for value in op.results:
            _note_name(value, seen, checker, op)


def _note_name(
    value: Value, seen: dict[str, Value], checker: Checker, op: Operation | None
) -> None:
    if value.name is None:
        return
    if value.name in seen and seen[value.name] is not value:
        checker.error(op, f"%{value.name} is defined twice")
    seen[value.name] = value


def _spell(value: Value) -> str:
    return f"%{value.name}" if value.name else f"a {value.type} value"


def _dominators(region: Region) -> dict[Block, set[Block]]:
    """Dominator sets by the classic iteration; regions are small."""
    blocks = region.blocks
    if not blocks:
        return {}
    entry = blocks[0]
    predecessors: dict[Block, list[Block]] = {b: [] for b in blocks}
    for block in blocks:
        for successor in block.successors:
            if successor in predecessors:
                predecessors[successor].append(block)
    everything = set(blocks)
    dominators: dict[Block, set[Block]] = {b: set(everything) for b in blocks}
    dominators[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for block in blocks[1:]:
            incoming = [dominators[p] for p in predecessors[block]]
            new = set.intersection(*incoming) if incoming else set()
            new.add(block)
            if new != dominators[block]:
                dominators[block] = new
                changed = True
    return dominators
