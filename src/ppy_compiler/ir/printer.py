"""The IR as text, one spelling per module.

Values are named in definition order: a value keeps the name it was given
when no earlier value in the function took it, and is numbered otherwise.
Attributes print sorted by key. So two modules with the same content print
to the same bytes, and a module printed after being parsed prints the same
text again.
"""

from __future__ import annotations

import json
import math

from .dialect import DialectRegistry
from .dialect import registry as default_registry
from .model import (
    Attribute,
    Block,
    Global,
    IRFunction,
    IRModule,
    Operation,
    Region,
    SourceLocation,
    SymbolRef,
    Value,
)
from .types import IRType

__all__ = ["print_attribute", "print_function", "print_module"]

_INDENT = "    "


def print_module(module: IRModule, registry: DialectRegistry | None = None) -> str:
    """The module's text, including the header the codec reads."""
    from .codec import IR_SCHEMA_VERSION

    lines = [f"ppyir {IR_SCHEMA_VERSION}", f"module @{module.name}"]
    for name, version in sorted(module.dialects.items()):
        lines.append(f"dialect {name} {version}")
    if module.attributes:
        lines.append(f"attrs {print_attribute(module.attributes)}")
    lines.append("")
    for name, item in sorted(module.globals.items()):
        lines.append(_global(name, item))
    if module.globals:
        lines.append("")
    for function in module.functions.values():
        lines.append(print_function(function, registry))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def print_function(function: IRFunction, registry: DialectRegistry | None = None) -> str:
    namer = _Namer()
    for block in function.body.blocks:
        for argument in block.arguments:
            namer.name(argument)
    for op in function.operations():
        for result in op.results:
            namer.name(result)
    prefix = {"public": "", "private": "private ", "extern": "extern "}[function.symbol.visibility]
    params = ", ".join(
        f"%{namer.name(function.entry.arguments[index])}: {t}"
        + (f" {print_attribute(attrs)}" if attrs else "")
        if function.entry is not None
        else f"{t}" + (f" {print_attribute(attrs)}" if attrs else "")
        for index, ((_name, t), attrs) in enumerate(
            zip(function.params, function.param_attributes, strict=True)
        )
    )
    results = _results(function.results)
    head = f"{prefix}func @{function.name}({params}) -> {results}"
    if function.attributes:
        head += f" attrs {print_attribute(function.attributes)}"
    if function.location is not None:
        head += f" loc({_location(function.location)})"
    if function.is_declaration:
        return head
    body = _region(function.body, namer, registry or default_registry(), 0)
    return f"{head} {{\n{body}}}"


def _results(results: tuple[IRType, ...]) -> str:
    if not results:
        return "()"
    if len(results) == 1:
        return str(results[0])
    return f"({', '.join(map(str, results))})"


def _global(name: str, item: Global) -> str:
    prefix = "" if item.constant else "mutable "
    visibility = {"public": "", "private": "private ", "extern": "extern "}[item.symbol.visibility]
    text = f"{visibility}{prefix}global @{name} : {item.type}"
    if item.value is not None:
        text += f" = {print_attribute(item.value)}"
    return text


def _region(region: Region, namer: _Namer, registry: DialectRegistry, depth: int) -> str:
    lines: list[str] = []
    for index, block in enumerate(region.blocks):
        # A function's entry block takes the parameters, which the signature
        # already spells; a nested region's entry lists its own.
        lines.append(_block_label(block, namer, depth, arguments=depth > 0 or index > 0))
        previous: SourceLocation | None = None
        for op in block.operations:
            # A location the previous operation already spelled is inherited.
            lines.append(_operation(op, namer, registry, depth + 1, previous))
            previous = op.location
    return "\n".join(lines) + "\n"


def _block_label(block: Block, namer: _Namer, depth: int, arguments: bool = True) -> str:
    label = f"{_INDENT * depth}^{block.name}"
    if block.arguments and arguments:
        label += "(" + ", ".join(f"%{namer.name(a)}: {a.type}" for a in block.arguments) + ")"
    return label + ":"


def _operation(
    op: Operation,
    namer: _Namer,
    registry: DialectRegistry,
    depth: int,
    previous: SourceLocation | None = None,
) -> str:
    spec = registry.op_spec(op.name)
    parts: list[str] = []
    if op.results:
        parts.append(", ".join(f"%{namer.name(r)}" for r in op.results) + " =")
    name = op.name
    attributes = dict(op.attributes)
    if spec is not None and spec.variant_attribute in attributes:
        name += f".{attributes.pop(spec.variant_attribute)}"
    parts.append(name)
    if spec is not None and spec.inline_attribute in attributes:
        parts.append(print_attribute(attributes.pop(spec.inline_attribute)))
    inputs = [f"%{namer.name(v)}" for v in op.operands]
    for successor in op.successors:
        target = f"^{successor.block.name}"
        if successor.arguments:
            target += "(" + ", ".join(f"%{namer.name(v)}" for v in successor.arguments) + ")"
        inputs.append(target)
    if inputs:
        parts.append(", ".join(inputs))
    if attributes:
        parts.append(print_attribute(attributes))
    if op.results:
        parts.append(": " + ", ".join(str(r.type) for r in op.results))
    if op.location is not None and op.location != previous:
        parts.append(f"loc({_location(op.location)})")
    line = _INDENT * depth + " ".join(parts)
    for region in op.regions:
        line += " {\n" + _region(region, namer, registry, depth) + _INDENT * depth + "}"
    return line


def _location(location: SourceLocation) -> str:
    return f"{json.dumps(location.file)}:{location.line}:{location.column}"


def print_attribute(value: Attribute) -> str:
    """One attribute value as the parser reads it back."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        text = repr(value)
        return text if ("." in text or "e" in text) else text + ".0"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, SymbolRef):
        return str(value)
    if isinstance(value, IRType):
        return f"!{value}"
    if isinstance(value, tuple):
        return "[" + ", ".join(print_attribute(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{" + ", ".join(f"{k} = {print_attribute(v)}" for k, v in sorted(value.items())) + "}"
        )
    raise TypeError(f"no attribute spelling for {value!r}")


class _Namer:
    """Deterministic value names within one function."""

    def __init__(self) -> None:
        self._names: dict[int, str] = {}
        self._taken: set[str] = set()
        self._counter = 0

    def name(self, value: Value) -> str:
        known = self._names.get(id(value))
        if known is not None:
            return known
        hint = _sanitize(value.name) if value.name else ""
        if not hint or hint in self._taken:
            while str(self._counter) in self._taken:
                self._counter += 1
            hint = str(self._counter)
            self._counter += 1
        self._taken.add(hint)
        self._names[id(value)] = hint
        return hint


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)
