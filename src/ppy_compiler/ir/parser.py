"""Reading the text the printer writes.

One cursor over the source; types are parsed by the type grammar's own
parser at that cursor, so there is one spelling of a type. Every value is
resolved when the function ends, so a branch may name a block or a value
that is defined later in the text; a name defined twice or never is an
error with its line.
"""

from __future__ import annotations

import json

from .dialect import DialectRegistry
from .dialect import registry as default_registry
from .model import (
    Attribute,
    Block,
    IRModule,
    Operation,
    Region,
    SourceLocation,
    Successor,
    SymbolRef,
    Value,
)
from .types import IRType, TypeError_, VoidType, _TypeParser

__all__ = ["ParseError", "parse_module"]


class ParseError(ValueError):
    def __init__(self, message: str, line: int) -> None:
        super().__init__(f"line {line}: {message}")
        self.line = line


def parse_module(text: str, registry: DialectRegistry | None = None) -> IRModule:
    """The module `text` spells; the header is validated by the codec."""
    return _Parser(text, registry or default_registry()).module()


class _Cursor:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    @property
    def line(self) -> int:
        return self.text.count("\n", 0, self.pos) + 1

    def error(self, message: str) -> ParseError:
        return ParseError(message, self.line)

    def skip(self) -> None:
        """Whitespace and `//` comments."""
        text = self.text
        while self.pos < len(text):
            if text[self.pos] in " \t\r\n":
                self.pos += 1
            elif text.startswith("//", self.pos):
                end = text.find("\n", self.pos)
                self.pos = len(text) if end < 0 else end
            else:
                break

    def peek(self, what: str = "") -> bool | str:
        self.skip()
        if what:
            return self.text.startswith(what, self.pos)
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def accept(self, what: str) -> bool:
        if self.peek(what):
            self.pos += len(what)
            return True
        return False

    def expect(self, what: str) -> None:
        if not self.accept(what):
            found = self.text[self.pos : self.pos + 12]
            raise self.error(f"expected {what!r}, found {found!r}")

    def word(self, allow_dots: bool = False) -> str:
        self.skip()
        start = self.pos
        while self.pos < len(self.text):
            c = self.text[self.pos]
            if c.isalnum() or c == "_" or (allow_dots and c == "."):
                self.pos += 1
            else:
                break
        if start == self.pos:
            raise self.error(f"expected a name, found {self.text[self.pos : self.pos + 12]!r}")
        return self.text[start : self.pos]

    def integer(self) -> int:
        self.skip()
        start = self.pos
        if self.peek() == "-":
            self.pos += 1
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        if start == self.pos or self.text[start : self.pos] == "-":
            raise self.error("expected an integer")
        return int(self.text[start : self.pos])

    def string(self) -> str:
        self.skip()
        if self.peek() != '"':
            raise self.error("expected a string")
        end = self.pos + 1
        while end < len(self.text):
            if self.text[end] == "\\":
                end += 2
                continue
            if self.text[end] == '"':
                break
            end += 1
        raw = self.text[self.pos : end + 1]
        self.pos = end + 1
        try:
            return json.loads(raw)
        except ValueError as error:
            raise self.error(f"bad string {raw!r}") from error

    def type_(self) -> IRType:
        self.skip()
        parser = _TypeParser(self.text, self.pos)
        try:
            result = parser.type_()
        except TypeError_ as error:
            raise self.error(str(error)) from error
        self.pos = parser.pos
        return result


class _Parser:
    def __init__(self, text: str, registry: DialectRegistry) -> None:
        self.cursor = _Cursor(text)
        self.registry = registry

    def module(self) -> IRModule:
        c = self.cursor
        c.expect("ppyir")
        schema = c.integer()
        c.expect("module")
        c.expect("@")
        module = IRModule(c.word(allow_dots=True), dialects={})
        module.attributes["_schema"] = schema
        while c.accept("dialect"):
            name = c.word()
            module.dialects[name] = c.integer()
        if c.accept("attrs"):
            attributes = self.attribute()
            assert isinstance(attributes, dict)
            module.attributes.update(attributes)
        while c.peek():
            visibility = "public"
            if c.accept("private"):
                visibility = "private"
            elif c.accept("extern"):
                visibility = "extern"
            constant = not c.accept("mutable")
            if c.accept("global"):
                self.global_(module, visibility, constant)
            elif c.accept("func"):
                self.function(module, visibility)
            else:
                raise c.error(f"expected `func` or `global`, found {c.text[c.pos : c.pos + 12]!r}")
        return module

    def global_(self, module: IRModule, visibility: str, constant: bool) -> None:
        c = self.cursor
        c.expect("@")
        name = c.word(allow_dots=True)
        c.expect(":")
        t = c.type_()
        value = None
        if c.accept("="):
            value = self.attribute()
        module.add_global(name, t, value, constant=constant, visibility=visibility)

    def function(self, module: IRModule, visibility: str) -> None:
        c = self.cursor
        c.expect("@")
        name = c.word(allow_dots=True)
        c.expect("(")
        params: list[tuple[str, IRType]] = []
        param_attributes: list[dict[str, Attribute]] = []
        while not c.accept(")"):
            if c.accept("%"):
                param_name = c.word()
                c.expect(":")
            else:
                param_name = f"arg{len(params)}"
            t = c.type_()
            attributes: dict[str, Attribute] = {}
            if c.peek("{"):
                parsed = self.attribute()
                assert isinstance(parsed, dict)
                attributes = parsed
            params.append((param_name, t))
            param_attributes.append(attributes)
            if not c.accept(","):
                c.expect(")")
                break
        c.expect("->")
        results: list[IRType] = []
        if c.accept("("):
            while not c.accept(")"):
                results.append(c.type_())
                if not c.accept(","):
                    c.expect(")")
                    break
        else:
            t = c.type_()
            if not isinstance(t, VoidType):
                results.append(t)
        attributes = {}
        if c.accept("attrs"):
            parsed = self.attribute()
            assert isinstance(parsed, dict)
            attributes = parsed
        location = self.location()
        function = module.add_function(
            name, params, results, visibility=visibility, attributes=attributes, location=location
        )
        function.param_attributes = param_attributes
        if c.accept("{"):
            scope = _Scope(self)
            self.region(function.body, scope, entry_params=function.params)
            c.expect("}")
            scope.resolve()

    def location(self) -> SourceLocation | None:
        c = self.cursor
        if not c.accept("loc("):
            return None
        file = c.string()
        c.expect(":")
        line = c.integer()
        c.expect(":")
        column = c.integer()
        c.expect(")")
        return SourceLocation(file, line, column)

    def region(
        self,
        region: Region,
        scope: _Scope,
        entry_params: tuple[tuple[str, IRType], ...] | None = None,
    ) -> None:
        c = self.cursor
        while c.peek("^"):
            c.expect("^")
            label = c.word()
            block = scope.define_block(label, region)
            if entry_params is not None and not region.blocks[1:] and not c.peek("("):
                # The entry block of a function takes the parameters.
                for param_name, t in entry_params:
                    scope.define(param_name, block.add_argument(t, param_name))
            elif c.accept("("):
                while not c.accept(")"):
                    c.expect("%")
                    value_name = c.word()
                    c.expect(":")
                    argument = block.add_argument(c.type_(), value_name)
                    scope.define(value_name, argument)
                    if not c.accept(","):
                        c.expect(")")
                        break
            c.expect(":")
            while c.peek() and not c.peek("^") and not c.peek("}"):
                block.append(self.operation(scope))

    def operation(self, scope: _Scope) -> Operation:
        c = self.cursor
        c.skip()
        line = c.line
        result_names: list[str] = []
        if c.peek("%"):
            while c.accept("%"):
                result_names.append(c.word())
                if not c.accept(","):
                    break
            c.expect("=")
        name = c.word(allow_dots=True)
        attributes: dict[str, Attribute] = {}
        spec = self.registry.op_spec(name)
        if spec is None and name.count(".") >= 2:
            base, _, variant = name.rpartition(".")
            base_spec = self.registry.op_spec(base)
            if base_spec is not None and base_spec.variant_attribute is not None:
                spec = base_spec
                name = base
                attributes[base_spec.variant_attribute] = variant
        if (
            spec is not None
            and spec.inline_attribute is not None
            and not any(c.peek(mark) for mark in ("%", "{", ":", "^", "loc("))
        ):
            attributes[spec.inline_attribute] = self.attribute()
        operand_names: list[str] = []
        while c.accept("%"):
            operand_names.append(c.word())
            if not c.accept(","):
                break
        successors: list[tuple[str, list[str]]] = []
        while c.accept("^"):
            target = c.word()
            arguments: list[str] = []
            if c.accept("("):
                while not c.accept(")"):
                    c.expect("%")
                    arguments.append(c.word())
                    if not c.accept(","):
                        c.expect(")")
                        break
            successors.append((target, arguments))
            if not c.accept(","):
                break
        if c.peek("{") and not c.peek("{\n"):
            parsed = self.attribute()
            assert isinstance(parsed, dict)
            attributes.update(parsed)
        result_types: list[IRType] = []
        if c.accept(":"):
            result_types.append(c.type_())
            while c.accept(","):
                result_types.append(c.type_())
        if len(result_types) != len(result_names):
            raise c.error(
                f"{name} names {len(result_names)} result(s) but gives {len(result_types)} type(s)"
            )
        location = self.location()
        op = Operation(name, (), result_types, attributes, (), location, result_names)
        for value_name, result in zip(result_names, op.results, strict=True):
            scope.define(value_name, result, line)
        scope.pending_operands(op, operand_names, line)
        for target, arguments in successors:
            successor = Successor(scope.block_ref(target))
            op.successors.append(successor)
            scope.pending_successor(successor, arguments, line)
        while c.peek("{"):
            c.expect("{")
            self.region(op.add_region(), scope)
            c.expect("}")
        return op

    def attribute(self) -> Attribute:
        c = self.cursor
        c.skip()
        if c.accept("{"):
            table: dict[str, Attribute] = {}
            while not c.accept("}"):
                key = c.word(allow_dots=True)
                c.expect("=")
                table[key] = self.attribute()
                if not c.accept(","):
                    c.expect("}")
                    break
            return table
        if c.accept("["):
            items: list[Attribute] = []
            while not c.accept("]"):
                items.append(self.attribute())
                if not c.accept(","):
                    c.expect("]")
                    break
            return tuple(items)
        if c.peek('"'):
            return c.string()
        if c.accept("@"):
            return SymbolRef(c.word(allow_dots=True))
        if c.accept("!"):
            return c.type_()
        if c.accept("true"):
            return True
        if c.accept("false"):
            return False
        if c.accept("nan"):
            return float("nan")
        if c.accept("-inf"):
            return float("-inf")
        if c.accept("inf"):
            return float("inf")
        return self.number()

    def number(self) -> int | float:
        c = self.cursor
        c.skip()
        start = c.pos
        text = c.text
        if c.pos < len(text) and text[c.pos] == "-":
            c.pos += 1
        while c.pos < len(text) and (text[c.pos].isdigit() or text[c.pos] in ".eE+-"):
            if text[c.pos] in "+-" and text[c.pos - 1] not in "eE":
                break
            c.pos += 1
        raw = text[start : c.pos]
        if not raw or raw == "-":
            raise c.error(f"expected an attribute value, found {text[start : start + 12]!r}")
        try:
            if any(ch in raw for ch in ".eE"):
                return float(raw)
            return int(raw)
        except ValueError as error:
            raise c.error(f"bad number {raw!r}") from error


class _Scope:
    """Names within one function, resolved once the function is read."""

    def __init__(self, parser: _Parser) -> None:
        self.parser = parser
        self.values: dict[str, Value] = {}
        self.blocks: dict[str, Block] = {}
        self._block_refs: dict[str, Block] = {}
        self._operands: list[tuple[Operation, list[str], int]] = []
        self._successors: list[tuple[Successor, list[str], int]] = []

    def define(self, name: str, value: Value, line: int | None = None) -> None:
        if name in self.values:
            raise ParseError(
                f"%{name} is defined twice", self.parser.cursor.line if line is None else line
            )
        self.values[name] = value

    def define_block(self, name: str, region: Region) -> Block:
        block = self._block_refs.get(name)
        if block is not None and block.region is not None:
            raise self.parser.cursor.error(f"^{name} is defined twice")
        if block is None:
            block = Block(name)
            self._block_refs[name] = block
        block.region = region
        region.blocks.append(block)
        self.blocks[name] = block
        return block

    def block_ref(self, name: str) -> Block:
        block = self._block_refs.get(name)
        if block is None:
            block = Block(name)
            self._block_refs[name] = block
        return block

    def pending_operands(self, op: Operation, names: list[str], line: int) -> None:
        self._operands.append((op, names, line))

    def pending_successor(self, successor: Successor, names: list[str], line: int) -> None:
        self._successors.append((successor, names, line))

    def resolve(self) -> None:
        for name, block in self._block_refs.items():
            if block.region is None:
                raise ParseError(f"branch to undefined block ^{name}", 0)
        for op, names, line in self._operands:
            for name in names:
                value = self.values.get(name)
                if value is None:
                    raise ParseError(f"%{name} is not defined", line)
                op.operands.append(value)
                value.uses.append((op, len(op.operands) - 1))
        for successor, names, line in self._successors:
            for name in names:
                value = self.values.get(name)
                if value is None:
                    raise ParseError(f"%{name} is not defined", line)
                successor.arguments.append(value)
                value.uses.append((successor, len(successor.arguments) - 1))
