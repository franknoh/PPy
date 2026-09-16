"""Byte-exact literals and formats for the shared standalone source emitter."""

from __future__ import annotations

from ...ir import Operation, Value


def string_literal(data: bytes) -> str:
    """ASCII source, fixed-width octal escapes: no locale or hex lookahead."""
    escapes = {34: '\\"', 92: "\\\\", 63: "\\?", 10: "\\n", 13: "\\r", 9: "\\t"}
    return (
        '"'
        + "".join(
            escapes.get(byte, chr(byte) if 32 <= byte < 127 else f"\\{byte:03o}") for byte in data
        )
        + '"'
    )


def string_symbol(value: Value) -> str | None:
    producer = value.owner
    if not isinstance(producer, Operation):
        return None
    if producer.name == "core.cast":
        return string_symbol(producer.operands[0])
    if (
        producer.name == "core.call_intrinsic"
        and producer.attributes.get("intrinsic") == "ppy.string_data"
    ):
        return str(producer.attributes["symbol"])
    return None


class PrintFormat:
    """One effect-local printf; NUL bytes travel through %c, never terminate it."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.text = bytearray()
        self.arguments: list[str] = []

    def literal(self, data: bytes) -> None:
        for byte in data:
            if byte == 0:
                self.value("%c", "0")
            else:
                self.text.extend(b"%%" if byte == 37 else bytes([byte]))

    def value(self, specifier: str, argument: str, macro: str = "") -> None:
        self.text.extend(specifier.encode("ascii"))
        if macro:
            self.parts.extend([string_literal(bytes(self.text)), macro])
            self.text.clear()
        self.arguments.append(argument)

    def finish(self) -> str:
        if self.text or not self.parts:
            self.parts.append(string_literal(bytes(self.text)))
        return ", ".join([" ".join(self.parts), *self.arguments])
