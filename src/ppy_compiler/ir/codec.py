"""`.ppyir`: the IR on disk, versioned.

The text the printer writes is the format: a header naming the schema
version and every dialect the module uses with its version, then the
module. Reading checks the header before anything else -- a newer schema,
an unknown dialect, or a dialect at a version this compiler does not have
is refused with the reason, not parsed on the hope that it is close
enough. Encoding is deterministic: the same module encodes to the same
bytes, whatever built it.

The format is experimental in 0.2.0: a reader may refuse what an older
writer wrote, and says so.
"""

from __future__ import annotations

from pathlib import Path

from .dialect import DialectRegistry
from .dialect import registry as default_registry
from .model import IRModule
from .parser import ParseError, parse_module
from .printer import print_module

__all__ = ["IR_SCHEMA_VERSION", "CodecError", "decode", "encode", "read", "write"]

#: The `.ppyir` schema. 1: the 0.2.0 format.
IR_SCHEMA_VERSION = 1


class CodecError(ValueError):
    """Text this compiler cannot read as IR, with the reason."""


def encode(module: IRModule, registry: DialectRegistry | None = None) -> str:
    """The module as `.ppyir` text."""
    return print_module(module, registry)


def decode(text: str, registry: DialectRegistry | None = None) -> IRModule:
    """The module `text` holds; refuses a schema or dialect it does not have."""
    registry = registry or default_registry()
    head = text.lstrip().split("\n", 1)[0].split()
    if len(head) != 2 or head[0] != "ppyir" or not head[1].isdigit():
        raise CodecError("not a .ppyir file: the first line names the schema, `ppyir N`")
    schema = int(head[1])
    if schema != IR_SCHEMA_VERSION:
        raise CodecError(
            f"the file is .ppyir schema {schema}; this compiler reads schema "
            f"{IR_SCHEMA_VERSION} -- emit it again with this compiler"
        )
    try:
        module = parse_module(text, registry)
    except ParseError as error:
        raise CodecError(str(error)) from error
    module.attributes.pop("_schema", None)
    for name, version in module.dialects.items():
        dialect = registry.dialect(name)
        if dialect is None:
            raise CodecError(f"the module uses dialect {name!r}, which this compiler does not have")
        if version > dialect.version:
            raise CodecError(
                f"the module uses dialect {name!r} at version {version}; this compiler "
                f"has version {dialect.version}"
            )
    return module


def write(module: IRModule, path: Path, registry: DialectRegistry | None = None) -> None:
    path.write_text(encode(module, registry), encoding="utf-8")


def read(path: Path, registry: DialectRegistry | None = None) -> IRModule:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CodecError(f"cannot read {path}: {error}") from error
    return decode(text, registry)
