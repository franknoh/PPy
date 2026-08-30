"""The PPY native ABI, as data: what a compiled artifact promises (spec 16.4).

This is the contract a binding manifest serializes and a runtime rebuilds.
Nothing here may depend on the compiler: the same dataclasses describe a
function to the lowering that emits it and to the runtime that calls it
years later.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["STATUS_FALLBACK", "STATUS_OK", "NativeParam", "NativeSignature"]

STATUS_OK = 0
STATUS_FALLBACK = 1

_ABI_NAMES = {"int": "i64", "float": "double", "bool": "i8"}


def _abi_name(scalar: str) -> str:
    return _ABI_NAMES[scalar]


@dataclass(frozen=True, slots=True)
class NativeParam:
    """One source-level parameter and the ABI atoms it expands to."""

    name: str
    kind: str
    element: str = ""
    elements: tuple[str, ...] = ()
    fields: tuple[tuple[str, str], ...] = ()
    class_name: str = ""

    @property
    def is_buffer(self) -> bool:
        return self.kind in {"list", "sequence", "view"}

    @property
    def is_object(self) -> bool:
        return self.kind == "object"

    @property
    def is_borrowed(self) -> bool:
        """A `ppy.Buffer[T]` is borrowed in place; a list is copied out.

        A Python list holds boxed elements, so there is no contiguous array to
        point at. A buffer-protocol object already has one (spec 6.4, 13.8).
        """
        return self.kind == "view"

    @property
    def is_tuple(self) -> bool:
        return self.kind == "tuple"

    @property
    def abi(self) -> tuple[str, ...]:
        if self.is_buffer:
            return (f"{_abi_name(self.element)}*", "i64")
        if self.is_tuple:
            return tuple(_abi_name(element) for element in self.elements)
        if self.is_object:
            return tuple(_abi_name(scalar) for _field, scalar in self.fields)
        return (_abi_name(self.kind),)

    def __str__(self) -> str:
        if self.is_buffer:
            borrow = " borrowed" if self.is_borrowed else ""
            return f"{_abi_name(self.element)}*{borrow} {self.name}, i64 {self.name}_len"
        if self.is_tuple:
            return ", ".join(
                f"{_abi_name(element)} {self.name}{index}"
                for index, element in enumerate(self.elements)
            )
        if self.is_object:
            return ", ".join(
                f"{_abi_name(scalar)} {self.name}_{field}" for field, scalar in self.fields
            )
        return f"{_abi_name(self.kind)} {self.name}"


@dataclass(frozen=True, slots=True)
class NativeSignature:
    """The PPY native ABI for one function (spec 16.4)."""

    qualname: str
    symbol: str
    parameters: tuple[NativeParam, ...]
    returns: tuple[str, ...]
    #: The body touches no Python object once its arguments are unpacked, so
    #: the boundary may drop the GIL around the call (spec 16.6).
    releases_gil: bool = False

    @property
    def ret(self) -> str:
        return self.returns[0] if len(self.returns) == 1 else "{" + ", ".join(self.returns) + "}"

    @property
    def returns_tuple(self) -> bool:
        return len(self.returns) > 1

    @property
    def params(self) -> tuple[str, ...]:
        return tuple(atom for parameter in self.parameters for atom in parameter.abi)

    def __str__(self) -> str:
        rendered = ", ".join(str(p) for p in self.parameters)
        return f"{self.ret} {self.symbol}({rendered})"
