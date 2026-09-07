"""The layout dialect: how a tensor's elements sit in memory.

`layout.row_major` and `layout.col_major` are the two contiguous orders;
`layout.strided<s0, s1, ...>` gives an element stride per axis, and may
end in `offset, K` (elements before the first) and `align, A` (bytes the
first element is aligned to). The tensor dialect carries one of these in
every tensor type and says nothing else about memory: an operation's
meaning is on the elements, the layout is how a backend reaches them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..dialect import Dialect, DialectRegistry
from ..types import DialectType

__all__ = ["COL_MAJOR", "ROW_MAJOR", "Layout", "LayoutDialect", "describe", "strided"]

ROW_MAJOR = DialectType("layout", "row_major")
COL_MAJOR = DialectType("layout", "col_major")


@dataclass(frozen=True, slots=True)
class Layout:
    """A layout read out of its type: strides per axis, or a contiguous order."""

    order: str = "row_major"
    strides: tuple[int, ...] | None = None
    offset: int = 0
    align: int = 0

    @property
    def contiguous(self) -> bool:
        return self.strides is None


def strided(strides: tuple[int, ...], *, offset: int = 0, align: int = 0) -> DialectType:
    args: list[int | str] = list(strides)
    if offset:
        args += ["offset", offset]
    if align:
        args += ["align", align]
    return DialectType("layout", "strided", tuple(args))


def describe(t: DialectType) -> Layout:
    """The `Layout` a layout type spells; raises `ValueError` when it spells none."""
    reason = verify_layout(t)
    if reason is not None:
        raise ValueError(reason)
    if t.name in {"row_major", "col_major"}:
        return Layout(order=t.name)
    strides: list[int] = []
    offset = 0
    align = 0
    args = list(t.args)
    while args:
        head = args.pop(0)
        if head == "offset":
            offset = int(args.pop(0))  # type: ignore[arg-type]
        elif head == "align":
            align = int(args.pop(0))  # type: ignore[arg-type]
        else:
            strides.append(int(head))  # type: ignore[arg-type]
    return Layout(order="strided", strides=tuple(strides), offset=offset, align=align)


def verify_layout(t: DialectType) -> str | None:
    if t.dialect != "layout":
        return f"{t} is not a layout"
    if t.name in {"row_major", "col_major"}:
        return None if not t.args else f"{t.name} takes no arguments"
    if t.name != "strided":
        return f"layout defines no type {t.name!r}"
    args = list(t.args)
    seen_option = False
    while args:
        head = args.pop(0)
        if head in {"offset", "align"}:
            seen_option = True
            if not args or not isinstance(args[0], int) or args[0] < 0:
                return f"layout.strided: `{head}` takes a non-negative integer"
            value = args.pop(0)
            if head == "align" and value and (value & (value - 1)):  # type: ignore[operator]
                return "layout.strided: `align` is a power of two"
        elif isinstance(head, int) and not seen_option:
            continue
        else:
            return f"layout.strided takes integer strides, then `offset` and `align`: not {head}"
    return None


class LayoutDialect(Dialect):
    name = "layout"
    version = 1

    def register_operations(self, registry: DialectRegistry) -> None:
        del registry  # a dialect of types alone

    def verify_type(self, t: DialectType) -> str | None:
        return verify_layout(t)
