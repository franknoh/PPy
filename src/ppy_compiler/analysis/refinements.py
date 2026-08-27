"""Refinements: additional proven facts about a value (spec 10.2)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

__all__ = ["IntRange", "Facts", "UNBOUNDED", "width_range"]


@dataclass(frozen=True, slots=True)
class IntRange:
    low: int | None = None
    high: int | None = None

    def __str__(self) -> str:
        lo = "-inf" if self.low is None else str(self.low)
        hi = "+inf" if self.high is None else str(self.high)
        return f"[{lo}, {hi}]"

    @property
    def is_unbounded(self) -> bool:
        return self.low is None and self.high is None

    def contains(self, other: "IntRange") -> bool:
        if self.low is not None and (other.low is None or other.low < self.low):
            return False
        if self.high is not None and (other.high is None or other.high > self.high):
            return False
        return True

    def join(self, other: "IntRange") -> "IntRange":
        low = None if self.low is None or other.low is None else min(self.low, other.low)
        high = None if self.high is None or other.high is None else max(self.high, other.high)
        return IntRange(low, high)

    def meet(self, other: "IntRange") -> "IntRange":
        low = self.low if other.low is None else (other.low if self.low is None else max(self.low, other.low))
        high = self.high if other.high is None else (other.high if self.high is None else min(self.high, other.high))
        return IntRange(low, high)

    def __add__(self, other: "IntRange") -> "IntRange":
        low = None if self.low is None or other.low is None else self.low + other.low
        high = None if self.high is None or other.high is None else self.high + other.high
        return IntRange(low, high)

    def __sub__(self, other: "IntRange") -> "IntRange":
        low = None if self.low is None or other.high is None else self.low - other.high
        high = None if self.high is None or other.low is None else self.high - other.low
        return IntRange(low, high)

    def __mul__(self, other: "IntRange") -> "IntRange":
        bounds = (self.low, self.high, other.low, other.high)
        if any(b is None for b in bounds):
            return UNBOUNDED
        products = [
            self.low * other.low, self.low * other.high,  # type: ignore[operator]
            self.high * other.low, self.high * other.high,  # type: ignore[operator]
        ]
        return IntRange(min(products), max(products))

    def negate(self) -> "IntRange":
        low = None if self.high is None else -self.high
        high = None if self.low is None else -self.low
        return IntRange(low, high)


UNBOUNDED = IntRange()


def width_range(bits: int, signed: bool) -> IntRange:
    if signed:
        return IntRange(-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    return IntRange(0, (1 << bits) - 1)


@dataclass(frozen=True, slots=True)
class Facts:
    """Proven facts attached to a value or binding."""

    int_range: IntRange | None = None
    length: int | None = None
    constant: Any = None
    has_constant: bool = False
    exact_class: str | None = None
    non_null: bool = False
    contiguous: bool = False
    no_alias: bool = False
    shape: tuple[int | str, ...] | None = None
    dtype: str | None = None
    width: tuple[int, bool] | None = None
    float_bits: int | None = None

    def with_(self, **updates: Any) -> "Facts":
        return replace(self, **updates)

    def merge(self, other: "Facts") -> "Facts":
        """Flow-merge: keep only facts both branches prove."""
        int_range = None
        if self.int_range is not None and other.int_range is not None:
            int_range = self.int_range.join(other.int_range)
        constant, has_constant = None, False
        if self.has_constant and other.has_constant and self.constant == other.constant:
            constant, has_constant = self.constant, True
        return Facts(
            int_range=int_range,
            length=self.length if self.length == other.length else None,
            constant=constant,
            has_constant=has_constant,
            exact_class=self.exact_class if self.exact_class == other.exact_class else None,
            non_null=self.non_null and other.non_null,
            contiguous=self.contiguous and other.contiguous,
            no_alias=self.no_alias and other.no_alias,
            shape=self.shape if self.shape == other.shape else None,
            dtype=self.dtype if self.dtype == other.dtype else None,
            width=self.width if self.width == other.width else None,
            float_bits=self.float_bits if self.float_bits == other.float_bits else None,
        )

    def describe(self) -> list[str]:
        out: list[str] = []
        if self.width is not None:
            bits, signed = self.width
            out.append(f"{'i' if signed else 'u'}{bits}")
        if self.float_bits is not None:
            out.append(f"f{self.float_bits}")
        if self.int_range is not None and not self.int_range.is_unbounded:
            out.append(f"range {self.int_range}")
        if self.length is not None:
            out.append(f"len == {self.length}")
        if self.has_constant:
            out.append(f"constant == {self.constant!r}")
        if self.exact_class:
            out.append(f"exact-class == {self.exact_class}")
        if self.non_null:
            out.append("non-null")
        if self.contiguous:
            out.append("contiguous")
        if self.no_alias:
            out.append("no-alias")
        if self.shape:
            out.append(f"shape == {self.shape}")
        if self.dtype:
            out.append(f"dtype == {self.dtype}")
        return out


EMPTY_FACTS = Facts()
