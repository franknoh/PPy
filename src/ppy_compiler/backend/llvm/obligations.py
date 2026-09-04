"""What an overflow guard would check, as a statement about integers.

A native loop runs its arithmetic on 64-bit words and guards every `+`, `-`,
and `*` that could overflow with a branch back to CPython. A guard that can
be shown never to fire can be left out. The showing is a question over
mathematical integers: given what the program established about the
operands -- the range an annotation declared, the bounds a `range()` gives
its induction variable -- can the true value of the chain lie outside the
machine word? This module states that question; `prover` answers it.

Nothing here imports a solver. An obligation is a small immutable tree that
prints, digests, and compares by content, so it can be tested without one
and cached under its own spelling.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Union

__all__ = [
    "I64_MAX",
    "I64_MIN",
    "BinOp",
    "Const",
    "Obligation",
    "Relation",
    "Term",
    "Var",
    "bounds",
    "fits_by_intervals",
    "spell",
    "variables",
]

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class Var:
    """One loaded value, with the range the analysis proved for it, if any."""

    name: str
    low: int | None = None
    high: int | None = None


@dataclass(frozen=True, slots=True)
class Const:
    value: int


@dataclass(frozen=True, slots=True)
class BinOp:
    """`left op right` over mathematical integers: no wrap, no width."""

    op: str
    left: Term
    right: Term


Term = Union[Var, Const, BinOp]  # noqa: UP007 - a runtime alias, used in isinstance checks


@dataclass(frozen=True, slots=True)
class Relation:
    """`left op right`, a fact the program established about its values."""

    left: Term
    op: str
    right: Term


@dataclass(frozen=True, slots=True)
class Obligation:
    """Show that `goal` fits a signed 64-bit word whenever `hypotheses` hold."""

    goal: Term
    hypotheses: tuple[Relation, ...] = ()

    def digest(self) -> str:
        text = spell(self.goal) + "\n" + "\n".join(_spell_relation(r) for r in self.hypotheses)
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()

    def __str__(self) -> str:
        if not self.hypotheses:
            return spell(self.goal)
        return f"{spell(self.goal)} given {', '.join(_spell_relation(r) for r in self.hypotheses)}"


def spell(term: Term) -> str:
    """The term as the program would write it, bounds beside each variable."""
    if isinstance(term, Const):
        return str(term.value)
    if isinstance(term, Var):
        if term.low is None and term.high is None:
            return term.name
        low = "-inf" if term.low is None else str(term.low)
        high = "+inf" if term.high is None else str(term.high)
        return f"{term.name}:[{low}, {high}]"
    return f"({spell(term.left)} {term.op} {spell(term.right)})"


def _spell_relation(relation: Relation) -> str:
    return f"{spell(relation.left)} {relation.op} {spell(relation.right)}"


def variables(term: Term) -> tuple[Var, ...]:
    """Every variable in the term, in order of first appearance."""
    found: list[Var] = []
    seen: set[str] = set()
    pending = [term]
    while pending:
        node = pending.pop()
        if isinstance(node, Var):
            if node.name not in seen:
                seen.add(node.name)
                found.append(node)
        elif isinstance(node, BinOp):
            pending.append(node.right)
            pending.append(node.left)
    return tuple(found)


def bounds(term: Term) -> tuple[int | None, int | None]:
    """The interval the term lies in, from the variables' own ranges alone.

    Interval arithmetic: the same reasoning the hoisted guard runs at loop
    entry on the actual corners, done here on the declared ones. It is
    exact for `+` and `-`, and a sound over-approximation for `*`. `None`
    on a side means unbounded on that side.
    """
    if isinstance(term, Const):
        return term.value, term.value
    if isinstance(term, Var):
        return term.low, term.high
    left_low, left_high = bounds(term.left)
    right_low, right_high = bounds(term.right)
    if term.op == "+":
        return _add(left_low, right_low), _add(left_high, right_high)
    if term.op == "-":
        return _sub(left_low, right_high), _sub(left_high, right_low)
    if term.op == "*":
        corners = [_mul(a, b) for a in (left_low, left_high) for b in (right_low, right_high)]
        if any(c is None for c in corners):
            return None, None
        known = [c for c in corners if c is not None]
        return min(known), max(known)
    return None, None


def fits_by_intervals(term: Term) -> bool:
    """Does the term provably fit a signed 64-bit word by intervals alone?

    The cheap answer, asked before any solver: a chain of annotated ranges
    that fits by interval arithmetic needs no proof search.
    """
    low, high = bounds(term)
    return low is not None and high is not None and low >= I64_MIN and high <= I64_MAX


def _add(a: int | None, b: int | None) -> int | None:
    return None if a is None or b is None else a + b


def _sub(a: int | None, b: int | None) -> int | None:
    return None if a is None or b is None else a - b


def _mul(a: int | None, b: int | None) -> int | None:
    return None if a is None or b is None else a * b
